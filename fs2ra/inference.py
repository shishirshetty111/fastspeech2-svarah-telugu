"""Text-only FastSpeech 2 synthesis (project/<Lang>/inference.py).

Unlike training and the teacher-forced validation samples, nothing here comes from
recorded audio: the phone sequence comes from the text front end, and duration, pitch
and energy are *predicted* by the variance adaptor from text (+ speaker) only.

The checkpoint's folder must hold the train-time stats.json / speakers.json / config.yaml
snapshot written by fs2ra.training (a milestone in ckpt/milestones/ uses its parent's).

Outputs (Results layout from the assignment PDF):
    <out_dir>/test_generated_audios/<id>.wav
    <out_dir>/variance_adaptor_plots/{duration,pitch,energy}/<id>.png
    <out_dir>/inference_summary.json   raw + spoken text, phones, predicted values
"""
import json
import os
import re

import numpy as np
import torch
from scipy.io import wavfile

from fs2ra.config import load_config, upstream_configs
from fs2ra.hf import REPO_ROOT
from fs2ra.symbols import check_phone_strings, register_language

# Phrase-final punctuation becomes a pause ("sp"), as in upstream synthesize.py; "10:45"
# and "3.5" are not split. A run of 2+ spaces between words (E3 "rhyth  m", T3 "అడ  విలో")
# is kept as a pause too, so the given spacing is audible rather than silently merged.
# No trailing "sp": training phone strings are trimmed of leading/trailing silence.
PHRASE_RE = re.compile(r"[,;.?!।]+(?=\s|$)|(?<=\S)[ 	]{2,}(?=\S)")


def _primary_stress(pron):
    vowels = [p for p in pron if p[-1].isdigit()]
    return next((i for i, v in enumerate(vowels) if v.endswith("1")), None)


class EnglishFrontEnd:
    """normalize_english + LibriSpeech lexicon / g2p_en, as for the training dictionary.

    Words with several lexicon pronunciations take the one MFA aligned most often in the
    training split (pronunciation_variants.json; e.g. TWENTY with its T, PARLIAMENTARY
    with its T), else the longest, as the lexicon often lists a reduced form first.
    Words in g2p_en's homograph table (e.g. "conduct") are disambiguated by an NLTK
    part-of-speech tag, as g2p_en does, then mapped to the lexicon variant with the same
    primary-stress position (noun K AA1 N D AH0 K T, verb K AH0 N D AH1 K T)."""

    def __init__(self, lexicon_path, variants_path=None):
        from fs2ra.text_en import EnglishG2P

        self.g2p = EnglishG2P(lexicon_path)
        import nltk

        try:  # nltk >= 3.8.2 tags with this resource; g2p_en only fetches the older one
            nltk.data.find("taggers/averaged_perceptron_tagger_eng/")
        except LookupError:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        self.variants = {}
        if variants_path and os.path.exists(variants_path):
            with open(variants_path, encoding="utf-8") as f:
                self.variants = json.load(f)

    def normalize(self, text):
        from fs2ra.text_en import normalize_english

        return normalize_english(text)

    def _choose(self, word, prons):
        seen = self.variants.get(word, {})
        best = max(prons, key=lambda p: (seen.get(" ".join(p), 0), len(p)))  # max: first on ties
        return best, "lexicon:train-aligned" if seen.get(" ".join(best)) else "lexicon:longest"

    def words(self, words):
        from nltk import pos_tag

        from fs2ra.text_en import EXTRA_PRONUNCIATIONS

        out = []
        for word, tag in pos_tag(words):
            prons, source = self.g2p.pronunciations(word)
            homograph = self.g2p._g2p.homograph2features.get(word)
            if homograph and word not in EXTRA_PRONUNCIATIONS:
                pron1, pron2, pos1 = homograph
                target = pron1 if tag.startswith(pos1) else pron2
                same = [p for p in prons if source == "lexicon" and _primary_stress(p) == _primary_stress(target)]
                pron = self._choose(word, same)[0] if same else target
                out.append((word, pron, "homograph(%s)" % tag))
            elif source == "lexicon" and len(prons) > 1:
                out.append((word, *self._choose(word, prons)))
            else:
                out.append((word, prons[0], source))
        return out


class TeluguFrontEnd:
    """normalize_telugu + rule-based grapheme-to-phoneme (same as the MFA dictionary)."""

    def normalize(self, text):
        from fs2ra.text_te import normalize_telugu

        return normalize_telugu(text)

    def words(self, words):
        from fs2ra.text_te import g2p

        return [(w, g2p(w), "rules") for w in words]


def front_end_for(cfg, inference_cfg=None):
    """inference_cfg: the project config's `inference` section (the checkpoint's config
    snapshot may predate it)."""
    if cfg["language"] == "english":
        return EnglishFrontEnd(cfg["preprocess"]["path"]["lexicon_path"],
                               (inference_cfg or {}).get("pronunciation_variants"))
    if cfg["language"] == "telugu":
        return TeluguFrontEnd()
    raise ValueError(cfg["language"])


def text_to_phones(raw, front_end, sp_kinds=None):
    """Raw text -> (spoken phrases, [(word, phones, source)], phones with "sp" between phrases).
    sp_kinds (optional list) receives, per "sp", "punct" (phrase punctuation) or "space"."""
    phrases, words, phones = [], [], []
    pos, pending = 0, ""
    for m in list(PHRASE_RE.finditer(raw)) + [None]:
        chunk = raw[pos: m.start() if m else len(raw)]
        spoken = front_end.normalize(chunk).split()
        if spoken:
            if phones:
                phones.append("sp")
                if sp_kinds is not None:
                    sp_kinds.append("punct" if pending.strip() else "space")
            phrases.append(" ".join(spoken))
            for word, pron, source in front_end.words(spoken):
                words.append({"word": word, "phones": " ".join(pron), "source": source})
                phones += pron
            pending = ""
        if m:
            pending += m.group()
            pos = m.end()
    return phrases, words, phones


def metadata_dir(ckpt_path):
    here = os.path.dirname(os.path.abspath(ckpt_path))
    for d in (here, os.path.dirname(here)):
        if all(os.path.exists(os.path.join(d, n)) for n in ("stats.json", "speakers.json")):
            return d
    raise FileNotFoundError("stats.json/speakers.json not found next to %s or in its parent folder "
                            "(copy them from the training ckpt folder)" % ckpt_path)


def load_model(ckpt_path, project_cfg, device):
    """Model built from the checkpoint folder's config snapshot (if present) and its
    train-time stats.json / speakers.json, so inference matches training."""
    from model import FastSpeech2  # upstream
    from utils.model import get_vocoder, load_checkpoint

    meta = metadata_dir(ckpt_path)
    snapshot = os.path.join(meta, "config.yaml")
    cfg = load_config(snapshot) if os.path.exists(snapshot) else project_cfg
    if cfg["language"] != project_cfg["language"]:
        raise ValueError("checkpoint is for %s, not %s" % (cfg["language"], project_cfg["language"]))
    register_language(cfg["language"])  # before the model is built (embedding size)
    configs = upstream_configs(cfg)
    configs[0]["path"]["preprocessed_path"] = meta  # stats.json / speakers.json
    model = FastSpeech2(configs[0], configs[1]).to(device)
    state = load_checkpoint(ckpt_path, device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    with open(os.path.join(meta, "speakers.json"), encoding="utf-8") as f:
        speakers = json.load(f)
    with open(os.path.join(meta, "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    return cfg, configs, model, get_vocoder(configs[1], device), speakers, stats, state.get("step")


# Fonts with Telugu glyphs (DejaVu Sans has none -> empty boxes in plot titles).
TELUGU_FONTS = ["C:/Windows/Fonts/Nirmala.ttc", "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansTelugu-Regular.ttf",
                "/System/Library/Fonts/Supplemental/Telugu Sangam MN.ttc"]
_title_family = None


def title_font_family():
    """['DejaVu Sans', <first available Telugu-capable font>] as a matplotlib fallback list."""
    global _title_family
    if _title_family is None:
        from matplotlib import font_manager

        _title_family = ["DejaVu Sans"]
        for path in TELUGU_FONTS:
            if os.path.exists(path):
                font_manager.fontManager.addfont(path)  # may already be known from the system scan
                names = sorted({f.name for f in font_manager.fontManager.ttflist
                                if os.path.normcase(os.path.abspath(f.fname)) == os.path.normcase(os.path.abspath(path))})
                name = "Nirmala UI" if "Nirmala UI" in names else (names[0] if names else None)
                if name:
                    _title_family.append(name)
                    break
    return _title_family


def plot_variance(kind, sid, raw, phones, frames, values, hop_s, path):
    """One predicted value per phone; one slot per phone (phones predicted to last 0
    frames stay visible and are marked "x")."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(phones))
    fig, ax = plt.subplots(figsize=(max(10, 0.16 * len(phones)), 3.6))
    if kind == "duration":
        ax.bar(x, frames, width=0.8, color="#4C78A8")
        ax.set_ylabel("predicted duration (frames)\n1 frame = %.1f ms" % (1000 * hop_s))
    else:
        ax.stairs(values, np.arange(len(phones) + 1) - 0.5, color="#E45756" if kind == "pitch" else "#54A24B",
                  lw=1.8, baseline=None)
        ax.set_ylabel("predicted F0 (Hz)" if kind == "pitch" else "predicted energy\n(mean frame L2 norm)")
        ax.set_ylim(min(0, values.min()), values.max() * 1.1 + 1e-6)
    ax.set_xticks(x)
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ax.set_xlim(-0.5, len(phones) - 0.5)
    for i, (p, f) in enumerate(zip(phones, frames)):
        if p == "sp":
            ax.axvspan(i - 0.5, i + 0.5, color="0.92", zorder=0)
        if f == 0:
            ax.plot(i, 0, "x", color="0.4", ms=4, clip_on=False)
    ax.set_title("%s: %s" % (sid, raw), fontsize=9, fontfamily=title_font_family())
    ax.set_xlabel("phones (grey band = pause \"sp\"; x = 0 frames)   predicted length %.2f s"
                  % (frames.sum() * hop_s), fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


VOCODERS = ("hifigan", "bigvgan")


def synthesize(project_cfg, ckpt_path, sentences, out_dir, speaker=None, device=None, pace=1.0, vocoder=None):
    """sentences: [{"id", "text"}] raw text. Returns the per-sentence summary list.
    pace: FastSpeech 2 duration control (upstream d_control): every predicted phone
    duration is multiplied by it before rounding; >1 = slower speech, 1.0 = as predicted.
    vocoder: "hifigan" (upstream universal HiFi-GAN, default) or "bigvgan" (fs2ra.vocoder_bigvgan);
    default from the config's inference.vocoder."""
    from utils.model import vocoder_infer  # upstream

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, configs, model, hifigan, speakers, stats, step = load_model(ckpt_path, project_cfg, device)
    preprocess_config, model_config, _ = configs
    vocoder_name = vocoder or project_cfg.get("inference", {}).get("vocoder", "hifigan")
    if vocoder_name not in VOCODERS:
        raise ValueError("vocoder must be one of %s" % (VOCODERS,))
    if vocoder_name == "bigvgan":
        from fs2ra.vocoder_bigvgan import bigvgan_infer, get_bigvgan

        voc = get_bigvgan(preprocess_config, device)

        def vocode(mel, n):
            return bigvgan_infer(mel, voc, preprocess_config, lengths=[n])[0]
    else:
        def vocode(mel, n):
            return vocoder_infer(mel, hifigan, model_config, preprocess_config, lengths=[n])[0]
    if configs[1]["multi_speaker"]:
        speaker = speaker or project_cfg.get("inference", {}).get("speaker")
        if speaker not in speakers:
            raise ValueError("speaker %r not in speakers.json (%d speakers)" % (speaker, len(speakers)))
    else:
        speaker = speaker or next(iter(speakers))
    speaker_id = speakers.get(speaker, 0)
    audio_cfg = preprocess_config["preprocessing"]
    sr, hop = audio_cfg["audio"]["sampling_rate"], audio_cfg["stft"]["hop_length"]
    _, _, p_mean, p_std = stats["pitch"]
    _, _, e_mean, e_std = stats["energy"]
    front_end = front_end_for(cfg, project_cfg.get("inference"))

    from text import text_to_sequence  # upstream, after register_language

    split = bool(project_cfg.get("inference", {}).get("split_at_punctuation"))

    def run(phone_list, sid):
        """One forward pass, no targets: d/p/e are the predictors' outputs (text-only)."""
        phone_str = "{" + " ".join(phone_list) + "}"
        check_phone_strings([phone_str], cfg["language"])
        seq = np.array(text_to_sequence(phone_str, []))
        with torch.no_grad():
            out = model(
                torch.tensor([speaker_id], device=device),
                torch.from_numpy(seq).long().unsqueeze(0).to(device),
                torch.tensor([len(seq)], device=device),
                len(seq),
                d_control=pace,
            )
        mel_len = int(out[9][0])
        frames = out[5][0, : len(seq)].cpu().numpy().astype(int)
        pitch_hz = out[2][0, : len(seq)].float().cpu().numpy() * p_std + p_mean
        energy = out[3][0, : len(seq)].float().cpu().numpy() * e_std + e_mean
        if frames.sum() != mel_len or mel_len == 0 or not np.isfinite(pitch_hz).all():
            raise RuntimeError("%s: inconsistent prediction (sum durations %d, mel %d)"
                               % (sid, frames.sum(), mel_len))
        wav = vocode(out[1].transpose(1, 2), mel_len * hop)
        return frames, pitch_hz, energy, wav

    summary = []
    for s in sentences:
        sp_kinds = []
        phrases, words, phones = text_to_phones(s["text"], front_end, sp_kinds)
        frames, pitch_hz, energy, wav = run(phones, s["id"])
        sp_idx = [i for i, p in enumerate(phones) if p == "sp"]
        cuts = [i for i, k in zip(sp_idx, sp_kinds) if k == "punct"] if split else []
        if cuts:
            # Phrase-level synthesis: each punctuation-delimited phrase is its own forward
            # pass, joined by digital silence as long as the pause the model predicted for
            # that "sp" in the whole-sentence pass. (In long sentences the decoder can voice
            # a predicted pause, e.g. Telugu T4's comma, although every training pause is silent.)
            parts_f, parts_p, parts_e, parts_w = [], [], [], []
            for a, b in zip([-1] + cuts, cuts + [len(phones)]):
                f, p, e, w = run(phones[a + 1: b], s["id"])
                parts_f.append(f), parts_p.append(p), parts_e.append(e), parts_w.append(w)
                if b < len(phones):  # the pause itself, from the whole-sentence prediction
                    parts_f.append(frames[b:b + 1]), parts_p.append(pitch_hz[b:b + 1])
                    parts_e.append(energy[b:b + 1])
                    parts_w.append(np.zeros(int(frames[b]) * hop, dtype=w.dtype))
            frames, pitch_hz, energy = (np.concatenate(x) for x in (parts_f, parts_p, parts_e))
            wav = np.concatenate(parts_w)
            assert len(frames) == len(phones) and len(wav) == frames.sum() * hop

        wav_path = os.path.join(out_dir, "test_generated_audios", s["id"] + ".wav")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        wavfile.write(wav_path, sr, wav)
        for kind, values in (("duration", frames), ("pitch", pitch_hz), ("energy", energy)):
            plot_variance(kind, s["id"], s["text"], phones, frames, values, hop / sr,
                          os.path.join(out_dir, "variance_adaptor_plots", kind, s["id"] + ".png"))
        summary.append({
            "id": s["id"], "raw_text": s["text"], "spoken_phrases": phrases, "words": words,
            "phones": phones, "pred_duration_frames": frames.tolist(),
            "pred_f0_hz": np.round(pitch_hz, 2).tolist(), "pred_energy": np.round(energy, 3).tolist(),
            "seconds": round(len(wav) / sr, 3), "wav": os.path.relpath(wav_path, out_dir).replace("\\", "/"),
        })
        print("%-4s %5.2f s  %3d phones  F0 %.0f-%.0f Hz  %s" % (
            s["id"], len(wav) / sr, len(phones), pitch_hz.min(), pitch_hz.max(), wav_path))

    ckpt_abs = os.path.abspath(ckpt_path)
    try:  # repo-relative when inside the repo (relpath raises across Windows drives)
        ckpt_rel = os.path.relpath(ckpt_abs, REPO_ROOT)
    except ValueError:
        ckpt_rel = ".."
    with open(os.path.join(out_dir, "inference_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": ckpt_abs if ckpt_rel.startswith("..") else ckpt_rel.replace("\\", "/"),
                   "step": step, "speaker": speaker, "pace": pace, "split_at_punctuation": split,
                   "vocoder": vocoder_name,
                   "device": str(device),
                   "sentences": summary},
                  f, ensure_ascii=False, indent=1)
    return summary
