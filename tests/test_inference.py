"""Text-only inference: front ends on the exact test sentences, and an end-to-end CPU
run (random-weight model saved as a checkpoint -> wav + variance plots) per language."""
import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, UPSTREAM_DIR

VOCODER = os.path.join(UPSTREAM_DIR, "hifigan", "generator_universal.pth.tar")
LEXICON = os.path.join(UPSTREAM_DIR, "lexicon", "librispeech-lexicon.txt")


def _sentences(lang):
    with open(os.path.join(REPO_ROOT, "project", lang, "test_sentences.json"), encoding="utf-8") as f:
        return {s["id"]: s["text"] for s in json.load(f)["sentences"]}


@pytest.fixture(scope="module")
def english():
    from fs2ra.inference import EnglishFrontEnd, text_to_phones

    fe = EnglishFrontEnd(LEXICON, os.path.join(REPO_ROOT, "project", "English", "pronunciation_variants.json"))
    return {k: text_to_phones(v, fe) for k, v in _sentences("English").items()}


def _pron(result, word):
    return next(w["phones"] for w in result[1] if w["word"] == word)


def test_conduct_noun_vs_verb(english):
    assert _pron(english["E2a"], "conduct") == "K AA1 N D AH0 K T"  # noun: stress on 1st syllable
    assert _pron(english["E2b"], "conduct") == "K AH0 N D AH1 K T"  # verb: reduced 1st, stress on 2nd


def test_variant_seen_in_training(english):
    # the lexicon lists the T-less forms first; training speakers mostly used the T
    assert _pron(english["E1"], "parliamentary") == "P AA2 R L AH0 M EH1 N T ER0 IY0"
    assert _pron(english["E4"], "twenty") == "T W EH1 N T IY0"


def test_e3_spacing_kept_and_e4_pause(english):
    # not spell-corrected; the double space is kept as a pause
    assert english["E3"][0] == ["the rhyth", "m of the orchestra changed unexpectedly"]
    phrases, _, phones = english["E4"]
    assert phrases[0].startswith("the meeting will start at ten forty five a_m on twelve november twenty twenty six")
    assert phones.count("sp") == 1 and phones[-1] != "sp"  # comma -> pause; no trailing pause


def test_telugu_front_end():
    from fs2ra.inference import TeluguFrontEnd, text_to_phones
    from fs2ra.text_te import PHONES

    for sid, text in _sentences("Telugu").items():
        phrases, words, phones = text_to_phones(text, TeluguFrontEnd())
        assert set(phones) <= set(PHONES) | {"sp"}, sid
        assert phones.count("sp") == (1 if sid in ("T3", "T4") else 0), sid  # T3 double space, T4 comma


E2E = r"""
import json, os, sys, torch
sys.path[:0] = [{repo!r}, {up!r}]
from fs2ra.config import load_config, upstream_configs
from fs2ra.symbols import register_language
from fs2ra.inference import synthesize
from model import FastSpeech2
cfg = load_config(os.path.join({repo!r}, "project", {lang!r}, "config.yaml"))
d = {tmp!r}
json.dump({{"pitch": [-2, 6, 150.0, 40.0], "energy": [-1.5, 6, 50.0, 30.0]}}, open(os.path.join(d, "stats.json"), "w"))
json.dump({{"spk_a": 0, "spk_b": 1}}, open(os.path.join(d, "speakers.json"), "w"))
register_language(cfg["language"])
p, m, _ = upstream_configs(cfg)
p["path"]["preprocessed_path"] = d
torch.manual_seed(0)
model = FastSpeech2(p, m)
with torch.no_grad():  # untrained predictors give ~0 frames per phone; aim for ~5
    model.variance_adaptor.duration_predictor.linear_layer.bias.fill_(1.8)
torch.save({{"model": model.state_dict(), "step": 7}}, os.path.join(d, "7.pth.tar"))
out = synthesize(cfg, os.path.join(d, "7.pth.tar"), [{{"id": "X1", "text": {text!r}}}], os.path.join(d, "out"),
                 speaker="spk_b", device=torch.device("cpu"))
print(json.dumps(out[0]["phones"]))
"""


@pytest.mark.skipif(not os.path.exists(VOCODER), reason="universal HiFi-GAN not unpacked (setup_upstream.py)")
@pytest.mark.parametrize("lang, sid", [("English", "E4"), ("Telugu", "T4")])
def test_end_to_end_cpu(tmp_path, lang, sid):
    """In a subprocess: symbol registration mutates upstream's global table."""
    code = E2E.format(repo=REPO_ROOT, up=UPSTREAM_DIR, lang=lang, tmp=str(tmp_path), text=_sentences(lang)[sid])
    run = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                         encoding="utf-8")
    assert run.returncode == 0, run.stderr[-3000:]
    out = tmp_path / "out"
    assert (out / "test_generated_audios" / "X1.wav").stat().st_size > 44
    for kind in ("duration", "pitch", "energy"):
        assert (out / "variance_adaptor_plots" / kind / "X1.png").exists()
    summary = json.loads((out / "inference_summary.json").read_text(encoding="utf-8"))
    s = summary["sentences"][0]
    assert summary["speaker"] == "spk_b" and summary["step"] == 7
    assert len(s["pred_duration_frames"]) == len(s["pred_f0_hz"]) == len(s["phones"])
