"""Export ground-truth-aligned (GTA, teacher-forced) mels + matching audio for
fine-tuning the HiFi-GAN vocoder on the acoustic model's own output (experiment).

For every train/val utterance of the preprocessed set:
  mels/<utt>.npy  (80, T) float32: the model's PostNet mel with ground-truth durations,
                  pitch and energy (eval mode) -- what the vocoder sees at inference,
                  minus the variance-prediction errors.
  wavs/<utt>.wav  int16 22.05 kHz: the same audio span the target mel was computed from
                  (upstream process_utterance: TextGrid-trimmed, first sum(durations)
                  frames), padded/cut to exactly T * hop samples so that frame i <-> samples
                  [i*hop, (i+1)*hop) as jik876/hifi-gan's fine-tuning loader assumes.
  training.txt / validation.txt   jik876 filelists ("<utt>|").

    python scripts/export_gta.py english --ckpt project/English/checkpoints/20000.pth.tar
"""
import argparse
import os
import random
import sys
import tarfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO_ROOT, os.path.join(REPO_ROOT, "third_party", "FastSpeech2")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from fs2ra import pipeline  # noqa: E402
from fs2ra.config import OUTPUT_ROOT, load_config  # noqa: E402
from fs2ra.inference import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", choices=["english", "telugu"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_val", type=int, default=100, help="validation utterances listed for HiFi-GAN validation")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N per split")
    ap.add_argument("--tar", default=None, help="default dist/<lang>_gta.tar")
    args = ap.parse_args()
    Lang = args.lang.capitalize()
    torch.set_num_threads(os.cpu_count() or 4)

    cfg = load_config(os.path.join(REPO_ROOT, "project", Lang, "config.yaml"))
    dirs = pipeline.work_dirs(cfg, False)
    pre_cfg = dict(cfg["preprocess"], path=dict(cfg["preprocess"]["path"], raw_path=dirs["corpus"],
                                                preprocessed_path=dirs["preprocessed"]))
    from preprocessor.preprocessor import Preprocessor  # upstream
    import librosa
    import tgt

    pre = Preprocessor(pre_cfg)
    _, configs, model, _, speakers, _, step = load_model(args.ckpt, cfg, torch.device("cpu"))
    from text import text_to_sequence  # upstream, after register_language (load_model)

    sr, hop = pre.sampling_rate, pre.hop_length
    p = dirs["preprocessed"]
    out = os.path.join(OUTPUT_ROOT, Lang, "gta_step%s" % step)
    os.makedirs(os.path.join(out, "mels"), exist_ok=True)
    os.makedirs(os.path.join(out, "wavs"), exist_ok=True)
    lists, l1 = {}, []
    for split in ("train", "val"):
        with open(os.path.join(p, split + ".txt"), encoding="utf-8") as f:
            rows = [l.split("|") for l in f.read().splitlines()]
        if args.limit:
            rows = rows[: args.limit]
        lists[split] = [r[0] for r in rows]
        for i, (base, spk, phones, _raw) in enumerate(rows):
            ld = lambda k: np.load(os.path.join(p, k, "%s-%s-%s.npy" % (spk, k, base)))  # noqa: E731
            mel_t, pitch, energy, dur = ld("mel"), ld("pitch"), ld("energy"), ld("duration")
            T = int(dur.sum())
            assert mel_t.shape[0] == T, base
            # the audio span of the target mel, exactly as upstream process_utterance
            tg = tgt.io.read_textgrid(os.path.join(p, "TextGrid", spk, base + ".TextGrid"))
            _, d2, start, end = pre.get_alignment(tg.get_tier_by_name("phones"))
            assert int(sum(d2)) == T, base
            wav, _ = librosa.load(os.path.join(dirs["corpus"], spk, base + ".wav"), sr=sr)
            wav = wav[int(sr * start): int(sr * end)][: T * hop]
            wav = np.pad(wav, (0, T * hop - len(wav)))
            seq = np.array(text_to_sequence(phones, []))
            with torch.no_grad():
                o = model(torch.tensor([speakers[spk]]), torch.from_numpy(seq).long().unsqueeze(0),
                          torch.tensor([len(seq)]), len(seq),
                          torch.from_numpy(mel_t).float().unsqueeze(0), torch.tensor([T]), T,
                          torch.from_numpy(pitch).float().unsqueeze(0),
                          torch.from_numpy(energy).float().unsqueeze(0),
                          torch.from_numpy(dur).long().unsqueeze(0))
            gta = o[1][0].numpy()  # PostNet output (T, 80)
            assert gta.shape == mel_t.shape, (base, gta.shape, mel_t.shape)
            l1.append(float(np.abs(gta - mel_t).mean()))
            np.save(os.path.join(out, "mels", base + ".npy"), gta.T.astype(np.float32))
            wavfile.write(os.path.join(out, "wavs", base + ".wav"), sr,
                          (np.clip(wav, -1, 1) * 32767).astype(np.int16))
            if i % 500 == 0:
                print(split, i, len(rows), "GTA L1 %.3f" % np.mean(l1), flush=True)

    val = list(lists["val"])
    random.Random(0).shuffle(val)
    with open(os.path.join(out, "training.txt"), "w") as f:
        f.writelines(u + "|\n" for u in lists["train"])
    with open(os.path.join(out, "validation.txt"), "w") as f:
        f.writelines(u + "|\n" for u in val[: args.n_val])
    print("GTA mel L1 vs target (all): %.4f over %d utterances" % (np.mean(l1), len(l1)))

    tar_path = args.tar or os.path.join(REPO_ROOT, "dist", "%s_gta.tar" % args.lang)
    with tarfile.open(tar_path, "w") as t:
        t.add(out, arcname="gta")
    print("wrote", tar_path, "%.1f MB" % (os.path.getsize(tar_path) / 2**20))


if __name__ == "__main__":
    main()
