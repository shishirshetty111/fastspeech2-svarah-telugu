"""Copy-synthesis check of the vocoders on held-out (validation) mel targets.

For N validation utterances, the ground-truth log-mel (as used in training) is vocoded by
HiFi-GAN universal and by BigVGAN-v2; the output wav is re-analysed with the same STFT
code as preprocessing, and compared with the input mel:
  mel L1   mean |log-mel(vocoded) - log-mel(target)| (lower = more faithful)
  MCD      mel-cepstral distortion in dB over 13 coefficients (c1..c13, DCT of the log-mel)

    python scripts/vocoder_eval.py english --n 40
Writes $FS2_OUTPUT_ROOT/<Lang>/vocoder_eval/{summary.json, <vocoder>/<utt>.wav}.
"""
import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO_ROOT, os.path.join(REPO_ROOT, "third_party", "FastSpeech2")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.fft import dct  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from fs2ra.config import OUTPUT_ROOT, load_config, upstream_configs  # noqa: E402
from fs2ra.sources import DATA_ROOT  # noqa: E402
from fs2ra.vocoder_bigvgan import bigvgan_infer, get_bigvgan  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", choices=["english", "telugu"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    Lang = args.lang.capitalize()

    import audio as Audio  # upstream
    from utils.model import get_vocoder, vocoder_infer

    cfg = load_config(os.path.join(REPO_ROOT, "project", Lang, "config.yaml"))
    p, m, _ = upstream_configs(cfg)
    s, a, mel_c = p["preprocessing"]["stft"], p["preprocessing"]["audio"], p["preprocessing"]["mel"]
    stft = Audio.stft.TacotronSTFT(s["filter_length"], s["hop_length"], s["win_length"], mel_c["n_mel_channels"],
                                   a["sampling_rate"], mel_c["mel_fmin"], mel_c["mel_fmax"])
    dev = torch.device("cpu")
    vocoders = {"hifigan_universal": (get_vocoder(m, dev), lambda x, v: vocoder_infer(x, v, m, p)),
                "bigvgan_v2_fmax8k": (get_bigvgan(p, dev), lambda x, v: bigvgan_infer(x, v, p))}

    pre = os.path.join(DATA_ROOT, args.lang, "preprocessed")
    with open(os.path.join(pre, "val.txt"), encoding="utf-8") as f:
        rows = [l.split("|") for l in f.read().splitlines()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    out_dir = os.path.join(OUTPUT_ROOT, Lang, "vocoder_eval")
    res = {k: {"mel_l1": [], "mcd": []} for k in vocoders}
    for i, (base, spk, *_rest) in enumerate(rows):
        target = np.load(os.path.join(pre, "mel", "%s-mel-%s.npy" % (spk, base))).T  # (80, T)
        x = torch.from_numpy(target).float().unsqueeze(0)
        for name, (voc, infer) in vocoders.items():
            wav = infer(x, voc)[0]
            os.makedirs(os.path.join(out_dir, name), exist_ok=True)
            if i < 6:
                wavfile.write(os.path.join(out_dir, name, base + ".wav"), a["sampling_rate"], wav)
            mel, _ = Audio.tools.get_mel_from_wav(wav.astype(np.float32) / a["max_wav_value"], stft)
            T = min(mel.shape[1], target.shape[1])
            res[name]["mel_l1"].append(float(np.abs(mel[:, :T] - target[:, :T]).mean()))
            c1, c2 = (dct(z[:, :T], type=2, axis=0, norm="ortho")[1:14] for z in (mel, target))
            res[name]["mcd"].append(float((10 / np.log(10)) * np.sqrt(2 * ((c1 - c2) ** 2).sum(0)).mean()))
        print(i, base, {k: round(v["mel_l1"][-1], 3) for k, v in res.items()}, flush=True)
    summary = {"lang": Lang, "n": len(rows), "seed": args.seed, "split": "val",
               "metrics": {k: {mk: [round(float(np.mean(mv)), 4), round(float(np.std(mv)), 4)]
                               for mk, mv in v.items()} for k, v in res.items()},
               "bigvgan_better_mel_l1": int(sum(b < h for b, h in zip(res["bigvgan_v2_fmax8k"]["mel_l1"],
                                                                         res["hifigan_universal"]["mel_l1"])))}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
