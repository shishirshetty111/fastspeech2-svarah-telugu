"""Rank English speakers by how well the trained model renders their voice, to choose the
inference speaker for the test sentences (the model is multi-speaker; the choice changes
nothing in training).

Score per speaker, on that speaker's VALIDATION utterances only (never trained on):
  gta_l1    mean |model mel - real mel| with the real durations/pitch/energy given
            (teacher-forced): how clearly the decoder reproduces that voice;
  text_l1   the same with predicted pitch/energy but real durations (so frames align):
            adds how well the predictors fit the speaker.
Also reported: training minutes per speaker. Speakers with < --min_val val utterances
are not ranked.

    python scripts/speaker_search.py --ckpt project/English/checkpoints/20000.pth.tar
writes $FS2_OUTPUT_ROOT/English/speaker_search.json
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO_ROOT, os.path.join(REPO_ROOT, "third_party", "FastSpeech2")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fs2ra import pipeline  # noqa: E402
from fs2ra.config import OUTPUT_ROOT, load_config  # noqa: E402
from fs2ra.inference import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--min_val", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    cfg = load_config(os.path.join(REPO_ROOT, "project", "English", "config.yaml"))
    p = pipeline.work_dirs(cfg, False)["preprocessed"]
    _, configs, model, _, speakers, _, step = load_model(args.ckpt, cfg, torch.device("cpu"))
    from text import text_to_sequence  # upstream, after register_language
    a = configs[0]["preprocessing"]
    sec_per_frame = a["stft"]["hop_length"] / a["audio"]["sampling_rate"]

    def rows(split):
        with open(os.path.join(p, split + ".txt"), encoding="utf-8") as f:
            return [l.split("|") for l in f.read().splitlines()]

    stats = {}
    for utt, spk, _, _ in rows("train"):
        d = np.load(os.path.join(p, "duration", "%s-duration-%s.npy" % (spk, utt)))
        stats.setdefault(spk, {"train_min": 0.0, "gta_l1": [], "text_l1": []})["train_min"] += d.sum() * sec_per_frame / 60
    for k, (utt, spk, ph, _) in enumerate(rows("val")):
        ld = lambda n: np.load(os.path.join(p, n, "%s-%s-%s.npy" % (spk, n, utt)))  # noqa: E731
        mel, pitch, energy, dur = ld("mel"), ld("pitch"), ld("energy"), ld("duration")
        seq = torch.from_numpy(np.array(text_to_sequence(ph, []))).long().unsqueeze(0)
        T = int(dur.sum())
        base = (torch.tensor([speakers[spk]]), seq, torch.tensor([seq.shape[1]]), seq.shape[1],
                torch.from_numpy(mel).float().unsqueeze(0), torch.tensor([T]), T)
        t = lambda x: torch.from_numpy(x).float().unsqueeze(0)  # noqa: E731
        with torch.no_grad():
            gta = model(*base, t(pitch), t(energy), torch.from_numpy(dur).long().unsqueeze(0))[1][0].numpy()
            txt = model(*base, None, None, torch.from_numpy(dur).long().unsqueeze(0))[1][0].numpy()
        s = stats[spk]
        s["gta_l1"].append(float(np.abs(gta - mel).mean()))
        s["text_l1"].append(float(np.abs(txt - mel).mean()))
        if k % 100 == 0:
            print("val", k, flush=True)

    table = []
    for spk, s in stats.items():
        if len(s["gta_l1"]) >= args.min_val:
            table.append({"speaker": spk, "train_min": round(s["train_min"], 2), "n_val": len(s["gta_l1"]),
                          "gta_l1": round(float(np.mean(s["gta_l1"])), 4),
                          "text_l1": round(float(np.mean(s["text_l1"])), 4)})
    table.sort(key=lambda r: r["gta_l1"])
    for i, r in enumerate(sorted(table, key=lambda r: r["text_l1"])):
        r["text_rank"] = i + 1
    out = os.path.join(OUTPUT_ROOT, "English", "speaker_search.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": os.path.relpath(args.ckpt, REPO_ROOT).replace("\\", "/"), "step": step,
                   "ranked_by": "gta_l1 (validation, teacher-forced)", "speakers": table}, f, indent=1)
    cur = cfg["inference"]["speaker"]
    print("%d speakers ranked (>= %d val utts); current %s" % (len(table), args.min_val, cur))
    print("%-4s %-34s %8s %5s %8s %8s %5s" % ("rank", "speaker", "trainmin", "nval", "gta_l1", "text_l1", "trank"))
    for i, r in enumerate(table, 1):
        if i <= 15 or r["speaker"] == cur:
            print("%-4d %-34s %8.2f %5d %8.4f %8.4f %5d" % (i, r["speaker"], r["train_min"], r["n_val"],
                                                          r["gta_l1"], r["text_l1"], r["text_rank"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
