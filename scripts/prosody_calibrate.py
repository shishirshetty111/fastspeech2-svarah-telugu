"""Calibrate and evaluate the English prosody post-rules (fs2ra/prosody.py) from data.

Amounts come from data only, never from the test sentences:
  * stressed-vowel floors and the split-word pause floor: real durations, TRAIN split;
  * ratios real/predicted (pitch range, stress accent, phrase-final lengthening and fall):
    text-only predictions vs real targets on the CALIBRATION half of the validation split.
The rules are then scored on the other (EVALUATION) half, which plays no part in setting them.
Halves: deterministic, by a hash of the utterance id.

    python scripts/prosody_calibrate.py --ckpt project/English/checkpoints/20000.pth.tar
writes project/English/prosody_rules.json and output/English/prosody_eval.json
"""
import argparse
import hashlib
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
from fs2ra.prosody import SP, apply_rules, base, final_vowels, is_vowel  # noqa: E402


def rows_of(p, split):
    with open(os.path.join(p, split + ".txt"), encoding="utf-8") as f:
        return [l.split("|") for l in f.read().splitlines()]


def real_targets(p, utt, spk, p_mean, p_std, e_mean, e_std):
    ld = lambda k: np.load(os.path.join(p, k, "%s-%s-%s.npy" % (spk, k, utt)))  # noqa: E731
    return ld("duration").astype(int), ld("pitch") * p_std + p_mean, ld("energy") * e_std + e_mean


def half(utt):
    return "calib" if int(hashlib.md5(utt.encode()).hexdigest(), 16) % 2 == 0 else "eval"


def zstats(phones, f0):
    v = np.array([x for ph, x in zip(phones, f0) if ph != SP])
    return v.mean(), v.std() or 1.0


def features(phones, d, f0):
    """Per-utterance prosody descriptors used both to calibrate and to evaluate."""
    m, s = zstats(phones, f0)
    vow = [i for i, p in enumerate(phones) if is_vowel(p)]
    st = [i for i in vow if phones[i].endswith("1")]
    un = [i for i in vow if not phones[i].endswith("1")]
    fin = set(final_vowels(phones))
    nonfin = [i for i in vow if i not in fin]
    z = (np.asarray(f0, float) - m) / s
    return {
        "f0_sd": s,
        "stress_gap_z": (z[st].mean() - z[un].mean()) if st and un else np.nan,
        "final_z": z[sorted(fin)].mean() if fin else np.nan,
        "final_dur": np.mean([d[i] for i in fin]) if fin else np.nan,
        "nonfinal_dur": np.mean([d[i] for i in nonfin]) if nonfin else np.nan,
        "stressed_dur": [d[i] for i in st],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "project", "English", "prosody_rules.json"))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N val utterances")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    cfg = load_config(os.path.join(REPO_ROOT, "project", "English", "config.yaml"))
    p = pipeline.work_dirs(cfg, False)["preprocessed"]
    _, configs, model, _, speakers, stats, step = load_model(args.ckpt, cfg, torch.device("cpu"))
    from text import text_to_sequence  # upstream, after register_language
    _, _, p_mean, p_std = stats["pitch"]
    _, _, e_mean, e_std = stats["energy"]

    # ---- train split: real stressed-vowel durations and mid-sentence pauses
    by_vowel, pauses = {}, []
    for utt, spk, ph, _ in rows_of(p, "train"):
        phones = ph.strip("{}").split()
        d, _, _ = real_targets(p, utt, spk, p_mean, p_std, e_mean, e_std)
        for i, x in enumerate(phones):
            if x.endswith("1"):
                by_vowel.setdefault(base(x), []).append(int(d[i]))
            elif x == SP and 0 < i < len(phones) - 1:
                pauses.append(int(d[i]))
    floors = {v: int(np.percentile(x, 10)) for v, x in sorted(by_vowel.items()) if len(x) >= 20}

    # ---- validation split: text-only predictions vs real
    val = rows_of(p, "val")[: args.limit or None]
    items = []
    for utt, spk, ph, raw in val:
        phones = ph.strip("{}").split()
        seq = np.array(text_to_sequence(ph, []))
        with torch.no_grad():
            o = model(torch.tensor([speakers[spk]]), torch.from_numpy(seq).long().unsqueeze(0),
                      torch.tensor([len(seq)]), len(seq))
        pd_ = o[5][0].numpy().astype(int)
        pf = o[2][0].numpy() * p_std + p_mean
        pe = o[3][0].numpy() * e_std + e_mean
        logd = o[4][0].numpy()  # log duration prediction; frames = round((exp - 1) * d_control)
        rd, rf, re_ = real_targets(p, utt, spk, p_mean, p_std, e_mean, e_std)
        items.append(dict(utt=utt, half=half(utt), phones=phones, pred=(pd_, pf, pe), logd=logd, real=(rd, rf, re_)))
    print("val utterances: %d (calib %d, eval %d)" % (
        len(items), sum(i["half"] == "calib" for i in items), sum(i["half"] == "eval" for i in items)))

    def med(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return float(np.median(xs))

    cal = [i for i in items if i["half"] == "calib"]
    fr = [features(i["phones"], i["real"][0], i["real"][1]) for i in cal]
    fp = [features(i["phones"], i["pred"][0], i["pred"][1]) for i in cal]
    pitch_scale = med([a["f0_sd"] / b["f0_sd"] for a, b in zip(fr, fp)])
    stress_gap = med([a["stress_gap_z"] - b["stress_gap_z"] for a, b in zip(fr, fp)])
    final_drop = -med([a["final_z"] - b["final_z"] for a, b in zip(fr, fp)])
    # final lengthening relative to the same utterance's non-final vowels (rate-independent)
    final_len = med([(a["final_dur"] / a["nonfinal_dur"]) / (b["final_dur"] / b["nonfinal_dur"])
                     for a, b in zip(fr, fp) if b["nonfinal_dur"] > 0 and b["final_dur"] > 0])
    # E: overall pace, so that predicted utterance length matches real on average
    dur_scale = 1.0 / med([i["pred"][0].sum() / max(i["real"][0].sum(), 1) for i in cal])
    rules = {
        "source": {"checkpoint": os.path.relpath(args.ckpt, REPO_ROOT).replace("\\", "/"), "step": step,
                   "train_utterances": len(rows_of(p, "train")), "calibration_utterances": len(cal),
                   "halves": "md5(utterance id) even = calibration, odd = evaluation"},
        "stress_floor_frames": floors,
        "pitch_scale": round(max(1.0, pitch_scale), 3),
        "stress_accent_sd": round(max(0.0, stress_gap), 3),
        "final_lengthen": round(max(1.0, final_len), 3),
        "final_drop_sd": round(max(0.0, final_drop), 3),
        "space_pause_min_frames": int(np.median(pauses)),
        "duration_scale": round(dur_scale, 3),
        "raw_estimates": {"pitch_scale": pitch_scale, "stress_gap_z": stress_gap, "final_drop_z": final_drop,
                          "final_lengthen": final_len, "duration_scale": dur_scale,
                          "n_train_pauses": len(pauses)},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=1)
    print(json.dumps({k: v for k, v in rules.items() if k != "stress_floor_frames"}, indent=1))

    # ---- evaluation half: does each rule move predictions towards real speech?
    ev = [i for i in items if i["half"] == "eval"]

    def score(variant):
        out = {k: [] for k in ("f0_sd_ratio", "stress_gap_err", "final_z_err", "final_len_ratio",
                               "stressed_short", "logdur_rmse", "f0_rmse_hz", "dur_bias")}
        for i in ev:
            ph, (pd_, pf, pe), (rd, rf, _) = i["phones"], i["pred"], i["real"]
            if "E" in variant:  # same as inference: d_control applied before rounding
                pd_ = np.clip(np.round((np.exp(i["logd"]) - 1) * rules["duration_scale"]), 0, None).astype(int)
            if variant:
                pd_, pf, pe, _ = apply_rules(ph, pd_, pf, pe, rules, enabled=variant)
            a, b = features(ph, rd, rf), features(ph, pd_, pf)
            out["f0_sd_ratio"].append(b["f0_sd"] / a["f0_sd"])
            out["stress_gap_err"].append(abs(b["stress_gap_z"] - a["stress_gap_z"]))
            out["final_z_err"].append(abs(b["final_z"] - a["final_z"]))
            if b["nonfinal_dur"] > 0 and a["nonfinal_dur"] > 0:
                out["final_len_ratio"].append((b["final_dur"] / b["nonfinal_dur"]) / (a["final_dur"] / a["nonfinal_dur"]))
            # share of primary-stressed vowels at <= 4 frames (46 ms)
            out["stressed_short"].extend(int(x <= 4) for x in b["stressed_dur"])
            nz = np.array([x != SP for x in ph])
            out["logdur_rmse"].append(np.sqrt(np.mean((np.log1p(pd_[nz]) - np.log1p(rd[nz])) ** 2)))
            out["f0_rmse_hz"].append(np.sqrt(np.mean((pf[nz] - rf[nz]) ** 2)))
            out["dur_bias"].append(pd_.sum() / max(rd.sum(), 1))
        return {k: round(med(v) if k not in ("stressed_short",) else float(np.mean(v)), 4) for k, v in out.items()}

    real_short = float(np.mean([int(x <= 4) for i in ev for x in features(i["phones"], *i["real"][:2])["stressed_dur"]]))
    result = {"n_eval": len(ev), "real_stressed_short_share": round(real_short, 4),
              "ideal": "f0_sd_ratio 1, final_len_ratio 1, errors/rmse lower is better, dur_bias 1",
              "metrics": {"none": score(""), **{v: score(v) for v in ("A", "B", "C", "D", "E", "P", "BE", "ABCDE")}}}
    # A rule is enabled only if it moves its own target towards real speech on the EVALUATION half.
    # P (pause from a run of spaces) cannot be tested on recorded speech (no such inputs): off.
    m0, close = result["metrics"]["none"], lambda a, b: abs(a) < abs(b) - 1e-4  # noqa: E731
    test = {"A": lambda m: close(m["stressed_short"] - real_short, m0["stressed_short"] - real_short),
            "B": lambda m: close(m["f0_sd_ratio"] - 1, m0["f0_sd_ratio"] - 1),
            "C": lambda m: close(m["stress_gap_err"], m0["stress_gap_err"]),
            "D": lambda m: close(m["final_z_err"], m0["final_z_err"]) or close(m["final_len_ratio"] - 1, m0["final_len_ratio"] - 1),
            "E": lambda m: close(m["dur_bias"] - 1, m0["dur_bias"] - 1)}
    rules["enabled"] = "".join(r for r in "ABCDE" if test[r](result["metrics"][r]))
    result["enabled"] = rules["enabled"]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=1)
    out_dir = os.path.join(OUTPUT_ROOT, "English")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "prosody_eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("enabled:", rules["enabled"] or "(none)")
    print("eval half: %d utterances; real share of stressed vowels <=4 frames %.3f" % (len(ev), real_short))
    keys = list(result["metrics"]["none"])
    print("%-6s" % "rules" + "".join("%16s" % k for k in keys))
    for v, m in result["metrics"].items():
        print("%-6s" % v + "".join("%16.4f" % m[k] for k in keys))


if __name__ == "__main__":
    main()
