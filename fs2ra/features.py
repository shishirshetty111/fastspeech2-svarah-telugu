"""FastSpeech 2 training targets from MFA alignments, reusing upstream
Preprocessor.process_utterance unchanged (mel, phone durations, phoneme-level
F0 and energy), with these project-level changes (see docs/UPSTREAM_NOTES.md):

1. MFA 3 leaves silences as *empty* intervals, which tgt drops by default; upstream
   (written for MFA 1.x "sil"/"sp") would then lose every internal pause from the
   duration sequence. We rewrite TextGrids with empty intervals labelled "sp".
2. Splits come from our manifests (not a random val_size); failed utterances are
   replaced from `reserve` until the hour targets are met.
3. Pitch/energy mean/std and the min/max used for the embedding bins are fitted on
   the final TRAIN set only (upstream fits on train+val), then applied to all.
4. Every saved utterance is validated: sum(durations) == mel frames, one pitch and
   energy value per phone, finite values, and no non-silence phone longer than
   data.max_phone_frames (filled pauses absorbed into a phone by the aligner).
"""
import json
import os
import re
from collections import Counter
from multiprocessing import Pool

import numpy as np
import tgt
from sklearn.preprocessing import StandardScaler

from fs2ra.inventory import normalize_text_key
from fs2ra.tables import read_csv, write_csv

SIL = "sp"
_PRE = None  # per-worker upstream Preprocessor


def convert_textgrids(mfa_dir, out_dir):
    """Copy MFA TextGrids with empty phone/word intervals labelled 'sp'."""
    n = 0
    for root, _, files in os.walk(mfa_dir):
        for name in files:
            if not name.endswith(".TextGrid"):
                continue
            tg = tgt.io.read_textgrid(os.path.join(root, name), include_empty_intervals=True)
            for tier in tg.tiers:
                for interval in tier._objects:
                    if not interval.text.strip():
                        interval.text = SIL
            rel = os.path.relpath(root, mfa_dir)
            os.makedirs(os.path.join(out_dir, rel), exist_ok=True)
            tgt.io.write_to_file(tg, os.path.join(out_dir, rel, name), format="long")
            n += 1
    return n


def _init_worker(preprocess_config):
    global _PRE
    from preprocessor.preprocessor import Preprocessor  # upstream

    _PRE = Preprocessor(preprocess_config)


def _process(item):
    speaker, utt_id = item
    tg_path = os.path.join(_PRE.out_dir, "TextGrid", speaker, utt_id + ".TextGrid")
    if not os.path.exists(tg_path):
        return utt_id, None, "no_alignment"
    try:
        ret = _PRE.process_utterance(speaker, utt_id)
    except Exception as e:  # logged, never silently dropped
        return utt_id, None, "error:%s" % type(e).__name__
    if ret is None:
        return utt_id, None, "rejected_by_upstream"  # empty span or <= 1 voiced frame
    info, pitch, energy, n_frames = ret
    return utt_id, {"info": info, "pitch": pitch, "energy": energy, "n_frames": n_frames}, "ok"


def _paths(out_dir, speaker, utt_id):
    return {k: os.path.join(out_dir, k, "%s-%s-%s.npy" % (speaker, k, utt_id))
            for k in ("mel", "pitch", "energy", "duration")}


def validate(out_dir, speaker, utt_id, phones, max_phone_frames=None):
    p = _paths(out_dir, speaker, utt_id)
    mel, pitch, energy, dur = (np.load(p[k]) for k in ("mel", "pitch", "energy", "duration"))
    problems = []
    speech = [d for d, ph in zip(dur, phones) if ph != SIL]
    if max_phone_frames and speech and max(speech) > max_phone_frames:
        # filled pauses / untranscribed speech absorbed into one phone (alignment outlier)
        problems.append("long_phone_%d_frames" % max(speech))
    if mel.ndim != 2 or mel.shape[1] != 80:
        problems.append("mel_shape")
    if int(dur.sum()) != mel.shape[0]:
        problems.append("duration_sum_%d_vs_mel_%d" % (dur.sum(), mel.shape[0]))
    if not (len(pitch) == len(energy) == len(dur)):
        problems.append("phone_level_lengths")
    if not all(np.isfinite(a).all() for a in (mel, pitch, energy)):
        problems.append("nonfinite")
    if (dur < 0).any():
        problems.append("negative_duration")
    return problems, {"phones": len(dur), "frames": int(mel.shape[0]), "zero_dur_phones": int((dur == 0).sum())}


def _select(index, ok_ids, split, target_hours, exclude_text_keys=frozenset()):
    """Successful utterances of `split`, topped up from successful reserve ones whose
    transcript does not occur in the other split (keeps train/val text-disjoint)."""
    chosen = [r for r in index if r["split"] == split and r["utt_id"] in ok_ids]
    hours = sum(float(r["duration_s"]) for r in chosen) / 3600
    topup = []
    if hours < target_hours:
        for r in index:
            if r["split"] == "reserve" and r["utt_id"] in ok_ids and not r.get("_used") \
                    and normalize_text_key(r["raw_text"]) not in exclude_text_keys:
                r["_used"] = True
                topup.append(r)
                hours += float(r["duration_s"]) / 3600
                if hours >= target_hours:
                    break
    return chosen + topup, topup


def stage_features(cfg, dirs, num_workers=None):
    pre_cfg = cfg["preprocess"]
    pre_cfg = dict(pre_cfg, path=dict(pre_cfg["path"], raw_path=dirs["corpus"],
                                      preprocessed_path=dirs["preprocessed"]))
    out = dirs["preprocessed"]
    n_tg = convert_textgrids(dirs["textgrid_mfa"], dirs["textgrid"])
    print("converted %d TextGrids (empty intervals -> '%s')" % (n_tg, SIL))
    for k in ("mel", "pitch", "energy", "duration"):
        os.makedirs(os.path.join(out, k), exist_ok=True)

    index = read_csv(dirs["index"])
    items = [(r["speaker"], r["utt_id"]) for r in index]
    with Pool(num_workers or max(1, (os.cpu_count() or 2) - 1), _init_worker, (pre_cfg,)) as pool:
        results = {u: (res, status) for u, res, status in pool.imap_unordered(_process, items, chunksize=8)}

    status_rows, ok = [], {}
    for r in index:
        res, status = results[r["utt_id"]]
        row = {"utt_id": r["utt_id"], "split": r["split"], "speaker": r["speaker"], "status": status}
        if res is not None:
            phones = res["info"].split("|")[2].strip("{}").split()
            problems, shape = validate(out, r["speaker"], r["utt_id"], phones,
                                       cfg["data"].get("max_phone_frames"))
            row.update(shape)
            if problems:
                row["status"] = "invalid:" + ";".join(problems)
            else:
                ok[r["utt_id"]] = res
        status_rows.append(row)

    targets = cfg["data"]["target_hours"]
    val_keys = {normalize_text_key(r["raw_text"]) for r in index if r["split"] == "val"}
    train, train_topup = _select(index, ok, "train", targets["train"], val_keys)
    train_keys = {normalize_text_key(r["raw_text"]) for r in train}
    val, val_topup = _select(index, ok, "val", targets["val"], train_keys)
    # the speaker embedding is only defined for training speakers
    train_speakers = {r["speaker"] for r in train}
    val_dropped = [r["utt_id"] for r in val if r["speaker"] not in train_speakers]
    val = [r for r in val if r["speaker"] in train_speakers]

    # statistics on final train set only
    pitch_scaler, energy_scaler = StandardScaler(), StandardScaler()
    for r in train:
        res = ok[r["utt_id"]]
        if len(res["pitch"]):
            pitch_scaler.partial_fit(res["pitch"].reshape(-1, 1))
        if len(res["energy"]):
            energy_scaler.partial_fit(res["energy"].reshape(-1, 1))
    stats_raw = {"pitch": (float(pitch_scaler.mean_[0]), float(pitch_scaler.scale_[0])),
                 "energy": (float(energy_scaler.mean_[0]), float(energy_scaler.scale_[0]))}

    # normalise every processed utterance; bin range (min/max) from train only
    train_ids = {r["utt_id"] for r in train}
    ranges = {k: [np.inf, -np.inf] for k in ("pitch", "energy")}
    for r in index:
        if r["utt_id"] not in ok:
            continue
        p = _paths(out, r["speaker"], r["utt_id"])
        for k in ("pitch", "energy"):
            mean, std = stats_raw[k]
            values = (np.load(p[k]) - mean) / std
            np.save(p[k], values)
            if r["utt_id"] in train_ids:
                ranges[k] = [min(ranges[k][0], float(values.min())), max(ranges[k][1], float(values.max()))]
    stats = {k: [ranges[k][0], ranges[k][1], stats_raw[k][0], stats_raw[k][1]] for k in ("pitch", "energy")}
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f)

    speakers = sorted({r["speaker"] for r in train})
    with open(os.path.join(out, "speakers.json"), "w") as f:
        json.dump({s: i for i, s in enumerate(speakers)}, f)
    for name, part in (("train", train), ("val", val)):
        with open(os.path.join(out, name + ".txt"), "w", encoding="utf-8", newline="\n") as f:
            for r in sorted(part, key=lambda r: r["utt_id"]):
                f.write(ok[r["utt_id"]]["info"] + "\n")

    write_csv(os.path.join(out, "feature_status.csv"), status_rows,
              columns=["utt_id", "split", "speaker", "status", "phones", "frames", "zero_dur_phones"])
    failed = [s for s in status_rows if s["status"] != "ok"]
    reason = lambda s: re.sub(r"_\d+(_vs_mel_\d+)?(_frames)?", "", s)  # long_phone_83_frames -> long_phone
    report = {
        "textgrids": n_tg,
        "processed_ok": len(ok),
        "failed": dict(sorted(Counter(reason(x["status"]) for x in failed).items())),
        "failed_by_split": {sp: sum(1 for x in failed if x["split"] == sp) for sp in ("train", "val", "reserve")},
        "final": {name: {"utterances": len(part), "hours": round(sum(float(r["duration_s"]) for r in part) / 3600, 4),
                         "topped_up_from_reserve": len(topup)}
                  for name, part, topup in (("train", train, train_topup), ("val", val, val_topup))},
        "speakers_in_train": len(speakers),
        "val_dropped_speaker_not_in_train": val_dropped,
        "stats_fit_on": "train only",
        "stats": stats,
    }
    with open(os.path.join(out, "preprocess_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("processed_ok", "failed", "final", "speakers_in_train")}, indent=1))
    return report
