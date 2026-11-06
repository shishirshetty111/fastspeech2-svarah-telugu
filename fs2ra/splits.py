"""Quality filtering and deterministic, duration-targeted train/val/reserve splits.

Rules:
* QC rejects are logged with a reason, never silently dropped.
* Utterances sharing a normalised transcript form one group and always land in
  the same split, so no sentence appears in both train and validation.
* Each speaker contributes to train and val in proportion to its usable hours
  (same-speaker, disjoint-utterance validation), so every val speaker is seen
  in training and the speaker embedding is defined for validation.
* Groups are drawn by (priority tier, sha1(seed:group_key)): lower tiers (cleaner
  material) first, ties broken by a seeded hash independent of file/row order.
* Utterances not selected go to `reserve` (disjoint from val) and may only be
  used to replace train utterances that fail forced alignment.
"""
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median


@dataclass
class QCConfig:
    min_duration_s: float = 1.0
    max_duration_s: float = 11.0
    min_rms_dbfs: float = -40.0
    max_clip_fraction: float = 0.001
    # robust z-score bound on letters/second, per speaker (catches text/audio mismatch)
    max_rate_robust_z: float = 4.0
    min_speaker_utts_for_rate: int = 20
    speakers: list = field(default_factory=list)  # empty = all speakers
    reject_text_regex: str = ""  # e.g. Latin letters in Telugu transcripts
    # priority tiers: utterances above this duration, or matching the regex, are
    # drawn after clean ones (they need a longer max_seq_len / text verbalisation)
    preferred_max_duration_s: float = 11.0
    deprioritize_text_regex: str = ""


def _robust_z(values):
    med = median(values)
    mad = median(abs(v - med) for v in values) or 1e-9
    return [(v - med) / (1.4826 * mad) for v in values], med, mad


def apply_qc(rows, cfg):
    """Returns (kept, rejected); each rejected row gets a `reject_reason`."""
    import re

    reject_re = re.compile(cfg.reject_text_regex) if cfg.reject_text_regex else None
    kept, rejected = [], []
    seen_audio = {}

    by_speaker = defaultdict(list)
    for r in rows:
        if r["duration_s"] > 0:
            by_speaker[r["speaker"]].append(r)
    all_rates = [r["letters_per_s"] for r in rows if r["duration_s"] > 0]
    rate_z = {}
    for spk, rs in by_speaker.items():
        pool = rs if len(rs) >= cfg.min_speaker_utts_for_rate else None
        if pool is not None:
            zs, _, _ = _robust_z([r["letters_per_s"] for r in pool])
        else:
            _, med, mad = _robust_z(all_rates)
            zs = [(r["letters_per_s"] - med) / (1.4826 * mad) for r in rs]
        for r, z in zip(rs, zs):
            rate_z[r["utt_id"]] = z

    for r in sorted(rows, key=lambda r: r["utt_id"]):
        reason = None
        if cfg.speakers and r["speaker"] not in cfg.speakers:
            reason = "speaker_not_selected"
        elif not r["text_key"]:
            reason = "empty_text"
        elif r["audio_sha1"] in seen_audio:
            reason = "duplicate_audio_of:" + seen_audio[r["audio_sha1"]]
        elif r["duration_s"] < cfg.min_duration_s:
            reason = "too_short"
        elif r["duration_s"] > cfg.max_duration_s:
            reason = "too_long"
        elif r["rms_dbfs"] < cfg.min_rms_dbfs:
            reason = "too_quiet"
        elif r["clip_fraction"] > cfg.max_clip_fraction:
            reason = "clipping"
        elif abs(rate_z[r["utt_id"]]) > cfg.max_rate_robust_z:
            reason = "speaking_rate_outlier"
        elif reject_re is not None and reject_re.search(r["text"]):
            reason = "unsupported_characters"
        seen_audio.setdefault(r["audio_sha1"], r["utt_id"])
        r["rate_robust_z"] = rate_z.get(r["utt_id"], float("nan"))
        if reason:
            rejected.append(dict(r, reject_reason=reason))
        else:
            tier = 1 if r["duration_s"] > cfg.preferred_max_duration_s else 0
            if cfg.deprioritize_text_regex and re.search(cfg.deprioritize_text_regex, r["text"]):
                tier = 2
            kept.append(dict(r, priority_tier=tier))
    return kept, rejected


def _group_order(key, seed):
    return hashlib.sha1(("%d:%s" % (seed, key)).encode("utf-8")).hexdigest()


def _fill(groups, targets, totals):
    """Greedy pass in fixed order: take a group if, for each of its speakers, it
    moves that speaker's total closer to (or no further past) its target."""
    taken, left = [], []
    for key, members in groups:
        add = defaultdict(float)
        for r in members:
            add[r["speaker"]] += r["duration_s"]
        ok = all(totals[s] + d <= targets.get(s, 0.0) + d / 2 for s, d in add.items())
        if ok:
            for s, d in add.items():
                totals[s] += d
            taken.append((key, members))
        else:
            left.append((key, members))
    return taken, left


def make_split(rows, train_hours, val_hours, seed):
    """rows: QC-passed inventory rows. Returns dict split -> list of rows."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["text_key"]].append(r)
    ordered = sorted(
        groups.items(),
        key=lambda kv: (max(r.get("priority_tier", 0) for r in kv[1]), _group_order(kv[0], seed)),
    )

    avail = defaultdict(float)
    for r in rows:
        avail[r["speaker"]] += r["duration_s"]
    total = sum(avail.values())
    need = (train_hours + val_hours) * 3600
    if total < need:
        raise ValueError("only %.2f h usable, need %.2f h" % (total / 3600, need / 3600))
    share = {s: a / total for s, a in avail.items()}
    val_t = {s: val_hours * 3600 * p for s, p in share.items()}
    train_t = {s: train_hours * 3600 * p for s, p in share.items()}

    val, rest = _fill(ordered, val_t, defaultdict(float))
    train, reserve = _fill(rest, train_t, defaultdict(float))

    out = {}
    for name, part in (("train", train), ("val", val), ("reserve", reserve)):
        out[name] = sorted((dict(r, split=name) for _, ms in part for r in ms),
                           key=lambda r: r["utt_id"])
    return out


def split_summary(splits):
    summary = {}
    for name, rows in splits.items():
        spk = defaultdict(float)
        for r in rows:
            spk[r["speaker"]] += r["duration_s"]
        summary[name] = {
            "utterances": len(rows),
            "hours": round(sum(spk.values()) / 3600, 4),
            "speakers": len(spk),
            "per_speaker_hours": {s: round(h / 3600, 4) for s, h in sorted(spk.items())},
        }
    return summary


def check_disjoint(splits, keys=("utt_id", "audio_sha1", "text_key")):
    """Raises AssertionError on any train/val overlap; reserve must not overlap val."""
    for k in keys:
        tr = {r[k] for r in splits["train"]}
        va = {r[k] for r in splits["val"]}
        rs = {r[k] for r in splits.get("reserve", [])}
        assert not tr & va, "train/val overlap on %s: %d" % (k, len(tr & va))
        assert not rs & va, "reserve/val overlap on %s: %d" % (k, len(rs & va))
        assert not rs & tr, "reserve/train overlap on %s: %d" % (k, len(rs & tr))
