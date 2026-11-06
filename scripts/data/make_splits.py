"""QC-filter the inventory and write deterministic ~5 h train / ~1 h val splits.

    python scripts/data/make_splits.py --language english telugu

Outputs per language:
  data/manifests/<lang>/{train,val,reserve,rejected}.csv   full rows incl. transcripts (git-ignored)
  project/<Lang>/splits/{train,val,reserve,rejected}.csv   no transcripts/audio (committed)
  project/<Lang>/splits/split_report.json                  config, sources, hours, QC counts, checks
"""
import argparse
import dataclasses
import hashlib
import json
import os
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from fs2ra.sources import DATA_ROOT, SOURCES  # noqa: E402
from fs2ra.splits import QCConfig, apply_qc, check_disjoint, make_split, split_summary  # noqa: E402
from fs2ra.tables import read_csv, write_csv  # noqa: E402

SEED = 1234
TRAIN_HOURS = 5.0
VAL_HOURS = 1.0

LANGUAGE_CONFIG = {
    # Svarah: 115 speaker proxies, ~3.8 min each -> multi-speaker model, every
    # speaker contributes to train and val. Digits/symbols need verbalisation,
    # so those utterances are drawn last (mostly end up in reserve).
    "english": {
        "project_dir": "English",
        "qc": QCConfig(
            min_duration_s=1.0,
            max_duration_s=15.0,
            preferred_max_duration_s=11.0,
            deprioritize_text_regex=r"[0-9%&+@$_]",
        ),
    },
    # IndicTTS Telugu: single-speaker male model. The female voice has 23.9 h but
    # only 3.6 h of QC-passed clips <= 15 s (most are 20-150 s); the male voice has
    # 8.1 h <= 15 s. Latin letters in transcripts are typos (Latin 'o' typed for
    # anusvara U+0C02) and are rejected rather than silently corrected.
    "telugu": {
        "project_dir": "Telugu",
        "qc": QCConfig(
            min_duration_s=1.0,
            max_duration_s=15.0,
            preferred_max_duration_s=11.0,
            speakers=["te_male"],
            reject_text_regex=r"[A-Za-z]",
        ),
    },
}

NUMERIC = ("duration_s", "letters_per_s", "rms_dbfs", "clip_fraction", "peak", "sample_rate", "row_index")
PUBLIC_COLUMNS = ["utt_id", "split", "speaker", "duration_s", "sample_rate", "shard", "row_index",
                  "source_path", "audio_sha1", "text_sha1", "priority_tier"]


def text_sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build(language):
    cfg = LANGUAGE_CONFIG[language]
    full_dir = os.path.join(DATA_ROOT, "manifests", language)
    rows = read_csv(os.path.join(full_dir, "inventory.csv"), numeric=NUMERIC)
    for r in rows:
        r["row_index"], r["sample_rate"] = int(r["row_index"]), int(r["sample_rate"])
    kept, rejected = apply_qc(rows, cfg["qc"])
    splits = make_split(kept, TRAIN_HOURS, VAL_HOURS, SEED)
    check_disjoint(splits)
    summary = split_summary(splits)

    pub_dir = os.path.join(REPO_ROOT, "project", cfg["project_dir"], "splits")
    for name, part in list(splits.items()) + [("rejected", rejected)]:
        for r in part:
            r["text_sha1"] = text_sha1(r["text"])
            r.setdefault("split", name)
        write_csv(os.path.join(full_dir, name + ".csv"), part, columns=list(part[0].keys()))
        cols = PUBLIC_COLUMNS + (["reject_reason"] if name == "rejected" else [])
        write_csv(os.path.join(pub_dir, name + ".csv"), part, columns=cols)

    reasons = Counter(r["reject_reason"].split(":")[0] for r in rejected)
    rej_hours = Counter()
    for r in rejected:
        rej_hours[r["reject_reason"].split(":")[0]] += r["duration_s"] / 3600
    tiers = {name: dict(Counter(int(r["priority_tier"]) for r in part)) for name, part in splits.items()}
    report = {
        "language": language,
        "source": {k: SOURCES[language][k] for k in ("repo_id", "revision", "license")},
        "seed": SEED,
        "targets_hours": {"train": TRAIN_HOURS, "val": VAL_HOURS},
        "qc_config": dataclasses.asdict(cfg["qc"]),
        "inventory": {"utterances": len(rows), "hours": round(sum(r["duration_s"] for r in rows) / 3600, 4),
                      "speakers": len({r["speaker"] for r in rows})},
        "qc_kept": {"utterances": len(kept), "hours": round(sum(r["duration_s"] for r in kept) / 3600, 4)},
        "qc_rejected": {k: {"utterances": reasons[k], "hours": round(rej_hours[k], 4)} for k in sorted(reasons)},
        "splits": summary,
        "priority_tiers_per_split": tiers,
        "checks": "train/val/reserve pairwise disjoint on utt_id, audio_sha1 and normalised text: passed",
    }
    with open(os.path.join(pub_dir, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("%s: train %.3f h (%d utts), val %.3f h (%d utts), reserve %.3f h, rejected %d"
          % (language, summary["train"]["hours"], summary["train"]["utterances"], summary["val"]["hours"],
             summary["val"]["utterances"], summary["reserve"]["hours"], len(rejected)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    for lang in parser.parse_args().language:
        build(lang)


if __name__ == "__main__":
    main()
