"""Integrity of the committed train/val/reserve manifests (project/<Lang>/splits)."""
import json
import os

import pytest

from conftest import REPO_ROOT
from fs2ra.tables import read_csv

LANGS = {"english": "English", "telugu": "Telugu"}


def split_dir(lang):
    return os.path.join(REPO_ROOT, "project", LANGS[lang], "splits")


def load(lang):
    d = split_dir(lang)
    if not os.path.exists(os.path.join(d, "split_report.json")):
        pytest.skip("splits for %s not built yet" % lang)
    parts = {n: read_csv(os.path.join(d, n + ".csv"), numeric=("duration_s",))
             for n in ("train", "val", "reserve", "rejected")}
    with open(os.path.join(d, "split_report.json"), encoding="utf-8") as f:
        return parts, json.load(f)


@pytest.mark.parametrize("lang", sorted(LANGS))
def test_disjoint(lang):
    parts, _ = load(lang)
    for key in ("utt_id", "audio_sha1", "text_sha1"):
        sets = {n: {r[key] for r in parts[n]} for n in ("train", "val", "reserve")}
        assert not sets["train"] & sets["val"], key
        assert not sets["reserve"] & sets["val"], key
        assert not sets["reserve"] & sets["train"], key
    all_ids = [r["utt_id"] for n in parts for r in parts[n]]
    assert len(all_ids) == len(set(all_ids))


@pytest.mark.parametrize("lang", sorted(LANGS))
def test_hours_and_bounds(lang):
    parts, report = load(lang)
    hours = {n: sum(r["duration_s"] for r in parts[n]) / 3600 for n in ("train", "val")}
    assert 4.9 <= hours["train"] <= 5.2
    assert 0.95 <= hours["val"] <= 1.1
    qc = report["qc_config"]
    for n in ("train", "val", "reserve"):
        assert all(qc["min_duration_s"] <= r["duration_s"] <= qc["max_duration_s"] for r in parts[n])
    for n in ("train", "val"):
        assert abs(report["splits"][n]["hours"] - hours[n]) < 1e-3
    total = sum(len(parts[n]) for n in parts)
    assert total == report["inventory"]["utterances"]


@pytest.mark.parametrize("lang", sorted(LANGS))
def test_val_speakers_seen_in_train(lang):
    parts, _ = load(lang)
    assert {r["speaker"] for r in parts["val"]} <= {r["speaker"] for r in parts["train"]}


@pytest.mark.parametrize("lang", sorted(LANGS))
def test_reproducible_from_inventory(lang):
    """Rebuilding from the raw-data inventory gives exactly the committed split."""
    from fs2ra.sources import DATA_ROOT
    from fs2ra.splits import apply_qc, make_split
    import sys

    inv = os.path.join(DATA_ROOT, "manifests", lang, "inventory.csv")
    if not os.path.exists(inv):
        pytest.skip("inventory not built (needs downloaded data)")
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "data"))
    import make_splits

    parts, _ = load(lang)
    rows = read_csv(inv, numeric=make_splits.NUMERIC)
    kept, _ = apply_qc(rows, make_splits.LANGUAGE_CONFIG[lang]["qc"])
    rebuilt = make_split(kept, make_splits.TRAIN_HOURS, make_splits.VAL_HOURS, make_splits.SEED)
    for n in ("train", "val", "reserve"):
        assert [r["utt_id"] for r in rebuilt[n]] == [r["utt_id"] for r in parts[n]]
