"""Integrity of the full preprocessed feature sets (skipped until built)."""
import json
import os

import numpy as np
import pytest

from fs2ra.inventory import normalize_text_key
from fs2ra.sources import DATA_ROOT
from fs2ra.tables import read_csv

LANGS = ["english", "telugu"]


def load(lang):
    pre = os.path.join(DATA_ROOT, lang, "preprocessed")
    if not os.path.exists(os.path.join(pre, "preprocess_report.json")):
        pytest.skip("%s features not built" % lang)
    lines = {}
    for split in ("train", "val"):
        with open(os.path.join(pre, split + ".txt"), encoding="utf-8") as f:
            lines[split] = [l.split("|") for l in f.read().splitlines()]
    index = {r["utt_id"]: r for r in read_csv(os.path.join(DATA_ROOT, lang, "corpus_index.csv"))}
    with open(os.path.join(pre, "preprocess_report.json")) as f:
        report = json.load(f)
    return pre, lines, index, report


@pytest.mark.parametrize("lang", LANGS)
def test_train_val_disjoint_by_id_and_text(lang):
    _, lines, index, _ = load(lang)
    ids = {s: {l[0] for l in lines[s]} for s in lines}
    keys = {s: {normalize_text_key(index[u]["raw_text"]) for u in ids[s]} for s in ids}
    assert not ids["train"] & ids["val"]
    assert not keys["train"] & keys["val"]


@pytest.mark.parametrize("lang", LANGS)
def test_hours_speakers_and_stats(lang):
    pre, lines, index, report = load(lang)
    hours = {s: sum(float(index[l[0]]["duration_s"]) for l in lines[s]) / 3600 for s in lines}
    assert 4.95 <= hours["train"] <= 5.1 and 0.98 <= hours["val"] <= 1.05
    with open(os.path.join(pre, "speakers.json")) as f:
        speakers = json.load(f)
    assert {l[1] for l in lines["val"]} <= set(speakers) == {l[1] for l in lines["train"]}
    assert report["stats_fit_on"] == "train only"


@pytest.mark.parametrize("lang", LANGS)
def test_every_target_consistent(lang):
    pre, lines, _, _ = load(lang)
    for basename, speaker, phones, _ in lines["train"] + lines["val"]:
        load_ = lambda k: np.load(os.path.join(pre, k, "%s-%s-%s.npy" % (speaker, k, basename)))
        dur, mel = load_("duration"), load_("mel")
        assert dur.sum() == mel.shape[0]
        assert len(phones.strip("{}").split()) == len(dur) == len(load_("pitch")) == len(load_("energy"))
