"""Synthetic-data tests for fs2ra.splits (no dataset download needed)."""
import random

import pytest

from fs2ra.inventory import count_letters, normalize_text_key
from fs2ra.splits import QCConfig, apply_qc, check_disjoint, make_split, split_summary


def fake_rows(n_speakers=6, per_speaker=400, seed=0, dup_text_every=7):
    rng = random.Random(seed)
    rows = []
    for s in range(n_speakers):
        for i in range(per_speaker):
            dur = rng.uniform(1.5, 10.0)
            text = "sentence %d" % (i // dup_text_every if i % dup_text_every == 0 else 10_000 + s * 1000 + i)
            rows.append({
                "utt_id": "spk%d_%04d" % (s, i), "speaker": "spk%d" % s, "duration_s": dur,
                "audio_sha1": "%d-%d" % (s, i), "text": text, "text_key": normalize_text_key(text),
                "letters_per_s": 14.0 + rng.uniform(-1, 1), "rms_dbfs": -22.0, "clip_fraction": 0.0,
            })
    return rows


def test_text_key_and_letter_count():
    assert normalize_text_key("  The orchestra's  CONDUCT. ") == "the orchestra s conduct"
    # Telugu vowel signs / virama are combining marks and must be counted
    assert count_letters("పట్టు") == 5


def test_qc_reasons():
    rows = fake_rows(n_speakers=1, per_speaker=50)
    rows[0]["duration_s"] = 0.3
    rows[1]["duration_s"] = 30.0
    rows[2]["rms_dbfs"] = -60.0
    rows[3]["clip_fraction"] = 0.05
    rows[4]["letters_per_s"] = 60.0
    rows[5]["audio_sha1"] = rows[6]["audio_sha1"]
    kept, rejected = apply_qc(rows, QCConfig())
    reasons = {r["utt_id"]: r["reject_reason"] for r in rejected}
    assert reasons["spk0_0000"] == "too_short"
    assert reasons["spk0_0001"] == "too_long"
    assert reasons["spk0_0002"] == "too_quiet"
    assert reasons["spk0_0003"] == "clipping"
    assert reasons["spk0_0004"] == "speaking_rate_outlier"
    assert reasons["spk0_0006"].startswith("duplicate_audio_of:spk0_0005")
    assert len(kept) + len(rejected) == len(rows)


def test_split_targets_disjoint_and_deterministic():
    rows = fake_rows()
    kept, _ = apply_qc(rows, QCConfig())
    a = make_split(kept, train_hours=2.0, val_hours=0.4, seed=1234)
    b = make_split(list(reversed(kept)), train_hours=2.0, val_hours=0.4, seed=1234)
    assert [r["utt_id"] for r in a["val"]] == [r["utt_id"] for r in b["val"]]
    assert [r["utt_id"] for r in a["train"]] == [r["utt_id"] for r in b["train"]]
    check_disjoint(a)
    s = split_summary(a)
    assert s["train"]["hours"] == pytest.approx(2.0, abs=0.05)
    assert s["val"]["hours"] == pytest.approx(0.4, abs=0.03)
    assert set(s["val"]["per_speaker_hours"]) <= set(s["train"]["per_speaker_hours"])
    assert s["train"]["utterances"] + s["val"]["utterances"] + s["reserve"]["utterances"] == len(kept)


def test_duplicate_texts_stay_in_one_split():
    kept, _ = apply_qc(fake_rows(), QCConfig())
    splits = make_split(kept, 2.0, 0.4, seed=7)
    where = {}
    for name, rs in splits.items():
        for r in rs:
            where.setdefault(r["text_key"], set()).add(name)
    assert all(len(v) == 1 for v in where.values())


def test_insufficient_data_raises():
    kept, _ = apply_qc(fake_rows(n_speakers=1, per_speaker=50), QCConfig())
    with pytest.raises(ValueError):
        make_split(kept, 5.0, 1.0, seed=0)


def test_speaker_keys():
    from fs2ra.inventory import svarah_speaker_key, telugu_speaker_key

    profile = {"gender": "Female", "age-group": "18-30", "primary_language": "Kannada",
               "native_place_state": "Karnataka", "native_place_district": "Hassan",
               "highest_qualification": "Post Graduate", "job_category": "Full Time",
               "occupation_domain": "Technology and Services"}
    a = svarah_speaker_key("281474976884635_f3269_chunk_0", profile)
    b = svarah_speaker_key("999_f1_chunk_3", profile)  # different recording, same person profile
    c = svarah_speaker_key("281474976884635_f3269_chunk_0", dict(profile, gender="Male"))
    assert a == b != c and a.startswith("en_")
    assert telugu_speaker_key("train_telugufemale_00001", {}) == "te_female"
    assert telugu_speaker_key("train_telugumale_02390", {}) == "te_male"
