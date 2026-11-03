"""Per-utterance inventory of a parquet-backed HF speech dataset.

One row per utterance with facts measured from the decoded audio (not the
dataset card): sample rate, channels, duration, level, clipping, and hashes for
duplicate detection. Audio is decoded one row group at a time, never all at once.
"""
import hashlib
import io
import os
import re
import unicodedata

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from fs2ra.sources import SOURCES, local_shards, raw_dir

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_text_key(text):
    """Grouping key for duplicate/near-duplicate transcripts (NOT a TTS front end):
    NFC, lowercase, punctuation removed, whitespace collapsed."""
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def count_letters(text):
    """Letters + combining marks (Telugu vowel signs/virama are category M)."""
    return sum(1 for c in unicodedata.normalize("NFC", text) if unicodedata.category(c)[0] in "LM")


SVARAH_PROFILE = ["gender", "age-group", "primary_language", "native_place_state",
                  "native_place_district", "highest_qualification", "job_category",
                  "occupation_domain"]


def svarah_speaker_key(stem, row):
    """Svarah has no speaker column. File stems are `<recording-id>_f<n>_chunk_<k>`,
    but the recording id is per session (2938 ids), not per speaker. Every
    recording id maps to exactly one demographic profile and there are 115
    distinct profiles vs. 117 speakers in the Svarah paper, so the profile is
    used as the speaker proxy (at most two speaker pairs may be merged)."""
    profile = "|".join(str(row.get(c)) for c in SVARAH_PROFILE)
    return "en_" + hashlib.sha1(profile.encode("utf-8")).hexdigest()[:8]


def telugu_speaker_key(stem, row):
    m = re.match(r"train_telugu(female|male)_\d+$", stem)
    return "te_" + m.group(1) if m else None


SPEAKER_KEY = {"english": svarah_speaker_key, "telugu": telugu_speaker_key}
META_COLUMNS = {
    "english": ["duration", "gender", "age-group", "primary_language", "native_place_state",
                "native_place_district", "highest_qualification", "job_category", "occupation_domain"],
    "telugu": ["gender"],
}


def audio_stats(wav):
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if len(wav) else 0.0
    return {
        "peak": peak,
        "rms_dbfs": 20 * np.log10(max(rms, 1e-10)),
        "clip_fraction": float(np.mean(np.abs(wav) >= 0.999)) if len(wav) else 0.0,
    }


def iter_inventory(language, limit=None):
    src = SOURCES[language]
    speaker_key = SPEAKER_KEY[language]
    n = 0
    for shard in local_shards(language):
        rel_shard = os.path.relpath(shard, raw_dir(language)).replace(os.sep, "/")
        pf = pq.ParquetFile(shard)
        row_index = 0
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                audio = row[src["audio_column"]]
                data = audio["bytes"]
                wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
                info = sf.info(io.BytesIO(data))
                stem = os.path.splitext(os.path.basename(audio["path"]))[0]
                text = row["text"]
                duration = len(wav) / sr
                rec = {
                    "utt_id": "%s_%s" % (language[:2], stem),
                    "source_path": audio["path"],
                    "shard": rel_shard,
                    "row_index": row_index,
                    "speaker": speaker_key(stem, row),
                    "recording_id": stem.split("_")[0] if language == "english" else "",
                    "sample_rate": sr,
                    "channels": info.channels,
                    "subtype": info.subtype,
                    "duration_s": duration,
                    "audio_sha1": hashlib.sha1(data).hexdigest(),
                    "text": text,
                    "text_key": normalize_text_key(text),
                    "n_letters": count_letters(text),
                    "letters_per_s": count_letters(text) / duration if duration > 0 else float("inf"),
                }
                rec.update(audio_stats(wav))
                for col in META_COLUMNS[language]:
                    rec["meta_" + col.replace("-", "_")] = row.get(col)
                yield rec
                row_index += 1
                n += 1
                if limit is not None and n >= limit:
                    return
