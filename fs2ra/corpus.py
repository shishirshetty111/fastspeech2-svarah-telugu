"""Export selected utterances as an MFA/upstream corpus:

    <corpus_dir>/<speaker>/<utt_id>.wav   mono, resampled to the model rate, peak-normalised
    <corpus_dir>/<speaker>/<utt_id>.lab   spoken-form transcript (language front end)
    <work_dir>/corpus_index.csv           utt_id, split, speaker, raw text, spoken text, ...

Upstream prepare_align scales by 32768 / max|x| (wraps to -32768 at the peak);
we peak-normalise to 0.95 instead. Pitch/energy are z-normalised later, so the
absolute level only affects the mel spectrogram consistently across utterances.
"""
import io
import os
from collections import defaultdict

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from fs2ra.sources import SOURCES, raw_dir

PEAK = 0.95


def _row_lookup(pf):
    """global row index -> (row_group, offset)"""
    starts, total = [], 0
    for rg in range(pf.num_row_groups):
        starts.append(total)
        total += pf.metadata.row_group(rg).num_rows
    return starts


def iter_audio(language, rows):
    """Yield (row, waveform float32, sample_rate) for manifest rows, reading each
    needed parquet row group once."""
    audio_col = SOURCES[language]["audio_column"]
    by_shard = defaultdict(list)
    for r in rows:
        by_shard[r["shard"]].append(r)
    for shard, shard_rows in sorted(by_shard.items()):
        pf = pq.ParquetFile(os.path.join(raw_dir(language), shard))
        starts = _row_lookup(pf)
        by_rg = defaultdict(list)
        for r in shard_rows:
            idx = int(r["row_index"])
            rg = max(i for i, s in enumerate(starts) if s <= idx)
            by_rg[rg].append((idx - starts[rg], r))
        for rg, items in sorted(by_rg.items()):
            table = pf.read_row_group(rg, columns=[audio_col]).column(audio_col).to_pylist()
            for offset, r in items:
                audio = table[offset]
                stem = os.path.splitext(os.path.basename(audio["path"]))[0]
                assert r["source_path"] == audio["path"], (r["utt_id"], audio["path"], stem)
                wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                yield r, wav, sr


def export_corpus(language, rows, corpus_dir, target_sr, normalize_text):
    """Write wav/lab pairs; returns index rows (one per exported utterance)."""
    index = []
    for r, wav, sr in iter_audio(language, rows):
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
        peak = float(np.max(np.abs(wav)))
        wav = wav / peak * PEAK if peak > 0 else wav
        spoken = normalize_text(r["text"])
        spk_dir = os.path.join(corpus_dir, r["speaker"])
        os.makedirs(spk_dir, exist_ok=True)
        sf.write(os.path.join(spk_dir, r["utt_id"] + ".wav"), wav, target_sr, subtype="PCM_16")
        with open(os.path.join(spk_dir, r["utt_id"] + ".lab"), "w", encoding="utf-8") as f:
            f.write(spoken + "\n")
        index.append({
            "utt_id": r["utt_id"], "split": r["split"], "speaker": r["speaker"],
            "duration_s": len(wav) / target_sr, "source_sample_rate": sr,
            "raw_text": r["text"], "spoken_text": spoken,
        })
    return index
