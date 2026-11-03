"""Pinned dataset sources. Revisions are the Hub commits inspected on 2026-11-03."""
import os

from fs2ra.hf import REPO_ROOT

DATA_ROOT = os.environ.get("FS2_DATA_ROOT", os.path.join(REPO_ROOT, "data"))

SOURCES = {
    "english": {
        "repo_id": "ai4bharat/Svarah",
        "revision": "ebbf7777fe771490696a3f7b007097606fa8c924",
        "gated": True,
        "license": "CC BY 4.0",
        "files": ["data/test-0000%d-of-00003.parquet" % i for i in range(3)],
        "audio_column": "audio_filepath",
    },
    "telugu": {
        "repo_id": "SPRINGLab/IndicTTS_Telugu",
        "revision": "7ab3c18491bef3f5d4e820c2701b5f78f54031d6",
        "gated": False,
        "license": "Indic TTS license (IIT Madras), https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf",
        "files": ["data/train-0000%d-of-00004.parquet" % i for i in range(4)],
        "audio_column": "audio",
    },
}


def raw_dir(language):
    return os.path.join(DATA_ROOT, "raw", language)


def local_shards(language):
    src = SOURCES[language]
    return [os.path.join(raw_dir(language), f) for f in src["files"]]
