"""Write project/English/pronunciation_variants.json: for each multi-pronunciation lexicon
word, how often MFA aligned each variant in the English *training* split. Used by
fs2ra.inference to pick the variant the model heard most (needs data/english/TextGrid_mfa).

    python scripts/pronunciation_variants.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO_ROOT, os.path.join(REPO_ROOT, "third_party", "FastSpeech2")]

from fs2ra.config import load_config  # noqa: E402
from fs2ra.lexicon_en import build_variant_table  # noqa: E402

cfg = load_config(os.path.join(REPO_ROOT, "project", "English", "config.yaml"))
pre = cfg["preprocess"]["path"]["preprocessed_path"]
with open(os.path.join(pre, "train.txt"), encoding="utf-8") as f:
    train_ids = [line.split("|")[0] for line in f]
build_variant_table(cfg["preprocess"]["path"]["lexicon_path"], os.path.join(cfg["data"]["work_dir"], "TextGrid_mfa"),
                    train_ids, os.path.join(REPO_ROOT, "project", "English", "pronunciation_variants.json"))
