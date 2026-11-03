import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_DIR = os.path.join(REPO_ROOT, "third_party", "FastSpeech2")
LJSPEECH_CKPT = os.path.join(REPO_ROOT, "pretrained", "LJSpeech", "900000.pth.tar")

if not os.path.isdir(UPSTREAM_DIR):
    raise RuntimeError("Upstream missing; run `python scripts/setup_upstream.py` first.")
sys.path.insert(0, UPSTREAM_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

requires_ljspeech_ckpt = pytest.mark.skipif(
    not os.path.exists(LJSPEECH_CKPT),
    reason="run `python scripts/setup_upstream.py --pretrained_ljspeech`",
)
sys.path.insert(0, REPO_ROOT)
