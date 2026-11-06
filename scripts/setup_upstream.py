"""Reproducibly fetch the pinned ming024/FastSpeech2 upstream, apply our compat
patch, unpack the bundled HiFi-GAN generators and (optionally) download the
upstream pretrained LJSpeech acoustic checkpoint used for the Phase 1 baseline.

Usage (repo root):
    python scripts/setup_upstream.py                    # upstream + patch + HiFi-GAN
    python scripts/setup_upstream.py --pretrained_ljspeech   # also the 900k baseline ckpt
    python scripts/setup_upstream.py --bigvgan          # also BigVGAN-v2 code + weights (optional vocoder)
"""
import argparse
import hashlib
import os
import subprocess
import sys
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_URL = "https://github.com/ming024/FastSpeech2"
UPSTREAM_COMMIT = "d4e79eb52e8b01d24703b2dfc0385544092958f3"
UPSTREAM_DIR = os.path.join(REPO_ROOT, "third_party", "FastSpeech2")
PATCH = os.path.join(REPO_ROOT, "patches", "ming024_fastspeech2_compat.patch")

# Google Drive file from the upstream README's "pretrained models" folder
# (folder id 1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F).
LJSPEECH_GDRIVE_ID = "1r3fYhnblBJ8hDKDSUDtidJ-BN-xAM9pe"
LJSPEECH_ZIP_SHA256 = "b64a44869cd2ff696acff3339280764a340f19167fdc4147d8b98ead9856689b"
LJSPEECH_CKPT_SHA256 = "8e24862525a53ba35ea22b3fa630d8c67808a99d7b66d90d47da26077003357d"
PRETRAINED_DIR = os.path.join(REPO_ROOT, "pretrained", "LJSpeech")


def run(cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_upstream():
    if not os.path.isdir(os.path.join(UPSTREAM_DIR, ".git")):
        run(["git", "clone", UPSTREAM_URL, UPSTREAM_DIR])
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM_DIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != UPSTREAM_COMMIT:
        run(["git", "checkout", UPSTREAM_COMMIT], cwd=UPSTREAM_DIR)

    # Apply the patch only if it is not already applied.
    already = subprocess.run(
        ["git", "apply", "--reverse", "--check", PATCH], cwd=UPSTREAM_DIR, capture_output=True
    ).returncode == 0
    if already:
        print("Compat patch already applied.")
    else:
        run(["git", "apply", PATCH], cwd=UPSTREAM_DIR)

    hifigan_dir = os.path.join(UPSTREAM_DIR, "hifigan")
    for name in ("generator_LJSpeech.pth.tar", "generator_universal.pth.tar"):
        if not os.path.exists(os.path.join(hifigan_dir, name)):
            print("Unzipping", name)
            with zipfile.ZipFile(os.path.join(hifigan_dir, name + ".zip")) as z:
                z.extractall(hifigan_dir)


def download_pretrained_ljspeech():
    ckpt = os.path.join(PRETRAINED_DIR, "900000.pth.tar")
    if os.path.exists(ckpt) and sha256(ckpt) == LJSPEECH_CKPT_SHA256:
        print("Pretrained LJSpeech checkpoint present and verified.")
        return
    os.makedirs(os.path.join(REPO_ROOT, "pretrained", "download"), exist_ok=True)
    zip_path = os.path.join(REPO_ROOT, "pretrained", "download", "LJSpeech_900000.zip")
    if not os.path.exists(zip_path):
        run([sys.executable, "-m", "gdown", LJSPEECH_GDRIVE_ID, "-O", zip_path])
    if sha256(zip_path) != LJSPEECH_ZIP_SHA256:
        sys.exit("SHA256 mismatch for %s; delete it and retry." % zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(PRETRAINED_DIR)
    if sha256(ckpt) != LJSPEECH_CKPT_SHA256:
        sys.exit("SHA256 mismatch for extracted %s" % ckpt)
    print("Pretrained LJSpeech checkpoint verified:", ckpt)


def setup_bigvgan():
    """NVIDIA BigVGAN code at the pinned commit + the fmax-8k 22 kHz weights (HF cache)."""
    sys.path.insert(0, REPO_ROOT)
    from fs2ra.vocoder_bigvgan import (BIGVGAN_COMMIT, BIGVGAN_DIR, BIGVGAN_REPO, BIGVGAN_REVISION,
                                       BIGVGAN_URL)

    if not os.path.isdir(os.path.join(BIGVGAN_DIR, ".git")):
        run(["git", "clone", BIGVGAN_URL, BIGVGAN_DIR])
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BIGVGAN_DIR, capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != BIGVGAN_COMMIT:
        run(["git", "checkout", BIGVGAN_COMMIT], cwd=BIGVGAN_DIR)
    from huggingface_hub import hf_hub_download

    for name in ("config.json", "bigvgan_generator.pt"):
        print("BigVGAN", name, "->", hf_hub_download(BIGVGAN_REPO, name, revision=BIGVGAN_REVISION))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ljspeech", action="store_true")
    parser.add_argument("--bigvgan", action="store_true")
    args = parser.parse_args()
    setup_upstream()
    if args.bigvgan:
        setup_bigvgan()
    if args.pretrained_ljspeech:
        download_pretrained_ljspeech()
