"""Pack the preprocessed training features of one language into a single tar for
upload to Google Drive (training on Colab/Kaggle needs only these, not raw audio,
MFA or the TextGrids).

    python scripts/package_features.py english        # -> dist/english_features.tar
Extract on the notebook with  tar -xf english_features.tar -C $FS2_DATA_ROOT
(recreates <lang>/preprocessed/...).
"""
import argparse
import hashlib
import os
import sys
import tarfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from fs2ra.sources import DATA_ROOT  # noqa: E402

FILES = ["train.txt", "val.txt", "speakers.json", "stats.json", "preprocess_report.json", "feature_status.csv"]
DIRS = ["mel", "pitch", "energy", "duration"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lang", choices=["english", "telugu"])
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "dist"))
    args = parser.parse_args()

    pre = os.path.join(DATA_ROOT, args.lang, "preprocessed")
    needed = set()
    for split in ("train.txt", "val.txt"):
        with open(os.path.join(pre, split), encoding="utf-8") as f:
            for line in f.read().splitlines():
                basename, speaker = line.split("|")[:2]
                needed |= {os.path.join(d, "%s-%s-%s.npy" % (speaker, d, basename)) for d in DIRS}
    missing = [p for p in needed if not os.path.exists(os.path.join(pre, p))]
    if missing:
        sys.exit("missing %d feature files, e.g. %s" % (len(missing), missing[0]))

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "%s_features.tar" % args.lang)
    arc_root = "%s/preprocessed" % args.lang
    with tarfile.open(out, "w") as tar:  # float features barely compress; plain tar
        for rel in FILES + sorted(needed):
            tar.add(os.path.join(pre, rel), arcname="%s/%s" % (arc_root, rel.replace(os.sep, "/")))
    sha = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    print("%s: %d feature files, %.1f MB, sha256 %s" % (out, len(needed), os.path.getsize(out) / 2**20, sha.hexdigest()))


if __name__ == "__main__":
    main()
