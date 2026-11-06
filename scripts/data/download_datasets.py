"""Download the pinned parquet shards of Svarah and/or IndicTTS Telugu.

    python scripts/data/download_datasets.py --language english telugu
Set FS2_DATA_ROOT to download elsewhere (e.g. a mounted Google Drive on Colab).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from huggingface_hub import hf_hub_download  # noqa: E402

from fs2ra.hf import load_hf_token  # noqa: E402
from fs2ra.sources import SOURCES, raw_dir  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    args = parser.parse_args()
    token = load_hf_token()
    for lang in args.language:
        src = SOURCES[lang]
        if src["gated"] and not token:
            sys.exit("%s is gated: accept its terms on the Hub and set HF_TOKEN" % src["repo_id"])
        for name in src["files"]:
            path = hf_hub_download(
                src["repo_id"], name, repo_type="dataset", revision=src["revision"],
                token=token, local_dir=raw_dir(lang),
            )
            print(lang, path, "%.1f MB" % (os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    main()
