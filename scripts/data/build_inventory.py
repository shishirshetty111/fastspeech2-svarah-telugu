"""Measure every utterance of the downloaded shards and write
data/manifests/<language>/inventory.csv (git-ignored: contains transcripts).

    python scripts/data/build_inventory.py --language english --limit 50   # smoke test
    python scripts/data/build_inventory.py --language english telugu
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fs2ra.inventory import iter_inventory  # noqa: E402
from fs2ra.sources import DATA_ROOT, SOURCES  # noqa: E402
from fs2ra.tables import write_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    args = parser.parse_args()
    for lang in args.language:
        t0 = time.time()
        rows = list(iter_inventory(lang, limit=args.limit))
        name = "inventory.csv" if args.limit is None else "inventory_first%d.csv" % args.limit
        out = os.path.join(DATA_ROOT, "manifests", lang, name)
        write_csv(out, rows)
        hours = sum(r["duration_s"] for r in rows) / 3600
        print("%s: %d utterances, %.3f h -> %s (%.0f s)" % (lang, len(rows), hours, out, time.time() - t0))


if __name__ == "__main__":
    main()
