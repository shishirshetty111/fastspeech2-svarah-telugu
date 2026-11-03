"""Tiny CSV helpers (UTF-8, header row) so manifests stay dependency-free."""
import csv
import os


def write_csv(path, rows, columns=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    columns = columns or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path, numeric=()):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in numeric:
            if k in r and r[k] != "":
                r[k] = float(r[k])
    return rows
