"""Train/validation loss curves from the CSV logs written by fs2ra.training."""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [("total", "Total loss"), ("mel", "Mel L1 (before PostNet)"), ("postnet_mel", "Mel L1 (after PostNet)"),
          ("duration", "Duration MSE (log domain)"), ("pitch", "Pitch MSE (normalised)"),
          ("energy", "Energy MSE (normalised)")]


def _read(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else {}


def plot_losses(train_csv, val_csv, out_png, title=None):
    train, val = _read(train_csv), _read(val_csv)
    if not train:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, label) in zip(axes.flat, PANELS):
        ax.plot(train["step"], train[key], label="train", lw=1)
        if val:
            ax.plot(val["step"], val[key], "o-", label="validation", ms=3, lw=1.2)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
