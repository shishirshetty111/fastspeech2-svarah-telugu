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


def plot_single(csv_path, out_png, split, title, mark_step=None):
    """One split per figure (Results layout: training_loss.png / validation_loss.png);
    mark_step: the step of the submitted checkpoint, drawn as a dashed line."""
    data = _read(csv_path)
    if not data:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, label) in zip(axes.flat, PANELS):
        style = dict(lw=1) if split == "train" else dict(marker="o", ms=3, lw=1.2, color="C1")
        ax.plot(data["step"], data[key], **style)
        if mark_step:
            ax.axvline(mark_step, ls="--", color="0.5", lw=1)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
