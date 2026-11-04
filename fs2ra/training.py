"""FastSpeech 2 training loop around the unchanged upstream model, loss, optimizer
and Dataset (ming024/FastSpeech2 train.py), adapted for preemptible notebooks:

1. Resume: the latest `<step>.pth.tar` in ckpt_path is picked up automatically; the
   optimizer schedule restarts from that step. Checkpoints are written atomically
   (tmp + rename) so a disconnect mid-write cannot corrupt the latest one.
2. Disk: only the last `keep_last` full checkpoints (model + Adam state) are kept;
   every `keep_every` steps a model-only copy is kept in ckpt_path/milestones.
3. Logging: CSV instead of TensorBoard. train_loss.csv holds the mean of each loss
   over the last log_step steps, val_loss.csv the full-validation-set means. On
   resume, rows after the restored step are dropped (they are recomputed).
4. Each validation also writes one teacher-forced sample (ground-truth durations,
   pitch and energy; mel from the model) as wav + png. Real text-only inference is
   project/<Lang>/inference.py.
5. Optional mixed precision (--amp): forward pass and loss under torch.autocast in
   bfloat16 where the GPU supports it (L4/A100), else float16 with a GradScaler (T4).
   Weights, optimizer state and checkpoints stay float32, so runs can switch between
   amp and fp32 (or GPUs) on resume. Validation and inference run in float32.
6. The train-only stats.json / speakers.json and the config are copied next to the
   checkpoints, so a checkpoint folder is self-contained for inference.

Teacher forcing (as upstream): during training the variance adaptor is fed the
*target* durations, pitch and energy, and the predictors are trained against them
with MSE; at inference the predicted values are used instead.
"""
import csv
import glob
import os
import re
import shutil
import time
from argparse import Namespace
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.io import wavfile
from torch.utils.data import DataLoader
from tqdm import tqdm

from fs2ra.config import upstream_configs
from fs2ra.symbols import check_phone_strings, register_language

LOSS_NAMES = ["total", "mel", "postnet_mel", "pitch", "energy", "duration"]
CKPT_RE = re.compile(r"^(\d+)\.pth\.tar$")


def latest_step(ckpt_dir):
    steps = [int(m.group(1)) for m in (CKPT_RE.match(f) for f in os.listdir(ckpt_dir)) if m] \
        if os.path.isdir(ckpt_dir) else []
    return max(steps, default=0)


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _optimizer_state(optimizer):
    # upstream sets lr as numpy float64; store plain floats so checkpoints written
    # under numpy 2 (Colab) load under numpy 1.x with weights_only=True
    state = optimizer._optimizer.state_dict()
    for group in state["param_groups"]:
        group["lr"] = float(group["lr"])
    return state


def _prune(ckpt_dir, keep_last):
    steps = sorted(int(m.group(1)) for m in (CKPT_RE.match(f) for f in os.listdir(ckpt_dir)) if m)
    for step in steps[:-keep_last]:
        os.remove(os.path.join(ckpt_dir, "%d.pth.tar" % step))


def _truncate_csv(path, max_step):
    """Keep rows with step <= max_step (all rows are dropped on a fresh start)."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    keep = [rows[0]] + [r for r in rows[1:] if int(r[0]) <= max_step] if rows else []
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(keep)


def _append_csv(path, header, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def _snapshot_metadata(cfg, config_path, ckpt_dir):
    """Front-end/normalisation files that inference must share with training."""
    pre = cfg["preprocess"]["path"]["preprocessed_path"]
    for name in ("stats.json", "speakers.json"):
        shutil.copyfile(os.path.join(pre, name), os.path.join(ckpt_dir, name))
    shutil.copyfile(config_path, os.path.join(ckpt_dir, "config.yaml"))


def evaluate(model, loss_fn, loader, device, max_batches=None):
    """Mean of each loss over the validation set (weighted by batch size)."""
    from utils.tools import to_device  # upstream

    sums, n, last = np.zeros(len(LOSS_NAMES)), 0, None
    model.eval()
    with torch.no_grad():
        for i, batchs in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            for batch in batchs:
                batch = to_device(batch, device)
                output = model(*(batch[2:]))
                losses = loss_fn(batch, output)
                sums += np.array([l.item() for l in losses]) * len(batch[0])
                n += len(batch[0])
                last = (batch, output)
    model.train()
    return sums / max(n, 1), last


def _write_sample(sample, vocoder, configs, out_dir, step):
    from utils.tools import synth_one_sample  # upstream

    preprocess_config, model_config, _ = configs
    batch, output = sample
    fig, wav_target, wav_pred, basename = synth_one_sample(batch, output, vocoder, model_config, preprocess_config)
    stem = os.path.join(out_dir, "step%07d_%s" % (step, basename))
    fig.savefig(stem + ".png", dpi=100)
    import matplotlib.pyplot as plt

    plt.close(fig)
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    if wav_pred is not None:
        wavfile.write(stem + "_teacher_forced.wav", sr, wav_pred)
        wavfile.write(stem + "_vocoded_target.wav", sr, wav_target)


def _worker_init(language, worker_id):
    register_language(language)  # spawned workers re-import upstream `text` without our phones


def _amp_dtype(amp, device):
    if not amp:
        return None
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16  # CPU autocast: only useful to exercise the code path


def train(cfg, config_path, total_step=None, dry_run=False, num_workers=2, seed=1234, amp=False):
    from dataset import Dataset  # upstream
    from model import FastSpeech2Loss
    from utils.model import get_model, get_param_num, get_vocoder
    from utils.tools import to_device

    from fs2ra.plots import plot_losses

    n_symbols = register_language(cfg["language"])  # before the model is built
    configs = upstream_configs(cfg)
    preprocess_config, model_config, train_config = configs
    step_cfg, opt_cfg = train_config["step"], train_config["optimizer"]
    paths = train_config["path"]
    max_val_batches = None
    if dry_run:  # tiny end-to-end run of every code path, into separate dirs
        for k in paths:
            paths[k] = paths[k].rstrip("/\\") + "_dryrun"
        step_cfg.update(log_step=2, val_step=3, synth_step=3, save_step=3)
        train_config["checkpoint"] = dict(train_config.get("checkpoint", {}), keep_last=2, keep_every=6)
        max_val_batches = 1
        num_workers = 0
    if total_step is not None:
        step_cfg["total_step"] = total_step
    total_step = step_cfg["total_step"]
    keep = train_config.get("checkpoint", {})
    keep_last, keep_every = keep.get("keep_last", 2), keep.get("keep_every", 10000)

    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    ckpt_dir, log_dir, result_dir = paths["ckpt_path"], paths["log_path"], paths["result_path"]
    os.makedirs(os.path.join(ckpt_dir, "milestones"), exist_ok=True)
    _snapshot_metadata(cfg, config_path, ckpt_dir)
    with open(os.path.join(log_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"preprocess": preprocess_config, "model": model_config, "train": train_config}, f,
                       sort_keys=False, allow_unicode=True)

    restore_step = latest_step(ckpt_dir)
    if restore_step >= total_step:
        print("already at step %d >= total_step %d; nothing to do" % (restore_step, total_step))
        return restore_step
    train_csv, val_csv = os.path.join(log_dir, "train_loss.csv"), os.path.join(log_dir, "val_loss.csv")
    _truncate_csv(train_csv, restore_step)
    _truncate_csv(val_csv, restore_step)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed + restore_step)
    amp_dtype = _amp_dtype(amp, device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    print("device %s | %s | precision %s | resume from step %d | total_step %d" % (
        device, torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "amp " + str(amp_dtype).replace("torch.", "") if amp_dtype else "float32", restore_step, total_step))

    train_set = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    val_set = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
    check_phone_strings(train_set.text + val_set.text, cfg["language"])
    batch_size, group_size = opt_cfg["batch_size"], 4  # upstream: sort within groups of 4 batches
    assert batch_size * group_size < len(train_set)
    loader_kw = dict(num_workers=num_workers, persistent_workers=num_workers > 0,
                     pin_memory=device.type == "cuda", worker_init_fn=partial(_worker_init, cfg["language"]))
    train_loader = DataLoader(train_set, batch_size=batch_size * group_size, shuffle=True,
                              collate_fn=train_set.collate_fn, **loader_kw)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=val_set.collate_fn,
                            **loader_kw)

    model, optimizer = get_model(Namespace(restore_step=restore_step), configs, device, train=True)
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)
    vocoder = get_vocoder(model_config, device)
    print("FastSpeech2 parameters: %d | symbols %d | train %d utts, val %d utts" % (
        get_param_num(model), n_symbols, len(train_set), len(val_set)))

    grad_acc, clip = opt_cfg["grad_acc_step"], opt_cfg["grad_clip_thresh"]
    step = restore_step + 1
    running, n_running, t0 = np.zeros(len(LOSS_NAMES)), 0, time.time()
    bar = tqdm(total=total_step, initial=restore_step, desc="train", dynamic_ncols=True)
    while step <= total_step:
        for batchs in train_loader:
            for batch in batchs:
                batch = to_device(batch, device)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    output = model(*(batch[2:]))
                    losses = loss_fn(batch, output)
                scaler.scale(losses[0] / grad_acc).backward()
                if step % grad_acc == 0:
                    scaler.unscale_(optimizer._optimizer)  # no-ops unless float16
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer._update_learning_rate()  # = upstream step_and_update_lr, via the scaler
                    scaler.step(optimizer._optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                running += np.array([l.item() for l in losses])
                n_running += 1

                if step % step_cfg["log_step"] == 0:
                    lr = optimizer._optimizer.param_groups[0]["lr"]
                    sec = (time.time() - t0) / n_running
                    mean = running / n_running
                    _append_csv(train_csv, ["step"] + LOSS_NAMES + ["lr", "sec_per_step"],
                                [step] + ["%.5f" % v for v in mean] + ["%.3e" % lr, "%.4f" % sec])
                    bar.write("step %d  " % step + "  ".join("%s %.4f" % kv for kv in zip(LOSS_NAMES, mean))
                              + "  lr %.2e  %.3fs/step" % (lr, sec))
                    running, n_running, t0 = np.zeros(len(LOSS_NAMES)), 0, time.time()

                if step % step_cfg["val_step"] == 0:
                    val, sample = evaluate(model, loss_fn, val_loader, device, max_val_batches)
                    _append_csv(val_csv, ["step"] + LOSS_NAMES, [step] + ["%.5f" % v for v in val])
                    bar.write("VAL step %d  " % step + "  ".join("%s %.4f" % kv for kv in zip(LOSS_NAMES, val)))
                    if step % step_cfg["synth_step"] == 0 and sample is not None:
                        _write_sample(sample, vocoder, configs, result_dir, step)
                    plot_losses(train_csv, val_csv, os.path.join(log_dir, "loss_curves.png"))
                    t0 = time.time()  # exclude validation from the step timing

                if step % step_cfg["save_step"] == 0 or step == total_step:
                    _atomic_save({"model": model.state_dict(), "optimizer": _optimizer_state(optimizer),
                                  "step": step}, os.path.join(ckpt_dir, "%d.pth.tar" % step))
                    if step % keep_every == 0 or step == total_step:
                        _atomic_save({"model": model.state_dict(), "step": step},
                                     os.path.join(ckpt_dir, "milestones", "%d.pth.tar" % step))
                    _prune(ckpt_dir, keep_last)

                bar.update(1)
                step += 1
                if step > total_step:
                    break
            if step > total_step:
                break
    bar.close()
    if os.path.exists(train_csv):
        plot_losses(train_csv, val_csv, os.path.join(log_dir, "loss_curves.png"))
    print("done: step %d, checkpoints in %s" % (total_step, ckpt_dir))
    return total_step


def find_checkpoints(ckpt_dir):
    """All saved steps (full and milestone), for model selection."""
    found = {}
    for path in glob.glob(os.path.join(ckpt_dir, "*.pth.tar")) + glob.glob(os.path.join(ckpt_dir, "milestones", "*.pth.tar")):
        m = CKPT_RE.match(os.path.basename(path))
        if m:
            found.setdefault(int(m.group(1)), path)
    return dict(sorted(found.items()))
