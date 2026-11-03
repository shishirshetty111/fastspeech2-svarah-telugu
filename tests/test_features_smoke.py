"""Preprocessed English smoke features are consumable by the upstream Dataset,
model and loss (one CPU forward/backward with teacher-forced targets).

Requires: python project/English/preprocess.py --smoke 30 10 --stage export align features
"""
import json
import os

import numpy as np
import pytest
import torch

from conftest import REPO_ROOT
from fs2ra.config import load_config
from fs2ra.sources import DATA_ROOT

SMOKE = os.path.join(DATA_ROOT, "english_smoke", "preprocessed")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(SMOKE, "train.txt")),
                                reason="English smoke features not built")


def configs():
    cfg = load_config(os.path.join(REPO_ROOT, "project", "English", "config.yaml"))
    pre, model, train = cfg["preprocess"], cfg["model"], cfg["train"]
    pre["path"]["preprocessed_path"] = SMOKE
    train["optimizer"]["batch_size"] = 4
    return pre, model, train


def test_all_targets_consistent():
    with open(os.path.join(SMOKE, "train.txt"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    with open(os.path.join(SMOKE, "val.txt"), encoding="utf-8") as f:
        lines += f.read().splitlines()
    for line in lines:
        basename, speaker, phones, _ = line.split("|")
        load = lambda k: np.load(os.path.join(SMOKE, k, "%s-%s-%s.npy" % (speaker, k, basename)))
        mel, dur, pitch, energy = load("mel"), load("duration"), load("pitch"), load("energy")
        assert dur.sum() == mel.shape[0] and mel.shape[1] == 80
        assert len(phones.strip("{}").split()) == len(dur) == len(pitch) == len(energy)


def test_stats_from_train_only():
    with open(os.path.join(SMOKE, "preprocess_report.json")) as f:
        assert json.load(f)["stats_fit_on"] == "train only"


def test_upstream_forward_backward():
    from dataset import Dataset
    from model import FastSpeech2, FastSpeech2Loss
    from utils.tools import to_device

    pre, model_cfg, train = configs()
    ds = Dataset("train.txt", pre, train, sort=True, drop_last=False)
    batch = to_device(ds.collate_fn([ds[i] for i in range(8)])[0], torch.device("cpu"))
    model = FastSpeech2(pre, model_cfg)
    loss_fn = FastSpeech2Loss(pre, model_cfg)
    model.train()
    output = model(*batch[2:])  # teacher forcing: true durations, pitch, energy
    losses = loss_fn(batch, output)
    assert all(torch.isfinite(x) for x in losses)
    losses[0].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # predicted mel length equals the target length under teacher forcing
    assert torch.equal(output[9], batch[7])
