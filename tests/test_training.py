"""Checkpoint/log bookkeeping used for resuming on preemptible notebooks."""
import csv
import os

import numpy as np
import torch

from fs2ra import training


def test_latest_step_prune_and_atomic_save(tmp_path):
    d = str(tmp_path)
    assert training.latest_step(d) == 0
    for step in (2000, 4000, 6000):
        training._atomic_save({"step": step}, os.path.join(d, "%d.pth.tar" % step))
    open(os.path.join(d, "stats.json"), "w").close()
    assert training.latest_step(d) == 6000
    training._prune(d, keep_last=2)
    assert sorted(os.listdir(d)) == ["4000.pth.tar", "6000.pth.tar", "stats.json"]
    assert not any(f.endswith(".tmp") for f in os.listdir(d))


def test_truncate_csv_drops_rows_after_restored_step(tmp_path):
    path = str(tmp_path / "train_loss.csv")
    for step in (100, 200, 300):
        training._append_csv(path, ["step", "total"], [step, 1.0])
    training._truncate_csv(path, 200)
    with open(path, newline="") as f:
        assert [r[0] for r in csv.reader(f)] == ["step", "100", "200"]
    training._truncate_csv(path, 0)  # fresh start keeps only the header
    training._append_csv(path, ["step", "total"], [100, 2.0])
    with open(path, newline="") as f:
        assert [r[0] for r in csv.reader(f)] == ["step", "100"]


def test_optimizer_state_has_plain_float_lr():
    class Wrapped:  # mimics upstream ScheduledOptim (numpy float64 lr)
        _optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])

    Wrapped._optimizer.param_groups[0]["lr"] = np.float64(1e-4)
    lr = training._optimizer_state(Wrapped)["param_groups"][0]["lr"]
    assert type(lr) is float and lr == 1e-4
