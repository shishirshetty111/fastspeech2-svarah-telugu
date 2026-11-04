"""Language-independent preprocessing stages: export -> align (MFA) -> features."""
import os
import shlex
import subprocess
import sys

from fs2ra.config import load_config
from fs2ra.corpus import export_corpus
from fs2ra.sources import DATA_ROOT
from fs2ra.tables import read_csv, write_csv

UPSTREAM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "FastSpeech2")


def work_dirs(cfg, smoke):
    work = cfg["data"]["work_dir"] + ("_smoke" if smoke else "")
    return {
        "work": work,
        "corpus": os.path.join(work, "corpus"),
        "textgrid_mfa": os.path.join(work, "TextGrid_mfa"),  # raw MFA output
        "textgrid": os.path.join(work, "preprocessed", "TextGrid"),  # silences labelled "sp"
        "preprocessed": os.path.join(work, "preprocessed"),
        "dictionary": os.path.join(work, "dictionary.txt"),
        "index": os.path.join(work, "corpus_index.csv"),
    }


def select_rows(cfg, smoke_train=0, smoke_val=0):
    """Full manifests (with text) for train, val and reserve, in manifest order.
    In smoke mode, the first N train and M val utterances only."""
    lang = cfg["language"]
    full_dir = os.path.join(DATA_ROOT, "manifests", lang)
    if not os.path.exists(os.path.join(full_dir, "train.csv")):
        sys.exit("Missing %s/train.csv: run scripts/data/build_inventory.py and make_splits.py" % full_dir)
    parts = {n: read_csv(os.path.join(full_dir, n + ".csv")) for n in ("train", "val", "reserve")}
    if smoke_train or smoke_val:
        return parts["train"][:smoke_train] + parts["val"][:smoke_val]
    return parts["train"] + parts["val"] + parts["reserve"]


def stage_export(cfg, rows, dirs, normalize_text, build_dictionary):
    sr = cfg["preprocess"]["preprocessing"]["audio"]["sampling_rate"]
    index = export_corpus(cfg["language"], rows, dirs["corpus"], sr, normalize_text)
    write_csv(dirs["index"], index)
    report = build_dictionary(index, dirs["dictionary"])
    print("exported %d utterances (%.3f h) -> %s" % (
        len(index), sum(r["duration_s"] for r in index) / 3600, dirs["corpus"]))
    return index, report


def mfa_command():
    """MFA executable prefix; override with FS2_MFA_CMD, e.g.
    'D:/tools/mfa/micromamba.exe run -p D:/tools/mfa/env mfa' (local) or 'mfa' (Colab conda)."""
    return shlex.split(os.environ.get("FS2_MFA_CMD", "mfa"), posix=(os.name != "nt"))


def stage_align(cfg, dirs, num_jobs=None, extra=()):
    """Pretrained model: `mfa align`. With align.train_acoustic_model (no pretrained model
    for the language, e.g. Telugu): `mfa train` on this corpus + dictionary, which also
    writes the alignments; the trained model is saved for reuse."""
    jobs = ["--num_jobs", str(num_jobs or max(1, (os.cpu_count() or 2) - 1))]
    if cfg["align"].get("train_acoustic_model"):
        model_out = os.path.join(dirs["work"], cfg["align"]["train_acoustic_model"])
        cmd = mfa_command() + [
            "train", dirs["corpus"], dirs["dictionary"], model_out, "--output_directory", dirs["textgrid_mfa"],
            "--clean", *jobs, "--output_format", "long_textgrid", *extra,
        ]
    else:
        cmd = mfa_command() + [
            "align", dirs["corpus"], dirs["dictionary"], cfg["align"]["acoustic_model"], dirs["textgrid_mfa"],
            "--clean", "--overwrite", *jobs, "--output_format", "long_textgrid", *extra,
        ]
    print("+", " ".join(cmd))
    # MFA has its own Python; don't leak this interpreter's venv into it (on Windows
    # the venv launcher exports __PYVENV_LAUNCHER__, which redirects the child's stdlib).
    env = {k: v for k, v in os.environ.items()
           if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    subprocess.run(cmd, check=True, env=env)


def load(cfg_path):
    return load_config(cfg_path)
