# Environment and compute

Constraint: **no paid compute by default**. Local machine for coding, CPU smoke tests and preprocessing; Colab Free or Kaggle GPU for training.

## Local machine (inspected 2026-11-02)
| Item | Finding |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| GPU | **None** (`nvidia-smi` absent, `torch.cuda.is_available() == False`) |
| Python | System 3.14.4 (too new for this stack); **project venv uses uv-managed CPython 3.11.15** |
| Tools | git 2.53, git-lfs present, ffmpeg 8.1, uv 0.11.6; no conda |
| Disk | D: 257 GB free; C: 36 GB free |
| Network | github.com and huggingface.co reachable; Google Drive (gdown) works |
| Credentials | `HF_TOKEN` in git-ignored `.env` (fine-grained read token), verified 2026-11-03 against gated Svarah |

## Version decision
Upstream pins (`torch 1.7`, `numpy 1.19`, `librosa 0.7.2`, `numba 0.48`) cannot be installed on Python ≥3.9 or on current Colab/Kaggle images. We use a modern stack (`project/requirements.txt`: torch 2.5.1, numpy 1.26.4, librosa 0.10.2, pyworld 0.3.5, …) plus a minimal upstream patch (`patches/`). Verified on CPU with the pretrained checkpoint (see PROGRESS.md).

numpy 2: the full local stack (preprocessing: librosa/pyworld/pyarrow) is pinned to numpy 1.26.4. **Training does not need it**: verified 2026-11-03 that `project/English/train.py --dry-run` runs and resumes under Python 3.12 + numpy 2.5.4 + torch 2.15 with only `Unidecode inflect` added (the Colab recipe below), and that a checkpoint written there loads under the local numpy 1.26 stack.

## Local setup (Windows PowerShell or bash, from repo root)
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe scripts/setup_upstream.py --pretrained_ljspeech
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe scripts/baseline_inference.py --text "Hello, how are you?"
```
(Without uv: `py -3.11 -m venv .venv` then `pip install …` equivalently. On Linux/macOS use `.venv/bin/python`.)

## Colab training (notebook written; not yet executed on a GPU)
Training on Colab uses **only the preprocessed features** (no raw audio, MFA or HF token).
1. Locally: `python scripts/package_features.py english` → `dist/english_features.tar` (555 MB, 16,696 `.npy` + train/val lists, train-only `stats.json`, `speakers.json`).
2. Upload it to `MyDrive/fs2ra/`. Code: either push this repo to GitHub and set `REPO_URL` in the notebook, or `git archive -o dist/code.zip HEAD` and upload `code.zip` there too.
3. Open `notebooks/train_colab.ipynb` (T4 GPU runtime), set `LANG`, run all. It mounts Drive, installs `Unidecode inflect`, runs `scripts/setup_upstream.py` (upstream clone + patch + HiFi-GAN), extracts the features to `/content/data` and runs `project/<Lang>/train.py` with `FS2_OUTPUT_ROOT=MyDrive/fs2ra/output`.
4. After a disconnect / usage limit, re-run all cells: training resumes from the newest checkpoint (at most `save_step` = 2,000 steps lost).

Drive space: features 555 MB + 2 full checkpoints (2 × 420 MB) + model-only milestones every 10k steps (140 MB each) ≈ 2.1 GB per language.

## Montreal Forced Aligner (MFA) — forced alignment for phone durations
MFA needs Kaldi binaries from conda-forge (not pip). Verified locally 2026-11-03: **MFA 3.4.3** in a self-contained micromamba env (no system installer, no PATH changes; delete `D:\tools\mfa` to remove).
```bash
mkdir -p /d/tools/mfa && cd /d/tools/mfa
curl -sSL -o micromamba.exe https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64
MAMBA_ROOT_PREFIX=D:/tools/mfa/root ./micromamba.exe create -y -p D:/tools/mfa/env -c conda-forge montreal-forced-aligner
export MAMBA_ROOT_PREFIX=D:/tools/mfa/root MFA_ROOT_DIR=D:/tools/mfa/mfa_root   # models live under D:/tools/mfa too
export FS2_MFA_CMD="D:/tools/mfa/micromamba.exe run -p D:/tools/mfa/env mfa"
D:/tools/mfa/micromamba.exe run -p D:/tools/mfa/env mfa model download acoustic english_us_arpa
```
The preprocessing scripts call `$FS2_MFA_CMD` and strip `__PYVENV_LAUNCHER__`/`PYTHONHOME`/`PYTHONPATH`/`VIRTUAL_ENV` from its environment (otherwise MFA's Python loads the project venv's stdlib on Windows: "SRE module mismatch").

Colab/Linux equivalent (not yet executed): `curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba`, then `bin/micromamba create -y -p /content/mfa -c conda-forge montreal-forced-aligner` and `FS2_MFA_CMD="/content/bin/micromamba run -p /content/mfa mfa"`. Alignment is CPU work and is planned locally; Colab only needs the preprocessed features.

## Still to discover
- GPU type/memory and real steps/sec on Colab (logged as `sec_per_step` in `train_loss.csv`).
- Telugu alignment approach (no pretrained Telugu MFA model; see PROGRESS.md).
