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

Known risk: Colab/Kaggle images ship **numpy 2.x**; `project/requirements.txt` downgrades to 1.26.4 (pip will do it; restart the runtime after install). numpy 2 compatibility is not yet verified.

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

## Colab / Kaggle (planned; not yet executed)
```bash
!git clone <this-repo-url> fs2 && cd fs2
!pip install -q -r requirements.txt          # keep the preinstalled CUDA torch
# Runtime -> Restart session (numpy downgrade), then:
!python scripts/setup_upstream.py --pretrained_ljspeech
!python -m pytest tests -q
!python scripts/baseline_inference.py --text "Hello, how are you?"
```
Data: `python scripts/data/download_datasets.py` (≈7.5 GB; set `FS2_DATA_ROOT` to a Drive path on Colab to keep it across sessions). Persist `preprocessed_data/`, checkpoints and logs to Google Drive (Colab) or `/kaggle/working` + dataset versions (Kaggle) so training can resume after disconnects. HF token via environment variable / Colab secrets only, never in tracked files.

## Still to discover
- GPU type/memory and real steps/sec from a short training run on Colab/Kaggle.
- MFA installation path (conda is the supported route; not installed locally) and a Telugu alignment approach.
