# FastSpeech 2 — English (Svarah) and Telugu (IndicTTS)

Work in progress. Phases 0–1 are done: the upstream reference implementation is pinned, patched for a modern Python/PyTorch stack, and verified with the upstream *pretrained* LJSpeech model. **No assignment model has been trained yet**; `English/` and `Telugu/` entrypoints are stubs until later phases.

Based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) @ `d4e79eb` (MIT) and its bundled HiFi-GAN (MIT). Our changes to upstream are in `../patches/`; details in `../docs/UPSTREAM_NOTES.md`.

## Setup (repo root)
```bash
uv venv --python 3.11 .venv            # or: py -3.11 -m venv .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe scripts/setup_upstream.py --pretrained_ljspeech
.venv/Scripts/python.exe -m pytest tests -q
```
Colab/Kaggle instructions: `../docs/ENVIRONMENT.md`.

## Baseline check (upstream pretrained model, not our training)
```bash
.venv/Scripts/python.exe scripts/baseline_inference.py --text "Hello, how are you?"
# -> outputs/baseline/baseline_hello.{wav,png,json}
```
