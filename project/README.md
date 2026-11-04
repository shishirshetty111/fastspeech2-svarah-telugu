# FastSpeech 2 — English (Svarah) and Telugu (IndicTTS)

Two FastSpeech 2 acoustic models **trained from scratch**: English on 5 h of Svarah, and Telugu on 5 h of IndicTTS Telugu. Each has a disjoint 1 h validation set. The vocoder is the pretrained HiFi-GAN "universal" model. Full write-up: [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf).

| | English | Telugu |
|---|---|---|
| Status | trained (50k steps); **submitted checkpoint: step 20,000** | in progress (front end and alignment pipeline done; training pending) |
| Checkpoint | [`English/checkpoints/20000.pth.tar`](English/checkpoints/) (Git LFS, 142 MB, model weights only) + `stats.json`, `speakers.json`, `config.yaml` | — |
| Results | [`Results/English/`](Results/English/) | — |

Based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) @ `d4e79eb` (MIT) and its bundled HiFi-GAN (MIT). Our changes to upstream are in `../patches/`, and the code around it is in `../fs2ra/`. Details: `../docs/UPSTREAM_NOTES.md`.

## Layout
```
English/, Telugu/
  preprocess.py   export -> MFA alignment -> mel / duration / pitch / energy targets
  train.py        resumable training (--dry-run, --amp); Colab: ../notebooks/train_colab.ipynb
  inference.py    text-only synthesis of test_sentences.json (+ variance plots)
  config.yaml     data, preprocessing, model, training and inference settings
  checkpoints/    submitted checkpoint + training-time stats.json / speakers.json / config.yaml
Results/<Lang>/
  training_loss.png, validation_loss.png
  test_generated_audios/<id>.wav
  variance_adaptor_plots/{duration,pitch,energy}/<id>.png
  inference_summary.json    phones, pronunciation sources, predicted durations / F0 / energy
Report/Technical_Report.pdf  (source: Technical_Report.md; build: python scripts/build_report.py)
```

## Quick start: synthesise with the trained English model
```bash
git lfs install && git clone https://github.com/shishirshetty111/fastspeech2-svarah-telugu.git
cd fastspeech2-svarah-telugu
python -m venv .venv && .venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/setup_upstream.py                          # clones + patches upstream, unpacks HiFi-GAN
python project/English/inference.py --out_dir my_outputs   # the 5 test sentences, CPU, about 1 min
python project/English/inference.py --text "Any English sentence." --ids demo --out_dir my_outputs
```
If `git lfs` was not installed before cloning, run `git lfs pull` to fetch the checkpoint.

## Full reproduction
The data download, splits, preprocessing (MFA 3.4.3), Colab training and inference commands are listed in Section 11 of the report and in `../docs/ENVIRONMENT.md`. Svarah is gated: put a read token in `.env` as `HF_TOKEN=...` (git-ignored). Tests: `python -m pytest tests -q`.

## Data and licences
- Svarah: CC BY 4.0.
- IndicTTS: IIT Madras licence.
- No audio or transcripts are redistributed. The split files contain only IDs and hashes.
- Upstream FastSpeech 2 and HiFi-GAN: MIT; their licence notices are kept in the upstream checkout.
