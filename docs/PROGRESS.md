# Implementation progress

Status (2026-11-03): **Phases 0, 1 and 2 complete.** Nothing has been trained. The only audio produced is from the *upstream* pretrained LJSpeech model (not an assignment deliverable). Data decisions: `docs/DATA.md`.

## Phase 0 — repo and environment inspection ✅
- [x] Read AGENTS.md, all docs, original PDF (visually, all 3 pages)
- [x] Cloned and pinned upstream `ming024/FastSpeech2 @ d4e79eb` (MIT; HiFi-GAN MIT) → `docs/UPSTREAM_NOTES.md`
- [x] Environment checked (no local GPU; Python 3.11 venv; network OK) → `docs/ENVIRONMENT.md`
- [x] Modern-stack compat patch + reproducible setup script + CPU tests

## Phase 1 — baseline pretrained English inference ✅
- [x] Upstream LJSpeech 900k checkpoint + upstream HiFi-GAN (LJSpeech) synthesised "Hello, how are you?" on CPU

## Phase 2 — dataset ingestion + manifests ✅
- [x] Svarah access via `HF_TOKEN` (`.env`, git-ignored; fine-grained read token) verified
- [x] Both datasets downloaded at pinned revisions; per-utterance inventories measured from decoded audio
- [x] Speaker identity resolved for both datasets; QC with logged reasons; deterministic 5 h / 1 h / reserve splits
- [x] Exact test sentences extracted and visually verified (`project/<Lang>/test_sentences.json`)
- [x] Tests: disjointness, hours, bounds, speaker coverage, exact rebuild, test-sentence fidelity

## Experiments log

### 2026-11-02 — environment + upstream
Commands (bash, repo root):
```
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
uv pip install --python .venv/Scripts/python.exe "numpy<2" librosa==0.10.2.post1 scipy soundfile pyworld tgt==1.5 PyYAML matplotlib scikit-learn tqdm unidecode inflect g2p-en==2.1.0 pypinyin tensorboard gdown pytest
git clone https://github.com/ming024/FastSpeech2 third_party/FastSpeech2   # HEAD d4e79eb
```
Upstream compat edits were then made and exported to `patches/ming024_fastspeech2_compat.patch`; verified that the patch applies cleanly to a fresh clone at d4e79eb (`git apply --check`).

### 2026-11-02 — pretrained checkpoint
- Listed upstream Drive folder `1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F` (from upstream README): `LJSpeech_900000.zip`, `LibriTTS_800000.pth.tar`, `AISHELL3_600000.pth.tar`.
- Downloaded `LJSpeech_900000.zip` (gdown id `1r3fYhnblBJ8hDKDSUDtidJ-BN-xAM9pe`, 354 MB, sha256 `b64a4486…689b`) → `pretrained/LJSpeech/900000.pth.tar` (sha256 `8e248625…357d`). Keys `model`, `optimizer`; 35,163,622 params; embedding 361×256.
- Plain `torch.load(weights_only=True)` fails only on numpy scalar/dtype globals (optimizer state); these are allowlisted instead of using `weights_only=False`.
- HiFi-GAN generators unzipped from upstream `hifigan/*.zip` (`generator_LJSpeech.pth.tar` sha256 `bb4b0cb7…05e4`).

### 2026-11-02 — baseline synthesis
```
.venv/Scripts/python.exe scripts/baseline_inference.py --text "Hello, how are you?"
```
- Exit 0, ~13 s wall on CPU. Phonemes: `{HH AH0 L OW1 sp HH AW1 AA1 R Y UW1}` (trailing `?` stripped by upstream front end).
- `outputs/baseline/baseline_hello.wav`: 74,284 bytes, 22050 Hz mono PCM_16, 1.683 s, peak 0.957, RMS 0.160, no non-finite samples.
- Predicted durations `[3,4,10,34,13,10,21,8,8,16,18]`, sum 145 = mel frames 145 = samples/256.
- Spectrogram plot (`baseline_hello.png`) shows harmonic structure and a low-energy pause at the `sp` token. **No human listening check yet** — please listen to the WAV.

### 2026-11-02 — tests
```
.venv/Scripts/python.exe -m pytest tests -q   → 5 passed (≈68 s)
```
Covers: patch applied; mel/STFT config == HiFi-GAN config; CPU mel/energy shapes and finiteness; checkpoint keys/shapes == upstream architecture; end-to-end baseline WAV + duration/mel/sample consistency.

### 2026-11-03 — Phase 2: data
Commands (repo root, bash):
```
uv pip install --python .venv/Scripts/python.exe pyarrow==17.0.0 huggingface_hub pymupdf   # pyarrow 26 needs numpy>=2
.venv/Scripts/python.exe scripts/data/download_datasets.py --language english telugu        # 1.1 GB + 6.4 GB
.venv/Scripts/python.exe scripts/data/build_inventory.py --language english --limit 50     # smoke test first
.venv/Scripts/python.exe scripts/data/build_inventory.py --language english telugu         # 9 s / 73 s
.venv/Scripts/python.exe scripts/data/make_splits.py --language english telugu
.venv/Scripts/python.exe -m pytest tests -q                                                  # 21 passed (~64 s)
```
Evidence (from `project/<Lang>/splits/split_report.json`):
- **English (Svarah @ ebbf777):** 6,656 utts / 9.614 h, all 16 kHz mono. No speaker column; recording ids are sessions (2,938), but map 1:1 to 115 demographic profiles (paper: 117 speakers) → profile hash = speaker proxy. QC pool 7.706 h (≤ 15 s). Split: **train 5.038 h (3,433 utts), val 1.028 h (777), reserve 1.641 h**; 113 speakers, all val speakers in train. Multi-speaker model required.
- **Telugu (IndicTTS_Telugu @ 7ab3c18):** card says ~8.74 h / 48 kHz; measured **36.712 h**, female 23.92 h @ 16 kHz (mostly 20–150 s paragraphs), male 12.79 h @ 48 kHz. Single-speaker **male** chosen (8.062 h QC pool ≤ 15 s vs. 3.63 h for female). Split: **train 5.000 h (2,623 utts), val 1.000 h (524), reserve 2.062 h**; all train/val clips ≤ 11 s.
- Disjoint on utt_id, audio sha1 and transcript; rebuild from inventory reproduces committed splits exactly.
- Test sentences: PyMuPDF logical text + 300 dpi render comparison. **Correction:** E3 is `rhyth  m` with two spaces (docs had one); T3 has `అడ  విలో` with two spaces. The PDF text layer seen by simpler extractors is corrupted for Telugu (e.g. `విశ్లషే ించారు`); the verified strings are in `project/Telugu/test_sentences.json`.

Files: `fs2ra/{hf,sources,inventory,splits,tables}.py`, `scripts/data/{download_datasets,build_inventory,make_splits}.py`, `project/{English,Telugu}/splits/*`, `project/{English,Telugu}/test_sentences.json`, `tests/{test_splits,test_manifests,test_test_sentences}.py`, `docs/DATA.md`.

## Current blockers / open issues
- MFA not installed (needs conda; plan to run on Colab/Kaggle or local Miniconda). Telugu alignment approach undecided.
- Upstream preprocessing fits normalisation stats on train+val and splits by count; must be replaced (see UPSTREAM_NOTES.md "Known upstream issues").
- Repository is not yet a git repo (`git init` not run).
- Colab/Kaggle flow written but not yet executed; numpy-2 images need the documented downgrade + restart.
- English val is all tier-0 (clean, ≤ 11 s) while train has 254 tier-1/2 utterances (11–15 s or digits); documented asymmetry.
- English train includes clips up to 15 s → `max_seq_len` must be raised above 1000 (≈ 1,300 frames) in the English model config.
- English digit transcripts (135 in train) need a verbalisation step consistent between MFA lexicon and training text; failures are replaced from reserve.
- Svarah speaker proxy may merge up to two speaker pairs (115 vs. 117).

## Next action (single task)
Phase 3: English preprocessing smoke test on 10–50 train utterances — export audio (16 → 22.05 kHz) + text for MFA, align with the English MFA model (on Colab/Kaggle or local Miniconda), extract mel/duration/F0/energy with the upstream definitions, fit pitch/energy stats on train only, and verify duration sums = mel frames.
