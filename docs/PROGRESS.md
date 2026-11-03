# Implementation progress

Status (2026-11-03): **Phases 0–2 complete; Phase 3 complete for English** (Telugu preprocessing is Phase 5). Nothing has been trained. The only audio produced is from the *upstream* pretrained LJSpeech model (not an assignment deliverable). Data decisions: `docs/DATA.md`.

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

## Phase 3 — English preprocessing ✅
- [x] MFA 3.4.3 installed (isolated micromamba env, `D:	ools\mfa`); `english_us_arpa` phones ⊆ upstream symbols
- [x] Shared English front end (`fs2ra/text_en.py`) for alignment, training and inference
- [x] 30+10 smoke test: export → align → features → upstream Dataset/model forward+backward on CPU
- [x] Full run: train 5.000 h / val 1.000 h, train-only stats, every target validated

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

### 2026-11-03 — Phase 3: English preprocessing
Commands (bash; `FS2_MFA_CMD`, `MAMBA_ROOT_PREFIX`, `MFA_ROOT_DIR` as in ENVIRONMENT.md):
```
mfa model download acoustic english_us_arpa
python project/English/preprocess.py --smoke 30 10 --stage export align features --num_jobs 4   # 40/40 ok, MFA 31 s
python -m pytest tests/test_features_smoke.py                                                   # upstream fwd/bwd OK
python project/English/preprocess.py --stage export align features --num_jobs 7                # 18 min
python project/English/preprocess.py --stage features                                          # after adding filters, 12.5 min
python -m pytest tests -q                                                                        # 39 passed (183 s)
```
Findings:
- **MFA 3 leaves silences unlabelled**; upstream (built for MFA 1.x) would silently drop every internal pause from the duration targets. Fixed by relabelling empty intervals `sp` before upstream feature extraction.
- MFA's Python picked up the project venv's stdlib on Windows (`__PYVENV_LAUNCHER__`); scrubbed from the subprocess env.
- Filled pauses absorbed into phones (up to 3.3 s `AH0`) → `max_phone_frames: 50` filter (437 rejects incl. reserve).
- Reserve top-up initially leaked 4 sentences into both train and val; top-up is now text-disjoint and tested.
- `row_index`/`sample_rate` were written as floats in the split CSVs; fixed (content unchanged).
- Visual check (mel + phone boundaries + pitch/energy) on a val utterance: boundaries match spectral transitions.
Result: train **5.000 h (3,417)**, val **1.000 h (757)**, 113 speakers; details in `docs/DATA.md`.

## Current blockers / open issues
- Telugu alignment: no pretrained Telugu MFA model; needs a Telugu G2P/lexicon and an MFA model trained on the Telugu corpus (Phase 5).
- Colab/Kaggle flow written but not yet executed; numpy-2 images need the documented downgrade + restart.
- English val is all tier-0 (clean, ≤ 11 s) while train has 254 tier-1/2 utterances (11–15 s or digits); documented asymmetry.
- English `max_seq_len` set to 1500 (longest utterance 1,272 frames).
- English verbalisation of digits is heuristic (e.g. 4-digit numbers read as years); mismatches with what speakers said are only caught if they cause alignment outliers.
- Heteronyms: inference uses the lexicon's first pronunciation, so E2a/E2b `conduct` would get the same stress — to be addressed in Phase 4 inference.
- Svarah speaker proxy may merge up to two speaker pairs (115 vs. 117).

## Next action (single task)
Phase 4: English training on Colab — `project/English/train.py` (wrapping upstream train loop with resumable Drive checkpoints, CSV loss logging, train/val loss plots), a feature-package script for upload, a Colab notebook, and a short CPU dry run here before using GPU time.
