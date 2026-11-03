# Conversation handoff — FastSpeech 2 research assignment

This document captures technical understanding and decisions discussed with the applicant. It is guidance, not proof that any code or experiment has already been run.

## Scope / intent
The task is an RA screening assignment from IIIT Hyderabad. Aim for a credible, reproducible end-to-end experimental implementation, not production speech service or an entirely new model. The user wants to minimize spending on compute and is considering Colab Free / Kaggle free GPUs. Do not assume a paid GPU, and do not imply that free capacity is guaranteed. As of 2026-11-02, the deadline is 2026-11-07.

## Architecture mental model
- Text normalization (with care on adversarial test samples), language-specific G2P/phonemization, ID/embedding lookup.
- Transformer/FFT encoder turns input phonemes into contextual features.
- Variance adaptor predicts **duration** (mel-frame count per phoneme), **pitch/F0**, **energy**. In phoneme-level setups pitch/energy targets are phoneme averages; in frame-level setups targets are frame-aligned—follow selected upstream config.
- Length regulator expands encoded phoneme vectors according to duration. Decoder predicts mel-spectrogram; postnet optionally refines. Compatible separately pretrained HiFi-GAN or other vocoder converts mel to waveform.
- Non-autoregressive acoustic decoding: frames predicted in parallel after expansion.

## One utterance: preprocessing sequence
1. Read waveform and transcript; identify speaker and basic metadata, validate matching text, trim problematic clips as documented, resample/mono if required by selected config.
2. Convert raw text into language-appropriate phoneme sequence and stable symbol IDs. Keep original transcript for traceability.
3. Run **forced alignment** (e.g. MFA with appropriate acoustic model and pronunciation lexicon) to map phonemes onto waveform time intervals. Convert into mel frame durations; reject bad/failed alignments with logged reasons.
4. Compute STFT on audio, apply mel filterbank and the exact selected implementation log/compression transform. An example is sample rate 22050, FFT 1024, hop 256, 80 mel channels, but **verify actual config** and don't assume a different frontend outputs identical features.
5. Estimate pitch F0 (often `pyworld` DIO/StoneMask or suitable extractor), handle unvoiced frames and interpolation explicitly; derive phoneme-level F0 if config requires.
6. Compute energy from STFT magnitude or implementation-specific method; optionally average by phoneme.
7. Calculate training-only normalizer statistics. Save mel, pitch, energy, duration arrays plus metadata (speaker ID, IDs, original text) in upstream-compatible format.
8. Check duration sums, mel lengths, frame alignment, shapes, NaNs, phoneme vocabulary coverage, sample rate, nonempty audio, split leakage.

## Training vs inference
- Training starts with text + human audio to build ground-truth mel, alignment durations, F0 and energy. Typical reference training feeds *ground-truth* duration/pitch/energy to condition decoder while optimizing predictor losses, and mel/postnet losses against actual mel. Ensure actual upstream implementation semantics are recorded.
- Inference starts with **text only** and selected speaker identity (for multispeaker); model predicts duration, pitch, energy, mel; vocoder renders audible speech. No original audio at inference.
- Report full objective: mel loss + (optional) postnet mel loss + duration + pitch + energy losses. Exact loss definitions from selected code, not invented.

## Existing code and pretrained weights
- Initial architecture/reference suggestion: https://github.com/ming024/FastSpeech2 . Verify its current setup commands and compatibility before choosing it.
- Other reference for Telugu exploration: https://github.com/smtiitm/Fastspeech2_HS and https://huggingface.co/multilingual-tts/EveryVoice-OpenBible-Telugu . They are *different toolkits / potentially incompatible checkpoint formats*. Never assume they load into ming024 code.
- Previous conversation considered from-scratch training, full fine-tuning, and LoRA. **LoRA is not required**. Pretrained-only inference is insufficient. Starting from pretrained acoustic weights could help compute, but only if compatibility and checkpoint/data provenance are verified. Inspect whether a Telugu checkpoint already trained on IndicTTS would contaminate validation data; if yes document or avoid.
- Using a compatible pretrained vocoder is recommended to save compute, but model mel settings must match vocoder requirements.

## Dataset concerns
- Svarah: ASR-focused Indian-accented English speech with many speakers (~9.6 h, 117). Existing HF repo may be gated; account acceptance and user-provided access may be necessary. A 5h single-speaker subset is likely unrealistic. Verify metadata, rights and speaker-balanced sampling; prefer an explicit multi-speaker strategy if using many speakers.
- IndicTTS Telugu: verify chosen mirror matches assignment intent; a public mirror is https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu . Public metadata includes audio/transcript and gender but gender is **not a guaranteed unique speaker identifier**; inspect provenance and full fields. Some long utterances may be less suitable; stratify and audit actual usable hours. Telugu G2P and a Telugu-compatible forced aligner/lexicon are necessary.
- Split by utterances; avoid duplicate recordings, duplicates across speakers and train/validation overlap. Depending on evaluation goal, support same-speaker disjoint utterance validation while documenting setup. Keep 5h/1h using **duration sums**, not row count.

## Specific evaluation concerns
- E1 unusually long English word; E2a/b `conduct` differing grammatical pronunciation; E3 `rhyth m` typo with space; E4 time, date and code-mixed `andaroo`.
- Telugu: T1 unusual/technical terms; T2a/b `పట్టు` in different contexts; T3 deliberately awkward spacing/orthography; T4 time/date and spacing. The PDF is the authoritative source for exact graphemes.
- For **each 5-language test sample**, export predicted duration, predicted pitch F0 and predicted energy (phoneme/frame level as configured). Plot with aligned phoneme labels where feasible; distinguish model outputs from ground truth and guard against NaNs, clipping and zero-length predictions.
- Produce actual audio and describe intelligibility, pronunciation, stress, pauses, unusual tokens, date/time reading, code mixing and failures. Don't fabricate listening judgments.

## Environment constraints
- Prefer CPU for metadata inspection/preprocessing when feasible and a free GPU for training. Free Colab/Kaggle availability, training speed and quota vary; test a short training run and calculate steps/sec before choosing full budget.
- Maintain resumable checkpoints, logging, configurable paths, deterministic dataset manifest, and reproducible notebook/CLI command instructions. Don't commit raw audio or authentication tokens. GitHub checkpoint file-size limits require LFS/external link.

## Unknowns Codex should investigate rather than guess
1. Exact compatible Python/PyTorch/librosa/pyworld/phonemizer/MFA versions and upstream command syntax.
2. Which Telugu forced aligner and text front-end will work with chosen IndicTTS audio and original test script.
3. How to handle multi-speaker Svarah, whether gender proxies in IndicTTS are reliable speaker IDs, and vocoder suitability.
4. Whether any pretrained Telugu checkpoint actually fits chosen framework; possible dataset overlap.
5. Which environment (local, Colab, Kaggle) will execute long training; Codex may only generate and smoke-test scripts.
