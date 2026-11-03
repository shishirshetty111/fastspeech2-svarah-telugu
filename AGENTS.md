# Repository agent instructions — FastSpeech 2 RA assignment

## Mission and authoritative sources
Build a reproducible **FastSpeech 2 text-to-speech** research assignment for **two separately trained models: English (Svarah) and Telugu (IndicTTS Telugu)**. The deadline in the original PDF is **2026-11-07**. Read `docs/FastSpeech2_Assignment.pdf` and `docs/ASSIGNMENT_REQUIREMENTS.md` first. The PDF is the authority if interpretations conflict. Also read `docs/CONTEXT_AND_DECISIONS.md` and `docs/IMPLEMENTATION_PLAN.md` before modifying code.

## What counts as done
- Actually train an English model on ~5 hours of Svarah, validate on ~1 disjoint hour.
- Actually train a Telugu model on ~5 hours of IndicTTS Telugu, validate on ~1 disjoint hour.
- Preserve all five specified **test utterances per language**, including the two `conduct`/`పట్టు` variants and unusual original spellings/spacing. Do not automatically spell-correct test text. Cross-check any PDF text extraction against the visually rendered PDF (especially Telugu glyphs).
- Generate 10 audio samples and predicted duration/F0/energy plots, save train/validation loss plots, produce the report, checkpoint files and reproducibility instructions matching the PDF's `project/` layout.
- Show both training and inference procedures; do not count pretrained-only synthesis as training.

## Working approach
- Use an established FastSpeech 2 codebase as a reference, initially evaluate `https://github.com/ming024/FastSpeech2`. Inspect exact versions, licenses, paths, pretrained weights and conventions before adapting. Track the upstream URL and commit; retain attribution and license notices. Reuse upstream code where possible instead of reimplementing the network.
- Run in **small verified phases**. First objective: understand repository and get one pretrained English sentence synthesized with a compatible vocoder; then get a 10-50-utterance preprocessing smoke test; then full train/validation preparation; then English training; then Telugu.
- Do not assume this coding environment has a GPU, has dataset credentials, or can access restricted network resources. Generate scripts and portable Colab/Kaggle instructions; report blockers rather than inventing success.
- Prefer real trainable model and clear experiments to speculative LoRA. Pretrained acoustic initialization/full fine-tuning is an **option**, not a mandate or an explicitly approved exception in the PDF. Full training from scratch is the clearest compliance route. If transfer learning is used, fully document origin, layer compatibility, changes and possible data overlap.
- Use a pretrained **compatible vocoder** where feasible. Validate mel/STFT configuration compatibility (sample rate, fft, hop, number of mel channels, log transforms).

## Data and modeling safeguards
- Data preparation: utterance-level audio/text manifest with speaker ID where available, exact hours and deterministic seed; disjoint 5h/1h subsets; no train/val overlap, and no stats fit on validation data. Respect dataset licenses and avoid committing raw audio or secret tokens.
- English Svarah is a multi-speaker, ASR-oriented dataset. Check speaker labels, acoustic consistency and suitability; don't silently treat as single-speaker TTS.
- IndicTTS Telugu may have two speakers. Verify available fields, audio hours, and speaker mapping before configuring model. Telugu front end and forced alignment are substantive tasks; don't pretend English dictionaries work on Telugu.
- Training targets: actual mel-spectrogram, aligned phoneme durations, F0, energy. Check that the sum of durations matches aligned mel frames and that all targets have sensible lengths; account for unvoiced F0 regions, sample-rate conversion, and normalization. The reference model's exact definitions take precedence over generic `librosa` examples.
- Training: actual durations/pitch/energy generally condition decoder; predictors learn to infer them. Inference: predictions come from text only. Explain this clearly in code/docs.
- Preserve phoneme mappings, tokenizer/front-end configuration, speaker metadata, normalization stats and vocoder config with checkpoints. Validate that train and inference share them.
- Code changes should be testable, CPU-checkable where feasible and reproducible. Never report a model trained, a waveform synthesized or a metric measured unless it actually ran and files were verified.

## Code quality and progress reporting
- Work in milestones; after each phase update `docs/PROGRESS.md` with commands executed, evidence, current failures and next actions. Use concise, precise status.
- Add `--dry-run` / smoke-test capability where reasonable; persist checkpoints and resume safely after GPU notebook disconnections.
- Avoid unnecessary dependencies, nonfunctional placeholders disguised as real results, destructive commands, paid infrastructure or large GPU jobs without explicit user approval.
- Keep `project/English` and `project/Telugu` public entrypoints (`preprocess.py`, `train.py`, `inference.py`, `config.yaml`) even if they import shared reusable modules.
- Do not commit credential files, datasets, large checkpoints accidentally; use git LFS or externally hosted public checkpoint links if needed, per assignment.

## First instruction
Read `prompts/FIRST_TASK.md`. Complete **only phase 0 and phase 1** (repo inspection, setup, baseline inference or accurate blockers). Do not launch multi-hour training or both-language implementation in the first task.
