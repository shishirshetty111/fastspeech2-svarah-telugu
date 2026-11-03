# Build plan — incremental, verifiable phases

Deadline: **2026-11-07**. No component should be called complete without evidence that it ran.

## Phase 0 — Repo and environment inspection (start here)
- Read PDF, context and AGENTS.md. Inspect / clone upstream `ming024/FastSpeech2` and pin commit. Check upstream config layout, tests, model/variance adaptor/forward hooks, inference/vocoder usage, licenses, Python and dependency compatibility. Do not install untrusted packages silently or launch GPU training.
- Check local Python, torch/cuda if available. Check disk, network, dataset credentials without printing secrets.
- Produce `docs/PROGRESS.md` and `docs/ENVIRONMENT.md` with findings, dependencies and exact setup/run commands. Define scripts for CPU smoke tests.
**Exit:** repo inspection summary + tested setup where possible, precise blockers otherwise.

## Phase 1 — Baseline pretrained English inference
- Use upstream pretrained English acoustic model and matching vocoder if accessible and permitted. Run or write a verifiable smoke test for 'Hello, how are you?' to prove components connect.
- Preserve configuration provenance; do not claim this is your trained result.
**Exit:** `baseline_hello.wav` with verified creation, or documented access/network/compatibility blocker and exact next command.

## Phase 2 — Dataset ingestion + manifests
- Implement data inspection that works with actual downloaded schema and gated Svarah access. Log sample rate, lengths, transcript quality, duplicate IDs, speaker metadata and audio hours.
- Generate reproducible 5h train/1h validation manifests per language using durations, deterministic seed and speaker policy. Validate disjointness.
**Exit:** machine-readable split manifests, stats, tests, no overlapping audio IDs.

## Phase 3 — English preprocessing smoke and full data
- Build/export text frontend, phonemes, MFA preparation and alignment, target extraction (mel, durations, F0, energy). Use selected upstream exact audio feature settings.
- Run on first 10-50 files, inspect arrays, durations vs mel, missing/nonfinite values, alignment and speaker mapping. Scale to all selected files only after checks pass.
**Exit:** 10-50-item CPU smoke test and then trainable 5h/1h feature datasets.

## Phase 4 — English training
- Wire `project/English/{preprocess,train,inference}.py` and `config.yaml` to shared implementation. Use a small dry run first; check loss and backprop; save/resume.
- Train within actual GPU budget and save true checkpoints/loss CSV+plots.
**Exit:** English trained checkpoint, validation loss, audio outputs and variance plots.

## Phase 5 — Telugu front end + training
- Define Telugu phoneme symbols, Unicode-aware text frontend and separate MFA alignment pipeline. Build custom pronunciation lexicon/acoustic alignment approach as needed; verify hand-audited examples; keep original and transformed forms separately.
- Adapt vocabulary/embeddings and speaker config, especially for any transfer learning; document all changed layers and provenance. Implement separate Telugu preprocessing, training and inference entrypoints and config.
- Smoke-test 10-50 records, then train and save checkpoint/loss curves.
**Exit:** actually trained Telugu model, syntheses and variance plots.

## Phase 6 — Tests and technical report
- Use source-PDF verified 5+5 exact test utterances. Generate actual audio files; predicted duration/F0/energy plots for each, plus train/val loss PNGs. Check all files open and numeric arrays are valid.
- Write `project/Report/Technical_Report.pdf` documenting methods, configuration, real metrics, subjective listening with clear limitations, detailed analysis per sentence, comparison between languages, and reproducibility instructions.
- Ensure public repo, licenses, checkpoints accessible by Git LFS/link; run a clean reproducibility pass.
**Exit:** submitted tree matches PDF with truthful artifacts.

## Time budgeting and mitigation
- Prioritize Phase 0-3 immediately; they typically reveal real blockers. Work on Telugu text front end in parallel with English training after Phase 3 if feasible.
- Use resumable, bounded GPU jobs. If compute or alignment is blocked, document evidence and an honestly partial result rather than generating fake plots or claiming success.
- Do not require LoRA. Transfer learning is allowed only as a documented choice pending interpretation of the original assignment.
