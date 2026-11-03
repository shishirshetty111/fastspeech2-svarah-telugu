# Copy this message as your FIRST Codex prompt

You are taking over a FastSpeech 2 RA screening assignment due 2026-11-07. The complete context is in AGENTS.md and docs/, including the original PDF. Read ALL of AGENTS.md, docs/ASSIGNMENT_REQUIREMENTS.md, docs/CONTEXT_AND_DECISIONS.md, docs/IMPLEMENTATION_PLAN.md and the original PDF before acting. Treat the PDF as authoritative.

Start with **Phase 0 and Phase 1 only**. Do not implement the entire Telugu pipeline or launch long GPU jobs yet.

1. Inspect the repository and the ming024/FastSpeech2 upstream implementation (clone into `third_party/FastSpeech2` if it is not present and permitted), recording exact commit/license, script paths, configuration and dependency constraints.
2. Assess environment compatibility and propose a reproducible local + Colab/Kaggle execution approach with **zero paid infrastructure** by default.
3. Set up / repair the minimal dependencies needed for baseline pretrained English inference. Download pretrained weights **only** from documented trusted URLs when accessible. Check the vocoder is compatible.
4. Actually synthesize `Hello, how are you?` with the upstream pretrained model if the environment permits. Verify the .wav exists and is nonempty. Do not claim success if blocked; report exact blocker and the commands I should execute.
5. Create/maintain minimal project structure, tests, `.gitignore`, `docs/PROGRESS.md`, and clear setup/run instructions; keep the assignment's `project/English` and `project/Telugu` entrypoints for later stages.
6. Explain how mel-spectrogram, duration, F0 and energy targets are obtained by THIS reference implementation and pinpoint the exact modules/functions to reuse.

After completing Phases 0–1, report changed files, exact commands actually executed, verification evidence, remaining blockers and the recommended single next task (dataset schema/manifest inspection). Do not fabricate checkpoints, losses, test utterance outputs or report results.
