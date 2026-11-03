# Start here — FastSpeech 2 context handoff for Codex

This folder is a **context/bootstrap package**, not a completed FastSpeech 2 implementation. No actual model has been trained, no checkpoint or audio created. Its purpose is to transfer the prior chat's scope, lessons, modeling choices, PDF requirements and engineering plan into a real repository that Codex can read.

## Package contents
- `AGENTS.md` — persistent instructions read by Codex while operating in the repo.
- `docs/FastSpeech2_Assignment.pdf` — original assignment, source of truth.
- `docs/ASSIGNMENT_REQUIREMENTS.md` — deliverables, English test sentences, Telugu transcription warning.
- `docs/CONTEXT_AND_DECISIONS.md` — scientific/engineering context and tradeoffs from the conversation.
- `docs/IMPLEMENTATION_PLAN.md` — phase-based work plan with verification criteria.
- `docs/PROGRESS.md`, `docs/ENVIRONMENT.md` — initial status and resource notes.
- `prompts/FIRST_TASK.md` and `prompts/SECOND_TASK.md` — copy/paste prompts.
- `project/` — directory skeleton aligned with assignment. Empty directories contain `.gitkeep` until results exist.

## Option A: Codex desktop app / IDE
1. Extract this ZIP into a new folder (e.g. `fastspeech2-ra/`). Open **that folder** as your Codex project/repository.
2. If the folder is not yet a git repo, initialize it (`git init`). Review `AGENTS.md` and original PDF.
3. Paste the message in `prompts/FIRST_TASK.md` into Codex. Codex should examine the upstream repository, plan and verify baseline inference — **not** attempt two huge training runs immediately.
4. Review its changes and command results, then progress to the second prompt. Run GPU experiments where free GPU is available (Colab/Kaggle), not automatically inside the coding environment.

## Option B: Codex CLI
After extracting the archive and opening a terminal **inside that extracted folder**:

```bash
git init
codex
```

Paste `prompts/FIRST_TASK.md` content as first message. If the CLI is not installed yet, use your preferred Codex desktop/IDE setup instead.

## Scope choices
- Initial upstream: https://github.com/ming024/FastSpeech2 (inspect exact requirements).
- Dataset URLs: https://huggingface.co/datasets/ai4bharat/Svarah and https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu . These should be verified as intended releases/usable data.
- The PDF's Telugu test cases need **visual verification** before creating exact-string UTF-8 manifests. Text extraction can alter combining characters.
- Training two models is required; transfer learning is a possible strategy, not a confirmed shortcut. Pretrained-only inference does NOT satisfy deliverables.
- Due 2026-11-07; prioritize code that works over excessive generalization.

## Why Codex does not already know the chat
Codex's project history is distinct from the current ChatGPT conversation. Repository context files give it the specific requirements and decisions in a durable, reviewable way.
