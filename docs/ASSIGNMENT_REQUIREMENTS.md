# Assignment requirements — transcription and checklist

**Source of truth:** `FastSpeech2_Assignment.pdf` in this folder (3 pages). Deadline: **07-11-2026**.

## Objective (from assignment)
Implement and evaluate FastSpeech 2 separately for English and Telugu, adapting an existing GitHub codebase. Explain modifications, particularly Telugu support. Train separate models, synthesize test sentences and analyze variance-adaptor duration, pitch and energy predictions.

## Dataset
- English: **Svarah**; approx. **5h training + 1h validation**.
- Telugu: **IndicTTS Telugu**; approx. **5h training + 1h validation**.
- State selection/subsampling strategy, preprocessing steps, and ensure no train/validation overlap.

## English test utterances — exact text from PDF
- **E1:** The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.
- **E2a:** The orchestra's conduct during the ceremony was praised by the audience.
- **E2b:** The researchers will conduct an experiment to evaluate the proposed method.
- **E3:** The rhyth  m of the orchestra changed unexpectedly. *(two spaces between `rhyth` and `m` in the PDF; exact strings in `project/English/test_sentences.json`)*
- **E4:** The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.

## Telugu test utterances
The PDF has **T1, T2a, T2b, T3, T4**, five utterances total. Some Telugu character sequences and word spacing are unusual by design, and PDF text extraction may split combining characters in different positions. **Read the original PDF page 2 visually and verify a UTF-8 test manifest manually before inference. Do not silently normalize or spell-correct**. Maintain raw given text and, if necessary, a separately documented spoken-form normalization for the model while preserving the raw input in reports. Verify the exact original wording against the visually rendered source.

## Required layout (relative to `project/`)
```
project/
  English/
    train.py
    preprocess.py
    inference.py
    config.yaml
    checkpoints/
  Telugu/
    train.py
    preprocess.py
    inference.py
    config.yaml
    checkpoints/
  Results/
    English/
      training_loss.png
      validation_loss.png
      test_generated_audios/
      variance_adaptor_plots/{duration,pitch,energy}/
    Telugu/
      training_loss.png
      validation_loss.png
      test_generated_audios/
      variance_adaptor_plots/{duration,pitch,energy}/
  Report/
    Technical_Report.pdf
  README.md
  requirements.txt
```

## All submission requirements
- Complete preprocessing/training/inference code and modified upstream files for both languages, plus configs.
- Both trained checkpoints, both train/validation loss plots, all **5 generated audios per language**, and duration/F0/energy plots for **each** test sentence.
- Technical report: architecture and modifications; data choice/sampling; preprocessing; train/validation setup; convergence and instability; subjective generated speech/pronunciation observations; per-sentence variance plots analysis; cross-language failures and limitations; software and commands for reproduction.
- Public GitHub repository and publicly accessible checkpoint artifacts. Large checkpoints can use Git LFS or accessible links per PDF.

## Notes requiring care
- This assignment says *train two models*, but does not explicitly require random initialization or prohibit transfer learning. Treat this as an interpretation to document, not permission to skip training.
- Exact Telugu test utterances must be visually checked, not assumed correct from automatic PDF extraction.
