# FastSpeech 2 - English (Svarah) and Telugu (IndicTTS)

Two FastSpeech 2 text-to-speech models trained from scratch:

- **English:** 5 h of Svarah for training, 1 h for validation.
- **Telugu:** 5 h of IndicTTS Telugu for training, 1 h for validation.

Audio is produced with the pretrained HiFi-GAN vocoder. For English, the vocoder was fine-tuned on the English model's own output. The full write-up is in [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf).

The code is based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (commit `d4e79eb`, MIT licence). The changes to that code are in `../patches/`, and the project code is in `../fs2ra/`.

## Folder layout

```
project/
  English/ and Telugu/
    preprocess.py        alignment and feature extraction (mel, duration, pitch, energy)
    train.py             training (resumes automatically after a disconnection)
    inference.py         speech from text, plus duration/pitch/energy plots
    config.yaml          all settings
    checkpoints/         submitted model (step 20,000), with its stats and speaker list
  Results/English/ and Results/Telugu/
    training_loss.png, validation_loss.png
    test_generated_audios/          the 5 test sentences
    variance_adaptor_plots/{duration,pitch,energy}/
    inference_summary.json          phonemes and predicted values for each sentence
  Report/Technical_Report.pdf
  README.md
  requirements.txt
```

## Checkpoints

The checkpoints are stored with Git LFS. Install Git LFS before cloning, or run `git lfs pull` after cloning.

- English: `English/checkpoints/20000.pth.tar` and the fine-tuned vocoder `English/checkpoints/vocoder_hifigan_finetuned_g02503000.pth.tar`
- Telugu: `Telugu/checkpoints/20000.pth.tar`

## Generate speech with the trained models

```bash
git lfs install
git clone https://github.com/shishirshetty111/fastspeech2-svarah-telugu.git
cd fastspeech2-svarah-telugu
python -m venv .venv
.venv/Scripts/activate                  # Linux/macOS: source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/setup_upstream.py        # downloads and patches the reference code and HiFi-GAN

python project/English/inference.py --out_dir my_outputs          # the 5 English test sentences
python project/Telugu/inference.py --out_dir my_outputs_te        # the 5 Telugu test sentences
python project/English/inference.py --text "Any English sentence." --ids demo --out_dir my_outputs
```

This runs on a normal CPU in about a minute. Optional settings:

- `--pace 1.2` speaks 20% slower.
- `--speaker <id>` picks another English voice from `speakers.json`. The default is `en_72a36e45`.

## Reproduce training from scratch

1. **Download and split the data.** Svarah requires access on Hugging Face: put a read token in a `.env` file as `HF_TOKEN=...`.

   ```bash
   python scripts/data/download_datasets.py
   python scripts/data/build_inventory.py
   python scripts/data/make_splits.py
   ```

2. **Install the Montreal Forced Aligner 3.4.3** from conda-forge (for example `micromamba create -p mfa_env -c conda-forge montreal-forced-aligner`). Then point the scripts to it with `FS2_MFA_CMD="micromamba run -p mfa_env mfa"`.

3. **Preprocess.**

   ```bash
   python project/English/preprocess.py --stage export align features
   python project/Telugu/preprocess.py --stage export
   # Telugu alignment trains its own MFA model: run notebooks/align_telugu_colab.ipynb on Colab
   python project/Telugu/preprocess.py --stage features
   ```

4. **Train on a GPU.** `notebooks/train_colab.ipynb` does this on Google Colab. Locally:

   ```bash
   python project/English/train.py --amp
   python project/Telugu/train.py --amp
   ```

5. **Fine-tune the English vocoder (optional).** Run `python scripts/export_gta.py english --ckpt project/English/checkpoints/20000.pth.tar`, then `notebooks/finetune_hifigan_colab.ipynb`.

6. **Run inference** as shown above.

To run the tests: `python -m pytest tests -q`.

## Data and licences

- Svarah: CC BY 4.0. IndicTTS: IIT Madras licence. No dataset audio or transcripts are included in this repository. The split files contain only clip IDs and hashes.
- FastSpeech 2 and HiFi-GAN code: MIT licence. Their licence notices are kept in the downloaded reference code.
