# Upstream reference: ming024/FastSpeech2

| Item | Value |
|---|---|
| URL | https://github.com/ming024/FastSpeech2 |
| Pinned commit | `d4e79eb52e8b01d24703b2dfc0385544092958f3` (2021-07-08, "update README and demo page"; latest on `master` at clone time) |
| License | MIT, © 2020 Chung-Ming Chien (`LICENSE`). Bundled `hifigan/` is MIT, © 2020 Jungil Kong (`hifigan/LICENSE`). Both notices must be retained in anything we vendor. |
| Local path | `third_party/FastSpeech2/` (git-ignored; reproduced by `scripts/setup_upstream.py`) |
| Our modifications | `patches/ming024_fastspeech2_compat.patch` (see below) |
| Paper variant | FastSpeech 2 **v1** (raw F0 as pitch feature, not CWT pitch spectrogram), plus a Tacotron-2 PostNet (not in paper) |

## Upstream layout (what matters)
| Path | Role |
|---|---|
| `prepare_align.py` → `preprocessor/{ljspeech,libritts,aishell3}.py` | Per-corpus: resample, peak-normalise, write `raw_data/<spk>/<id>.wav` + cleaned `.lab` for MFA |
| `preprocess.py` → `preprocessor/preprocessor.py::Preprocessor` | Reads MFA TextGrids → durations; computes mel, F0, energy; stats; writes `train.txt`/`val.txt` |
| `audio/stft.py::TacotronSTFT`, `audio/tools.py::get_mel_from_wav` | Mel + energy extraction |
| `text/symbols.py`, `text/__init__.py::text_to_sequence`, `text/cleaners.py`, `text/cmudict.py` | Symbol table (ARPAbet + pinyin + silences, 360 symbols, +1 pad → 361 embeddings), `{PH PH}` parsing |
| `synthesize.py::preprocess_english` | Inference G2P: `lexicon/librispeech-lexicon.txt`, fallback `g2p_en` |
| `model/fastspeech2.py::FastSpeech2` | Encoder → (speaker emb) → VarianceAdaptor → Decoder → Linear → PostNet |
| `model/modules.py::VarianceAdaptor / VariancePredictor / LengthRegulator` | Duration, pitch, energy predictors, bucketised embeddings, length regulation |
| `model/loss.py::FastSpeech2Loss` | Total loss |
| `transformer/` | FFT blocks, PostNet |
| `dataset.py::Dataset / TextDataset` | Loads `.npy` targets via `train.txt`/`val.txt` lines `id|speaker|{phones}|raw_text` |
| `train.py`, `evaluate.py`, `model/optimizer.py::ScheduledOptim` | Training loop, validation, Noam-style warmup + step annealing |
| `utils/model.py::get_model/get_vocoder/vocoder_infer` | Checkpoint + HiFi-GAN loading, mel→wav |
| `utils/tools.py::synth_samples/plot_mel/expand` | Inference outputs, mel+F0+energy plot |
| `config/<dataset>/{preprocess,model,train}.yaml` | Three-file config split |

## How THIS implementation builds training targets (LJSpeech config)
Audio front end: `sampling_rate 22050`, `filter_length (n_fft) 1024`, `hop 256`, `win 1024`, `80` mel bins, `fmin 0`, `fmax 8000`. These exactly match `hifigan/config.json` (asserted in `tests/test_upstream_compat.py::test_mel_config_matches_hifigan`).

1. **Duration** — `Preprocessor.get_alignment`: reads the MFA TextGrid `phones` tier. Leading silences (`sil/sp/spn`) are trimmed (start time recorded), trailing silences trimmed; internal silences kept as phones. Each phone's frame count is `round(end*sr/hop) - round(start*sr/hop)` (rounding at boundaries, so the sum is consistent). The waveform is then cut to `[start, end]`.
2. **Mel-spectrogram** — `TacotronSTFT.mel_spectrogram`: reflect-pad by n_fft/2, STFT magnitude via conv1d with Hann-windowed Fourier basis, `mel_basis @ |STFT|` (librosa Slaney mel filters), then `log(clamp(x, 1e-5))` (natural log, `dynamic_range_compression`). Truncated to `sum(duration)` frames. Stored as `(T, 80)`.
3. **Energy** — same STFT: `energy = L2 norm of the linear magnitude spectrum per frame` (`torch.norm(magnitudes, dim=1)`), truncated to `sum(duration)`. With `feature: phoneme_level`, averaged over each phone's frames.
4. **F0** — `pyworld.dio` with `frame_period = hop/sr*1000 ms` (≈11.61 ms, matching mel frames), refined by `pyworld.stonemask`; truncated to `sum(duration)`. With `phoneme_level`: unvoiced frames (F0 = 0) are **linearly interpolated** between voiced frames (edges held at first/last voiced value), then averaged per phone. Utterances with ≤1 voiced frame are dropped.
5. **Normalisation** — `StandardScaler.partial_fit` over (IQR-outlier-filtered) pitch/energy values, then every saved file is z-scored in place; `stats.json` holds `[min, max, mean, std]` (min/max are post-normalisation and define the 256 bucket boundaries for the pitch/energy embeddings).
6. **Metadata** — `train.txt`/`val.txt` lines `basename|speaker|{phones}|raw_text`; `speakers.json` maps speaker → index.

### Training vs inference (verified in code)
- **Training** (`train.py:82`, `model(*batch[2:])`): the batch includes mel, pitch, energy and duration targets. `VarianceAdaptor` still runs all three predictors (for their losses) but the **ground-truth** pitch/energy are bucketised into the embeddings and the **ground-truth** durations drive the `LengthRegulator` (teacher forcing).
- **Inference**: targets are `None`, so predicted pitch/energy select the embedding buckets and durations are `clamp(round(exp(log_d) - 1) * d_control, 0)`.
- **Loss** (`model/loss.py`): `L1(mel) + L1(postnet mel) + MSE(log(d+1)) + MSE(pitch) + MSE(energy)`, all masked; pitch/energy in normalised units.

## Modules to reuse unchanged (or nearly)
`model/*`, `transformer/*`, `hifigan/*`, `audio/*` (patched), `utils/*` (patched), `dataset.py`, `train.py`/`evaluate.py` loop logic, `Preprocessor.process_utterance` target-extraction logic. Language-specific work goes into: a corpus loader replacing `preprocessor/ljspeech.py`, the symbol table/front end (Telugu), and split logic.

## Compat patch applied (`patches/ming024_fastspeech2_compat.patch`)
| File | Change | Why |
|---|---|---|
| `audio/stft.py` | conv1d on the input's device instead of `.cuda()` | upstream preprocessing crashed without a GPU |
| `audio/stft.py`, `audio/audio_processing.py` | keyword args to `librosa.filters.mel`, `pad_center` | librosa ≥0.10 made them keyword-only |
| `utils/model.py` | `load_checkpoint`: `map_location` + `weights_only=True` with only numpy scalar/dtype allowlisted | CPU loading of GPU ckpts; avoid arbitrary pickle execution |
| `utils/model.py` | HiFi-GAN paths relative to the `hifigan` package | upstream required CWD = repo root |
| `model/modules.py` | energy embedding uses `e_control` (was `p_control`) | upstream bug: `--energy_control` was ignored and pitch control scaled energy |

## Known upstream issues to fix in later phases (not patched yet)
- `preprocessor/ljspeech.py` etc.: `librosa.load(path, sampling_rate)` positional `sr` breaks on librosa ≥0.10. We will write our own corpus loaders anyway.
- `Preprocessor.process_utterance`: `librosa.load(wav_path)` silently resamples to librosa's default 22050 rather than the config value — fine for 22050 but a hidden coupling.
- `Preprocessor.build_from_path`: pitch/energy scalers are fit on **all** utterances (train + val) and the val split is a random `val_size` count. This violates our "no stats fit on validation" rule and the 5 h/1 h duration-based split; must be replaced by manifest-driven splits with train-only stats.
- `Preprocessor.build_from_path` also iterates `os.listdir` (order not deterministic across filesystems) and `random.shuffle` without a seed.
- `train.py`: saves only every `save_step` (100k), uses tensorboard only (no CSV/PNG loss curves), `nn.DataParallel`, no data-loader resume. Need frequent resumable checkpoints for Colab/Kaggle and CSV loss logging for the required plots.
- `synthesize.py` single mode uses the raw text (first 100 chars) as the output filename — invalid on Windows for `?`, `:`; we pass explicit utterance IDs.
- `VariancePredictor` hard-codes `padding=1` in the second conv (only correct for `kernel_size: 3`).
- Symbol table has no Telugu symbols; the `{...}` phone parser assumes space-separated phone tokens — Telugu needs a new phone set and embedding size (checkpoint transfer would need embedding resize).
- `synthesize.py::preprocess_english` strips only trailing punctuation and relies on `g2p_en` for numbers/OOV; E4 (`10:45 AM`, `12 November 2026`, `andaroo`) and E3 (`rhyth m`) behaviour must be documented, not silently normalised.
