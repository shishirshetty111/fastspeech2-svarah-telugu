# Data selection, subsampling and splits

Everything below was measured from the decoded audio (`scripts/data/build_inventory.py`), not taken from dataset cards. Split numbers are copied from `project/<Lang>/splits/split_report.json`, which is the authoritative record.

## Common procedure
1. **Download** pinned parquet shards (`scripts/data/download_datasets.py`, revisions in `fs2ra/sources.py`). Svarah is gated: `HF_TOKEN` from the environment or a git-ignored `.env`; the token is never printed or committed.
2. **Inventory** every utterance, decoding one parquet row group at a time: sample rate, channels, duration, peak, RMS dBFS, clipped fraction, sha1 of the audio bytes, letters/second (Unicode letters + combining marks), normalised transcript key (NFC, lowercase, punctuation removed; used for grouping only, not as a TTS front end).
3. **QC** (`fs2ra/splits.py::apply_qc`); every reject is written with a reason to `project/<Lang>/splits/rejected.csv`:
   `too_short` (< 1 s), `too_long` (> 15 s), `too_quiet` (RMS < −40 dBFS, a crude SNR proxy), `clipping` (> 0.1 % samples at full scale), `speaking_rate_outlier` (|robust z| > 4 on letters/s within the speaker; catches transcript/audio mismatch), `duplicate_audio`, `unsupported_characters` (language-specific), `speaker_not_selected`.
4. **Split** (`fs2ra/splits.py::make_split`), seed 1234:
   - Utterances with the same normalised transcript form one group and always go to the same split → no sentence occurs in both train and val.
   - Each speaker contributes to train and val in proportion to its usable hours (same-speaker, disjoint-utterance validation). The speaker embedding is therefore defined for every validation utterance.
   - Groups are drawn in order (priority tier, sha1(seed:text key)); tier 0 = clean text ≤ 11 s, tier 1 = 11–15 s (needs `max_seq_len` > 1000 frames, since 1000 frames ≈ 11.6 s at 22.05 kHz/hop 256), tier 2 = text that needs verbalisation (digits/symbols). Validation is filled first, then train, both to duration targets (5.0 h / 1.0 h); a group is taken only if it moves its speaker's total closer to that speaker's target.
   - Everything else QC-passed is `reserve`: disjoint from val (and train), used **only** to top up train after forced-alignment failures (Phase 3), never for validation or normalisation statistics.
5. **Checks**: pairwise disjointness on utt_id, audio sha1 and transcript (raw sha1 in committed files; normalised key at build time), duration bounds, hour targets, val speakers ⊆ train speakers, and exact rebuild from the inventory (`tests/test_manifests.py`).
6. **What is committed**: `project/<Lang>/splits/*.csv` contain IDs, speaker, duration, source shard/row/path, audio sha1 and transcript sha1, but no audio and no transcripts (licensing; Svarah is CC BY 4.0 but IndicTTS has its own licence). Full manifests with text are regenerated locally in `data/manifests/<lang>/`.

Normalisation statistics (pitch/energy mean/std) must be fit on **train only** in Phase 3; upstream fits them on train+val and must be changed.

## English — Svarah (`ai4bharat/Svarah` @ `ebbf777`, CC BY 4.0)
**Measured inventory:** 6,656 utterances, **9.614 h**, all 16 kHz mono PCM16 (decoded durations equal the dataset `duration` field exactly). The Hub split is named `test` only because Svarah is an ASR benchmark; it is the whole corpus. Read and spontaneous speech, short command words (e.g. "jump" ×31, "visual" ×30), 769 transcripts with digits, no non-ASCII characters, no duplicate audio.

**Speaker identity.** There is no speaker column. File stems are `<recording-id>_f<n>_chunk_<k>`, but there are 2,938 recording ids (sessions), not speakers. Every recording id maps to exactly one demographic profile (gender, age group, native language, state, district, qualification, job category, occupation domain), and there are **115 distinct profiles** vs. 117 speakers reported in the Svarah paper. We use the profile hash as the speaker id (`en_<sha1[:8]>`); at most two pairs of speakers with identical profiles may be merged. 3.8 min median per speaker (0.3–24.5 min).

**Consequence for modelling.** No single speaker has anywhere near 5 h, so a single-speaker setup is impossible. Plan: upstream's multi-speaker mode (`multi_speaker: True`, speaker embedding table from `speakers.json`), all speakers in train and val; test sentences synthesised with one fixed, documented speaker id (chosen in Phase 4 by training data amount/validation quality). Indian-accented, multi-condition recordings will limit quality compared with a studio TTS corpus; this is a property of the assigned dataset.

**Sample rate.** 16 kHz will be resampled to 22.05 kHz to keep the upstream/HiFi-GAN feature configuration (n_fft 1024, hop 256, 80 mels, fmax 8 kHz). Because fmax = 8 kHz equals the 16 kHz Nyquist frequency, the mel band carries no empty region; HiFi-GAN will see the same mel definition it was trained on.

**QC result** (max 15 s, digits/symbols de-prioritised to tier 2):

| | utterances | hours |
|---|---|---|
| inventory | 6,656 | 9.614 |
| too_short (< 1 s) | 1,215 | 0.197 |
| too_long (> 15 s) | 261 | 1.451 |
| too_quiet | 160 | 0.234 |
| speaking_rate_outlier (mostly digit strings, e.g. phone numbers) | 24 | 0.027 |
| **QC-passed pool** | **4,996** | **7.706** |

**Split** (seed 1234):

| split | utterances | hours | speakers | tiers (0/1/2) |
|---|---|---|---|---|
| train | 3,433 | 5.038 | 113 | 3179 / 119 / 135 |
| val | 777 | 1.028 | 113 | 777 / 0 / 0 |
| reserve | 786 | 1.641 | 112 | 128 / 136 / 522 |

Two speaker proxies lost all their utterances to QC. Val is entirely tier 0 because it is filled first; train contains 254 tier-1/2 utterances. This asymmetry is deliberate (cleanest material for the validation measurement) and noted for the report.

Alternatives considered: 11 s cap (6.54 h pool, too little alignment headroom); rejecting all digit transcripts (5.55–6.45 h pool). Chosen: 15 s cap with digit transcripts kept but drawn last.

## Telugu — IndicTTS Telugu (`SPRINGLab/IndicTTS_Telugu` @ `7ab3c18`, Indic TTS licence, IIT Madras)
Not gated. Hugging Face mirror of the IIT Madras Indic TTS Telugu monolingual recordings; the mirror's licence pointer is the original Indic TTS licence (https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf), which must be respected — no audio or transcripts are committed.

**The dataset card is wrong on two counts** (measured from decoded audio):

| | card | measured |
|---|---|---|
| total duration | ~8.74 h (M 4.47, F 4.27) | **36.71 h** (M 12.79, F 23.92) |
| sample rate | 48 kHz | female **16 kHz**, male **48 kHz** |

**Speakers.** Two: file stems `train_telugufemale_NNNNN` / `train_telugumale_NNNNN`, and the `gender` label (0/1) agrees with the stem for all 8,576 rows, so here gender *is* a reliable speaker id (one speaker per gender per the source). 1,311 sentences were recorded by both speakers.

| speaker | utts | hours | sr | ≤ 11 s | ≤ 15 s | 20–150 s |
|---|---|---|---|---|---|---|
| female | 4,043 | 23.92 | 16 kHz | 1.89 h | 3.77 h | 16.83 h |
| male | 4,533 | 12.79 | 48 kHz | 6.21 h | 8.10 h | 3.24 h |

Female clips are mostly long story paragraphs (median well above 15 s; up to 151 s); all female files are peak-normalised to exactly 0.650. Male clips are mostly sentences.

**Decision: single-speaker male model.** The male voice alone yields an 8.06 h QC-passed pool with clips ≤ 15 s, whereas female clips ≤ 15 s give only 3.63 h; using the female voice would need ~20 s clips (≈ 1,720 frames, ~3× the attention cost) or segmenting long recordings (requires alignment first). A single consistent voice is also the standard TTS setting and avoids mixing a 16 kHz and a 48 kHz recording chain. The female data is unused (logged as `speaker_not_selected`). Male audio will be resampled 48 → 22.05 kHz.

**QC result** (male, max 15 s, Latin letters rejected):

| | utterances | hours |
|---|---|---|
| male inventory | 4,533 | 12.79 |
| too_long (> 15 s) | 717 | 4.684 |
| speaking_rate_outlier | 12 | 0.011 |
| unsupported_characters | 11 | 0.032 |
| too_short | 10 | 0.002 |
| clipping | 1 | 0.001 |
| **QC-passed pool** | **3,782** | **8.062** |

The rate filter catches real transcript/audio mismatches, e.g. `telugumale_02390`: 2.8 s of audio with a long multi-clause transcript (71.7 letters/s vs. a median of 10.9). Low-rate rejects are short story titles read slowly. The Latin-letter rejects are transcription typos in which Latin `o` was typed instead of the anusvara `ం` (U+0C02), e.g. `యిoది`; they are rejected, not silently corrected. Two transcripts contain U+FEFF (BOM) characters that the Telugu front end must strip (Phase 5). No digits occur in the Telugu transcripts — but T4 contains `2026`, `12వ` and `10:45`, so number verbalisation is purely an inference-time front-end problem for Telugu.

**Split** (seed 1234):

| split | utterances | hours | tiers (0/1) |
|---|---|---|---|
| train | 2,623 | 5.000 | 2623 / 0 |
| val | 524 | 1.000 | 524 / 0 |
| reserve | 635 | 2.062 | 102 / 533 |

All train/val clips are ≤ 11 s, so the upstream `max_seq_len: 1000` suffices for Telugu.

**Transfer-learning caveat for later:** any pretrained Telugu acoustic model trained on IndicTTS (e.g. IIT Madras Fastspeech2_HS) has very likely seen these exact male recordings, including our validation set. If such weights are ever used, validation loss would be contaminated; this must be documented or avoided.
