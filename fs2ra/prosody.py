"""Generic, data-calibrated prosody post-rules for English text-only inference.

The acoustic model predicts phone durations, pitch and energy (pass 1). These rules adjust
those predictions; the decoder then runs again with the adjusted values supplied as
variance targets (pass 2). The phones themselves are never changed.

Every rule fires on information the front end produces for any input (ARPAbet stress
digits, phone positions, pause types) -- nothing is word-specific -- and every amount is
derived from data by scripts/prosody_calibrate.py (stored in prosody_rules.json), which also
enables only the rules that move their own measure towards real speech on held-out
validation utterances ("enabled"):

  A  stressed-vowel floor   a primary-stressed vowel is at least as long as the 10th
                            percentile of real durations of that vowel when stressed
                            (train split)
  B  pitch range            per sentence, pitch deviations from the sentence mean are scaled
                            by k = median(real F0 sd / predicted F0 sd) on calibration data
  C  stress accent          primary-stressed vowels are raised by the gap between the real
                            and predicted stressed-minus-unstressed F0 difference (in
                            sentence-sd units), if the prediction under-marks stress
  D  phrase-final           the last vowel before a pause or the sentence end is lengthened by
                            real/predicted final-lengthening ratio and lowered by the real/
                            predicted gap in its F0 (sentence-sd units)
  E  overall pace          predicted durations are scaled (before rounding, via the model's
                            d_control, in fs2ra.inference) by 1 / median(predicted / real
                            utterance length) on calibration data
  J  join split word       a pause from a run of spaces (not punctuation) gets 0 frames, so the
                            pieces are spoken together (phones unchanged; a policy, not
                            calibrated; mutually exclusive with P)
  P  split-word pause       a pause from a run of spaces (not punctuation) lasts at least the
                            median real mid-sentence pause of the train split
"""
import numpy as np

SP = "sp"


def is_vowel(p):
    return p[-1:].isdigit()


def base(p):
    return p[:-1] if is_vowel(p) else p


def final_vowels(phones):
    """Index of the last vowel before each pause and before the end."""
    out, last = [], None
    for i, p in enumerate(phones):
        if p == SP:
            if last is not None:
                out.append(last)
            last = None
        elif is_vowel(p):
            last = i
    if last is not None:
        out.append(last)
    return out


def sentence_stats(phones, f0):
    idx = [i for i, p in enumerate(phones) if p != SP]
    v = np.asarray(f0, float)[idx]
    return float(v.mean()), float(v.std()) or 1.0


def apply_rules(phones, dur, f0_hz, energy, rules, sp_kinds=None, enabled="ABCDEP"):
    """phones: list; dur (frames, ints), f0_hz, energy: per-phone arrays (pass-1 predictions).
    Returns adjusted (dur, f0_hz, energy) and a list of human-readable changes."""
    d = np.asarray(dur, float).copy()
    f = np.asarray(f0_hz, float).copy()
    e = np.asarray(energy, float).copy()
    log = []
    stressed = [i for i, p in enumerate(phones) if p.endswith("1")]
    finals = final_vowels(phones)

    if "A" in enabled:
        for i in stressed:
            floor = rules["stress_floor_frames"].get(base(phones[i]))
            if floor and d[i] < floor:
                log.append("A %s#%d %d->%d frames" % (phones[i], i, d[i], floor))
                d[i] = floor
    if "D" in enabled and rules.get("final_lengthen", 1.0) > 1.0:
        for i in finals:
            new = round(d[i] * rules["final_lengthen"])
            if new > d[i]:
                log.append("D %s#%d %d->%d frames" % (phones[i], i, d[i], new))
                d[i] = new
    if "P" in enabled and sp_kinds:
        sps = [i for i, p in enumerate(phones) if p == SP]
        for i, kind in zip(sps, sp_kinds):
            floor = rules.get("space_pause_min_frames", 0)
            if kind == "space" and d[i] < floor:
                log.append("P sp#%d %d->%d frames" % (i, d[i], floor))
                d[i] = floor
    if "J" in enabled and sp_kinds:
        # J (join): a pause that comes only from a run of spaces, not from punctuation, is
        # given 0 frames, so the pieces are spoken together. Phones are unchanged: only the
        # pause duration is. Policy choice (no recorded speech has such inputs to calibrate on).
        for i, kind in zip([i for i, p in enumerate(phones) if p == SP], sp_kinds):
            if kind == "space" and d[i] > 0:
                log.append("J sp#%d %d->0 frames" % (i, d[i]))
                d[i] = 0

    m, s = sentence_stats(phones, f)
    if "B" in enabled and rules.get("pitch_scale", 1.0) != 1.0:
        k = rules["pitch_scale"]
        f = m + k * (f - m)
        log.append("B pitch deviations x%.2f around %.0f Hz" % (k, m))
        s = s * k
    if "C" in enabled and rules.get("stress_accent_sd", 0) > 0:
        for i in stressed:
            f[i] += rules["stress_accent_sd"] * s
        log.append("C +%.0f Hz on %d stressed vowels" % (rules["stress_accent_sd"] * s, len(stressed)))
    if "D" in enabled and rules.get("final_drop_sd", 0) > 0:
        for i in finals:
            f[i] -= rules["final_drop_sd"] * s
        log.append("D -%.0f Hz on %d phrase-final vowels" % (rules["final_drop_sd"] * s, len(finals)))
    return d.astype(int), f, e, log
