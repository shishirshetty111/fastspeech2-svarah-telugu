"""Telugu text front end shared by alignment (MFA transcripts/dictionary), training
and inference.

Telugu script is an abugida with a near-phonemic spelling, so pronunciation is
rule-based (no lexicon):
- a consonant letter carries the inherent vowel /a/ unless followed by a vowel
  sign (matra) or the virama (్), which removes it; Telugu keeps the inherent vowel
  word-finally (no schwa deletion, unlike Hindi);
- anusvara (ం) becomes the nasal homorganic with the next consonant (ng/ny/nn/n/m),
  and /m/ before vowels, approximants, sibilants, h and word-finally;
- visarga (ః) -> h; candrabindu (ఁ) is dropped; vocalic r (ఋ, ృ) -> r u, as spoken.
Phones are ASCII labels (aspiration 'h', retroflex doubled, long vowels doubled):
see PHONES. Geminates are two identical phones (e.g. పట్టు -> p a tt tt u).

normalize_telugu() returns the spoken-form word string: NFC, invisible characters
removed, numbers / times / ordinals ("12వ") written as Telugu words, punctuation
removed. Raw text is always kept separately.
"""
import re
import unicodedata

VOWELS = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"]
CONSONANTS = ["k", "kh", "g", "gh", "ng", "ch", "chh", "j", "jh", "ny", "tt", "tth", "dd", "ddh", "nn",
              "t", "th", "d", "dh", "n", "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "ss", "s",
              "h", "ll", "rx", "ts", "dz"]
PHONES = VOWELS + CONSONANTS

INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"], "ఋ": ["r", "u"],
    "ౠ": ["r", "uu"], "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"], "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}
VOWEL_SIGNS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"], "ృ": ["r", "u"], "ౄ": ["r", "uu"],
    "ె": ["e"], "ే": ["ee"], "ై": ["ai"], "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
}
CONSONANT_LETTERS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ల": "l", "వ": "v", "శ": "sh", "ష": "ss", "స": "s", "హ": "h",
    "ళ": "ll", "ఱ": "rx", "ౘ": "ts", "ౙ": "dz",
}
VIRAMA, ANUSVARA, VISARGA, CANDRABINDU = "్", "ం", "ః", "ఁ"
_NASAL_BEFORE = {}
for _nasal, _group in (("ng", "k kh g gh"), ("ny", "ch chh j jh"), ("nn", "tt tth dd ddh nn"),
                       ("n", "t th d dh n"), ("m", "p ph b bh m")):
    for _p in _group.split():
        _NASAL_BEFORE[_p] = _nasal
_INVISIBLE = dict.fromkeys(map(ord, "﻿​‌‍­"))


def g2p(word):
    """Phones for one Telugu word; raises ValueError on characters outside the script."""
    phones, pending = [], None  # pending: consonant still waiting for its vowel

    def flush(vowel=("a",)):
        nonlocal pending
        if pending is not None:
            phones.append(pending)
            phones.extend(vowel)
            pending = None

    for ch in word:
        if ch in CONSONANT_LETTERS:
            flush()
            pending = CONSONANT_LETTERS[ch]
        elif ch == VIRAMA:
            flush(vowel=())
        elif ch in VOWEL_SIGNS:
            if pending is None:
                raise ValueError("vowel sign without consonant in %r" % word)
            flush(vowel=VOWEL_SIGNS[ch])
        elif ch in INDEPENDENT_VOWELS:
            flush()
            phones.extend(INDEPENDENT_VOWELS[ch])
        elif ch == ANUSVARA:
            flush()
            phones.append(ANUSVARA)  # resolved below
        elif ch == VISARGA:
            flush()
            phones.append("h")
        elif ch == CANDRABINDU:
            flush()
        else:
            raise ValueError("unsupported character %r (U+%04X) in %r" % (ch, ord(ch), word))
    flush()
    return [(_NASAL_BEFORE.get(phones[i + 1], "m") if i + 1 < len(phones) else "m") if p == ANUSVARA else p
            for i, p in enumerate(phones)]


# --- numbers -------------------------------------------------------------------
_UNITS = "సున్నా ఒకటి రెండు మూడు నాలుగు ఐదు ఆరు ఏడు ఎనిమిది తొమ్మిది".split()
_TEENS = "పది పదకొండు పన్నెండు పదమూడు పద్నాలుగు పదిహేను పదహారు పదిహేడు పద్దెనిమిది పందొమ్మిది".split()
_TENS = [None, None] + "ఇరవై ముప్పై నలభై యాభై అరవై డెబ్బై ఎనభై తొంభై".split()


def number_words(n):
    """Cardinal number in Telugu words (0 <= n < 100000)."""
    if n < 10:
        return _UNITS[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS[tens] + ("" if unit == 0 else " " + _UNITS[unit])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if hundreds == 1:
            return "వంద" if rest == 0 else "నూట " + number_words(rest)
        return _UNITS[hundreds] + (" వందలు" if rest == 0 else " వందల " + number_words(rest))
    if n < 100000:
        thousands, rest = divmod(n, 1000)
        if thousands == 1:
            return "వెయ్యి" + ("" if rest == 0 else " " + number_words(rest))
        return number_words(thousands) + (" వేలు" if rest == 0 else " వేల " + number_words(rest))
    raise ValueError("number too large: %d" % n)


def _say_number(digits):
    if len(digits) > 5 or (len(digits) > 1 and digits[0] == "0"):
        return " ".join(_UNITS[int(d)] for d in digits)  # ids, phone numbers: digit by digit
    return number_words(int(digits))


def _number_with_suffix(m):
    words, suffix = _say_number(m.group(1)), m.group(2)
    if suffix.startswith("వ") and words.endswith("ు"):
        return words[:-1] + suffix  # ordinal: పన్నెండు + వ -> పన్నెండవ
    return words + (" " + suffix if suffix else "")


_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")
_NUM_RE = re.compile("(\\d+)([ఀ-౿]*)")
_PUNCT_RE = re.compile("[^ఀ-౿\\s]")


def normalize_telugu(text):
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    text = _TIME_RE.sub(lambda m: " %s %s " % (_say_number(m.group(1)), _say_number(m.group(2).lstrip("0") or "0")),
                        text)
    text = _NUM_RE.sub(_number_with_suffix, text)
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def build_telugu_dictionary(index_rows, out_path):
    """MFA dictionary (word<TAB>phones) for every word of the exported corpus."""
    import os
    from collections import Counter

    from fs2ra.tables import write_csv

    counts = Counter(w for r in index_rows for w in r["spoken_text"].split())
    lines, report = [], []
    for word in sorted(counts):
        phones = g2p(word)
        lines.append("%s\t%s" % (word, " ".join(phones)))
        report.append({"word": word, "count": counts[word], "pronunciation": " ".join(phones)})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    write_csv(os.path.splitext(out_path)[0] + "_words.csv", report)
    used = Counter(p for r in report for p in r["pronunciation"].split() for _ in range(r["count"]))
    print("dictionary: %d words, %d phones used -> %s" % (len(report), len(used), out_path))
    return report
