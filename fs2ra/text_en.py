"""English text front end shared by alignment (MFA transcripts/dictionary),
training and inference, so all three see the same words and phonemes.

normalize_english() returns the spoken-form word string; the raw text is always
kept separately. Pronunciations come from the upstream LibriSpeech lexicon with
g2p_en for out-of-vocabulary words, matching upstream synthesize.py.
"""
import re
import unicodedata

from text.cleaners import expand_abbreviations  # upstream (third_party/FastSpeech2 on sys.path)
from text.numbers import normalize_numbers

_DIGITS = "zero one two three four five six seven eight nine".split()

# ARPAbet letter names, for spelled-out acronyms (lexicon "A" lists AH0 first).
LETTER_NAMES = {
    "a": "EY1", "b": "B IY1", "c": "S IY1", "d": "D IY1", "e": "IY1", "f": "EH1 F",
    "g": "JH IY1", "h": "EY1 CH", "i": "AY1", "j": "JH EY1", "k": "K EY1", "l": "EH1 L",
    "m": "EH1 M", "n": "EH1 N", "o": "OW1", "p": "P IY1", "q": "K Y UW1", "r": "AA1 R",
    "s": "EH1 S", "t": "T IY1", "u": "Y UW1", "v": "V IY1", "w": "D AH1 B AH0 L Y UW0",
    "x": "EH1 K S", "y": "W AY1", "z": "Z IY1",
}
# Pseudo-words emitted by the normaliser; they take priority over the lexicon.
EXTRA_PRONUNCIATIONS = {"a_m": [["EY1", "EH1", "M"]], "p_m": [["P", "IY1", "EH1", "M"]]}
ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")


def spelled(word):
    return " ".join(LETTER_NAMES[c] for c in word).split()


_ONES = "zero one two three four five six seven eight nine ten eleven twelve".split()


def _say_digits(m):
    return " " + " ".join(_DIGITS[int(d)] for d in m.group(0)) + " "


def _decade(number):
    words = normalize_numbers(number)  # "nineteen ninety", "twenty twenty"
    return words[:-1] + "ies" if words.endswith("y") else words + "s"


def _time(m):
    hour, minute = int(m.group(1)), int(m.group(2))
    words = normalize_numbers(str(hour))
    if minute == 0:
        words += " o'clock"
    elif minute < 10:
        words += " oh " + _ONES[minute]
    else:
        words += " " + normalize_numbers(str(minute))
    suffix = (m.group(3) or "").lower().replace(".", "")
    if suffix in ("am", "pm"):
        words += " " + suffix[0] + "_m"
    return " " + words + " "


_RULES = [
    (re.compile(r"_"), " "),  # raw underscores first; a_m / p_m are inserted below
    # 10:45 AM -> ten forty-five a_m ; 9:05 -> nine oh five
    (re.compile(r"\b(\d{1,2}):(\d{2})(?:\s*(AM|PM|am|pm|a\.m\.|p\.m\.))?(?![\w:])"), _time),
    # phone / PIN / account numbers (5+ digits, no separators) are read digit by digit
    (re.compile(r"(?<![\d,.])\d{5,}(?![\d,.]*\d)"), _say_digits),
    (re.compile(r"%"), " percent "),
    (re.compile(r"&"), " and "),
    (re.compile(r"\+"), " plus "),
    (re.compile(r"@"), " at "),
]


def normalize_english(text):
    """Raw transcript/test sentence -> lowercase spoken words separated by single
    spaces (apostrophes inside words kept). Deterministic; no spelling correction."""
    text = unicodedata.normalize("NFKC", text).replace("\r", " ").replace("\n", " ")
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    text = re.sub(r"\b(\d*0)s\b", lambda m: _decade(m.group(1)), text)  # 1990s -> nineteen nineties
    # split digit/letter runs (9Organic, RX100) but keep ordinals (11th) for normalize_numbers
    text = re.sub(r"(\d)(?!(?:st|nd|rd|th)\b)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = normalize_numbers(text)  # years, ordinals, $, decimals, commas, plain cardinals
    text = expand_abbreviations(text.lower()).replace("-", " ")
    text = re.sub(r"[^a-z'_ ]", " ", text)  # "_" only survives in a_m / p_m
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)  # quotes, not in-word apostrophes
    return re.sub(r"\s+", " ", text).strip()


def read_lexicon(path):
    """word (lowercase) -> list of pronunciations (lists of ARPAbet phones), file order."""
    lexicon = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                lexicon.setdefault(parts[0].lower(), []).append(parts[1:])
    return lexicon


class EnglishG2P:
    """LibriSpeech lexicon first, g2p_en for OOV words (as upstream synthesize.py)."""

    def __init__(self, lexicon_path):
        from g2p_en import G2p

        self.lexicon = read_lexicon(lexicon_path)
        self._g2p = G2p()

    def pronunciations(self, word):
        if word in EXTRA_PRONUNCIATIONS:
            return EXTRA_PRONUNCIATIONS[word], "extra"
        if word in self.lexicon:
            return self.lexicon[word], "lexicon"
        phones = [p for p in self._g2p(word) if p.strip() and re.match(r"^[A-Z]+[0-2]?$", p)]
        return [phones], "g2p_en"
