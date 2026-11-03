"""English front end: spoken-form normalisation and pronunciation sources."""
import os

import pytest

from conftest import UPSTREAM_DIR
from fs2ra.text_en import ACRONYM_RE, EnglishG2P, normalize_english, spelled

LEXICON = os.path.join(UPSTREAM_DIR, "lexicon", "librispeech-lexicon.txt")


@pytest.mark.parametrize("raw,spoken", [
    ("The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
     "the meeting will start at ten forty five a_m on twelve november twenty twenty six so andaroo should arrive on time"),
    ("The rhyth  m of the orchestra changed unexpectedly.", "the rhyth m of the orchestra changed unexpectedly"),
    ("The orchestra's conduct", "the orchestra's conduct"),
    ("pincode 593081", "pincode five nine three zero eight one"),
    ("from 11th to 14th century", "from eleventh to fourteenth century"),
    ("the 1990s and the 80s", "the nineteen nineties and the eighties"),
    ("at 9:05 pm", "at nine oh five p_m"),
    ("50% off & 2+2 @ home", "fifty percent off and two plus two at home"),
    ("Dr. Rao paid $5", "doctor rao paid five dollars"),
    ("my_upi", "my upi"),
])
def test_normalize_english(raw, spoken):
    assert normalize_english(raw) == spoken


def test_acronyms():
    assert ACRONYM_RE.findall("ESIC and AICTE, not An or I") == ["ESIC", "AICTE"]
    assert spelled("upi") == ["Y", "UW1", "P", "IY1", "AY1"]


def test_pronunciation_sources():
    g2p = EnglishG2P(LEXICON)
    assert g2p.pronunciations("a_m") == ([["EY1", "EH1", "M"]], "extra")
    prons, source = g2p.pronunciations("conduct")
    assert source == "lexicon" and len(prons) == 2  # noun and verb stress both kept for MFA
    prons, source = g2p.pronunciations("andaroo")
    assert source == "g2p_en" and prons[0]
