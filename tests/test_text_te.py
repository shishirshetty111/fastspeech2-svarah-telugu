"""Telugu rule-based front end and symbol-table extension."""
import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, UPSTREAM_DIR
from fs2ra.text_te import PHONES, g2p, normalize_telugu, number_words


@pytest.mark.parametrize("word, phones", [
    ("పట్టు", "p a tt tt u"),            # geminate + inherent vowel suppressed by virama
    ("రంగం", "r a ng g a m"),             # anusvara: homorganic before g, m word-finally
    ("ఉంచండి", "u ny ch a nn dd i"),      # palatal and retroflex nasal
    ("నవంబర్", "n a v a m b a r"),        # word-final virama: no vowel
    ("కృషి", "k r u ss i"),               # vocalic r -> r u
    ("దుఃఖం", "d u h kh a m"),            # visarga
    ("ఔషధాల", "au ss a dh aa l a"),
    ("న్యూరోఫార్మకోలాజికల్", "n y uu r oo ph aa r m a k oo l aa j i k a l"),
])
def test_g2p(word, phones):
    assert " ".join(g2p(word)) == phones
    assert set(g2p(word)) <= set(PHONES)


def test_g2p_rejects_other_scripts():
    with pytest.raises(ValueError):
        g2p("abc")


@pytest.mark.parametrize("n, words", [
    (0, "సున్నా"), (12, "పన్నెండు"), (26, "ఇరవై ఆరు"), (45, "నలభై ఐదు"), (100, "వంద"),
    (125, "నూట ఇరవై ఐదు"), (300, "మూడు వందలు"), (1000, "వెయ్యి"), (2026, "రెండు వేల ఇరవై ఆరు"),
])
def test_number_words(n, words):
    assert number_words(n) == words


def test_normalize_t4_and_spacing():
    with open(os.path.join(REPO_ROOT, "project", "Telugu", "test_sentences.json"), encoding="utf-8") as f:
        t = {s["id"]: s["text"] for s in json.load(f)["sentences"]}
    t4 = normalize_telugu(t["T4"])
    assert "రెండు వేల ఇరవై ఆరు నవంబర్ పన్నెండవ తేదీ" in t4
    assert "ఉదయం పది నలభై ఐదు గంటలకు" in t4
    assert not any(c.isdigit() or c in ",.:" for c in t4)
    # T3's double space is kept in the raw text; the front end splits on whitespace
    assert normalize_telugu(t["T3"]).split()[1:3] == ["అడ", "విలో"]
    for s in t.values():
        for w in normalize_telugu(s).split():
            g2p(w)


def test_symbol_registration_isolated():
    """Run in a subprocess: registration mutates upstream's global symbol table."""
    code = (
        "from fs2ra.symbols import register_language, check_phone_strings\n"
        "import text\n"
        "from text.symbols import symbols\n"
        "assert register_language('english') == 360\n"
        "from fs2ra.text_te import PHONES\n"
        "n = register_language('telugu')\n"
        "ids = [text._symbol_to_id['@' + p] for p in PHONES]\n"
        "assert len(set(ids)) == len(PHONES) and n == len(symbols) and max(ids) < n\n"
        "check_phone_strings(['{' + ' '.join(PHONES) + ' sp}'], 'telugu')\n"
        "print(n)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=os.pathsep.join([REPO_ROOT, UPSTREAM_DIR])))
    assert out.returncode == 0, out.stderr
    assert int(out.stdout) > 360
