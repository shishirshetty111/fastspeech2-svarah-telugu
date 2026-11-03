"""The assignment's test sentences must be preserved exactly as given in the PDF."""
import json
import os
import re
import unicodedata

import pytest

from conftest import REPO_ROOT

PDF = os.path.join(REPO_ROOT, "docs", "FastSpeech2_Assignment.pdf")


def load(lang):
    with open(os.path.join(REPO_ROOT, "project", lang, "test_sentences.json"), encoding="utf-8") as f:
        return {s["id"]: s["text"] for s in json.load(f)["sentences"]}


def test_ids_and_invariants():
    en, te = load("English"), load("Telugu")
    assert list(en) == ["E1", "E2a", "E2b", "E3", "E4"]
    assert list(te) == ["T1", "T2a", "T2b", "T3", "T4"]
    assert "rhyth  m" in en["E3"] and "అడ  విలో" in te["T3"]  # deliberate double spaces
    assert "conduct" in en["E2a"] and "conduct" in en["E2b"]
    assert "పట్టు" in te["T2a"] and "పట్టు" in te["T2b"]
    for text in list(en.values()) + list(te.values()):
        assert unicodedata.normalize("NFC", text) == text and text == text.strip()
    assert all(re.fullmatch(r"[ఀ-౿0-9 ,.:]+", t) for t in te.values())


def test_sentences_match_pdf_text():
    pymupdf = pytest.importorskip("pymupdf")
    lines = []
    for page in pymupdf.open(PDF):
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                text = "".join(s["text"] for s in line["spans"])
                if not re.fullmatch(r"\s*[ET]\d\s*", text):  # drop table ID cells
                    lines.append(text)
    # Undo layout wrapping: one trailing layout space per line, no space at mid-word span splits.
    pdf_text = "".join(lines)
    for text in list(load("English").values()) + list(load("Telugu").values()):
        assert text in pdf_text, text
