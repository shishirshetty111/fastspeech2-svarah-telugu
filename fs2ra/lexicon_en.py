"""Build the MFA pronunciation dictionary for the English corpus vocabulary."""
import os
from collections import Counter

from fs2ra.tables import write_csv
from fs2ra.text_en import ACRONYM_RE, EnglishG2P, spelled


def build_english_dictionary(lexicon_path):
    g2p = EnglishG2P(lexicon_path)

    def build(index_rows, out_path):
        """index_rows: corpus index (raw_text, spoken_text). All-caps OOV tokens in the
        raw text (ESIC, AICTE, COVID) get both a spelled-out and a g2p_en variant;
        MFA picks whichever matches the audio."""
        counts = Counter(w for r in index_rows for w in r["spoken_text"].split())
        acronyms = {t.lower() for r in index_rows for t in ACRONYM_RE.findall(r["raw_text"])}
        lines, report = [], []
        for word in sorted(counts):
            prons, source = g2p.pronunciations(word)
            if source == "g2p_en" and word in acronyms and word.isalpha():
                prons, source = [spelled(word)] + prons, "acronym:spelled+g2p_en"
            for phones in prons:
                lines.append("%s\t%s" % (word, " ".join(phones)))
            report.append({"word": word, "count": counts[word], "source": source,
                           "pronunciations": " | ".join(" ".join(p) for p in prons)})
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        write_csv(os.path.splitext(out_path)[0] + "_words.csv", report)
        oov = [r for r in report if r["source"] != "lexicon"]
        print("dictionary: %d words (%d g2p_en OOV, %d tokens) -> %s" % (
            len(report), len(oov), sum(r["count"] for r in oov), out_path))
        return report

    return build


def aligned_variant_counts(textgrid_dir, utt_ids):
    """word -> Counter of the pronunciations MFA chose for it in these utterances
    (words tier vs phones tier of the raw MFA TextGrids)."""
    import glob

    import tgt

    counts = {}
    for path in glob.glob(os.path.join(textgrid_dir, "*", "*.TextGrid")):
        if os.path.basename(path)[: -len(".TextGrid")] not in utt_ids:
            continue
        tg = tgt.io.read_textgrid(path)
        phones = tg.get_tier_by_name("phones").intervals
        for w in tg.get_tier_by_name("words").intervals:
            if w.text:
                pron = " ".join(p.text for p in phones
                                if p.start_time >= w.start_time - 1e-6 and p.end_time <= w.end_time + 1e-6)
                counts.setdefault(w.text.lower(), Counter())[pron] += 1
    return counts


def build_variant_table(lexicon_path, textgrid_dir, train_ids, out_path):
    """For words with several lexicon pronunciations, how often each was chosen by MFA in
    the *training* alignments. Inference uses the most frequent one (the LibriSpeech
    lexicon often lists a reduced form first, e.g. TWENTY "T W EH1 N IY0" before
    "T W EH1 N T IY0", which training speakers used 96 vs 19 times)."""
    import json

    from fs2ra.text_en import read_lexicon

    lexicon = read_lexicon(lexicon_path)
    counts = aligned_variant_counts(textgrid_dir, set(train_ids))
    table = {w: dict(c.most_common()) for w, c in sorted(counts.items())
             if len(lexicon.get(w, [])) > 1}
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(table, f, indent=0, sort_keys=True)
    print("variant table: %d multi-pronunciation words seen in training -> %s" % (len(table), out_path))
    return table
