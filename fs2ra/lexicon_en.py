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
