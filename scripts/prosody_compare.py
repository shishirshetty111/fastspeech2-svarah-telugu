"""Render English audio with and without the prosody post-rules, for listening.

Sentences: the five test sentences, held-out validation sentences of the inference speaker
(evaluation half of scripts/prosody_calibrate.py -- never used to set the rules; the real
recording is copied alongside for reference) and a few new sentences.

    python scripts/prosody_compare.py --ckpt project/English/checkpoints/20000.pth.tar
output: $FS2_OUTPUT_ROOT/English/prosody_compare/<variant>/... and index.txt
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO_ROOT, os.path.join(REPO_ROOT, "third_party", "FastSpeech2")]

import torch  # noqa: E402

from fs2ra import pipeline  # noqa: E402
from fs2ra.config import OUTPUT_ROOT, load_config  # noqa: E402
from fs2ra.inference import synthesize  # noqa: E402

NEW = ["Please call me back when you reach the station.",
       "The weather was pleasant, so we walked along the river.",
       "Could you send the report to my manager before Friday?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rules", default=os.path.join(REPO_ROOT, "project", "English", "prosody_rules.json"))
    ap.add_argument("--n_val", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(os.path.join(REPO_ROOT, "project", "English", "config.yaml"))
    speaker = cfg["inference"]["speaker"]
    dirs = pipeline.work_dirs(cfg, False)
    out = os.path.join(OUTPUT_ROOT, "English", "prosody_compare")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(REPO_ROOT, "project", "English", "test_sentences.json"), encoding="utf-8") as f:
        sents = [{"id": s["id"], "text": s["text"]} for s in json.load(f)["sentences"]]
    with open(os.path.join(dirs["preprocessed"], "val.txt"), encoding="utf-8") as f:
        val = [l.split("|") for l in f.read().splitlines()]
    val = [r for r in val if r[1] == speaker and int(hashlib.md5(r[0].encode()).hexdigest(), 16) % 2 == 1]
    val = sorted(val, key=lambda r: -len(r[3]))[: args.n_val]  # longest: most prosody to hear
    os.makedirs(os.path.join(out, "real_recordings"), exist_ok=True)
    for k, r in enumerate(val, 1):
        sents.append({"id": "V%d" % k, "text": r[3]})
        shutil.copy(os.path.join(dirs["corpus"], r[1], r[0] + ".wav"),
                    os.path.join(out, "real_recordings", "V%d.wav" % k))
    sents += [{"id": "N%d" % k, "text": t} for k, t in enumerate(NEW, 1)]

    with open(args.rules, encoding="utf-8") as f:
        rules = json.load(f)
    variants = {"0_none": "none", "1_rules_" + rules["enabled"]: args.rules}
    for r in rules["enabled"]:  # each enabled rule alone, to hear what it does
        if len(rules["enabled"]) > 1:
            p = os.path.join(out, "rules_only_%s.json" % r)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(dict(rules, enabled=r), f, indent=1)
            variants["2_only_" + r] = p
    for name, rp in variants.items():
        print("==", name)
        synthesize(cfg, args.ckpt, sents, os.path.join(out, name), device=torch.device("cpu"), prosody_rules=rp)
    with open(os.path.join(out, "index.txt"), "w", encoding="utf-8") as f:
        f.write("speaker %s; rules %s (enabled %s)\n" % (speaker, args.rules, rules["enabled"]))
        for s in sents:
            f.write("%s\t%s\n" % (s["id"], s["text"]))
    print("outputs in", out)


if __name__ == "__main__":
    main()
