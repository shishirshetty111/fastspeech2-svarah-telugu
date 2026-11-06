"""Telugu FastSpeech 2 text-only inference (assignment layout: project/Telugu/inference.py).

    python project/Telugu/inference.py --ckpt project/Telugu/checkpoints/50000.pth.tar
    python project/Telugu/inference.py --ckpt <ckpt> --out_dir project/Results/Telugu   # final results
    python project/Telugu/inference.py --ckpt <ckpt> --ids T1 --text "any other sentence"

Synthesizes the five test sentences in test_sentences.json exactly as given (raw text is
normalized by the front end, never spell-corrected), or one --text. Duration, pitch and
energy are predicted from text only; see fs2ra/inference.py for the output layout.
The checkpoint folder (or its parent, for milestones/) needs the stats.json, speakers.json
and config.yaml that training copies next to the checkpoints.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO_ROOT, os.path.join(REPO_ROOT, "third_party", "FastSpeech2")]

from fs2ra.config import OUTPUT_ROOT, load_config  # noqa: E402
from fs2ra.inference import synthesize  # noqa: E402
from fs2ra.training import find_checkpoints  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    parser.add_argument("--ckpt", help="default: latest step in project/Telugu/checkpoints")
    parser.add_argument("--out_dir", help="default: $FS2_OUTPUT_ROOT/Telugu/inference/step<N>")
    parser.add_argument("--ids", nargs="*", help="subset of test sentence ids")
    parser.add_argument("--text", help="synthesize this text instead (id from --ids, default 'custom')")
    parser.add_argument("--speaker", help="speaker name from speakers.json (multi-speaker models)")
    parser.add_argument("--pace", type=float, default=1.0,
                        help="duration control: scale predicted phone durations (>1 slower; default 1.0)")
    parser.add_argument("--vocoder", help="hifigan | bigvgan | path to a fine-tuned HiFi-GAN g_* file "
                                          "(default: inference.vocoder in config.yaml, else hifigan)")
    args = parser.parse_args()

    ckpt = args.ckpt
    if not ckpt:
        found = find_checkpoints(os.path.join(HERE, "checkpoints"))
        if not found:
            sys.exit("no checkpoint given and none in %s" % os.path.join(HERE, "checkpoints"))
        ckpt = found[max(found)]
    if args.text:
        sentences = [{"id": (args.ids or ["custom"])[0], "text": args.text}]
    else:
        with open(os.path.join(HERE, "test_sentences.json"), encoding="utf-8") as f:
            sentences = [s for s in json.load(f)["sentences"] if not args.ids or s["id"] in args.ids]
    step = os.path.basename(ckpt).split(".")[0]
    out_dir = args.out_dir or os.path.join(OUTPUT_ROOT, "Telugu", "inference", "step" + step)
    synthesize(load_config(args.config), ckpt, sentences, out_dir, speaker=args.speaker, pace=args.pace, vocoder=args.vocoder)
    print("outputs in", out_dir)


if __name__ == "__main__":
    main()
