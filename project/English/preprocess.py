"""English (Svarah) preprocessing: export corpus -> MFA alignment -> FastSpeech 2 targets.

    python project/English/preprocess.py --stage export align features             # full 5 h + 1 h (+ reserve)
    python project/English/preprocess.py --smoke 30 10 --stage export align features   # 30 train + 10 val

Set FS2_MFA_CMD to the MFA executable (see docs/ENVIRONMENT.md). Outputs go to
$FS2_DATA_ROOT/english[_smoke]/ (git-ignored).
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO_ROOT, os.path.join(REPO_ROOT, "third_party", "FastSpeech2")]

from fs2ra import pipeline  # noqa: E402
from fs2ra.lexicon_en import build_english_dictionary  # noqa: E402
from fs2ra.text_en import normalize_english  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--stage", nargs="+", choices=["export", "align", "features"], required=True)
    parser.add_argument("--smoke", nargs=2, type=int, metavar=("N_TRAIN", "N_VAL"),
                        help="only the first N train and M val utterances, in a separate work dir")
    parser.add_argument("--num_jobs", type=int, default=None)
    args = parser.parse_args()

    cfg = pipeline.load(args.config)
    smoke = args.smoke is not None
    dirs = pipeline.work_dirs(cfg, smoke)
    if "export" in args.stage:
        rows = pipeline.select_rows(cfg, *(args.smoke or (0, 0)))
        build_dict = build_english_dictionary(cfg["preprocess"]["path"]["lexicon_path"])
        pipeline.stage_export(cfg, rows, dirs, normalize_english, build_dict)
    if "align" in args.stage:
        pipeline.stage_align(cfg, dirs, args.num_jobs)
    if "features" in args.stage:
        from fs2ra.features import stage_features

        stage_features(cfg, dirs)


if __name__ == "__main__":
    main()
