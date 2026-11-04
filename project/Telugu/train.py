"""Telugu (IndicTTS, male) FastSpeech 2 training, resumable (assignment layout: project/Telugu/train.py).

    python project/Telugu/train.py --dry-run                 # few CPU steps through every code path
    python project/Telugu/train.py                           # full run; resumes from the latest checkpoint
    python project/Telugu/train.py --total_step 60000        # override config train.step.total_step
    python project/Telugu/train.py --amp                     # mixed precision on GPU (faster)

Needs the features from `preprocess.py` under $FS2_DATA_ROOT/telugu/preprocessed.
Checkpoints, CSV logs, loss plots and teacher-forced validation samples go to
$FS2_OUTPUT_ROOT/Telugu/{ckpt,log,result} (point FS2_OUTPUT_ROOT at Google Drive on Colab).
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO_ROOT, os.path.join(REPO_ROOT, "third_party", "FastSpeech2")]

from fs2ra.config import load_config  # noqa: E402
from fs2ra.training import train  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--total_step", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="tiny run into *_dryrun output dirs")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="mixed precision (bf16 on L4/A100, fp16 on T4)")
    args = parser.parse_args()
    train(load_config(args.config), args.config, total_step=args.total_step, dry_run=args.dry_run,
          num_workers=args.num_workers, amp=args.amp)


if __name__ == "__main__":
    main()
