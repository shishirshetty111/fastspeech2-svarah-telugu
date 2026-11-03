"""Phase 1 baseline: synthesize one sentence with the upstream pretrained LJSpeech
FastSpeech 2 checkpoint (ming024/FastSpeech2, 900k steps) + upstream HiFi-GAN.

This is NOT a model trained for the assignment; it only proves that the upstream
text front end, acoustic model, variance adaptor and vocoder connect end to end.

Usage (from the repo root, local or Colab/Kaggle):
    python scripts/baseline_inference.py --text "Hello, how are you?" \
        --ckpt pretrained/LJSpeech/900000.pth.tar --out_dir outputs/baseline
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_UPSTREAM = os.path.join(REPO_ROOT, "third_party", "FastSpeech2")


def load_configs(upstream, dataset="LJSpeech"):
    cfg_dir = os.path.join(upstream, "config", dataset)
    configs = []
    for name in ("preprocess", "model", "train"):
        with open(os.path.join(cfg_dir, name + ".yaml"), "r") as f:
            configs.append(yaml.load(f, Loader=yaml.FullLoader))
    preprocess_config = configs[0]
    # Upstream paths are relative to its own root; make them CWD-independent.
    for key in ("lexicon_path", "preprocessed_path"):
        preprocess_config["path"][key] = os.path.join(
            upstream, preprocess_config["path"][key]
        )
    return tuple(configs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hello, how are you?")
    parser.add_argument("--utt_id", default="baseline_hello")
    parser.add_argument(
        "--ckpt", default=os.path.join(REPO_ROOT, "pretrained", "LJSpeech", "900000.pth.tar")
    )
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "outputs", "baseline"))
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    args = parser.parse_args()

    upstream = os.path.abspath(args.upstream)
    sys.path.insert(0, upstream)
    import torch
    from model import FastSpeech2
    from synthesize import preprocess_english
    from utils.model import get_vocoder, load_checkpoint
    from utils.tools import synth_samples, to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config, model_config, train_config = load_configs(upstream)

    model = FastSpeech2(preprocess_config, model_config).to(device)
    model.load_state_dict(load_checkpoint(args.ckpt, device)["model"], strict=True)
    model.eval()
    vocoder = get_vocoder(model_config, device)

    sequence = preprocess_english(args.text, preprocess_config)
    batch = (
        [args.utt_id],
        [args.text],
        np.array([0]),
        np.array([sequence]),
        np.array([len(sequence)]),
        len(sequence),
    )
    batch = to_device(batch, device)

    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        # Inference: no targets are passed, so duration/pitch/energy come from
        # the variance-adaptor predictors (text-only conditioning).
        output = model(*batch[2:])
        synth_samples(
            batch, output, vocoder, model_config, preprocess_config, args.out_dir
        )

    src_len = int(output[8][0])
    durations = output[5][0, :src_len].cpu().numpy()
    summary = {
        "text": args.text,
        "phoneme_ids": sequence.tolist(),
        "device": str(device),
        "checkpoint": os.path.relpath(args.ckpt, REPO_ROOT),
        "predicted_durations_frames": durations.tolist(),
        "predicted_pitch_normalized": output[2][0, :src_len].cpu().numpy().tolist(),
        "predicted_energy_normalized": output[3][0, :src_len].cpu().numpy().tolist(),
        "mel_frames": int(output[9][0]),
        "sum_durations": float(durations.sum()),
    }
    with open(os.path.join(args.out_dir, args.utt_id + ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", os.path.join(args.out_dir, args.utt_id + ".wav"))


if __name__ == "__main__":
    main()
