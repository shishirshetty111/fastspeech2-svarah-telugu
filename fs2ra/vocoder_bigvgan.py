"""Optional vocoder: NVIDIA BigVGAN-v2 (MIT), universal, 22.05 kHz / 80 mels / hop 256 /
fmax 8 kHz -- the same mel definition as our FastSpeech 2 targets (n_fft 1024, win 1024,
fmin 0, fmax 8000, natural log of the magnitude mel clamped at 1e-5). Inference only:
nothing is trained or fine-tuned.

Code: github.com/NVIDIA/BigVGAN @ BIGVGAN_COMMIT, cloned to third_party/BigVGAN by
scripts/setup_upstream.py --bigvgan. Weights: huggingface.co/BIGVGAN_REPO @ BIGVGAN_REVISION.
"""
import importlib
import os
import sys

import numpy as np
import torch

from fs2ra.hf import REPO_ROOT

BIGVGAN_URL = "https://github.com/NVIDIA/BigVGAN.git"
BIGVGAN_COMMIT = "7d2b454564a6c7d014227f635b7423881f14bdac"
BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x"
BIGVGAN_REVISION = "a9d8b711c344f3c49cc1a05d0c8b6741cf733aa8"
BIGVGAN_DIR = os.path.join(REPO_ROOT, "third_party", "BigVGAN")

# BigVGAN's top-level modules shadow upstream FastSpeech 2's "utils" package; import them
# in isolation and restore sys.modules afterwards.
_SHADOWED = ("utils", "env", "activations", "alias_free_activation", "bigvgan", "meldataset")


def _import_bigvgan():
    if not os.path.exists(os.path.join(BIGVGAN_DIR, "bigvgan.py")):
        raise FileNotFoundError("BigVGAN code missing: python scripts/setup_upstream.py --bigvgan")
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in _SHADOWED}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, BIGVGAN_DIR)
    try:
        return importlib.import_module("bigvgan"), importlib.import_module("env")
    finally:
        sys.path.remove(BIGVGAN_DIR)
        for k in [k for k in sys.modules if k.split(".")[0] in _SHADOWED]:
            del sys.modules[k]
        sys.modules.update(saved)


def check_compatible(h, preprocess_config):
    """The checkpoint's mel definition must equal the FastSpeech 2 one."""
    a, s, m = (preprocess_config["preprocessing"][k] for k in ("audio", "stft", "mel"))
    ours = (a["sampling_rate"], s["filter_length"], s["hop_length"], s["win_length"], m["n_mel_channels"],
            m["mel_fmin"], m["mel_fmax"])
    theirs = (h.sampling_rate, h.n_fft, h.hop_size, h.win_size, h.num_mels, h.fmin, h.fmax)
    if ours != theirs:
        raise ValueError("BigVGAN mel config %s != FastSpeech 2 %s" % (theirs, ours))


def get_bigvgan(preprocess_config, device):
    import json

    from huggingface_hub import hf_hub_download

    bigvgan, env = _import_bigvgan()
    # files fetched directly (BigVGAN.from_pretrained breaks with newer huggingface_hub)
    with open(hf_hub_download(BIGVGAN_REPO, "config.json", revision=BIGVGAN_REVISION)) as f:
        h = env.AttrDict(json.load(f))
    check_compatible(h, preprocess_config)
    model = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    state = torch.load(hf_hub_download(BIGVGAN_REPO, "bigvgan_generator.pt", revision=BIGVGAN_REVISION),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state["generator"])
    model.remove_weight_norm()
    return model.eval().to(device)


def bigvgan_infer(mels, vocoder, preprocess_config, lengths=None):
    """Same interface as upstream utils.model.vocoder_infer: mels (B, 80, T) log-mel ->
    list of int16 waveforms, scaled like upstream (max_wav_value)."""
    with torch.inference_mode():
        wavs = vocoder(mels).squeeze(1)
    max_wav = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    wavs = (wavs.float().cpu().numpy() * max_wav).clip(-max_wav, max_wav - 1).astype(np.int16)
    out = [w for w in wavs]
    if lengths is not None:
        out = [w[:n] for w, n in zip(out, lengths)]
    return out
