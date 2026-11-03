"""CPU checks that the patched upstream works on a modern stack and that the
acoustic-feature front end matches the HiFi-GAN vocoder configuration."""
import json
import os

import numpy as np
import soundfile as sf
import torch

from conftest import LJSPEECH_CKPT, REPO_ROOT, UPSTREAM_DIR, requires_ljspeech_ckpt
from baseline_inference import load_configs


def test_compat_patch_applied():
    with open(os.path.join(UPSTREAM_DIR, "model", "modules.py")) as f:
        src = f.read()
    assert "energy_target, src_mask, e_control" in src
    with open(os.path.join(UPSTREAM_DIR, "audio", "stft.py")) as f:
        assert ".cuda()," not in f.read()


def test_mel_config_matches_hifigan():
    preprocess_config, _, _ = load_configs(UPSTREAM_DIR)
    pp = preprocess_config["preprocessing"]
    with open(os.path.join(UPSTREAM_DIR, "hifigan", "config.json")) as f:
        voc = json.load(f)
    assert pp["audio"]["sampling_rate"] == voc["sampling_rate"]
    assert pp["stft"]["filter_length"] == voc["n_fft"]
    assert pp["stft"]["hop_length"] == voc["hop_size"]
    assert pp["stft"]["win_length"] == voc["win_size"]
    assert pp["mel"]["n_mel_channels"] == voc["num_mels"]
    assert pp["mel"]["mel_fmin"] == voc["fmin"]
    assert pp["mel"]["mel_fmax"] == voc["fmax"]


def test_stft_mel_and_energy_on_cpu():
    import audio as Audio

    pp = load_configs(UPSTREAM_DIR)[0]["preprocessing"]
    stft = Audio.stft.TacotronSTFT(
        pp["stft"]["filter_length"],
        pp["stft"]["hop_length"],
        pp["stft"]["win_length"],
        pp["mel"]["n_mel_channels"],
        pp["audio"]["sampling_rate"],
        pp["mel"]["mel_fmin"],
        pp["mel"]["mel_fmax"],
    )
    sr, hop = pp["audio"]["sampling_rate"], pp["stft"]["hop_length"]
    t = np.arange(sr) / sr
    wav = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    mel, energy = Audio.tools.get_mel_from_wav(wav, stft)
    assert mel.shape == (80, len(wav) // hop + 1)
    assert energy.shape == (mel.shape[1],)
    assert np.isfinite(mel).all() and np.isfinite(energy).all()
    # log(clamp(x, 1e-5)) dynamic range compression floor
    assert mel.min() >= np.log(1e-5) - 1e-4


@requires_ljspeech_ckpt
def test_checkpoint_matches_upstream_architecture():
    from model import FastSpeech2
    from utils.model import load_checkpoint

    preprocess_config, model_config, _ = load_configs(UPSTREAM_DIR)
    model = FastSpeech2(preprocess_config, model_config)
    state = load_checkpoint(LJSPEECH_CKPT, torch.device("cpu"))["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected


@requires_ljspeech_ckpt
def test_baseline_inference_writes_wav(tmp_path):
    import subprocess
    import sys

    script = os.path.join(REPO_ROOT, "scripts", "baseline_inference.py")
    subprocess.run(
        [sys.executable, script, "--text", "Hello, how are you?", "--out_dir", str(tmp_path)],
        check=True,
    )
    wav, sr = sf.read(tmp_path / "baseline_hello.wav")
    with open(tmp_path / "baseline_hello.json") as f:
        summary = json.load(f)
    assert sr == 22050 and len(wav) > sr // 2
    assert np.isfinite(wav).all() and np.abs(wav).max() > 0.05
    assert summary["sum_durations"] == summary["mel_frames"] == len(wav) // 256
