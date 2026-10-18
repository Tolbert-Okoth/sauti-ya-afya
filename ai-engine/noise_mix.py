"""Mix noise into recordings at a chosen signal-to-noise ratio (training augmentation and benchmarks)."""
import glob
import os

import numpy as np

SR = 16000
SYNTHETIC = ("white_hiss", "mains_hum")


def noise_groups(noise_dir):
    """{group name: [files]} from one sub-folder per group."""
    return {g: sorted(glob.glob(os.path.join(noise_dir, g, "*")))
            for g in sorted(os.listdir(noise_dir)) if os.path.isdir(os.path.join(noise_dir, g))}


def synthetic_noise(kind, n, rng):
    t = np.arange(n) / SR
    if kind == "white_hiss":
        return rng.standard_normal(n).astype(np.float32)
    if kind == "mains_hum":
        base = 50 if rng.random() < 0.8 else 60
        return sum(np.sin(2 * np.pi * base * k * t + rng.uniform(0, 6.28)) / k for k in range(1, 8)).astype(np.float32)
    raise ValueError(kind)


def mix(y, noise, snr_db, rng):
    if len(noise) < len(y):
        noise = np.tile(noise, int(np.ceil(len(y) / len(noise))))
    start = rng.integers(0, len(noise) - len(y) + 1)
    noise = noise[start:start + len(y)]
    p_signal, p_noise = np.mean(y ** 2) + 1e-12, np.mean(noise ** 2) + 1e-12
    mixed = y + noise * np.sqrt(p_signal / (p_noise * 10 ** (snr_db / 10)))
    peak = np.max(np.abs(mixed))
    return (mixed / peak * 0.9 if peak > 0.99 else mixed).astype(np.float32)   # no clipping from mixing
