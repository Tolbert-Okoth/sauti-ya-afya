"""Denoising must remove noise WITHOUT removing the sounds that carry the diagnosis."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import denoise as dn  # noqa: E402

SR = dn.SR
T = np.arange(SR * 10) / SR


def rng():
    return np.random.default_rng(0)


def breath(r):
    noise = np.convolve(r.standard_normal(len(T)), np.ones(6) / 6, "same")
    return noise * (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * T)) * 0.05


def hiss(r):
    return r.standard_normal(len(T)) * 0.01


def band_energy(y, lo, hi):
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    return spectrum[(freqs >= lo) & (freqs < hi)].sum()


def db(x):
    return 10 * np.log10(x)


def test_crackles_are_preserved_and_not_flagged():
    r = rng()
    length = int(0.008 * SR)
    k = np.arange(length) / SR
    burst = np.sin(2 * np.pi * 500 * k) * np.exp(-k / 0.002) * 0.4
    crackles = np.zeros(len(T))
    positions = r.choice(len(T) - length, 60, replace=False)
    for p in positions:
        crackles[p:p + length] += burst
    signal = (breath(r) + hiss(r) + crackles).astype(np.float32)
    clean, mask, _ = dn.denoise(signal)
    kept = np.mean([np.sum(clean[p:p + length] ** 2) / np.sum(signal[p:p + length] ** 2) for p in positions])
    assert db(kept) > -3.0
    assert mask.mean() < 0.02


def test_wheeze_is_preserved_and_not_notched():
    r = rng()
    wheeze = 0.08 * np.sin(2 * np.pi * (400 * T + 20 * np.sin(2 * np.pi * 0.5 * T))) * (np.sin(2 * np.pi * 0.3 * T) > 0.3)
    signal = (breath(r) + hiss(r) + wheeze).astype(np.float32)
    clean, mask, report = dn.denoise(signal)
    assert db(band_energy(clean, 350, 450) / band_energy(signal, 350, 450)) > -1.0
    assert report["hum_removed_hz"] == []
    assert mask.mean() < 0.02


def test_steady_hiss_is_reduced():
    r = rng()
    background = hiss(r)
    signal = (breath(r) + background).astype(np.float32)
    clean, _, _ = dn.denoise(signal)
    # Above 2.5 kHz the synthetic breath is weak, so this band is mostly hiss
    assert db(band_energy(clean, 3000, 7000) / band_energy(signal, 3000, 7000)) < -4.0


def test_mains_hum_is_removed():
    r = rng()
    hum = sum(0.05 / k * np.sin(2 * np.pi * 50 * k * T) for k in range(1, 8))
    signal = (breath(r) + hiss(r) + hum).astype(np.float32)
    clean, _, report = dn.denoise(signal)
    assert report["hum_removed_hz"][:3] == [50, 100, 150]
    assert db(band_energy(clean, 98, 102) / band_energy(signal, 98, 102)) < -15


def test_friction_burst_is_flagged_only_where_it_happens():
    r = rng()
    signal = breath(r) + hiss(r)
    signal[30000:36000] += r.standard_normal(6000) * 0.6
    _, mask, report = dn.denoise(signal.astype(np.float32))
    assert mask[30500:35500].mean() > 0.9
    assert mask[:28000].mean() < 0.01
    assert report["friction_fraction"] > 0


def test_short_input_passes_through():
    y = np.zeros(100, dtype=np.float32)
    clean, mask, _ = dn.denoise(y)
    assert len(clean) == 100 and not mask.any()
