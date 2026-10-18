"""
Lung-sound noise reduction for single-microphone (phone) recordings.

Design rule from the literature: generic "remove all noise" enhancement also
removes crackles and wheezes, which ARE the diagnosis. So each noise type gets
a targeted treatment, attenuation is capped, and sounds that can't be
separated safely (voices, crying, stethoscope/phone friction) are DETECTED and
excluded from analysis instead of being "repaired".

| Noise                        | Treatment                                              |
|------------------------------|--------------------------------------------------------|
| Mains hum (50/60 Hz + harm.) | narrow notch filters, only if persistent across the clip |
| Steady background (hiss, fan, room tone) | Wiener-style gain vs. a noise floor estimated from the quietest frames, capped at -12 dB |
| Heart sounds                 | low band (<200 Hz) attenuated during detected heart beats |
| Voices / crying / airborne   | detected by energy above 2.5 kHz (chest-coupled lung sound has little), frames muted |
| Friction / phone movement    | detected as loud, broadband, sustained (>60 ms) bursts, frames muted; short crackles (<20 ms) are not matched |
| Handling rumble / hiss       | 100-2000 Hz band-pass (audio_pipeline.bandpass)          |
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as F_audio

SR = 16000
N_FFT = 512          # 32 ms frames
HOP = 128            # 8 ms hop
FRAME_SEC = HOP / SR

# Stationary noise
NOISE_PERCENTILE = 10
NOISE_SMOOTH_FRAMES = 15         # 120 ms
OVER_SUBTRACTION = 1.5
GAIN_FLOOR = 10 ** (-12 / 20)      # never attenuate steady noise by more than 12 dB

# Heart sounds
HEART_BAND = (20, 150)
HEART_ATTENUATE_BELOW = 200
HEART_FACTOR = 4.0                 # low-band energy x median
HEART_GAIN = 10 ** (-12 / 20)

# Artifacts (muted and counted)
ARTIFACT_GAIN = 10 ** (-20 / 20)
HIGH_BAND = (2500, 7500)
LUNG_BAND = (100, 2000)
AIRBORNE_ABSOLUTE = 0.5            # high-band / lung-band energy
AIRBORNE_MIN_SEC = 0.10
FRICTION_FACTOR = 25.0             # broadband energy x median
FRICTION_FLATNESS = 0.30
FRICTION_MIN_SEC = 0.06

# Hum
HUM_BASES = (50, 60)
HUM_MAX_FREQ = 2000
HUM_PEAK_RATIO = 10.0              # peak vs. neighbouring spectrum (10 dB)
HUM_PERSISTENCE = 0.8              # fraction of 1 s segments where the peak is present
HUM_Q = 35.0

_window = torch.hann_window(N_FFT)


def _band(freqs, lo, hi):
    return (freqs >= lo) & (freqs < hi)


def _runs_at_least(mask, min_len):
    """Keep only runs of True that are at least min_len long."""
    out = np.zeros_like(mask)
    start = None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out[start:i] = True
            start = None
    return out


def _dilate(mask, n):
    if n <= 0 or not mask.any():
        return mask
    kernel = np.ones(2 * n + 1)
    return np.convolve(mask.astype(float), kernel, mode="same") > 0


def detect_hum(y):
    """Return the mains-hum harmonic frequencies present throughout the clip."""
    seg = SR  # 1 s segments -> 1 Hz resolution
    if len(y) < seg:
        return []
    n_seg = len(y) // seg
    win = np.hanning(seg)
    spectra = np.array([np.abs(np.fft.rfft(y[i * seg:(i + 1) * seg] * win)) ** 2 for i in range(n_seg)])
    freqs = np.fft.rfftfreq(seg, 1 / SR)

    def peak_ratio(psd, f):
        peak = psd[_band(freqs, f - 1.5, f + 1.5)].max()
        side = psd[_band(freqs, f - 15, f - 4) | _band(freqs, f + 4, f + 15)]
        return peak / (np.median(side) + 1e-20)

    best = []
    for base in HUM_BASES:
        found = []
        for f in range(base, HUM_MAX_FREQ + 1, base):
            ratios = np.array([peak_ratio(s, f) for s in spectra])
            if np.median(ratios) >= HUM_PEAK_RATIO and np.mean(ratios >= HUM_PEAK_RATIO / 2) >= HUM_PERSISTENCE:
                found.append(f)
        # Hum is a harmonic family starting low; a lone steady tone (e.g. a wheeze) is not notched
        if len(found) >= 2 and found[0] <= 3 * base and len(found) > len(best):
            best = found
    return best


def denoise(y):
    """
    y: mono float32 at 16 kHz.
    Returns (clean, artifact_mask, report):
      clean          denoised signal, same length
      artifact_mask  per-sample bool, True where voices/crying/friction were detected and muted
      report         what was found, for the API response
    """
    y = np.asarray(y, dtype=np.float32)
    y = y - float(np.mean(y)) if len(y) else y
    report = {"hum_removed_hz": [], "heart_sound_fraction": 0.0,
              "airborne_noise_fraction": 0.0, "friction_fraction": 0.0, "noise_floor_db": None}
    if len(y) < N_FFT * 4:
        return y, np.zeros(len(y), dtype=bool), report

    # 1. Mains hum
    hum = detect_hum(y)
    wave = torch.from_numpy(y)
    for f in hum:
        wave = F_audio.bandreject_biquad(wave, SR, central_freq=float(f), Q=HUM_Q)
    report["hum_removed_hz"] = hum

    # 2. STFT
    X = torch.stft(wave, N_FFT, HOP, window=_window, return_complex=True, center=True)
    P = (X.abs() ** 2).numpy()
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    n_frames = P.shape[1]

    # 3. Steady background: Wiener-style gain against the quietest frames
    # Percentile of TIME-SMOOTHED power: a raw per-frame percentile underestimates a
    # fluctuating noise floor ~10x (exponential statistics); 120 ms smoothing removes most of that bias
    P_smooth = F.avg_pool1d(torch.from_numpy(P)[None].float(), NOISE_SMOOTH_FRAMES, stride=1,
                            padding=NOISE_SMOOTH_FRAMES // 2, count_include_pad=False)[0].numpy()
    noise = np.percentile(P_smooth, NOISE_PERCENTILE, axis=1, keepdims=True) + 1e-12
    gain = np.clip(1.0 - OVER_SUBTRACTION * noise / (P + 1e-12), GAIN_FLOOR, 1.0)
    gain = F.avg_pool2d(torch.from_numpy(gain)[None, None].float(), 3, stride=1, padding=1,
                        count_include_pad=False)[0, 0].numpy()
    report["noise_floor_db"] = round(float(10 * np.log10(noise[_band(freqs, *LUNG_BAND)].mean())), 1)

    # 4. Heart sounds: low-band bursts
    low = P[_band(freqs, *HEART_BAND)].sum(axis=0)
    loud_low = low > HEART_FACTOR * np.median(low)
    # Heart sounds are short bursts (S1/S2 ~40-200 ms); longer runs are breathing, left alone
    heart = _dilate(_runs_at_least(loud_low, round(0.04 / FRAME_SEC)) & ~_runs_at_least(loud_low, round(0.2 / FRAME_SEC)), 3)
    gain[np.ix_(freqs < HEART_ATTENUATE_BELOW, heart)] *= HEART_GAIN
    report["heart_sound_fraction"] = round(float(heart.mean()), 3)

    # 5. Artifacts we cannot separate safely. Energies are measured ABOVE the steady
    #    noise floor, so constant microphone hiss is never mistaken for a voice.
    excess = np.maximum(P - OVER_SUBTRACTION * noise, 0.0)
    e_lung = excess[_band(freqs, *LUNG_BAND)].sum(axis=0) + 1e-12
    e_high = excess[_band(freqs, *HIGH_BAND)].sum(axis=0)
    floor_high = noise[_band(freqs, *HIGH_BAND)].sum()
    airborne = (e_high > AIRBORNE_ABSOLUTE * e_lung) & (e_high > floor_high)
    airborne = _dilate(_runs_at_least(airborne, round(AIRBORNE_MIN_SEC / FRAME_SEC)), 2)

    broad = P[_band(freqs, 100, 4000)] + 1e-12
    e_total = broad.sum(axis=0)
    flatness = np.exp(np.mean(np.log(broad), axis=0)) / np.mean(broad, axis=0)
    friction = (e_total > FRICTION_FACTOR * np.median(e_total)) & (flatness > FRICTION_FLATNESS)
    friction = _dilate(_runs_at_least(friction, round(FRICTION_MIN_SEC / FRAME_SEC)), 2)

    artifacts = airborne | friction
    gain[:, artifacts] = np.minimum(gain[:, artifacts], ARTIFACT_GAIN)
    report["airborne_noise_fraction"] = round(float(airborne.mean()), 3)
    report["friction_fraction"] = round(float(friction.mean()), 3)

    # 6. Rebuild the signal
    clean = torch.istft(X * torch.from_numpy(gain).float(), N_FFT, HOP, window=_window,
                        center=True, length=len(y)).numpy().astype(np.float32)
    mask = np.repeat(artifacts, HOP)[:len(y)]
    if len(mask) < len(y):
        mask = np.pad(mask, (0, len(y) - len(mask)), constant_values=bool(artifacts[-1]) if n_frames else False)
    return clean, mask, report
