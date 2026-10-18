"""
Shared audio -> spectrogram pipeline.

Training (train_robust.py) and serving (analyzer.py) MUST both use these
functions so the model sees identical inputs in both places. If you change
anything here, bump PIPELINE_VERSION and retrain: analyzer.py refuses to load
a checkpoint built with a different version.
"""
import subprocess

import numpy as np
import torch
import torchaudio.functional as F_audio
import torchaudio.transforms as T
from PIL import Image
from torchvision import transforms

from denoise import denoise

PIPELINE_VERSION = 4   # v3: denoise.py before windowing; v4: mel spectrogram capped at 4 kHz

SAMPLE_RATE = 16000
WINDOW_SECONDS = 5
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
MIN_WINDOW_SAMPLES = SAMPLE_RATE * 2   # a trailing piece shorter than 2 s is dropped
MAX_SECONDS = 30                       # only the first 30 s of a recording is analysed

# Quality gates (on the raw, un-normalised signal)
SILENCE_RMS = 0.005      # below this a window is treated as silence
CLIP_LEVEL = 0.99        # |sample| at or above this counts as clipped
MAX_CLIPPED_FRACTION = 0.05
MAX_ARTIFACT_FRACTION = 0.4   # voices/crying/friction share above which a window is dropped

TOP_DB = 80.0
FFMPEG_TIMEOUT_SECONDS = 60

# f_max 4 kHz: stethoscope datasets (e.g. SPRSound, 8 kHz) have nothing above it, phones do;
# capping keeps both looking the same to the model. Lung sounds of interest are < 2 kHz.
_mel = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512, power=2.0, f_max=4000.0)
_to_db = T.AmplitudeToDB(stype='power', top_db=TOP_DB)

image_to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class AudioDecodeError(Exception):
    pass


def load_audio(file_path, max_seconds=MAX_SECONDS):
    """Decode any audio file to mono float32 at 16 kHz using ffmpeg.

    Only local files are allowed (protocol whitelist), so an uploaded playlist
    or similar container cannot make ffmpeg fetch URLs or read other files.
    """
    command = [
        'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-protocol_whitelist', 'file,pipe',
        '-i', file_path,
        '-t', str(max_seconds),
        '-vn', '-f', 's16le', '-acodec', 'pcm_s16le',
        '-ar', str(SAMPLE_RATE), '-ac', '1',
        '-threads', '1', '-',
    ]
    try:
        proc = subprocess.run(command, capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise AudioDecodeError("Audio decoding timed out")
    if proc.returncode != 0:
        raise AudioDecodeError(f"Could not decode audio: {proc.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def split_windows(y):
    """Split a signal into 5 s windows; a trailing piece >= 2 s is zero-padded."""
    windows = []
    for start in range(0, len(y), WINDOW_SAMPLES):
        w = y[start:start + WINDOW_SAMPLES]
        if len(w) < MIN_WINDOW_SAMPLES:
            continue
        if len(w) < WINDOW_SAMPLES:
            w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))
        windows.append(w)
    return windows


def window_quality(w, artifact_fraction=0.0):
    """Returns (ok, reason). Checked on the raw signal before any normalisation."""
    rms = float(np.sqrt(np.mean(w ** 2)))
    if rms < SILENCE_RMS:
        return False, "silent"
    if float(np.mean(np.abs(w) >= CLIP_LEVEL)) > MAX_CLIPPED_FRACTION:
        return False, "clipped"
    if artifact_fraction > MAX_ARTIFACT_FRACTION:
        return False, "noisy"
    return True, ""


def analysis_windows(y, return_starts=False):
    """Denoise the whole recording, then window it.

    Returns (usable_clean_windows, rejected_reason_counts, total_windows, noise_report), plus
    the start time in seconds of each usable window when return_starts is True.
    Quality is judged on the raw signal; the model sees the denoised one.
    """
    clean, artifact_mask, report = denoise(y)
    raw_windows = split_windows(y)
    clean_windows = split_windows(clean)
    mask_windows = split_windows(artifact_mask.astype(np.float32))
    usable, starts, rejected = [], [], {}
    for i, (raw_w, clean_w, mask_w) in enumerate(zip(raw_windows, clean_windows, mask_windows)):
        ok, reason = window_quality(raw_w, float(mask_w.mean()))
        if ok:
            usable.append(clean_w)
            starts.append(i * WINDOW_SECONDS)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    if return_starts:
        return usable, rejected, len(raw_windows), report, starts
    return usable, rejected, len(raw_windows), report


def bandpass(waveform):
    """100-2000 Hz band: keeps wheezes and crackles, drops handling rumble and hiss."""
    filtered = F_audio.highpass_biquad(waveform, SAMPLE_RATE, cutoff_freq=100.0)
    return F_audio.lowpass_biquad(filtered, SAMPLE_RATE, cutoff_freq=2000.0)


def spectrogram_image(w):
    """5 s window -> 8-bit greyscale mel spectrogram (as an RGB PIL image).

    dB values are taken relative to the window's own peak, so the image does not
    depend on recording volume (phone distance, mic gain).
    """
    waveform = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).unsqueeze(0)
    waveform = bandpass(waveform)
    db = _to_db(_mel(waveform)).squeeze(0)
    db = db - db.max()
    s_norm = torch.clamp((db + TOP_DB) / TOP_DB, 0.0, 1.0)
    pixels = (s_norm * 255.0).round().to(torch.uint8).numpy()
    pixels = np.flipud(pixels)  # low frequencies at the bottom
    return Image.fromarray(np.ascontiguousarray(pixels)).convert('RGB')


def window_tensor(w):
    return image_to_tensor(spectrogram_image(w))
