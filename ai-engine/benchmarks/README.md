# Noise-robustness benchmarks

Produced by `evaluate_noise.py` on the 110 held-out TEST recordings (28 patients). Noise clips: ESC-50 folds 4-5
(voices = crying baby, laughing + Free Spoken Digit Dataset speech; animals = rooster, dog; impulsive = door knock,
footsteps, clock tick; steady = engine, rain, wind) plus synthetic white hiss and 50 Hz mains hum. SNR is noise
relative to the whole recording: 10 dB = noise clearly audible, 0 dB = as loud as the breathing.
None of these clips are used in training (training augmentation uses ESC-50 folds 1-3 and no speech).

| File | Model | Pipeline |
|---|---|---|
| `noise_v3model_no_denoise.json` | `sauti_mobilenet_v3.pth` (audit fix) | v2, no denoising |
| `noise_v4model_denoise.json` | `sauti_mobilenet_v4.pth` (not kept) | v3, `denoise.py` |
| `noise_v5model_denoise_noisetrain.json` | `sauti_mobilenet_v5.pth` (**current**) | v3 + noise-augmented training |

## Macro-recall (mean of per-class sensitivity)

| Condition | v3 model, no denoise | v4 model, denoise | v5 model, denoise + noise training |
|---|---|---|---|
| clean | 0.83 | 0.78 | 0.90 |
| animals 10 / 0 dB | 0.54 / 0.46 | 0.48 / 0.40 | 0.78 / 0.75 |
| impulsive 10 / 0 dB | 0.57 / 0.52 | 0.56 / 0.47 | 0.73 / 0.68 |
| steady 10 / 0 dB | 0.33 / 0.32 | 0.42 / 0.35 | 0.63 / 0.48 |
| voices 10 / 0 dB | 0.37 / 0.34 | 0.36 / 0.37 | 0.58 / 0.51 |
| white hiss 10 / 0 dB | 0.35 / 0.34 | 0.35 / 0.33 | 0.79 / 0.62 |
| mains hum 10 / 0 dB | 0.74 / 0.74 | 0.81 / 0.78 | 0.86 / 0.86 |
| **mean of noisy conditions** | **0.469** | **0.473** | **0.689** |
| mean noisy Pneumonia sensitivity | 0.38 | 0.28 | 0.76 |

**Finding:** denoising alone does not make the model robust. It helps with hum and steady background noise but
hurts pneumonia detection under animal and voice noise, and slightly lowers clean performance. Both models fail
completely on microphone hiss (everything is called Normal). A model that never saw noise in training cannot
interpret leftover noise or muted gaps.

**Noise-augmented training fixes most of it (v5):** best in every condition including clean audio; pneumonia
sensitivity under noise doubles (0.38 -> 0.76) and hiss no longer turns every patient "Normal". Remaining weak
spots: loud steady noise and voices, where healthy recordings are often called abnormal (Normal sensitivity
0.25-0.42).
