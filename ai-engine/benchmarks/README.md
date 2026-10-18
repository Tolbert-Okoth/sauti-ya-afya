# Noise-robustness benchmarks

Produced by `evaluate_noise.py` on the 110 held-out TEST recordings (28 patients). Noise clips: ESC-50 folds 4-5
(voices = crying baby, laughing + Free Spoken Digit Dataset speech; animals = rooster, dog; impulsive = door knock,
footsteps, clock tick; steady = engine, rain, wind) plus synthetic white hiss and 50 Hz mains hum. SNR is noise
relative to the whole recording: 10 dB = noise clearly audible, 0 dB = as loud as the breathing.
None of these clips are used in training (training augmentation uses ESC-50 folds 1-3 and no speech).

| File | Model | Pipeline |
|---|---|---|
| `noise_v3model_no_denoise.json` | `sauti_mobilenet_v3.pth` (audit fix) | v2, no denoising |
| `noise_v4model_denoise.json` | `sauti_mobilenet_v4.pth` | v3, `denoise.py` |

## Macro-recall (mean of per-class sensitivity)

| Condition | v3 model, no denoise | v4 model, denoise |
|---|---|---|
| clean | 0.83 | 0.78 |
| animals 10 / 0 dB | 0.54 / 0.46 | 0.48 / 0.40 |
| impulsive 10 / 0 dB | 0.57 / 0.52 | 0.56 / 0.47 |
| steady 10 / 0 dB | 0.33 / 0.32 | 0.42 / 0.35 |
| voices 10 / 0 dB | 0.37 / 0.34 | 0.36 / 0.37 |
| white hiss 10 / 0 dB | 0.35 / 0.34 | 0.35 / 0.33 |
| mains hum 10 / 0 dB | 0.74 / 0.74 | 0.81 / 0.78 |
| **mean of noisy conditions** | **0.469** | **0.473** |
| mean noisy Pneumonia sensitivity | 0.38 | 0.28 |

**Finding:** denoising alone does not make the model robust. It helps with hum and steady background noise but
hurts pneumonia detection under animal and voice noise, and slightly lowers clean performance. Both models fail
completely on microphone hiss (everything is called Normal). A model that never saw noise in training cannot
interpret leftover noise or muted gaps; the next experiment trains with noise augmentation.
