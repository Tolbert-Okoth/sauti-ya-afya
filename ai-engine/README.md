# SautiYaAfya AI Engine

FastAPI service that classifies a lung-sound recording as **Asthma/wheeze**, **Normal** or **Pneumonia** pattern.

> ⚠️ **Screening aid only.** The model detects **sounds** (crackles, wheeze-type sounds, normal breathing) in a single
> recording. It was trained and tested on children (SPRSound, Shanghai, 1 month–18 years) recorded with a stethoscope,
> not yet on phone recordings from Kenya. A diagnosis needs the whole child: several chest spots, breathing rate,
> fever and danger signs (WHO/IMCI). Do not use it for clinical decisions until validated in the target setting.

**What the classes mean** (names kept for API compatibility; `biomarkers.sound_finding` says it plainly):

| API class | Sound detected |
|---|---|
| `Pneumonia` | crackles (fine or coarse) |
| `Asthma` | wheeze-type: wheeze, rhonchi, stridor |
| `Normal` | normal breath sounds |

## How a recording is analysed

1. `audio_pipeline.load_audio` decodes the upload with ffmpeg (local files only) to 16 kHz mono, first 30 s.
2. `denoise.py` cleans the whole recording (details below) and marks stretches covered by voices, crying or
   phone friction.
3. The signal is cut into 5 s windows. Silent, clipped or mostly-noise windows are skipped. If none is usable, the
   job fails with `retake: true` and a message telling the health worker what to fix and to record again (never a
   "Low risk" result).
4. Each window is band-passed (100–2000 Hz) and turned into an 8-bit mel spectrogram, scaled relative to its own peak so
   phone volume doesn't matter.
5. MobileNetV2 scores each window; the **mean of the window probabilities** is reported unchanged.
6. Decision (`analyzer.decide`), using the admin confidence threshold (clamped to 0.50–0.99):

   | Top class | ≥ threshold | < threshold |
   |---|---|---|
   | Pneumonia / Asthma | that class, **High** | "Suspected …", **Medium** |
   | Normal | Normal, **Low** | "Inconclusive", **Medium** |

7. Extras that never change the probabilities:
   - `acoustic_flags`: heuristic wheeze-like / crackle-like segment counts (unvalidated thresholds).
   - `symptoms_reported` / `danger_signs`: parsed from the symptom text (English + Kiswahili, negation-aware).
     Any WHO/IMCI danger sign sets the risk to **High**.
   - `noise_report`: what the denoiser found (hum frequencies, share of heart-sound, voice and friction frames).

## Per-child analysis (`POST /analyze_child`)

The app records **2–4 chest spots** per child and sends them together, with the breathing rate and signs:

| Form field | Meaning |
|---|---|
| `files` | 1–4 recordings (one per chest spot) |
| `age_months` | age in months (IMCI cut-offs depend on it) |
| `breaths_per_minute` | counted over a full minute (optional, but needed for the fast-breathing check) |
| `symptoms` | free text plus ticked IMCI signs (danger signs, chest indrawing, stridor, blue lips) |

Each spot is scored separately (`biomarkers.spots`). Crackle and wheeze scores are combined across spots with
rules chosen on validation children (`evaluate_children.py`, target ≥80% sensitivity), then `classify_child`
applies WHO IMCI logic (`imci.py`): general danger sign → refer urgently; under 2 months with fast breathing,
chest indrawing or crackles → refer urgently; fast breathing / chest indrawing → IMCI pneumonia signs (or, with
wheeze, a bronchodilator trial and reassessment); crackles alone → clinician review / count breathing; wheeze alone
→ IMCI wheeze pathway; otherwise Low. The response adds `explanation` (plain text for the health worker).

On the 129 held-out children (`benchmarks/child_level_v7model.json`), up to 4 spots each:

| Finding | Combine rule | Sensitivity | Specificity | Children with finding |
|---|---|---|---|---|
| Crackles | mean of the 2 highest spots, ≥ 0.49 | 0.79 | 0.76 | 29 |
| Wheeze-type | mean of spots, ≥ 0.28 | 0.44 | 0.94 | 16 |

Wheeze detection per child is weak (few children with wheeze in the data); breathing rate and IMCI signs still
drive the risk level. The single-recording `POST /analyze` endpoint is unchanged.

## Noise handling

Removing "all noise" blindly also removes crackles and wheezes, so each noise type is treated separately
(`denoise.py`, tested in `tests/test_denoise.py` to keep crackles and wheezes intact):

| Noise | Treatment |
|---|---|
| Mains hum (50/60 Hz and harmonics) | narrow notch filters, only when the hum persists through the clip |
| Steady background (hiss, fan, engine, rain) | gain against a noise floor estimated from the quietest moments, capped at −12 dB |
| Heart sounds | low band (<200 Hz) attenuated during short heart-beat bursts |
| Voices, crying, animals (airborne) | detected by energy above 2.5 kHz relative to the lung band, muted |
| Friction / phone movement | loud, broadband, sustained (>60 ms) bursts are muted; short crackles are not matched |

Denoising alone did **not** make the model robust: what did was training on recordings with noise mixed in
(`--noise-aug-dir`), so the model learns what leftover noise looks like. See `benchmarks/README.md`.

Training (`train_robust.py`) imports the same `audio_pipeline.py`, so training and serving inputs are identical. The
checkpoint stores `PIPELINE_VERSION`; the engine refuses to start with a mismatched or missing model.

## Running

```bash
pip install -r requirements.txt          # plus ffmpeg on the system (render-build.sh installs it)
gunicorn -c gunicorn_config.py main:app  # or: python main.py
```

| Env var | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | `*` | Comma-separated frontend origins for CORS. **Set this in production.** |
| `ENGINE_API_KEY` | unset | If set, `/analyze` and `/status` require header `X-API-Key`. A key shipped in browser JS is not secret: for real protection call the engine from the Node server. |
| `MAX_UPLOAD_MB` | `10` | Upload size limit. |
| `JOB_TTL_SECONDS` | `3600` | Finished jobs are forgotten after this. |
| `MAX_JOBS` | `500` | Jobs held in memory at once. |
| `MODEL_FILE` | `sauti_mobilenet_v7.pth` | Checkpoint to load. |

Keep **one** worker: the job store is in memory.

## Retraining

```bash
pip install -r requirements-train.txt
python setup_phase4.py                  # only if ../raw_data_phase4 is missing
# noise clips for augmentation: one sub-folder per group (we used ESC-50 folds 1-3: voices, animals, impulsive, steady)
python train_robust.py --epochs 12 --noise-aug-dir /path/to/noise_train --noise-copies 2
python evaluate_noise.py --noise-dir /path/to/noise_eval   # benchmark with DIFFERENT noise clips
python -m pytest -q tests
```

- Train / validation / test are split **by patient** (`patient_key`), so the three stethoscope-filter copies of a KAUH
  recording, or several recordings of one ICBHI patient, never straddle splits.
- Files known to be mislabelled are listed with reasons in `data_exclusions.txt`.
- The epoch with the best validation macro-recall is kept; test metrics are reported per class (sensitivity and
  precision) at recording level.

## Current model (`sauti_mobilenet_v7.pth`)

Pipeline v4, trained only on children: SPRSound recordings labelled per 5 s window from paediatricians' sound
annotations (`python train_robust.py --labels sound --sprsound-dir ... --noise-aug-dir ...`), with noise augmentation.
Held-out test: **1,219 recordings from 129 children** never seen in training (`benchmarks/children_v7model.json`).

| Sound | Sensitivity | Precision |
|---|---|---|
| Wheeze-type (`Asthma`) | 0.65 | 0.53 |
| Normal | 0.75 | 0.90 |
| Crackles (`Pneumonia`) | 0.63 | 0.39 |

Macro-recall 0.675 (the previous, adult-trained engine scored 0.37 on these children). Main weakness: false
crackle alarms (about 6 in 10 crackle calls are normal recordings). Not yet measured: phone recordings, noise
robustness on children, and per-child accuracy when several chest spots are combined.
