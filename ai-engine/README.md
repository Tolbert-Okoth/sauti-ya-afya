# SautiYaAfya AI Engine

FastAPI service that classifies a lung-sound recording as **Asthma/wheeze**, **Normal** or **Pneumonia** pattern.

> ⚠️ **Screening aid only.** The model is trained mostly on **adult** recordings (ICBHI 2017, KAUH and a few public clips).
> Its held-out results (`model_metrics.json`) say nothing about performance in children. Do not use it for clinical decisions
> until it has been validated on pediatric data from the target setting.

## How a recording is analysed

1. `audio_pipeline.load_audio` decodes the upload with ffmpeg (local files only) to 16 kHz mono, first 30 s.
2. The signal is cut into 5 s windows. Silent or clipped windows are skipped. If none is usable, the job fails with
   `retake: true` and a message telling the health worker to record again (never a "Low risk" result).
3. Each window is band-passed (100–2000 Hz) and turned into an 8-bit mel spectrogram, scaled relative to its own peak so
   phone volume doesn't matter.
4. MobileNetV2 scores each window; the **mean of the window probabilities** is reported unchanged.
5. Decision (`analyzer.decide`), using the admin confidence threshold (clamped to 0.50–0.99):

   | Top class | ≥ threshold | < threshold |
   |---|---|---|
   | Pneumonia / Asthma | that class, **High** | "Suspected …", **Medium** |
   | Normal | Normal, **Low** | "Inconclusive", **Medium** |

6. Extras that never change the probabilities:
   - `acoustic_flags`: heuristic wheeze-like / crackle-like segment counts (unvalidated thresholds).
   - `symptoms_reported` / `danger_signs`: parsed from the symptom text (English + Kiswahili, negation-aware).
     Any WHO/IMCI danger sign sets the risk to **High**.

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
| `MODEL_FILE` | `sauti_mobilenet_v3.pth` | Checkpoint to load. |

Keep **one** worker: the job store is in memory.

## Retraining

```bash
pip install -r requirements-train.txt
python setup_phase4.py                  # only if ../raw_data_phase4 is missing
python train_robust.py --epochs 12      # writes sauti_mobilenet_v3.pth + model_metrics.json
python -m pytest -q tests
```

- Train / validation / test are split **by patient** (`patient_key`), so the three stethoscope-filter copies of a KAUH
  recording, or several recordings of one ICBHI patient, never straddle splits.
- Files known to be mislabelled are listed with reasons in `data_exclusions.txt`.
- The epoch with the best validation macro-recall is kept; test metrics are reported per class (sensitivity and
  precision) at recording level.

## Current model (`sauti_mobilenet_v3.pth`)

Held-out test set: 110 recordings from 28 patients never seen in training (`model_metrics.json`).

| | Sensitivity (recall) | Precision |
|---|---|---|
| Asthma / wheeze | 0.89 | 0.94 |
| Normal | 0.67 | 0.80 |
| Pneumonia | 0.92 | 0.82 |

Accuracy 0.85, macro-recall 0.83. With only 28 test patients these numbers carry wide error bars (a few patients
can move them by 5–10 points), and they are adult data. Known weakness: very quiet recordings (≈10% of normal
level) degrade because the recording's own noise floor dominates the spectrogram.
