# AI Engine: audit, fixes and research log

This document records the work done on the SautiYaAfya AI engine (`ai-engine/`) and the CHW recording screen
(`client/client/src/components/SmartRecorder.js`) in October 2026: what was wrong, what was changed and why, every
experiment including the ones that failed, how to reproduce the results, and what is still open.

> **Status in one paragraph.** The engine now detects **sounds** (crackles, wheeze-type sounds, normal breathing) in
> children's lung recordings and classifies the **child** by combining 2-4 chest spots with breathing rate and WHO IMCI
> signs. On 129 held-out children it catches 79% of children with crackles while correctly clearing 76% of children
> without them. It is trained and tested on stethoscope recordings from Shanghai (SPRSound); it has **not yet been
> validated on phone recordings from Kenya**, which is the most important next step. It is a screening aid, not a
> diagnostic device.

## Contents

1. [Branches and how to deploy](#1-branches-and-how-to-deploy)
2. [Phase 1: audit of the original engine](#2-phase-1-audit-of-the-original-engine)
3. [Phase 2: fixes](#3-phase-2-fixes)
4. [Phase 3: noise](#4-phase-3-noise)
5. [Phase 4: children's data](#5-phase-4-childrens-data)
6. [Phase 5: per-child analysis and the app](#6-phase-5-per-child-analysis-and-the-app)
7. [All results in one place](#7-all-results-in-one-place)
8. [Data sources and licences](#8-data-sources-and-licences)
9. [Reproducing the work](#9-reproducing-the-work)
10. [Open issues and next steps](#10-open-issues-and-next-steps)
11. [References](#11-references)

---

## 1. Branches and how to deploy

| Branch | Contains | Status |
|---|---|---|
| `fix/ai-engine-audit` | Phase 2 fixes, adult model v3 | superseded |
| `feat/noise-reduction` | + Phase 3 noise handling, adult model v5 | superseded (fails on children, see Phase 4) |
| `feat/pediatric-data` | + Phases 4-5: children's model v7, `/analyze_child`, 4-spot app | **merge this one** |

Each branch builds on the previous one, so merging `feat/pediatric-data` into `main` brings in everything.

**After merging:**

1. Render (AI engine): start command `gunicorn -c gunicorn_config.py main:app`; set `ALLOWED_ORIGINS` to the frontend
   URL(s). Optional: `ENGINE_API_KEY`, `MAX_UPLOAD_MB`, `JOB_TTL_SECONDS`, `MAX_JOBS` (see `ai-engine/README.md`).
2. Vercel (client): no new settings. The app now calls `POST /analyze_child`, so deploy the engine first.
3. Keep **one** engine worker: jobs are held in memory.

---

## 2. Phase 1: audit of the original engine

Probes on the original engine (30 recordings per class from its own training data, so a best case) and code review.

### Critical

| # | Finding | Evidence |
|---|---|---|
| 1 | The spectrogram given to the network was **black and white**: `s_norm.byte()` ran before `* 255`, so every pixel was 0 or 255 | pixel values `[0 255]` only |
| 2 | Training and serving prepared audio differently (serving min-max scaled and zeroed small values) | `analyzer.py` vs `train_robust.py` |
| 3 | **No held-out evaluation**: a train/test split was created and ignored; reported accuracy was training accuracy. KAUH recordings exist 3 times (three stethoscope filters), so even a random split would leak | `train_robust.py` |
| 4 | Training data was **mostly adult** (ICBHI 2017, KAUH ages 12-90), with mislabelled files (pulmonary oedema, bronchiectasis and wheeze clips in the pneumonia folder) | file names |
| 5 | Shown "probabilities" were often **hard-coded** by rules (`[0.01, 0.98, 0.01]`, raised to at least 0.80, symptom bonuses without renormalising; one result summed to 1.096) | `analyzer.py` |

### High

| # | Finding |
|---|---|
| 6 | One pneumonia-like 5 s chunk overrode all normal chunks (pushes toward antibiotics) |
| 7 | Quiet wheezes were forced to "Normal" (rule order); 8/10 wheeze recordings at 30% volume came out Normal |
| 8 | Symptom keywords ignored negation ("no fever" raised pneumonia), missed "wheezing"/"coughing", no Kiswahili |
| 9 | Silence, very short clips or a missing model returned "success" with **Low** risk |
| 10 | Random test-time augmentation: the same file gave different results |
| 11 | ~15 hand-tuned thresholds with no validation |

### Not working as intended / security / operations

- The admin threshold was accepted but never used; symptoms were never sent to the engine; `llm_bridge.py` was never
  called and read fields the analyzer did not produce, with an unreachable severe-tachypnoea branch and wrong IMCI age bands.
- Public engine with CORS `*` and credentials; no upload size limit; client file name used in the path; ffmpeg could
  follow playlists (SSRF / local file read); unbounded in-memory job store; patient symptoms written to logs.
- `gunicorn_config.py` lacked the uvicorn worker class; unpinned requirements; stray files and three unused models.

---

## 3. Phase 2: fixes

Branch `fix/ai-engine-audit`.

| Area | Change |
|---|---|
| One pipeline | `audio_pipeline.py` is imported by training and serving: ffmpeg decode (local files only), 5 s windows, quality gate, 100-2000 Hz band-pass, 8-bit mel spectrogram relative to the window's peak (volume-independent). The checkpoint stores `PIPELINE_VERSION`; the engine refuses a mismatched or missing model |
| Honest training | `train_robust.py`: split **by patient** (KAUH filter copies grouped), mislabelled files excluded (`data_exclusions.txt`), best epoch chosen on validation, per-class sensitivity / precision on held-out test saved with the model |
| Honest output | Real averaged probabilities; rules no longer overwrite them; deterministic. Below the admin threshold: "Suspected X" / "Inconclusive" (Medium), never Low. Unusable recordings fail with a "record again" message (`retake: true`) |
| Symptoms | Negation-aware English + Kiswahili parser; symptoms never change probabilities; WHO/IMCI danger signs force High risk |
| API | Upload cap, server-chosen file names, extension allow-list, ffmpeg `-protocol_whitelist file,pipe`, expiring job store, one analysis at a time, optional `X-API-Key`, configurable CORS, no patient data in logs |
| Ops | Uvicorn worker in gunicorn, pinned requirements, stale files removed, tests (`ai-engine/tests/`) |

Result (adult data, 28 held-out patients): accuracy 0.85; sensitivity asthma 0.89, normal 0.67, pneumonia 0.92.

---

## 4. Phase 3: noise

Branch `feat/noise-reduction`. Goal: remove noise from breathing sounds without removing what matters.

**Design rule from the literature:** generic "remove all noise" enhancement also removes crackles and wheezes, which are
the findings. So `denoise.py` treats each noise type separately and caps how much it attenuates:

| Noise | Treatment |
|---|---|
| Mains hum 50/60 Hz + harmonics | narrow notches, only when the hum persists through the clip and is a harmonic family |
| Steady background (hiss, fan, engine, rain) | Wiener-style gain against a noise floor from the quietest moments (time-smoothed percentile, which removes a ~10x estimation bias), capped at -12 dB |
| Heart sounds | < 200 Hz attenuated during short (40-200 ms) low-frequency bursts |
| Voices, crying, animals | detected by energy above 2.5 kHz relative to the lung band, measured above the noise floor; muted |
| Friction / phone movement | loud, broadband, sustained (> 60 ms) bursts muted; crackles (< 20 ms) are not matched |

`tests/test_denoise.py` checks crackles keep their energy (-0.8 dB), wheezes are untouched, hiss drops, hum drops
~25-35 dB, friction is flagged only where it happens. Two detector bugs found by these tests were fixed (voice
detector firing on every inhale; heart detector treating inhales as heartbeats).

**Benchmark** (`ai-engine/evaluate_noise.py`): held-out patients mixed with **unseen** noise (ESC-50 folds 4-5 + spoken
digits; training augmentation uses folds 1-3 and no speech) at 10 dB and 0 dB SNR, plus synthetic hiss and hum.

| Model | Clean | Mean of 12 noisy conditions | Pneumonia sensitivity under noise |
|---|---|---|---|
| v3, no denoising | 0.83 | 0.47 | 0.38 |
| v4, denoising only | 0.78 | 0.47 | **0.28 (worse)** |
| v5, denoising + **noise-augmented training** | **0.90** | **0.69** | **0.76** |

**Finding:** denoising alone did not help; the model had never seen leftover noise or muted gaps. Training on
recordings with noise mixed in (`--noise-aug-dir`) is what made it robust. Both v3 and v4 called every recording
Normal under microphone hiss; v5 does not.

---

## 5. Phase 4: children's data

Branch `feat/pediatric-data`.

### The adult model fails on children

[SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) (CC-BY-4.0): Shanghai Children's Medical Center,
ages 1 month-18 years, 6,567 recordings, 20.8 h, labelled by 11 paediatricians, with patient diagnoses in its
patient summaries. The current adult-trained model v5 on 574 held-out children's recordings: **macro-recall 0.37**
(pneumonia 0.22, asthma 0.02, normal 0.88) - it calls most sick children healthy.

### Training on diagnoses fails too (experiment v6, not kept)

Labelling every recording of a child with the child's diagnosis also stayed at chance (validation macro-recall 0.37).
SPRSound's own per-recording annotations show why:

| Child's diagnosis | Recordings that sound normal | with crackles | with wheeze-type |
|---|---|---|---|
| Pneumonia | 70% | 18% | 11% |
| Asthma | 80% | 3% | 15% |
| Healthy | 87% | 1% | 6% |

A child is recorded at several chest spots and most spots sound normal, so a per-recording diagnosis label is mostly
noise. **The model must learn what is audible, and the diagnosis must be made per child.**

### Sound-pattern model (v7, current)

`pediatric_data.py` + `train_robust.py --labels sound`: each 5 s window labelled from the paediatricians' event
annotations (crackles → `Pneumonia` class, wheeze/rhonchi/stridor → `Asthma` class, normal breath → `Normal`); 860
children, split by child (403 train, 129 validation, 129 test after capping normal recordings); noise augmentation;
pipeline v4 caps the mel spectrogram at 4 kHz so 8 kHz stethoscope audio and phone audio look the same.

| On 129 held-out children (1,219 recordings) | Adult engine v5 | **v7** |
|---|---|---|
| Macro-recall | 0.37 | **0.675** |
| Wheeze-type detected | 0.02 | 0.65 |
| Crackles detected | 0.22 | 0.63 |
| Normal correctly normal | 0.88 | 0.75 |

The API class names are kept for compatibility; `biomarkers.sound_finding` says plainly what was heard.

Also fixed here: the silence cut-off (RMS 0.005) rejected 11% of good children's recordings (quiet stethoscope
audio, RMS ~0.002-0.003); lowered to 0.001, now 0% rejected.

---

## 6. Phase 5: per-child analysis and the app

### Engine: `POST /analyze_child`

1-4 chest-spot recordings + `age_months` + `breaths_per_minute` + `symptoms`. Each spot is scored; crackle and wheeze
scores are combined with rules chosen on validation children (`evaluate_children.py`, target ≥ 80% sensitivity); then
`analyzer.classify_child` applies WHO IMCI (`imci.py`):

| Situation | Result | Risk |
|---|---|---|
| General danger sign (unable to drink, vomits everything, convulsions, lethargic, stridor, cyanosis) | Severe illness - refer urgently | High |
| Under 2 months with fast breathing, chest indrawing or crackles | Possible serious infection - refer urgently | High |
| Fast breathing for age (≥60 / ≥50 / ≥40 per minute for <2 / 2-11 / 12-59 months) or chest indrawing | IMCI pneumonia signs; with wheeze only: bronchodilator trial, reassess | High |
| Crackles without fast breathing | Crackles heard - count breathing / clinician review | Medium |
| Wheeze without fast breathing | IMCI wheeze pathway | Medium |
| Nothing heard, no signs | No crackles or wheeze heard | Low |

| Per child, 129 held-out children | Combine rule | Sensitivity | Specificity |
|---|---|---|---|
| Crackles (29 children) | mean of the 2 highest spots ≥ 0.49 | **0.79** | **0.76** |
| Wheeze-type (16 children) | mean of spots ≥ 0.28 | 0.44 | 0.94 |

Combining spots raised crackle specificity from 0.61 (single recording) to 0.76 at the same sensitivity. Wheeze per
child is weak (few children with wheeze in the data).

### App: `SmartRecorder.js`

- 4 chest spots (left/right upper/lower back), 15 s each with auto-stop and countdown, minimum 5 s; at least 2 recommended.
- Breaths-per-minute field and IMCI sign toggles (sent as phrases the engine's parser recognises).
- **Microphone opened with echo cancellation, noise suppression and auto gain OFF**: browser voice processing removes
  breath sounds and crackles.
- Result shows the engine's plain-language explanation and what was heard; score bars labelled as crackles / wheeze.
- The first recorded spot is archived to the Node server (it stores one file per patient).
- The admin confidence slider no longer affects the per-child result (validated thresholds are used).

### Children under noise (v7)

400 test recordings with unseen noise: clean 0.70, **mean under noise 0.56** (worst: loud voices 0.47). Degrades,
never collapses. No recording was rejected even with loud voices: the voice detector looks above 2.5 kHz and the
children's stethoscope audio only reaches 4 kHz.

---

## 7. All results in one place

Raw numbers live in `ai-engine/benchmarks/` (JSON) and `ai-engine/model_metrics.json`.

| Model | Trained on | Tested on | Headline |
|---|---|---|---|
| original | adult, no held-out test | its own training data | binary spectrograms, rule-overwritten outputs |
| v3 | adult, patient split | 28 adult patients | macro-recall 0.83 |
| v4 | + denoising | 28 adult patients + noise | clean 0.78, noisy 0.47 |
| v5 | + noise augmentation | 28 adult patients + noise | clean 0.90, noisy 0.69; **children 0.37** |
| v6 | + children, diagnosis labels | children | chance (0.37), not kept |
| **v7** | children, sound labels, noise aug. | 129 children | **0.675 per recording; crackles per child 0.79 / 0.76** |

All test sets are patients never used for training or threshold selection.

---

## 8. Data sources and licences

| Data | Used for | Licence / terms | In the repo? |
|---|---|---|---|
| ICBHI 2017 (via Kaggle), KAUH, assorted clips (`raw_data_phase4/`) | adult models v3-v5 | per source | yes (430 MB) |
| [SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) | children's model v7 | CC-BY-4.0, cite the TBioCAS 2022 paper | no (4.4 GB, clone it) |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | noise augmentation (folds 1-3) and benchmark (folds 4-5) | CC BY-NC 3.0: **non-commercial**; replace before commercial retraining | no |
| [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) | voice noise in the benchmark only | CC BY-SA 4.0 | no |

The ICBHI age table needs a login, so ICBHI patients are tagged age "unknown".

---

## 9. Reproducing the work

```bash
cd ai-engine
pip install -r requirements-train.txt          # plus ffmpeg on the system

# Data
git clone --depth 1 https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound /data/SPRSound
bash tools/build_noise_sets.sh /data/noise     # ESC-50 folds 1-3 -> train, 4-5 + speech -> eval

# Children's sound-pattern model (current engine, ~3 h on 4 CPU cores)
python train_robust.py --labels sound --epochs 8 --sprsound-dir /data/SPRSound \
    --max-train-per-class 1300 --noise-aug-dir /data/noise/noise_train --noise-copies 1
# writes sauti_mobilenet_v7.pth + model_metrics.json (rename MODEL_PATH in train_robust.py for a new version)

# Evaluations
python evaluate_children.py --sprsound-dir /data/SPRSound                  # per-child thresholds + results
python evaluate_noise.py --model sauti_mobilenet_v7.pth --labels sound --sprsound-dir /data/SPRSound \
    --max-recordings 400 --noise-dir /data/noise/noise_eval --snr 10 0     # children under noise
python -m pytest -q tests                                                  # 45 tests
```

Splits are deterministic (seed 42). If `evaluate_children.py` gives new thresholds, copy them into
`CRACKLE_COMBINE/CRACKLE_THRESHOLD/WHEEZE_COMBINE/WHEEZE_THRESHOLD` in `analyzer.py`.

---

## 10. Open issues and next steps

In priority order:

1. **Validate on phone recordings from Kenya.** All children's numbers come from stethoscope recordings in Shanghai.
   Collect a few hundred phone recordings of children (4 spots each) with a clinician's sound labels and diagnosis,
   and measure. This also shows whether the voice/crying detector works on phone audio.
2. **More wheeze data.** Per-child wheeze sensitivity is 0.44 (only ~70 asthma children in SPRSound).
3. **Too many false crackle alarms on single recordings** (precision 0.39). Per-child combining helps (specificity 0.76);
   more data and a stronger model are next.
4. **Replace ESC-50** with commercially licensed or self-recorded clinic noise before any commercial use.
5. **Server-side access control**: the engine is called directly from the browser; route it through the Node server
   with Firebase auth for real protection.
6. Store all chest-spot recordings (the Node server keeps only the first); adapt the doctor dashboard to show spots.
7. `llm_bridge.py` is fixed but still not wired into the app.
8. Kiswahili symptom phrases should be reviewed by a native speaker; the admin threshold slider should be removed or
   repurposed.

---

## 11. References

- Zhang Q. et al., *SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database*, IEEE TBioCAS 2022 -
  [GitHub](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound), [IEEE DataPort](https://ieee-dataport.org/node/12530)
- *Improving the Robustness and Clinical Applicability of Automatic Respiratory Sound Classification Using Deep
  Learning-Based Audio Enhancement* - [arXiv 2407.13895](https://arxiv.org/abs/2407.13895)
- *A machine learning approach to the development and prospective evaluation of a pediatric lung sound classification
  model* - [Scientific Reports](https://www.nature.com/articles/s41598-023-27399-5)
- *Artificial Intelligence Models for Pediatric Lung Sound Analysis: Systematic Review and Meta-Analysis* -
  [JMIR](https://www.sciencedirect.com/org/science/article/pii/S143888712500559X)
- PERCH digital auscultation study - [PMC5531306](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5531306/)
- Johns Hopkins smart stethoscope (noise removal without losing diagnostic sounds) -
  [news](https://www.hopkinsmedicine.org/news/articles/2016/11/smart-stethoscope-zeros-in-on-lung-sounds),
  [US9848848B2](https://patents.google.com/patent/US9848848B2/en)
- *StethAid: A Digital Auscultation Platform for Pediatrics* - [Sensors](https://doi.org/10.3390/s23125750)
- WHO, *Integrated Management of Childhood Illness (IMCI) chart booklet* (fast-breathing cut-offs, danger signs)
- ESC-50 - [GitHub](https://github.com/karolpiczak/ESC-50); Free Spoken Digit Dataset -
  [GitHub](https://github.com/Jakobovski/free-spoken-digit-dataset)
