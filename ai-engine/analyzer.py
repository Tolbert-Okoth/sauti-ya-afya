import os

# 🚀 FORCE SINGLE THREADING (Render free tier)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import re
from collections import Counter

import numpy as np
import torch
from torchvision import models

import audio_pipeline as ap
import imci

torch.set_num_threads(1)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.environ.get("MODEL_FILE", "sauti_mobilenet_v7.pth"))
CLASSES = ['Asthma', 'Normal', 'Pneumonia']
DEFAULT_THRESHOLD = 0.75
MIN_THRESHOLD, MAX_THRESHOLD = 0.5, 0.99
DISCLAIMER = "AI Analysis Only. Consult a Doctor."
# The model detects audible sound patterns; the class names are kept for API compatibility.
SOUND_FINDING = {"Asthma": "wheeze", "Normal": "normal", "Pneumonia": "crackles"}


class RecordingUnusable(Exception):
    """The recording cannot be analysed; the health worker should record again."""


# ---------------------------------------------------------------------------
# 🧠 MODEL (fails loudly: a missing or mismatched model must never look like "Low risk")
# ---------------------------------------------------------------------------
def _load_model(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("pipeline_version") != ap.PIPELINE_VERSION:
        raise RuntimeError(
            f"Model was trained with pipeline v{checkpoint.get('pipeline_version')}, "
            f"engine uses v{ap.PIPELINE_VERSION}. Retrain with train_robust.py.")
    if checkpoint.get("classes") != CLASSES:
        raise RuntimeError(f"Model classes {checkpoint.get('classes')} != {CLASSES}")
    net = models.mobilenet_v2(weights=None)
    net.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(net.last_channel, len(CLASSES)))
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net, checkpoint.get("metrics", {})


print("🔄 Loading Medical Brain...")
model, MODEL_METRICS = _load_model(MODEL_PATH)
print(f"✅ Model loaded: {os.path.basename(MODEL_PATH)} (pipeline v{ap.PIPELINE_VERSION})")


# ---------------------------------------------------------------------------
# 🔬 DSP HEURISTICS — informational flags only, they never change probabilities.
# Thresholds are unvalidated and are kept for clinician context.
# ---------------------------------------------------------------------------
def count_transients_tkeo(y, block_size=320):
    """Teager-Kaiser energy bursts (crackle-like explosive events)."""
    if np.max(np.abs(y)) < 1e-6:
        return 0
    tkeo = np.abs(y[1:-1] ** 2 - y[:-2] * y[2:])
    thresh = np.mean(tkeo) * 8.0
    n_blocks = len(tkeo) // block_size
    blocks = tkeo[:n_blocks * block_size].reshape(n_blocks, block_size).max(axis=1)
    return int(np.sum(blocks > thresh))


def dsp_features(w):
    """Computed on a peak-normalised copy so results don't depend on volume."""
    y = w / (np.max(np.abs(w)) + 1e-9)
    spectrum = np.abs(np.fft.rfft(y)) + 1e-10
    freqs = np.fft.rfftfreq(len(y), 1 / ap.SAMPLE_RATE)
    centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / np.sum(spectrum)))
    flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
    zcr = float(np.count_nonzero(np.diff(y > 0)) / len(y))
    return {"zcr": zcr, "centroid": centroid, "bandwidth": bandwidth,
            "flatness": flatness, "transients": count_transients_tkeo(y)}


def dsp_flags(f):
    flags = []
    if 600 < f["centroid"] < 2500 and f["bandwidth"] < 1000 and f["flatness"] < 0.25 and f["transients"] < 2:
        flags.append("wheeze_like")
    if f["transients"] >= 5 and f["flatness"] < 0.42 and f["zcr"] <= 0.18:
        flags.append("crackle_like")
    return flags


# ---------------------------------------------------------------------------
# 🏥 SYMPTOMS — English + Kiswahili, negation-aware. Symptoms never change the
# acoustic probabilities; WHO/IMCI danger signs escalate the risk level.
# ---------------------------------------------------------------------------
SYMPTOM_PATTERNS = {
    "fever": r"fever|feverish|febrile|hot body|homa",
    "cough": r"cough\w*|kikohozi|kukohoa|anakohoa",
    "difficulty_breathing": r"short(ness)? of breath|breathless\w*|(difficult\w*|trouble|hard|fast|rapid|labou?red) (in )?breathing|"
                            r"kupumua kwa shida|shida ya kupumua|kupumua haraka|anapumua (kwa )?shida",
    "wheeze": r"wheez\w*|whistl\w*|mluzi|miluzi",
    "chest_pain": r"chest pain|pain in (the )?chest|maumivu ya kifua",
    "chest_tightness": r"tight\w*( chest)?|kifua kubana|kubana kifua",
    "crackles": r"crackl\w*",
}
DANGER_SIGN_PATTERNS = {
    "unable_to_drink_or_breastfeed": r"(unable|can'?t|cannot|not able) to (drink|feed|breastfeed)|hawezi kunywa|hawezi kunyonya",
    "vomits_everything": r"vomit\w* everything|anatapika kila kitu",
    "convulsions": r"convuls\w*|seizure\w*|fits?\b|degedege",
    "lethargic_or_unconscious": r"letharg\w*|unconscious|not (waking|responsive)|amelegea|hana fahamu",
    "chest_indrawing": r"(chest|lower chest|subcostal) (wall )?in-?drawing|indrawing|retraction\w*",
    "stridor": r"stridor",
    "cyanosis": r"cyanos\w*|blue (lips|skin|tongue)|midomo ya bluu",
}
NEGATION = re.compile(r"\b(no|not|without|denies|denied|absent|negative for|hakuna|hana|bila|si)\b")


def parse_symptoms(text):
    """Returns (present_symptoms, danger_signs). Clauses with a negation word are ignored."""
    present, danger = [], []
    if not text:
        return present, danger
    for clause in re.split(r"[,;.\n]|\bbut\b|\blakini\b", text.lower()):
        if NEGATION.search(clause) and not re.search(r"\b(not able|can'?t|cannot|hawezi|hana fahamu)\b", clause):
            continue
        for name, pattern in SYMPTOM_PATTERNS.items():
            if name not in present and re.search(r"\b(" + pattern + r")", clause):
                present.append(name)
        for name, pattern in DANGER_SIGN_PATTERNS.items():
            if name not in danger and re.search(r"\b(" + pattern + r")", clause):
                danger.append(name)
    return present, danger


# ---------------------------------------------------------------------------
# 🩺 ANALYSIS
# ---------------------------------------------------------------------------
def clamp_threshold(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    if not np.isfinite(value):
        return DEFAULT_THRESHOLD
    return min(MAX_THRESHOLD, max(MIN_THRESHOLD, value))


def decide(probs, threshold):
    """Diagnosis + risk from the model's averaged probabilities.

    Not confident enough (top probability below the admin threshold) never
    returns Low risk: it becomes "Suspected X" or "Inconclusive" (Medium).
    """
    top = max(probs, key=probs.get)
    confident = probs[top] >= threshold
    if top == "Normal":
        return ("Normal", "Low") if confident else ("Inconclusive", "Medium")
    return (top, "High") if confident else (f"Suspected {top}", "Medium")


@torch.no_grad()
def score_recording(file_path):
    """Decode, denoise and score one recording.

    Returns (probs {class: p}, usable windows, rejected reason Counter, total windows, noise report).
    Raises RecordingUnusable with advice for the health worker when nothing usable is left.
    """
    y = ap.load_audio(file_path)
    usable, rejected, n_windows, noise_report = ap.analysis_windows(y)
    rejected = Counter(rejected)
    if n_windows == 0:
        raise RecordingUnusable("Recording is too short. Please record at least 5 seconds.")
    if not usable:
        worst = rejected.most_common(1)[0][0]
        message = {
            "silent": "too quiet - hold the phone firmly against the chest",
            "clipped": "distorted (too loud) - move the phone slightly away",
            "noisy": "covered by background noise (voices, crying or phone movement) - "
                     "find a quieter spot and keep the phone still",
        }[worst]
        raise RecordingUnusable(f"Recording is {message} and record again.")

    batch = torch.stack([ap.window_tensor(w) for w in usable])
    mean_probs = torch.softmax(model(batch), dim=1).mean(dim=0)
    probs = {c: float(p) for c, p in zip(CLASSES, mean_probs)}
    return probs, usable, rejected, n_windows, noise_report


def analyze_audio(file_path, symptoms="", sensitivity_threshold=DEFAULT_THRESHOLD):
    threshold = clamp_threshold(sensitivity_threshold)
    probs, usable, rejected, n_windows, noise_report = score_recording(file_path)

    diagnosis, risk = decide(probs, threshold)

    acoustic_flags = Counter()
    for w in usable:
        acoustic_flags.update(dsp_flags(dsp_features(w)))

    present, danger = parse_symptoms(symptoms)
    notes = []
    if danger:
        risk = "High"
        notes.append("IMCI danger sign reported: refer urgently regardless of the sound analysis.")
    if rejected:
        notes.append(f"{sum(rejected.values())} of {n_windows} segments were unusable ({dict(rejected)}).")

    return {
        "status": "success",
        "biomarkers": {
            "ai_diagnosis": diagnosis,
            # What was actually heard. "Pneumonia" means crackles were detected, "Asthma" wheeze-type
            # sounds; a diagnosis needs the whole child (several chest spots, breathing rate, fever).
            "sound_finding": SOUND_FINDING[max(probs, key=probs.get)],
            "prob_pneumonia": round(probs["Pneumonia"], 3),
            "prob_asthma": round(probs["Asthma"], 3),
            "prob_normal": round(probs["Normal"], 3),
            "threshold_used": threshold,
            "segments_analyzed": len(usable),
            "segments_rejected": sum(rejected.values()),
            "acoustic_flags": dict(acoustic_flags),
            "symptoms_reported": present,
            "danger_signs": danger,
            "noise_report": noise_report,
        },
        "visualizer": {"spectrogram_image": ""},
        "preliminary_assessment": f"{diagnosis} Pattern",
        "risk_level_output": risk,
        "notes": notes,
        "disclaimer": DISCLAIMER,
    }



# ---------------------------------------------------------------------------
# 👶 PER-CHILD ANALYSIS: several chest spots + breathing rate + signs (WHO IMCI)
# ---------------------------------------------------------------------------
# How spot scores are combined and the thresholds, chosen on VALIDATION children by
# evaluate_children.py (screening target: >= 80% sensitivity); see benchmarks/README.md.
# Test children (129): crackles sensitivity 0.79 / specificity 0.76; wheeze 0.44 / 0.94 (only 16 children with wheeze).
CRACKLE_COMBINE, CRACKLE_THRESHOLD = "top2_mean", 0.49
WHEEZE_COMBINE, WHEEZE_THRESHOLD = "mean", 0.28
MAX_SPOTS = 4


def combine_spots(values, strategy):
    values = sorted(values, reverse=True)
    if strategy == "max":
        return values[0]
    if strategy == "top2_mean":
        return float(np.mean(values[:2]))
    return float(np.mean(values))


def classify_child(crackles, wheeze, fast_breathing, age_months, present, danger):
    """IMCI-style classification of the whole child. Returns (assessment, risk, ai_diagnosis, explanation)."""
    general = [d for d in danger if d in imci.GENERAL_DANGER_SIGNS]
    indrawing = any(d in imci.PNEUMONIA_SIGNS for d in danger)
    fever = "fever" in present
    heard = ("crackles and wheeze" if crackles and wheeze else "crackles" if crackles
             else "wheeze-type sounds" if wheeze else "no crackles or wheeze")
    sound_dx = "Pneumonia" if crackles else "Asthma" if wheeze else "Normal"

    if general:
        return ("Severe illness - refer urgently", "High", sound_dx,
                f"IMCI general danger sign reported ({', '.join(general)}). Refer urgently. Lung sounds: {heard}.")
    if age_months is not None and age_months < 2 and (fast_breathing or indrawing or crackles):
        return ("Young infant: possible serious infection - refer urgently", "High", sound_dx,
                f"Under 2 months with {'fast breathing' if fast_breathing else 'chest indrawing' if indrawing else 'crackles'}."
                " IMCI: possible serious bacterial infection, refer urgently.")
    if fast_breathing or indrawing:
        sign = "fast breathing" if fast_breathing else "chest indrawing"
        if wheeze and not crackles:
            return ("Wheeze with " + sign + " - bronchodilator trial, reassess", "High", "Asthma",
                    f"Wheeze-type sounds with {sign}. IMCI: give a rapid-acting bronchodilator trial and reassess "
                    "breathing before classifying as pneumonia.")
        return ("IMCI pneumonia signs" + (" with crackles" if crackles else ""), "High", "Pneumonia",
                f"{sign.capitalize()} meets the IMCI pneumonia sign. Lung sounds: {heard}. Follow the local IMCI pneumonia protocol.")
    if crackles:
        if fast_breathing is None:
            return ("Crackles heard - count breathing rate", "Medium", "Suspected Pneumonia",
                    "Crackles were heard. Count the breathing rate for a full minute and look for chest indrawing to classify.")
        return ("Crackles heard" + (" with fever" if fever else "") + " - clinician review", "Medium", "Suspected Pneumonia",
                "Crackles were heard but breathing is not fast for age. Have a clinician review"
                + (", especially with fever." if fever else "; reassess if the child gets worse."))
    if wheeze:
        return ("Wheeze heard - IMCI wheeze pathway", "Medium", "Suspected Asthma",
                "Wheeze-type sounds were heard without fast breathing. Follow the IMCI wheeze guidance; refer recurrent wheeze.")
    note = " Count the breathing rate for a full minute to complete the IMCI check." if fast_breathing is None else ""
    return ("No crackles or wheeze heard", "Low", "Normal",
            "No crackles or wheeze-type sounds were detected at the recorded spots." + note)


def analyze_child(file_paths, symptoms="", age_months=None, breaths_per_minute=None):
    """Analyse 1-4 chest-spot recordings of ONE child and classify the child (IMCI-style)."""
    spots, scored = [], []
    for i, path in enumerate(file_paths[:MAX_SPOTS], start=1):
        try:
            probs, usable, rejected, n_windows, _ = score_recording(path)
        except (RecordingUnusable, ap.AudioDecodeError) as e:
            spots.append({"spot": i, "usable": False, "message": str(e)})
            continue
        scored.append(probs)
        spots.append({"spot": i, "usable": True, "sound_finding": SOUND_FINDING[max(probs, key=probs.get)],
                      "prob_crackles": round(probs["Pneumonia"], 3), "prob_wheeze": round(probs["Asthma"], 3),
                      "prob_normal": round(probs["Normal"], 3), "segments_analyzed": len(usable)})
    if not scored:
        raise RecordingUnusable(next(s["message"] for s in spots))

    crackle_score = combine_spots([p["Pneumonia"] for p in scored], CRACKLE_COMBINE)
    wheeze_score = combine_spots([p["Asthma"] for p in scored], WHEEZE_COMBINE)
    crackles, wheeze = crackle_score >= CRACKLE_THRESHOLD, wheeze_score >= WHEEZE_THRESHOLD
    present, danger = parse_symptoms(symptoms)
    fast = imci.is_fast_breathing(age_months, breaths_per_minute)
    assessment, risk, ai_dx, explanation = classify_child(crackles, wheeze, fast, age_months, present, danger)

    notes = []
    if len(scored) < 2:
        notes.append("Only one usable chest spot: record at least 2 spots for a reliable result.")
    unusable = [s["spot"] for s in spots if not s["usable"]]
    if unusable:
        notes.append(f"Spot(s) {unusable} could not be analysed; see each spot's message.")
    return {
        "status": "success",
        "biomarkers": {
            "ai_diagnosis": ai_dx,
            "sound_finding": ("crackles and wheeze" if crackles and wheeze else "crackles" if crackles
                              else "wheeze" if wheeze else "normal"),
            # Child-level scores combined across spots; they do not sum to 1.
            "prob_pneumonia": round(crackle_score, 3),
            "prob_asthma": round(wheeze_score, 3),
            "prob_normal": round(float(np.mean([p["Normal"] for p in scored])), 3),
            "spots": spots,
            "age_months": age_months,
            "breaths_per_minute": breaths_per_minute,
            "fast_breathing_cutoff": imci.fast_breathing_cutoff(age_months),
            "fast_breathing": fast,
            "symptoms_reported": present,
            "danger_signs": danger,
            "thresholds": {"crackles": [CRACKLE_COMBINE, CRACKLE_THRESHOLD], "wheeze": [WHEEZE_COMBINE, WHEEZE_THRESHOLD]},
        },
        "visualizer": {"spectrogram_image": ""},
        "preliminary_assessment": assessment,
        "explanation": explanation,
        "risk_level_output": risk,
        "notes": notes,
        "disclaimer": DISCLAIMER,
    }
