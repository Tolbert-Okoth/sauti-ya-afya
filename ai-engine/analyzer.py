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

torch.set_num_threads(1)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.environ.get("MODEL_FILE", "sauti_mobilenet_v5.pth"))
CLASSES = ['Asthma', 'Normal', 'Pneumonia']
DEFAULT_THRESHOLD = 0.75
MIN_THRESHOLD, MAX_THRESHOLD = 0.5, 0.99
DISCLAIMER = "AI Analysis Only. Consult a Doctor."


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
def analyze_audio(file_path, symptoms="", sensitivity_threshold=DEFAULT_THRESHOLD):
    threshold = clamp_threshold(sensitivity_threshold)
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
    window_probs = torch.softmax(model(batch), dim=1)
    mean_probs = window_probs.mean(dim=0)
    probs = {c: float(p) for c, p in zip(CLASSES, mean_probs)}

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

