# analyzer.py - Upgraded version (safer, less overconfident, reduced false pneumonia calls)
# Focus: much higher specificity for pneumonia, soft bonuses instead of hard overwrites,
# persistence requirements, better quality gating, capped symptom influence

import os
import re

# FORCE SINGLE THREADING
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F_audio
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import gc
import subprocess
import time
import random

# SAFE SCIPY IMPORT
try:
    from scipy.stats import kurtosis, entropy
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️ Scipy not found. Some advanced stats disabled.")
    SCIPY_AVAILABLE = False

torch.set_num_threads(1)

print("🔄 Loading Lung Sound Analyzer (Vesicular Shield 2.1 - Safer Edition)...")
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None)

CLASSES = ['Asthma', 'Normal', 'Pneumonia']
SEVERITY_SCORE = {'Pneumonia': 3, 'Asthma': 2, 'Normal': 1, 'Unknown': 0}

# Symptom influence — much softer and capped
SYMPTOM_RISK_BONUS = {
    'fever': 0.08, 'pain': 0.10, 'breath': 0.12,
    'cough': 0.06, 'whistle': 0.15, 'tight': 0.10,
    'wheeze': 0.18, 'crackle': 0.15
}
MAX_SYMPTOM_BONUS = 0.25

CRACKLE_WEIGHT = 0.6

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(model.last_channel, 3)
)

MODEL_PATH = 'sauti_mobilenet_v2_robust.pth'
ai_available = False

try:
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        ai_available = True
        print("✅ Model loaded (quantized head)")
    else:
        print(f"❌ Model file missing: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Model load failed: {e}")

preprocess_ai = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def calculate_rms(chunk):
    return np.sqrt(np.mean(chunk**2))


def apply_bandpass_filter(waveform, sr=16000):
    try:
        filtered = F_audio.highpass_biquad(waveform, sr, cutoff_freq=100.0)
        filtered = F_audio.lowpass_biquad(filtered, sr, cutoff_freq=2200.0)
        return filtered
    except:
        return waveform


def count_transients_tkeo(y_chunk):
    try:
        if np.max(np.abs(y_chunk)) < 0.025:
            return 0

        y_sq = y_chunk[1:-1] ** 2
        y_cross = y_chunk[:-2] * y_chunk[2:]
        tkeo_energy = y_sq - y_cross
        tkeo_abs = np.abs(tkeo_energy)

        avg_energy = np.mean(tkeo_abs)
        thresh = max(avg_energy * 10.0, 0.0008)  # raised from 8.0 / 0.0005

        block_size = 320
        n_blocks = len(tkeo_abs) // block_size
        count = 0

        for i in range(n_blocks):
            if np.max(tkeo_abs[i*block_size : (i+1)*block_size]) > thresh:
                count += 1

        if count > 40:  # artifact / machine-gun fire guard
            return 0
        return count
    except:
        return 0


def calculate_spectral_flux(y_chunk, sr=16000):
    try:
        n_fft = 512
        hop_length = 256
        window = np.hanning(n_fft)

        n_frames = (len(y_chunk) - n_fft) // hop_length
        if n_frames < 3:
            return 0.0

        flux_sum = 0.0
        prev_spectrum = None

        for i in range(n_frames):
            start = i * hop_length
            frame = y_chunk[start : start + n_fft] * window
            spectrum = np.abs(np.fft.rfft(frame))
            spectrum = spectrum / (np.linalg.norm(spectrum) + 1e-9)

            if prev_spectrum is not None:
                flux = np.linalg.norm(spectrum - prev_spectrum)
                flux_sum += flux
            prev_spectrum = spectrum

        return flux_sum / n_frames if n_frames > 0 else 0.0
    except:
        return 0.0


def extract_physics_features_lite(y_chunk, sr=16000):
    try:
        zero_crossings = np.nonzero(np.diff(y_chunk > 0))[0]
        zcr = len(zero_crossings) / len(y_chunk) if len(y_chunk) > 0 else 0.0

        spectrum = np.abs(np.fft.rfft(y_chunk)) + 1e-10
        freqs = np.fft.rfftfreq(len(y_chunk), 1/sr)

        spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * spectrum) / np.sum(spectrum))

        rms = np.sqrt(np.mean(y_chunk**2))
        peak = np.max(np.abs(y_chunk))
        crest_factor = peak / (rms + 1e-9)

        spectral_flux = calculate_spectral_flux(y_chunk, sr)

        log_spectrum = np.log(spectrum + 1e-10)
        geom_mean = np.exp(np.mean(log_spectrum))
        arith_mean = np.mean(spectrum)
        spectral_flatness = geom_mean / arith_mean

        harmonic_ratio = 1.0 - spectral_flatness

        kurt = kurtosis(y_chunk) if SCIPY_AVAILABLE else 0.0
        ent = entropy(np.abs(y_chunk) + 1e-10) if SCIPY_AVAILABLE else 0.0

        mad = np.mean(np.abs(y_chunk - np.mean(y_chunk)))
        transients = count_transients_tkeo(y_chunk)

        return (zcr, harmonic_ratio, spectral_flatness, kurt, ent, mad,
                transients, spectral_centroid, spectral_bandwidth, crest_factor, spectral_flux)
    except:
        return (0.0,)*11


def generate_spectrogram(y_chunk, sr=16000):
    try:
        waveform = torch.from_numpy(y_chunk).unsqueeze(0).float()
        waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min() + 1e-8)
        waveform[torch.abs(waveform) < 0.012] = 0
        waveform = apply_bandpass_filter(waveform, sr)

        mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_mels=128, n_fft=2048, hop_length=512, power=2.0
        )
        spectrogram = mel_transform(waveform)
        spectrogram_db = T.AmplitudeToDB(stype='power', top_db=80)(spectrogram)
        s_norm = (spectrogram_db + 80) / 80.0
        s_norm = torch.clamp(s_norm, 0, 1)
        s_norm = (s_norm * 255).byte().squeeze(0).numpy()
        s_norm = np.flipud(s_norm)
        return Image.fromarray(s_norm.astype(np.uint8)).convert('RGB')
    except:
        return None


def predict_with_tta(model, input_tensor):
    t1 = input_tensor
    t2 = input_tensor.clone(); f = random.randint(0, 100); t2[:,:,f:f+10,:] = 0
    t3 = input_tensor.clone(); t = random.randint(0, 100); t3[:,:,:,t:t+10] = 0
    batch = torch.cat([t1, t2, t3], dim=0)
    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)
    return avg_probs


def soft_bonus(probs, bonus_tensor):
    """Apply soft bonus/penalty in logit space and renormalize"""
    logits = torch.log(probs.clamp(min=1e-8))
    logits += bonus_tensor
    new_probs = F.softmax(logits, dim=0)
    return new_probs


def analyze_audio(file_path, symptoms="", sensitivity_threshold=0.75):
    try:
        print("--- [START] Safer Lung Sound Analysis (2025 edition) ---")

        # Convert to 16kHz mono 30s max
        command = [
            'ffmpeg', '-y', '-i', file_path, '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', '-t', '30', '-threads', '1',
            '-preset', 'ultrafast', '-loglevel', 'error', '-'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0:
            raise Exception(f"FFmpeg failed: {err.decode()}")

        y_full = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        chunks = [y_full[i:i+80000] for i in range(0, len(y_full), 80000) if len(y_full[i:i+80000]) > 16000]

        if not chunks:
            return {"status": "error", "message": "No usable audio content"}

        # Quality gating
        rms_values = [calculate_rms(c) for c in chunks]
        mean_rms = np.mean(rms_values)
        valid_chunk_ratio = sum(1 for r in rms_values if r > 0.010) / len(chunks)

        if mean_rms < 0.012 or valid_chunk_ratio < 0.40:
            print(f"   ⚠️ Low quality: mean RMS={mean_rms:.3f}, valid ratio={valid_chunk_ratio:.2f}")
            return {
                "status": "error",
                "message": "Recording too quiet or contains too much silence/noise"
            }

        final_diagnosis = "Inconclusive"
        highest_severity = 1
        valid_chunks = 0
        probs_list = []
        pneumonia_evidence_chunks = 0
        asthma_evidence_chunks = 0
        wheeze_count = 0
        total_transients = 0

        # Tuned thresholds (much stricter for crackles)
        STRONG_CRACKLE_MIN   = 12
        MODERATE_CRACKLE_MIN = 8
        GLOBAL_STRONG_CRACKLE = 22
        GLOBAL_MOD_CRACKLE   = 15
        MIN_CHUNKS_FOR_STRONG = max(2, len(chunks) // 4)

        for idx, chunk in enumerate(chunks):
            rms = calculate_rms(chunk)
            if rms < 0.010:
                continue

            img = generate_spectrogram(chunk)
            if not img or not ai_available:
                continue

            input_tensor = preprocess_ai(img).unsqueeze(0)
            probs = predict_with_tta(model, input_tensor)
            probs_list.append(probs)

            feats = extract_physics_features_lite(chunk)
            zcr, h_ratio, flatness, kurt, ent, mad, transients, cent, bw, crest, flux = feats
            total_transients += transients

            winner_idx = torch.argmax(probs).item()
            chunk_diag = CLASSES[winner_idx]
            conf = float(probs[winner_idx])

            # ───────────────────────────────────────────────
            # Define strong characteristic patterns
            # ───────────────────────────────────────────────

            is_friction_artifact = zcr > 0.20

            is_strong_wheeze = (
                550 < cent < 2400 and
                bw < 1100 and
                flatness < 0.28 and
                zcr > 0.07 and
                transients <= 3 and
                flux < 0.7
            )

            is_potential_strong_crackle = (
                transients >= STRONG_CRACKLE_MIN and
                flatness < 0.40 and
                crest > 11 and
                350 < cent < 1500 and
                flux > 0.9
            )

            is_moderate_crackle = (
                transients >= MODERATE_CRACKLE_MIN and
                is_potential_strong_crackle
            )

            is_very_clean_normal = (
                zcr < 0.085 and
                cent < 700 and
                rms < 0.045 and
                transients <= 1 and
                h_ratio < 0.55 and
                not is_strong_wheeze
            )

            is_golden_normal = (
                0.04 < zcr < 0.14 and
                0.32 < flatness < 0.62 and
                transients <= 4 and
                not is_strong_wheeze and
                not is_potential_strong_crackle
            )

            # ───────────────────────────────────────────────
            # Apply soft physics-based adjustments
            # ───────────────────────────────────────────────

            if is_very_clean_normal or is_golden_normal:
                probs = soft_bonus(probs, torch.tensor([-0.8, +1.4, -0.8]))
                chunk_diag = "Normal"
                chunk_sev = 1

            elif is_strong_wheeze and not is_friction_artifact:
                probs = soft_bonus(probs, torch.tensor([+1.3, -0.5, -0.9]))
                wheeze_count += 1
                asthma_evidence_chunks += 1
                chunk_sev = 2

            elif is_potential_strong_crackle and not is_friction_artifact:
                probs = soft_bonus(probs, torch.tensor([-0.6, -0.6, +1.5]))
                pneumonia_evidence_chunks += 1
                chunk_sev = 3

            elif is_moderate_crackle and flux > 1.1:
                probs = soft_bonus(probs, torch.tensor([-0.4, -0.4, +1.1]))
                pneumonia_evidence_chunks += 1
                chunk_sev = 2.5

            else:
                # No strong physics override — trust model more
                chunk_sev = SEVERITY_SCORE.get(chunk_diag, 1)
                if conf < 0.58:
                    chunk_diag = "Uncertain"

            print(f"   Chunk {idx+1:2d}: {chunk_diag:<12}  conf:{conf:5.2f}  sev:{chunk_sev:4.1f} "
                  f"  trans:{transients:3d}  cent:{cent:5.0f}  flat:{flatness:5.3f}")

            valid_chunks += 1
            highest_severity = max(highest_severity, chunk_sev)

        if valid_chunks == 0:
            return {"status": "error", "message": "No valid analyzable chunks"}

        # ───────────────────────────────────────────────
        # Global aggregation
        # ───────────────────────────────────────────────

        avg_probs_tensor = torch.mean(torch.stack(probs_list), dim=0)
        averaged_probs = {k: float(v) for k, v in zip(CLASSES, avg_probs_tensor)}

        # Persistence-based final lean
        if pneumonia_evidence_chunks >= MIN_CHUNKS_FOR_STRONG and total_transients >= GLOBAL_STRONG_CRACKLE:
            averaged_probs["Pneumonia"] = max(averaged_probs["Pneumonia"], 0.74)
            final_diagnosis = "Pneumonia"
        elif pneumonia_evidence_chunks >= max(2, len(chunks)//5) and total_transients >= GLOBAL_MOD_CRACKLE:
            averaged_probs["Pneumonia"] = max(averaged_probs["Pneumonia"], 0.65)
            if averaged_probs["Pneumonia"] > 0.62:
                final_diagnosis = "Pneumonia"

        elif asthma_evidence_chunks >= MIN_CHUNKS_FOR_STRONG and wheeze_count >= 2:
            averaged_probs["Asthma"] = max(averaged_probs["Asthma"], 0.72)
            final_diagnosis = "Asthma"

        else:
            winner = max(averaged_probs, key=averaged_probs.get)
            max_p = averaged_probs[winner]
            if max_p < 0.63:
                final_diagnosis = "Normal (low confidence)"
            else:
                final_diagnosis = winner

        # Global transient veto (very strong crackle signal across file)
        if total_transients > GLOBAL_STRONG_CRACKLE and averaged_probs["Pneumonia"] < 0.70:
            print(f"   Global TKEO alert: {total_transients} transients → Pneumonia lean")
            averaged_probs["Pneumonia"] = max(averaged_probs["Pneumonia"], 0.71)
            final_diagnosis = "Pneumonia"

        # ───────────────────────────────────────────────
        # Symptom influence — capped & soft
        # ───────────────────────────────────────────────

        risk_bonus = 0.0
        matched_symptoms = []
        symptoms_lower = symptoms.lower()

        for key, bonus in SYMPTOM_RISK_BONUS.items():
            if re.search(r'\b' + re.escape(key) + r'\b', symptoms_lower):
                risk_bonus += bonus
                matched_symptoms.append(key)

        risk_bonus = min(risk_bonus, MAX_SYMPTOM_BONUS)

        if risk_bonus > 0 and len(matched_symptoms) > 0:
            print(f"   Symptoms influence: +{risk_bonus:.2f}  (matched: {', '.join(matched_symptoms)})")
            pneumonia_boost = risk_bonus * 1.0
            asthma_boost   = risk_bonus * 0.75
            averaged_probs["Pneumonia"] += pneumonia_boost
            averaged_probs["Asthma"]   += asthma_boost
            total = sum(averaged_probs.values())
            for k in averaged_probs:
                averaged_probs[k] /= total

            # Re-evaluate winner after symptom adjustment
            new_winner = max(averaged_probs, key=averaged_probs.get)
            if averaged_probs[new_winner] > 0.64:
                final_diagnosis = new_winner

        # Final safety net
        if final_diagnosis == "Inconclusive":
            final_diagnosis = "Normal"

        risk_label = "Low"
        if "Pneumonia" in final_diagnosis:
            risk_label = "High"
        elif "Asthma" in final_diagnosis:
            risk_label = "Medium-High"
        elif final_diagnosis == "Normal (low confidence)":
            risk_label = "Low – uncertain"

        print(f"--- [FINISH] Diagnosis: {final_diagnosis}   Risk: {risk_label} ---")

        return {
            "status": "success",
            "biomarkers": {
                "ai_diagnosis": final_diagnosis,
                "prob_pneumonia": round(averaged_probs["Pneumonia"], 3),
                "prob_asthma": round(averaged_probs["Asthma"], 3),
                "prob_normal": round(averaged_probs["Normal"], 3)
            },
            "visualizer": {"spectrogram_image": ""},
            "preliminary_assessment": f"{final_diagnosis} pattern detected",
            "risk_level_output": risk_label,
            "disclaimer": "This is NOT a medical diagnosis. Consult a qualified healthcare professional."
        }

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return {"status": "error", "message": str(e)}