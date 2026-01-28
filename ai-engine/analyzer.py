import os

# 🚀 FORCE SINGLE THREADING (Crucial for Free Tier Servers)
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
import librosa # 📚 Added for Physics Math (ZCR/Harmonics)

# 🛑 LIMIT TORCH THREADS
torch.set_num_threads(1) 

print("🔄 Loading Filtered Medical Brain...")
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None) 

# 🛠️ CLASS ORDER (Must match training folder order)
CLASSES = ['Asthma', 'Normal', 'Pneumonia']
SEVERITY_SCORE = {'Pneumonia': 3, 'Asthma': 2, 'Normal': 1, 'Unknown': 0}

# 🏥 HYBRID SYMPTOM WEIGHTS
SYMPTOM_RISK_BONUS = {
    'fever': 0.10, 'pain': 0.15, 'breath': 0.15, 
    'cough': 0.05, 'whistle': 0.20, 'tight': 0.15
}

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
        # Quantize for speed/memory on CPU
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        ai_available = True
        print("✅ Filtered AI Model Loaded")
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ AI Load Error: {e}")

preprocess_ai = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def calculate_rms(chunk):
    """Measure volume to detect silence"""
    return np.sqrt(np.mean(chunk**2))

def apply_bandpass_filter(waveform, sr=16000):
    """Medical Filter: 100Hz (Heartbeat) to 2000Hz (Hiss)"""
    try:
        filtered = F_audio.highpass_biquad(waveform, sr, cutoff_freq=100.0)
        filtered = F_audio.lowpass_biquad(filtered, sr, cutoff_freq=2000.0)
        return filtered
    except:
        return waveform

def extract_physics_features(y_chunk, sr=16000):
    """
    Extracts the 'Truth' features:
    - ZCR (Zero Crossing Rate) -> Detects Crackles (Pneumonia)
    - Harmonic Ratio -> Detects Whistling (Asthma)
    """
    try:
        # ZCR (Spikiness/Crackles)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y_chunk))
        
        # Harmonic Ratio (Musicality/Wheeze)
        # Use simple HPSS decomposition
        y_harm, y_perc = librosa.effects.hpss(y_chunk)
        harmonic_energy = np.sum(np.abs(y_harm))
        percussive_energy = np.sum(np.abs(y_perc))
        harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-6)
        
        return zcr, harmonic_ratio
    except:
        return 0.0, 0.0

def generate_spectrogram(y_chunk, sr=16000):
    try:
        waveform = torch.from_numpy(y_chunk).unsqueeze(0)
        waveform = apply_bandpass_filter(waveform, sr)
        mel_transform = T.MelSpectrogram(sample_rate=sr, n_mels=128, n_fft=2048, hop_length=512, power=2.0)
        spectrogram = mel_transform(waveform)
        spectrogram_db = T.AmplitudeToDB(stype='power', top_db=80)(spectrogram)
        s_norm = (spectrogram_db + 80) / 80.0  
        s_norm = torch.clamp(s_norm, 0, 1)     
        s_norm = s_norm.byte().squeeze(0).numpy() * 255 
        s_norm = np.flipud(s_norm)
        return Image.fromarray(s_norm.astype(np.uint8)).convert('RGB')
    except:
        return None

def predict_with_tta(model, input_tensor):
    """Test Time Augmentation: Average 3 predictions for stability"""
    t1 = input_tensor
    t2 = input_tensor.clone()
    f_start = random.randint(0, 100)
    t2[:, :, f_start:f_start+10, :] = 0 
    
    t3 = input_tensor.clone()
    t_start = random.randint(0, 100)
    t3[:, :, :, t_start:t_start+10] = 0  
    
    batch = torch.cat([t1, t2, t3], dim=0)
    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)
    return avg_probs

def analyze_audio(file_path, symptoms="", sensitivity_threshold=0.75):
    try:
        start_time = time.time()
        print(f"--- [START] Analysis Job (Filtered + TTA + Physics Veto) ---")
        
        # 1. Load Audio with FFmpeg
        command = [
            'ffmpeg', '-y', '-i', file_path, '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', '-t', '30', '-threads', '1', 
            '-preset', 'ultrafast', '-loglevel', 'error', '-'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0: raise Exception(f"FFmpeg Error: {err.decode()}")

        y_full = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        
        chunks = []
        CHUNK_SIZE = 80000 # 5 seconds
        for i in range(0, len(y_full), CHUNK_SIZE):
            chunk = y_full[i : i + CHUNK_SIZE]
            if len(chunk) > 16000: chunks.append(chunk)

        final_diagnosis = "Inconclusive"
        highest_severity = -1
        valid_chunks = 0
        averaged_probs = {"Asthma": 0.0, "Normal": 0.0, "Pneumonia": 0.0}

        # 2. Analyze Chunks
        for idx, chunk in enumerate(chunks):
            # A. Silence Check
            if calculate_rms(chunk) < 0.005: continue 

            # B. Generate AI Inputs
            img = generate_spectrogram(chunk)
            
            if ai_available and img:
                input_tensor = preprocess_ai(img).unsqueeze(0)
                probs = predict_with_tta(model, input_tensor)
                
                # Unpack raw AI probabilities
                # CLASSES = ['Asthma', 'Normal', 'Pneumonia']
                p_asthma = float(probs[0])
                p_normal = float(probs[1])
                p_pneumonia = float(probs[2])

                # C. CALCULATE PHYSICS FEATURES (The Truth)
                zcr, harmonic_ratio = extract_physics_features(chunk)

                # ---------------------------------------------------------
                # 🛡️ THE MEDICAL HIERARCHY (PHYSICS VETO LOGIC)
                # ---------------------------------------------------------
                
                # RULE 1: CRACKLE CHECK (The "Pneumonia" Trump Card)
                # If ZCR is high (> 0.15), it is Pneumonia. Period.
                # Crackles override wheezes.
                if zcr > 0.15:
                    winner_idx = 2 # Pneumonia
                    winner_prob = 0.95 # Force High Confidence
                    chunk_diagnosis = "Pneumonia"
                    print(f"   ⚠️ HIERARCHY: High Crackles (ZCR={zcr:.2f}) -> Forcing Pneumonia.")

                # RULE 2: WHEEZE CHECK (The "Asthma" Test)
                # Only if NO crackles. If Harmonic Ratio is high (> 0.25), it is Asthma.
                elif harmonic_ratio > 0.25:
                    winner_idx = 0 # Asthma
                    winner_prob = 0.95 # Force High Confidence
                    chunk_diagnosis = "Asthma"
                    print(f"   ⚠️ HIERARCHY: Pure Wheeze (Harmonic={harmonic_ratio:.2f}) -> Forcing Asthma.")

                # RULE 3: DEFAULT TO AI (With Warning Zones)
                # If physics are subtle, trust the Trained AI.
                else:
                    winner_idx = torch.argmax(probs).item()
                    chunk_diagnosis = CLASSES[winner_idx]
                    winner_prob = float(probs[winner_idx])

                    # Apply Warning Zone (Only if Physics didn't override)
                    if winner_prob < 0.40:
                        chunk_diagnosis = "Normal"
                        chunk_severity = 1
                    elif winner_prob < 0.60:
                        chunk_diagnosis = "Normal" # Default to Normal but flag as "Suspected" in severity logic
                        chunk_severity = 1.5 
                        # We allow "Suspected" logic to be handled by severity score below
                    
                # ---------------------------------------------------------

                # Assign Severity based on final decision
                if chunk_diagnosis == "Normal" and winner_prob < 0.60:
                     chunk_severity = 1.5 if winner_prob > 0.4 else 1
                else:
                     chunk_severity = SEVERITY_SCORE.get(chunk_diagnosis, 0)
                
                print(f"   🔹 Chunk {idx+1}: {chunk_diagnosis} (Conf: {winner_prob:.2f} | Sev: {chunk_severity} | ZCR: {zcr:.2f} | Harm: {harmonic_ratio:.2f})")
                valid_chunks += 1

                # Update Global Diagnosis if this chunk is more severe
                if chunk_severity > highest_severity:
                    highest_severity = chunk_severity
                    final_diagnosis = chunk_diagnosis
                    # Store probabilities for final output
                    averaged_probs = {"Asthma": p_asthma, "Normal": p_normal, "Pneumonia": p_pneumonia}
                    
                    # If we forced a diagnosis via Physics, update stats to match
                    if chunk_diagnosis == "Pneumonia" and zcr > 0.15:
                         averaged_probs = {"Asthma": 0.05, "Normal": 0.05, "Pneumonia": 0.90}
                    elif chunk_diagnosis == "Asthma" and harmonic_ratio > 0.25:
                         averaged_probs = {"Asthma": 0.90, "Normal": 0.05, "Pneumonia": 0.05}
        
        if valid_chunks == 0: final_diagnosis = "Inconclusive"
        elif final_diagnosis == "Inconclusive": final_diagnosis = "Normal"

        # 🏥 SYMPTOM CHECK
        if symptoms:
            symptoms_lower = symptoms.lower()
            risk_bonus = 0.0
            for key, bonus in SYMPTOM_RISK_BONUS.items():
                if key in symptoms_lower: risk_bonus += bonus
            
            if risk_bonus > 0:
                print(f"   ⚠️ Symptoms Bonus: +{risk_bonus:.2f}")
                averaged_probs["Pneumonia"] = min(0.99, averaged_probs["Pneumonia"] + risk_bonus)
                averaged_probs["Asthma"] = min(0.99, averaged_probs["Asthma"] + (risk_bonus * 0.8))
                
                # Re-Evaluate Winner after symptoms
                new_winner = max(averaged_probs, key=averaged_probs.get)
                new_prob = averaged_probs[new_winner]
                
                # If symptoms pushed confidence > 60%, upgrade severity
                if new_prob > 0.60:
                     highest_severity = max(highest_severity, SEVERITY_SCORE.get(new_winner, 0))
                
                if SEVERITY_SCORE.get(new_winner, 0) >= SEVERITY_SCORE.get(final_diagnosis, 0):
                    final_diagnosis = new_winner

        # 🎯 FINAL VERDICT FORMATTING
        risk_label = "Low"
        
        # Handle "Suspected" Logic
        if highest_severity == 1.5 and final_diagnosis == "Normal":
             final_diagnosis = "Suspected Respiratory Issue"
             risk_label = "Medium"
        elif final_diagnosis == "Normal":
            risk_label = "Low"
        elif highest_severity == 1.5:
            final_diagnosis = f"Suspected {final_diagnosis}"
            risk_label = "Medium"
        else:
            risk_label = "High"

        print(f"--- [SUCCESS] Verdict: {final_diagnosis} (Risk: {risk_label}) ---")
        return {
            "status": "success",
            "biomarkers": {
                "tonality_score": 0.0,
                "ai_diagnosis": final_diagnosis,
                "prob_pneumonia": round(averaged_probs["Pneumonia"], 3),
                "prob_asthma": round(averaged_probs["Asthma"], 3),
                "prob_normal": round(averaged_probs["Normal"], 3)
            },
            "visualizer": { "spectrogram_image": "" },
            "preliminary_assessment": f"{final_diagnosis} Pattern",
            "risk_level_output": risk_label
        }

    except Exception as e:
        print(f"❌ ANALYZER ERROR: {e}")
        return {"status": "error", "message": str(e)}