import os

# 🚀 FORCE SINGLE THREADING (Crucial for Free Tier)
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
# ❌ REMOVED: import librosa (Too heavy)

# 🛑 LIMIT TORCH THREADS
torch.set_num_threads(1) 

print("🔄 Loading Lite Medical Brain...")
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None) 

# 🛠️ CLASS ORDER
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

def extract_physics_features_lite(y_chunk, sr=16000):
    """
    LIGHTWEIGHT PHYSICS EXTRACTION (No Librosa)
    Uses pure Numpy FFT to detect Wheezes vs Crackles.
    """
    try:
        # 1. ZCR (Zero Crossing Rate) - Crackle Detector
        # Count how many times signal crosses 0 axis
        zero_crossings = np.nonzero(np.diff(y_chunk > 0))[0]
        zcr = len(zero_crossings) / len(y_chunk)
        
        # 2. Harmonicity (Spectral Flatness) - Wheeze Detector
        # Wheeze = Tone (Spiky Spectrum). Crackle = Noise (Flat Spectrum).
        # We calculate "Spectral Flatness" and invert it.
        
        # Fast Fourier Transform
        spectrum = np.abs(np.fft.rfft(y_chunk))
        spectrum = spectrum + 1e-10 # Avoid divide by zero
        
        # Geometric Mean / Arithmetic Mean (Wiener Entropy)
        log_spectrum = np.log(spectrum)
        geom_mean = np.exp(np.mean(log_spectrum))
        arith_mean = np.mean(spectrum)
        
        spectral_flatness = geom_mean / arith_mean
        
        # High Flatness (1.0) = Noise. Low Flatness (0.0) = Tone.
        # We want Harmonic Ratio: 1.0 = Pure Tone (Wheeze).
        harmonic_ratio = 1.0 - spectral_flatness
        
        return zcr, harmonic_ratio
    except Exception as e:
        print(f"Physics Error: {e}")
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
        print(f"--- [START] Analysis Job (Lite Mode: No Librosa) ---")
        
        # 1. Load Audio with FFmpeg (Very Light)
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

                # C. CALCULATE PHYSICS FEATURES (LITE MODE)
                zcr, harmonic_ratio = extract_physics_features_lite(chunk)

                # ---------------------------------------------------------
                # 🛡️ THE MEDICAL HIERARCHY (PHYSICS VETO LOGIC)
                # ---------------------------------------------------------
                
                # RULE 1: CRACKLE CHECK (The "Pneumonia" Trump Card)
                if zcr > 0.15:
                    winner_idx = 2 # Pneumonia
                    winner_prob = 0.95 
                    chunk_diagnosis = "Pneumonia"
                    print(f"   ⚠️ HIERARCHY: High Crackles (ZCR={zcr:.2f}) -> Forcing Pneumonia.")
                    averaged_probs = {"Asthma": 0.05, "Normal": 0.05, "Pneumonia": 0.90}

                # RULE 2: WHEEZE CHECK (The "Asthma" Test)
                # Harmonic Ratio (inverted flatness) > 0.5 means very tonal
                elif harmonic_ratio > 0.5:
                    winner_idx = 0 # Asthma
                    winner_prob = 0.95 
                    chunk_diagnosis = "Asthma"
                    print(f"   ⚠️ HIERARCHY: Pure Wheeze (Harmonic={harmonic_ratio:.2f}) -> Forcing Asthma.")
                    averaged_probs = {"Asthma": 0.90, "Normal": 0.05, "Pneumonia": 0.05}

                # RULE 3: DEFAULT TO AI
                else:
                    winner_idx = torch.argmax(probs).item()
                    chunk_diagnosis = CLASSES[winner_idx]
                    winner_prob = float(probs[winner_idx])

                    if winner_prob < 0.40:
                        chunk_diagnosis = "Normal"
                        chunk_severity = 1
                    elif winner_prob < 0.60:
                        chunk_diagnosis = "Normal"
                        chunk_severity = 1.5 
                    
                # ---------------------------------------------------------

                # Assign Severity
                if chunk_diagnosis == "Normal" and winner_prob < 0.60:
                     chunk_severity = 1.5 if winner_prob > 0.4 else 1
                else:
                     chunk_severity = SEVERITY_SCORE.get(chunk_diagnosis, 0)
                
                print(f"   🔹 Chunk {idx+1}: {chunk_diagnosis} (Conf: {winner_prob:.2f} | Sev: {chunk_severity} | ZCR: {zcr:.2f} | Harm: {harmonic_ratio:.2f})")
                valid_chunks += 1

                # Update Global Diagnosis
                if chunk_severity > highest_severity:
                    highest_severity = chunk_severity
                    final_diagnosis = chunk_diagnosis
                    if zcr <= 0.15 and harmonic_ratio <= 0.5: # Don't overwrite if physics forced it
                         averaged_probs = {"Asthma": p_asthma, "Normal": p_normal, "Pneumonia": p_pneumonia}
        
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
                
                new_winner = max(averaged_probs, key=averaged_probs.get)
                new_prob = averaged_probs[new_winner]
                
                if new_prob > 0.60:
                     highest_severity = max(highest_severity, SEVERITY_SCORE.get(new_winner, 0))
                
                if SEVERITY_SCORE.get(new_winner, 0) >= SEVERITY_SCORE.get(final_diagnosis, 0):
                    final_diagnosis = new_winner

        # 🎯 FINAL VERDICT
        risk_label = "Low"
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
                "ai_diagnosis": final_diagnosis,
                "prob_pneumonia": round(averaged_probs["Pneumonia"], 3),
                "prob_asthma": round(averaged_probs["Asthma"], 3),
                "prob_normal": round(averaged_probs["Normal"], 3)
            },
            "risk_level_output": risk_label
        }

    except Exception as e:
        print(f"❌ ANALYZER ERROR: {e}")
        return {"status": "error", "message": str(e)}