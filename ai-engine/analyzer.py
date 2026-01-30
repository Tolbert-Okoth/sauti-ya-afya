import os
import re 

# 🚀 FORCE SINGLE THREADING
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

# 🛡️ SAFE IMPORT FOR SCIPY
try:
    from scipy.stats import kurtosis, entropy
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️ Scipy not found. Fallback to basic math.")
    SCIPY_AVAILABLE = False

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
    'cough': 0.05, 'whistle': 0.20, 'tight': 0.15,
    'wheeze': 0.25, 'crackle': 0.20  
}

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
    return np.sqrt(np.mean(chunk**2))

def apply_bandpass_filter(waveform, sr=16000):
    try:
        filtered = F_audio.highpass_biquad(waveform, sr, cutoff_freq=100.0)
        filtered = F_audio.lowpass_biquad(filtered, sr, cutoff_freq=2000.0)
        return filtered
    except:
        return waveform

def count_transients_tkeo(y_chunk):
    """
    Teager-Kaiser Energy Operator (TKEO) Detector.
    detects 'Explosive' energy bursts while suppressing smooth background noise.
    Formula: Psi[n] = x[n]^2 - x[n-1]*x[n+1]
    """
    try:
        # 1. Basic Safety
        if np.max(np.abs(y_chunk)) < 0.02: return 0

        # 2. Calculate TKEO Energy Profile
        # We use array slicing to implement x[n]^2 - x[n-1]*x[n+1] efficiently
        # current^2 - (prev * next)
        y_sq = y_chunk[1:-1] ** 2
        y_cross = y_chunk[:-2] * y_chunk[2:]
        tkeo_energy = y_sq - y_cross
        
        # 3. Rectify (Absolute Energy)
        tkeo_abs = np.abs(tkeo_energy)

        # 4. Dynamic Thresholding (Crest Factor on Energy)
        # We look for bursts that are 6x the average energy level.
        # This is incredibly robust against "Hissy" breath noise.
        avg_energy = np.mean(tkeo_abs)
        thresh = max(avg_energy * 6.0, 0.0005) # Floor to prevent detecting silence
        
        # 5. Count Bursts (Block-wise)
        # We check 20ms blocks (320 samples)
        block_size = 320 
        n_blocks = len(tkeo_abs) // block_size
        count = 0
        
        for i in range(n_blocks):
            # If the MAX energy in this block exceeds threshold, it's a pop.
            if np.max(tkeo_abs[i*block_size : (i+1)*block_size]) > thresh:
                count += 1
        
        # 6. Artifact Guard (Machine Gun fire)
        if count > 45: return 0 
        return count
    except:
        return 0

def extract_physics_features_lite(y_chunk, sr=16000):
    try:
        zero_crossings = np.nonzero(np.diff(y_chunk > 0))[0]
        zcr = len(zero_crossings) / len(y_chunk)
        
        spectrum = np.abs(np.fft.rfft(y_chunk)) + 1e-10
        log_spectrum = np.log(spectrum)
        geom_mean = np.exp(np.mean(log_spectrum))
        arith_mean = np.mean(spectrum)
        spectral_flatness = geom_mean / arith_mean
        harmonic_ratio = 1.0 - spectral_flatness
        
        if SCIPY_AVAILABLE:
            kurt = kurtosis(y_chunk)
            ent = entropy(np.abs(y_chunk) + 1e-10)
        else:
            kurt = 0.0; ent = 0.0
        
        mad = np.mean(np.abs(y_chunk - np.mean(y_chunk)))
        
        # 🔥 SWITCHED TO TKEO DETECTOR 🔥
        transients = count_transients_tkeo(y_chunk)
        
        return zcr, harmonic_ratio, spectral_flatness, kurt, ent, mad, transients
    except:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

def generate_spectrogram(y_chunk, sr=16000):
    try:
        waveform = torch.from_numpy(y_chunk).unsqueeze(0)
        waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min() + 1e-8)
        waveform[torch.abs(waveform) < 0.01] = 0 
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
    t1 = input_tensor
    t2 = input_tensor.clone(); f = random.randint(0, 100); t2[:,:,f:f+10,:] = 0
    t3 = input_tensor.clone(); t = random.randint(0, 100); t3[:,:,:,t:t+10] = 0
    batch = torch.cat([t1, t2, t3], dim=0)
    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)
    return avg_probs

def analyze_audio(file_path, symptoms="", sensitivity_threshold=0.75):
    try:
        print(f"--- [START] Analysis Job (TKEO Powered) ---")
        
        command = ['ffmpeg', '-y', '-i', file_path, '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-t', '30', '-threads', '1', '-preset', 'ultrafast', '-loglevel', 'error', '-']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0: raise Exception(f"FFmpeg Error: {err.decode()}")

        y_full = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        chunks = []
        for i in range(0, len(y_full), 80000):
            chunk = y_full[i : i + 80000]
            if len(chunk) > 16000: chunks.append(chunk)

        final_diagnosis = "Inconclusive"
        highest_severity = -1
        valid_chunks = 0
        probs_list = [] 
        averaged_probs = {"Asthma": 0.0, "Normal": 0.0, "Pneumonia": 0.0}
        
        physics_override = False
        total_transients = 0
        pneumonia_chunks_detected = 0 

        for idx, chunk in enumerate(chunks):
            rms = calculate_rms(chunk)
            if rms < 0.005: continue 

            img = generate_spectrogram(chunk)
            if ai_available and img:
                input_tensor = preprocess_ai(img).unsqueeze(0)
                probs = predict_with_tta(model, input_tensor)
                probs_list.append(probs) 
                
                zcr, harmonic_ratio, spectral_flatness, kurt, ent, mad, transients = extract_physics_features_lite(chunk)
                total_transients += transients

                winner_idx = torch.argmax(probs).item()
                chunk_diagnosis = CLASSES[winner_idx]
                winner_prob = float(probs[winner_idx])

                # 🛡️ THE NORMALCY SHIELD
                if chunk_diagnosis == "Normal" and winner_prob > 0.85:
                    pneumonia_pop_threshold = 8 
                    pneumonia_harm_limit = 0.5 
                else:
                    pneumonia_pop_threshold = 5 # TKEO is precise, we can trust 5 pops
                    pneumonia_harm_limit = 0.65 

                # 1. HYBRID PNEUMONIA CHECK
                force_pneumonia = False
                
                # Guards (The Spotters)
                is_friction_noise = (zcr > 0.21)
                not_too_flat = (spectral_flatness < 0.42)
                is_crisp = (zcr > 0.15) 

                # ✨ GOLDEN ZONE ✨
                is_golden_normal = (0.05 < zcr < 0.15) and (0.30 < spectral_flatness < 0.60) and (transients < 4)

                if is_golden_normal:
                     print(f"   ✨ GOLDEN ZONE: Physics matches Healthy Breath (ZCR={zcr:.2f}). Forcing Normal.")
                     chunk_diagnosis = "Normal"
                     chunk_severity = 1
                     probs_list[-1] = torch.tensor([0.05, 0.90, 0.05]) 
                     
                elif is_friction_noise:
                     print(f"   ⚠️ ARTIFACT: Extreme Friction (ZCR={zcr:.2f}). Ignoring Pops.")
                else:
                    # TKEO + ZCR Logic
                    if transients >= pneumonia_pop_threshold and is_crisp and not_too_flat:
                        print(f"   ⚠️ HIERARCHY: TKEO Crackles ({transients} bursts, ZCR={zcr:.2f}) -> Forcing Pneumonia.")
                        force_pneumonia = True
                    
                    elif transients >= 4 and harmonic_ratio < pneumonia_harm_limit and is_crisp and chunk_diagnosis != "Normal" and not_too_flat:
                        print(f"   ⚠️ HIERARCHY: Moderate TKEO Crackles ({transients} bursts) -> Forcing Pneumonia.")
                        force_pneumonia = True

                if force_pneumonia:
                    chunk_diagnosis = "Pneumonia"
                    chunk_severity = 3
                    winner_prob = 0.90 
                    probs_list[-1] = torch.tensor([0.05, 0.05, 0.90]) 
                    physics_override = True
                    pneumonia_chunks_detected += 1
                
                # 2. STANDARD AI PREDICTION
                else:
                    if chunk_diagnosis == "Pneumonia" and winner_prob > 0.60:
                        if spectral_flatness > 0.42:
                            print(f"   ℹ️ INFO: AI=Pneumonia, but Sound is Flat ({spectral_flatness:.2f}). Downgrading to Normal.")
                            chunk_diagnosis = "Normal"
                            chunk_severity = 1
                            probs_list[-1] = torch.tensor([0.10, 0.80, 0.10])
                        else:
                            pneumonia_chunks_detected += 1

                    # 🛡️ ASTHMA SANITY CHECK
                    if chunk_diagnosis == "Asthma":
                        veto_triggered = False
                        
                        if spectral_flatness > 0.35:
                            if transients > 1 and is_crisp and not is_friction_noise and not_too_flat: 
                                veto_triggered = True
                                new_diag = "Pneumonia"
                            else: 
                                print(f"   ℹ️ INFO: AI=Asthma, but Sound is Flat. Likely Normal Breath.")
                                chunk_diagnosis = "Normal"
                                chunk_severity = 1
                                winner_prob = 0.60
                                probs_list[-1] = torch.tensor([0.20, 0.60, 0.20]) 
                        
                        elif transients > 2 and is_crisp and not is_friction_noise: 
                            veto_triggered = True
                            new_diag = "Pneumonia"

                        if veto_triggered:
                            print(f"   🛡️ VETO: AI=Asthma, but TKEO detected bursts ({transients}).")
                            chunk_diagnosis = new_diag
                            chunk_severity = 3
                            winner_prob = 0.85 
                            probs_list[-1] = torch.tensor([0.10, 0.05, 0.85]) 
                            physics_override = True
                            pneumonia_chunks_detected += 1
                        else:
                            if chunk_diagnosis != "Normal": chunk_severity = 2 
                    
                    elif chunk_diagnosis == "Normal": chunk_severity = 1
                    elif winner_prob < 0.60: 
                        chunk_diagnosis = f"Suspected {chunk_diagnosis}"
                        chunk_severity = 1.5 
                    else: chunk_severity = SEVERITY_SCORE.get(chunk_diagnosis, 0)

                print(f"   🔹 Chunk {idx+1}: {chunk_diagnosis} (Conf: {winner_prob:.2f} | Sev: {chunk_severity})")
                valid_chunks += 1

                if chunk_severity > highest_severity:
                    highest_severity = chunk_severity
        
        if valid_chunks == 0:
            final_diagnosis = "Inconclusive"
        else:
            if probs_list:
                avg_probs_tensor = torch.mean(torch.stack(probs_list), dim=0)
                if avg_probs_tensor.dim() > 1: avg_probs_tensor = avg_probs_tensor.squeeze()
                averaged_probs = {k: float(v) for k, v in zip(CLASSES, avg_probs_tensor)}
                
                # 3. CONSENSUS LOGIC
                if pneumonia_chunks_detected > 0:
                    final_diagnosis = "Pneumonia"
                    if averaged_probs["Pneumonia"] < 0.5:
                        averaged_probs["Pneumonia"] = 0.75
                        averaged_probs["Asthma"] = min(averaged_probs["Asthma"], 0.20)
                
                elif physics_override:
                     if highest_severity == 3: final_diagnosis = "Pneumonia"
                     elif highest_severity == 2: final_diagnosis = "Asthma"
                     else: final_diagnosis = "Normal"
                
                else:
                    new_winner = max(averaged_probs, key=averaged_probs.get)
                    max_prob = averaged_probs[new_winner]
                    if max_prob < 0.50: final_diagnosis = "Normal" 
                    else: final_diagnosis = new_winner

            if final_diagnosis == "Inconclusive": final_diagnosis = "Normal"
            
            # GLOBAL CHECK (12 Bursts)
            avg_harm = np.mean([extract_physics_features_lite(c, 16000)[1] for c in chunks]) if chunks else 0
            avg_flat = np.mean([extract_physics_features_lite(c, 16000)[2] for c in chunks]) if chunks else 0
            
            if total_transients > 12 and avg_harm < 0.65 and avg_flat < 0.42 and final_diagnosis != "Pneumonia":
                 print(f"   ⚠️ Global TKEO Check: {total_transients} bursts detected. Overriding to Pneumonia.")
                 final_diagnosis = "Pneumonia"
                 averaged_probs["Pneumonia"] = 0.85

        if symptoms:
            matched = []
            risk_bonus = 0.0
            for key, bonus in SYMPTOM_RISK_BONUS.items():
                if re.search(r'\b' + re.escape(key) + r'\b', symptoms.lower()):
                    risk_bonus += bonus; matched.append(key)
            
            if risk_bonus > 0:
                print(f"   ⚠️ Symptoms Bonus: +{risk_bonus:.2f} (Matched: {matched})")
                averaged_probs["Pneumonia"] = min(0.99, averaged_probs["Pneumonia"] + risk_bonus)
                averaged_probs["Asthma"] = min(0.99, averaged_probs["Asthma"] + (risk_bonus * 0.8))
                
                new_winner = max(averaged_probs, key=averaged_probs.get)
                if averaged_probs[new_winner] > 0.60:
                     highest_severity = max(highest_severity, SEVERITY_SCORE.get(new_winner, 0))
                
                if SEVERITY_SCORE.get(new_winner, 0) >= SEVERITY_SCORE.get(final_diagnosis.replace("Suspected ", ""), 0): 
                    final_diagnosis = new_winner

        risk_label = "Low"
        if "Suspected" in final_diagnosis: risk_label = "Medium"
        elif final_diagnosis == "Normal": risk_label = "Low"
        elif final_diagnosis == "Inconclusive": risk_label = "Low"
        else: risk_label = "High"

        print(f"--- [SUCCESS] Verdict: {final_diagnosis} (Risk: {risk_label}) ---")
        return {
            "status": "success",
            "biomarkers": {
                "ai_diagnosis": final_diagnosis,
                "prob_pneumonia": round(averaged_probs["Pneumonia"], 3),
                "prob_asthma": round(averaged_probs["Asthma"], 3),
                "prob_normal": round(averaged_probs["Normal"], 3)
            },
            "visualizer": { "spectrogram_image": "" },
            "preliminary_assessment": f"{final_diagnosis} Pattern",
            "risk_level_output": risk_label,
            "disclaimer": "AI Analysis Only. Consult a Doctor."
        }

    except Exception as e:
        print(f"❌ ANALYZER ERROR: {e}")
        return {"status": "error", "message": str(e)}