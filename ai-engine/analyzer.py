import os

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

# 🛑 LIMIT TORCH THREADS
torch.set_num_threads(1) 

print("🔄 Loading Filtered Medical Brain...")
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
    try:
        filtered = F_audio.highpass_biquad(waveform, sr, cutoff_freq=100.0)
        filtered = F_audio.lowpass_biquad(filtered, sr, cutoff_freq=2000.0)
        return filtered
    except:
        return waveform

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
    # 🛠️ FIXED: Removed extra trailing comma/colon causing IndexError
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
        print(f"--- [START] Analysis Job (Filtered + TTA + Warning Zone) ---")
        
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
        CHUNK_SIZE = 80000 
        for i in range(0, len(y_full), CHUNK_SIZE):
            chunk = y_full[i : i + CHUNK_SIZE]
            if len(chunk) > 16000: chunks.append(chunk)

        final_diagnosis = "Inconclusive"
        highest_severity = -1
        valid_chunks = 0
        averaged_probs = {"Asthma": 0.0, "Normal": 0.0, "Pneumonia": 0.0}

        for idx, chunk in enumerate(chunks):
            if calculate_rms(chunk) < 0.005: continue 

            img = generate_spectrogram(chunk)
            
            if ai_available and img:
                input_tensor = preprocess_ai(img).unsqueeze(0)
                probs = predict_with_tta(model, input_tensor)
                
                winner_idx = torch.argmax(probs).item()
                chunk_diagnosis = CLASSES[winner_idx]
                winner_prob = float(probs[winner_idx])
                
                # 🛡️ NEW LOGIC: The "Warning Zone"
                if winner_prob < 0.40:
                    # Too weak (<40%), assume noise
                    chunk_diagnosis = "Normal"
                    chunk_severity = 1
                elif winner_prob < 0.60:
                    # ⚠️ WARN USER (40% - 60%)
                    # Keep the diagnosis (e.g. Asthma) but assign "Medium" severity
                    chunk_severity = 1.5 
                else:
                    # 🚨 HIGH CONFIDENCE (>60%)
                    chunk_severity = SEVERITY_SCORE.get(chunk_diagnosis, 0)
                
                print(f"   🔹 Chunk {idx+1}: {chunk_diagnosis} (Conf: {winner_prob:.2f} | Sev: {chunk_severity})")
                valid_chunks += 1

                if chunk_severity > highest_severity:
                    highest_severity = chunk_severity
                    final_diagnosis = chunk_diagnosis
                    averaged_probs = {"Asthma": float(probs[0]), "Normal": float(probs[1]), "Pneumonia": float(probs[2])}
        
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
        
        if final_diagnosis == "Normal":
            risk_label = "Low"
        elif highest_severity == 1.5:
            # The "Grey Area" case
            final_diagnosis = f"Suspected {final_diagnosis}"
            risk_label = "Medium"
        else:
            # The "High Confidence" case
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