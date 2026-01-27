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
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import gc 
import subprocess 
import time
import random # Needed for TTA

# 🛑 LIMIT TORCH THREADS
torch.set_num_threads(1) 

print("🔄 Loading Robust Brain...")
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None) 

# 🛠️ CLASS ORDER
CLASSES = ['Asthma', 'Normal', 'Pneumonia']

# Map diagnosis to severity score for comparison
SEVERITY_SCORE = {'Pneumonia': 3, 'Asthma': 2, 'Normal': 1, 'Unknown': 0}

# 🏥 HYBRID SYMPTOM WEIGHTS (The "Doctor" Logic)
# If these words appear in the symptoms string, we add this bonus to the risk score.
SYMPTOM_RISK_BONUS = {
    'fever': 0.10,       # +10% risk
    'pain': 0.15,        # +15% risk (chest pain)
    'breath': 0.15,      # +15% risk (shortness of breath)
    'cough': 0.05,       # +5% risk
    'sweat': 0.05,       # +5% risk (night sweats)
    'weight': 0.10       # +10% risk (weight loss)
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
        print("✅ Robust AI Model Loaded")
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
    """Calculate Root Mean Square (Energy/Volume) of the audio chunk"""
    return np.sqrt(np.mean(chunk**2))

def generate_spectrogram(y_chunk, sr=16000):
    try:
        waveform = torch.from_numpy(y_chunk).unsqueeze(0)
        
        mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_mels=128, n_fft=2048, hop_length=512, power=2.0
        )
        spectrogram = mel_transform(waveform)
        spectrogram_db = T.AmplitudeToDB(stype='power', top_db=80)(spectrogram)
        
        # Fixed Normalization (Energy-Aware)
        s_norm = (spectrogram_db + 80) / 80.0  
        s_norm = torch.clamp(s_norm, 0, 1)     
        s_norm = s_norm.byte().squeeze(0).numpy() * 255 
        
        s_norm = np.flipud(s_norm)
        return Image.fromarray(s_norm.astype(np.uint8)).convert('RGB')
    except:
        return None

# 🧠 TTA: TEST TIME AUGMENTATION
# "Ask the AI 3 times with slight variations to be sure"
def predict_with_tta(model, input_tensor):
    # 1. Original
    t1 = input_tensor
    
    # 2. Frequency Mask (Block a horizontal stripe)
    t2 = input_tensor.clone()
    f_dim = t2.shape[2]
    f_mask_width = random.randint(5, 15)
    f_start = random.randint(0, f_dim - f_mask_width)
    t2[:, :, f_start:f_start+f_mask_width, :] = 0 # Zero out frequency band

    # 3. Time Mask (Block a vertical stripe)
    t3 = input_tensor.clone()
    t_dim = t3.shape[3]
    t_mask_width = random.randint(5, 15)
    t_start = random.randint(0, t_dim - t_mask_width)
    t3[:, :, :, t_start:t_start+t_mask_width] = 0 # Zero out time band

    # Batch them up for efficiency
    batch = torch.cat([t1, t2, t3], dim=0) # Shape: [3, 3, 224, 224]

    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        # Average the 3 predictions
        avg_probs = torch.mean(probs, dim=0)
    
    return avg_probs

def analyze_audio(file_path, symptoms="", sensitivity_threshold=0.75):
    try:
        start_time = time.time()
        print(f"--- [START] Analysis Job (Robust Engine + TTA + Hybrid) ---")
        
        # 1. DECODE AUDIO
        command = [
            'ffmpeg', '-y', '-i', file_path,
            '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-t', '30', 
            '-threads', '1',  
            '-preset', 'ultrafast',
            '-loglevel', 'error', '-'
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0: raise Exception(f"FFmpeg Error: {err.decode()}")

        y_full = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 2. SMART SLICING
        CHUNK_SIZE = 80000 
        chunks = []
        for i in range(0, len(y_full), CHUNK_SIZE):
            chunk = y_full[i : i + CHUNK_SIZE]
            if len(chunk) > 16000: 
                chunks.append(chunk)

        print(f"--- [STEP 2] Sliced into {len(chunks)} chunks ---")

        # 3. INFERENCE LOOP WITH TTA
        final_diagnosis = "Inconclusive"
        highest_severity = -1
        valid_chunks = 0
        averaged_probs = {"Asthma": 0.0, "Normal": 0.0, "Pneumonia": 0.0}

        for idx, chunk in enumerate(chunks):
            # Energy Check
            rms = calculate_rms(chunk)
            if rms < 0.005: 
                print(f"   🔸 Chunk {idx+1}: Skipped (Too Silent - RMS: {rms:.4f})")
                continue

            img = generate_spectrogram(chunk)
            
            if ai_available and img:
                # 🚀 USE TTA PREDICTION
                input_tensor = preprocess_ai(img).unsqueeze(0)
                probs = predict_with_tta(model, input_tensor)
                
                p_asthma = float(probs[0])
                p_normal = float(probs[1])
                p_pneumonia = float(probs[2])

                winner_idx = torch.argmax(probs).item()
                chunk_diagnosis = CLASSES[winner_idx]
                winner_prob = float(probs[winner_idx])
                
                chunk_severity = SEVERITY_SCORE.get(chunk_diagnosis, 0)
                
                print(f"   🔹 Chunk {idx+1}: {chunk_diagnosis} (TTA Conf: {winner_prob:.2f})")
                valid_chunks += 1

                if chunk_severity > highest_severity:
                    highest_severity = chunk_severity
                    final_diagnosis = chunk_diagnosis
                    averaged_probs = {
                        "Asthma": p_asthma, 
                        "Normal": p_normal, 
                        "Pneumonia": p_pneumonia
                    }
                elif chunk_severity == highest_severity and winner_prob > averaged_probs[chunk_diagnosis]:
                    averaged_probs = {
                        "Asthma": p_asthma, 
                        "Normal": p_normal, 
                        "Pneumonia": p_pneumonia
                    }
            
            del img
            gc.collect()
        
        # 4. FINAL VERDICT LOGIC
        if valid_chunks == 0:
            final_diagnosis = "Inconclusive"
        elif final_diagnosis == "Inconclusive":
             final_diagnosis = "Normal"

        # 🏥 5. HYBRID DIAGNOSIS ADJUSTMENT
        # Adjust probabilities based on text symptoms
        symptom_risk_added = 0.0
        if symptoms:
            symptoms_lower = symptoms.lower()
            for key, bonus in SYMPTOM_RISK_BONUS.items():
                if key in symptoms_lower:
                    symptom_risk_added += bonus
            
            if symptom_risk_added > 0:
                print(f"   ⚠️ Symptoms Detected ('{symptoms}'): Adding +{symptom_risk_added:.2f} Risk Bonus")
                # Boost Pneumonia/Asthma scores, Decrease Normal score
                averaged_probs["Pneumonia"] = min(0.99, averaged_probs["Pneumonia"] + symptom_risk_added)
                averaged_probs["Asthma"] = min(0.99, averaged_probs["Asthma"] + (symptom_risk_added * 0.8))
                averaged_probs["Normal"] = max(0.01, averaged_probs["Normal"] - symptom_risk_added)
                
                # Re-Evaluate Winner after Symptom Boost
                new_winner = max(averaged_probs, key=averaged_probs.get)
                if new_winner != final_diagnosis and SEVERITY_SCORE[new_winner] > SEVERITY_SCORE[final_diagnosis]:
                    print(f"   🔄 HYBRID OVERRIDE: Changed {final_diagnosis} -> {new_winner} due to symptoms.")
                    final_diagnosis = new_winner

        elapsed = time.time() - start_time
        print(f"--- [SUCCESS] Verdict: {final_diagnosis} ({elapsed:.2f}s) ---")

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
            "risk_level_output": "High" if final_diagnosis != "Normal" else "Low"
        }

    except Exception as e:
        print(f"❌ ANALYZER ERROR: {e}")
        return {"status": "error", "message": str(e)}