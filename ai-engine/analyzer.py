import os

# 🚀 FORCE SINGLE THREADING
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchaudio # 🚀 FAST ENGINE
import torchaudio.transforms as T
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import gc 
import subprocess 
import time

# 🛑 LIMIT TORCH THREADS
torch.set_num_threads(1) 

print("🔄 Loading Robust Brain...")
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None) 

# 🛠️ IMPORTANT: Must match Training Class Order (Alphabetical)
# 0: asthma_wheeze, 1: normal, 2: pneumonia
CLASSES = ['Asthma', 'Normal', 'Pneumonia']

# Map diagnosis to severity score for comparison
SEVERITY_SCORE = {'Pneumonia': 3, 'Asthma': 2, 'Normal': 1, 'Unknown': 0}

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(model.last_channel, 3)
)

# ✅ NEW ROBUST MODEL
MODEL_PATH = 'sauti_mobilenet_v2_robust.pth'
ai_available = False

try:
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        # Quantize for speed
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

def generate_spectrogram(y_chunk, sr=16000):
    """Helper function to generate spectrogram for a single chunk"""
    try:
        waveform = torch.from_numpy(y_chunk).unsqueeze(0)
        # Tuned to match Librosa
        mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_mels=128, n_fft=2048, hop_length=512, power=2.0
        )
        spectrogram = mel_transform(waveform)
        spectrogram_db = T.AmplitudeToDB(stype='power', top_db=80)(spectrogram)
        
        s_min, s_max = spectrogram_db.min(), spectrogram_db.max()
        s_norm = 255 * (spectrogram_db - s_min) / (s_max - s_min)
        s_norm = s_norm.byte().squeeze(0).numpy()
        s_norm = np.flipud(s_norm)
        return Image.fromarray(s_norm).convert('RGB')
    except:
        return None

def analyze_audio(file_path, sensitivity_threshold=0.75):
    try:
        start_time = time.time()
        print(f"--- [START] Analysis Job (Robust Engine) ---")
        
        # 1. EXTENDED RECORDING TIME (30 Seconds)
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
        print(f"--- [STEP 1] Audio Decoded ({len(y_full)} samples) ---")

        # 2. SMART SLICING LOGIC
        CHUNK_SIZE = 80000 # 5 seconds
        total_samples = len(y_full)
        chunks = []
        
        if total_samples <= CHUNK_SIZE:
            chunks.append(y_full)
        else:
            for i in range(0, total_samples, CHUNK_SIZE):
                chunk = y_full[i : i + CHUNK_SIZE]
                if len(chunk) > 16000: # Only process if > 1 second
                    chunks.append(chunk)

        print(f"--- [STEP 2] Sliced into {len(chunks)} chunks ---")

        # 3. MULTI-PASS INFERENCE
        final_diagnosis = "Inconclusive"
        highest_severity = -1
        averaged_probs = {"Asthma": 0.0, "Normal": 0.0, "Pneumonia": 0.0}

        for idx, chunk in enumerate(chunks):
            img = generate_spectrogram(chunk)
            
            if ai_available and img:
                with torch.no_grad():
                    input_tensor = preprocess_ai(img).unsqueeze(0)
                    outputs = model(input_tensor)
                    probs = F.softmax(outputs, dim=1)[0]
                    
                    p_asthma = float(probs[0])
                    p_normal = float(probs[1])
                    p_pneumonia = float(probs[2])
                    
                    winner_idx = torch.argmax(probs).item()
                    chunk_diagnosis = CLASSES[winner_idx]
                    winner_prob = float(probs[winner_idx])
                    
                    # SAFETY CHECK: Ignore low confidence guesses
                    if winner_prob < 0.50:
                        chunk_diagnosis = "Inconclusive"
                        chunk_severity = 0
                    else:
                        chunk_severity = SEVERITY_SCORE.get(chunk_diagnosis, 0)

                    print(f"   🔹 Chunk {idx+1}: {chunk_diagnosis} (Conf: {winner_prob:.2f})")

                    # Logic: Take the HIGHEST SEVERITY chunk found
                    if chunk_severity > highest_severity:
                        highest_severity = chunk_severity
                        final_diagnosis = chunk_diagnosis
                        averaged_probs = {
                            "Asthma": p_asthma, 
                            "Normal": p_normal, 
                            "Pneumonia": p_pneumonia
                        }
            
            del img
            gc.collect()
        
        # Fallback if everything was inconclusive
        if final_diagnosis == "Inconclusive":
             # Default to Normal if nothing bad was found, but mark probabilities low
             final_diagnosis = "Normal" 

        elapsed = time.time() - start_time
        print(f"--- [SUCCESS] Final Verdict: {final_diagnosis} ({elapsed:.2f}s) ---")

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