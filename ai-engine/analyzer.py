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

# 🛑 LIMIT TORCH THREADS
torch.set_num_threads(1) 

print("🔄 Loading Robust Brain...")
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None) 

# 🛠️ CLASS ORDER
CLASSES = ['Asthma', 'Normal', 'Pneumonia']

# Map diagnosis to severity score for comparison
SEVERITY_SCORE = {'Pneumonia': 3, 'Asthma': 2, 'Normal': 1, 'Unknown': 0}

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
        
        # 🛠️ FIXED: Standardized Spectrogram Settings
        mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_mels=128, n_fft=2048, hop_length=512, power=2.0
        )
        spectrogram = mel_transform(waveform)
        spectrogram_db = T.AmplitudeToDB(stype='power', top_db=80)(spectrogram)
        
        # 🛠️ CRITICAL FIX: FIXED NORMALIZATION
        # Instead of stretching min/max dynamically, we clamp to a standard range.
        # This prevents "Silence" from looking like "Static/Pneumonia".
        s_norm = (spectrogram_db + 80) / 80.0  # Map -80dB..0dB to 0..1
        s_norm = torch.clamp(s_norm, 0, 1)     # Clip anything outside
        s_norm = s_norm.byte().squeeze(0).numpy() * 255 # Convert to 0-255
        
        # Flip Y-axis (Low freq at bottom)
        s_norm = np.flipud(s_norm)
        return Image.fromarray(s_norm.astype(np.uint8)).convert('RGB')
    except:
        return None

def analyze_audio(file_path, sensitivity_threshold=0.75):
    try:
        start_time = time.time()
        print(f"--- [START] Analysis Job (Robust Engine) ---")
        
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
        CHUNK_SIZE = 80000 # 5 seconds
        chunks = []
        for i in range(0, len(y_full), CHUNK_SIZE):
            chunk = y_full[i : i + CHUNK_SIZE]
            if len(chunk) > 16000: # Only process if > 1 second
                chunks.append(chunk)

        print(f"--- [STEP 2] Sliced into {len(chunks)} chunks ---")

        # 3. INFERENCE LOOP
        final_diagnosis = "Inconclusive"
        highest_severity = -1
        valid_chunks = 0
        averaged_probs = {"Asthma": 0.0, "Normal": 0.0, "Pneumonia": 0.0}

        for idx, chunk in enumerate(chunks):
            # 🛠️ CHECK ENERGY: Skip if too quiet (RMS < 0.005)
            rms = calculate_rms(chunk)
            if rms < 0.005: 
                print(f"   🔸 Chunk {idx+1}: Skipped (Too Silent - RMS: {rms:.4f})")
                continue

            img = generate_spectrogram(chunk)
            
            if ai_available and img:
                with torch.no_grad():
                    input_tensor = preprocess_ai(img).unsqueeze(0)
                    outputs = model(input_tensor)
                    probs = F.softmax(outputs, dim=1)[0]
                    
                    winner_idx = torch.argmax(probs).item()
                    winner_prob = float(probs[winner_idx])
                    chunk_diagnosis = CLASSES[winner_idx]
                    
                    # Store probabilities if this is the "worst" chunk so far
                    chunk_severity = SEVERITY_SCORE.get(chunk_diagnosis, 0)
                    
                    print(f"   🔹 Chunk {idx+1}: {chunk_diagnosis} (Conf: {winner_prob:.2f})")
                    valid_chunks += 1

                    # Logic: Prioritize Disease Detection
                    if chunk_severity > highest_severity:
                        highest_severity = chunk_severity
                        final_diagnosis = chunk_diagnosis
                        averaged_probs = {
                            "Asthma": float(probs[0]), 
                            "Normal": float(probs[1]), 
                            "Pneumonia": float(probs[2])
                        }
                    # Tie-breaker: If severity is same, take higher confidence
                    elif chunk_severity == highest_severity and winner_prob > averaged_probs[chunk_diagnosis]:
                        averaged_probs = {
                            "Asthma": float(probs[0]), 
                            "Normal": float(probs[1]), 
                            "Pneumonia": float(probs[2])
                        }
            
            del img
            gc.collect()
        
        # 4. FINAL VERDICT LOGIC
        if valid_chunks == 0:
            final_diagnosis = "Inconclusive" # Audio was pure silence
        elif final_diagnosis == "Inconclusive":
             final_diagnosis = "Normal" # Default to normal if audio exists but no disease found

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