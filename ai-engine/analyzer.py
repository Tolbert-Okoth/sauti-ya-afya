import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

import librosa
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import gc 
import subprocess 
import os
import time

# 🛑 LIMIT THREADS (Crucial for Render Free Tier)
torch.set_num_threads(1) 

print("🔄 Loading Lite Brain...")
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None) 
CLASSES = ['Asthma', 'Normal', 'Pneumonia']
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(model.last_channel, 3)
)
MODEL_PATH = 'sauti_mobilenet_v2_multiclass.pth'
ai_available = False

try:
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        # Quantize to reduce RAM by ~50%
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        ai_available = True
        print("✅ AI Model Loaded (Quantized)")
    else:
        print("⚠️ Model file not found.")
except Exception as e:
    print(f"⚠️ AI Load Error: {e}")

preprocess_ai = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def analyze_audio(file_path, sensitivity_threshold=0.75):
    try:
        start_time = time.time()
        print(f"--- [START] Analysis Job Started ---")
        
        # 1. TURBO FFMPEG (The Fix for Free Tier Deadlocks)
        # -threads 1: Forces single-threaded processing to match the 0.1 vCPU limit.
        # -preset ultrafast: Sacrifices tiny bits of quality for maximum speed.
        command = [
            'ffmpeg', '-y', '-i', file_path,
            '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-t', '5', 
            '-threads', '1',  # 🚀 CRITICAL FIX
            '-preset', 'ultrafast', # 🚀 SPEED BOOST
            '-loglevel', 'error', '-'
        ]

        print("--- [STEP 1] Running Turbo FFmpeg... ---")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg Error: {err.decode()}")

        # Load Audio
        y = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        sr = 16000
        print(f"--- [STEP 2] Decoded {len(y)} samples ({time.time() - start_time:.2f}s) ---")

        # 2. LIGHTWEIGHT DSP
        rms_energy = float(np.mean(librosa.feature.rms(y=y)))
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        tonality_score = float(1.0 - np.mean(spectral_flatness)) 
        
        print("--- [STEP 3] DSP Complete ---")

        # 3. INTERNAL SPECTROGRAM (Aggressive Memory Cleanup)
        
        # A. Raw Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        del y 
        gc.collect() 

        # B. DB Conversion
        S_dB = librosa.power_to_db(S, ref=np.max)
        del S
        gc.collect()

        # C. Normalization
        s_min, s_max = S_dB.min(), S_dB.max()
        s_norm = 255 * (S_dB - s_min) / (s_max - s_min)
        s_norm = s_norm.astype(np.uint8)
        del S_dB
        gc.collect()

        # D. Image Creation (Internal Use Only)
        img_data = np.flipud(s_norm)
        img = Image.fromarray(img_data).convert('RGB')
        
        del s_norm
        del img_data
        gc.collect()

        print("--- [STEP 4] Internal Image Created (Not Saving to Output) ---")

        # 4. AI INFERENCE
        ai_diagnosis = "Unknown"
        ai_probs = {"Asthma": 0.0, "Normal": 0.0, "Pneumonia": 0.0}
        
        if ai_available:
            with torch.no_grad():
                input_tensor = preprocess_ai(img).unsqueeze(0)
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                ai_probs["Asthma"] = float(probs[0])
                ai_probs["Normal"] = float(probs[1])
                ai_probs["Pneumonia"] = float(probs[2])
                winner_idx = torch.argmax(probs).item()
                ai_diagnosis = CLASSES[winner_idx]
        
        # 🗑️ FINAL CLEANUP
        del img
        gc.collect()
        
        elapsed = time.time() - start_time
        print(f"--- [STEP 5] AI Result: {ai_diagnosis} (Total Time: {elapsed:.2f}s) ---")

        return {
            "status": "success",
            "biomarkers": {
                "tonality_score": round(tonality_score, 3),
                "ai_diagnosis": ai_diagnosis,
                "prob_pneumonia": round(ai_probs["Pneumonia"], 3),
                "prob_asthma": round(ai_probs["Asthma"], 3),
                "prob_normal": round(ai_probs["Normal"], 3)
            },
            "visualizer": { 
                # ✅ EMPTY STRING: Verdict Only Mode
                "spectrogram_image": "" 
            },
            "preliminary_assessment": f"{ai_diagnosis} Pattern",
            "risk_level_output": "High" if ai_diagnosis != "Normal" else "Low"
        }

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"status": "error", "message": str(e)}