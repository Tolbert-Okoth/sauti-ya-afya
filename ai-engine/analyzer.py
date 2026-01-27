import os

# 🚀 FORCE SINGLE THREADING (Prevention)
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
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        ai_available = True
        print("✅ AI Model Loaded (Quantized)")
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
        print(f"--- [START] Analysis Job (Fast Torchaudio Mode) ---")
        
        # 1. TURBO FFMPEG
        command = [
            'ffmpeg', '-y', '-i', file_path,
            '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-t', '5', 
            '-threads', '1',  
            '-preset', 'ultrafast',
            '-loglevel', 'error', '-'
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg Error: {err.decode()}")

        y_np = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        print(f"--- [STEP 1] Audio Decoded ({len(y_np)} samples) ---")

        # 2. GENERATE SPECTROGRAM (Torchaudio TUNED to match Librosa)
        img = None
        try:
            waveform = torch.from_numpy(y_np).unsqueeze(0) 
            
            # 🛠️ TUNING: Match Librosa Defaults exactly
            mel_transform = T.MelSpectrogram(
                sample_rate=16000,
                n_mels=128,
                n_fft=2048,      # ✅ MATCHES LIBROSA DEFAULT
                hop_length=512,  # ✅ MATCHES LIBROSA DEFAULT
                power=2.0        # ✅ MATCHES LIBROSA DEFAULT
            )
            spectrogram = mel_transform(waveform)
            
            # Amplitude to DB
            spectrogram_db = T.AmplitudeToDB(stype='power', top_db=80)(spectrogram)
            
            # Normalize to 0-255
            s_min, s_max = spectrogram_db.min(), spectrogram_db.max()
            s_norm = 255 * (spectrogram_db - s_min) / (s_max - s_min)
            s_norm = s_norm.byte().squeeze(0).numpy()
            
            # Flip Y-axis (Standard spectrogram view)
            s_norm = np.flipud(s_norm)
            img = Image.fromarray(s_norm).convert('RGB')
            
            print("--- [STEP 2] Spectrogram Generated (Torchaudio Tuned) ---")

        except Exception as e:
            print(f"⚠️ SPECTROGRAM FAILED: {e}")
            img = Image.new('RGB', (224, 224), color='black')

        # 3. AI INFERENCE
        ai_diagnosis = "Unknown"
        ai_probs = {"Asthma": 0.0, "Normal": 0.0, "Pneumonia": 0.0}
        
        if ai_available and img:
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
        del y_np
        gc.collect()
        
        elapsed = time.time() - start_time
        print(f"--- [SUCCESS] Verdict: {ai_diagnosis} ({elapsed:.2f}s) ---")

        return {
            "status": "success",
            "biomarkers": {
                "tonality_score": 0.0,
                "ai_diagnosis": ai_diagnosis,
                "prob_pneumonia": round(ai_probs["Pneumonia"], 3),
                "prob_asthma": round(ai_probs["Asthma"], 3),
                "prob_normal": round(ai_probs["Normal"], 3)
            },
            "visualizer": { "spectrogram_image": "" },
            "preliminary_assessment": f"{ai_diagnosis} Pattern",
            "risk_level_output": "High" if ai_diagnosis != "Normal" else "Low"
        }

    except Exception as e:
        print(f"❌ ANALYZER ERROR: {e}")
        return {"status": "error", "message": str(e)}