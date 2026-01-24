import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

import librosa
import numpy as np
import base64
import io
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import gc 
import subprocess 
import os

# 🛑 LIMIT THREADS (Prevents CPU/RAM spikes on Render Free Tier)
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
        print("--- [STEP 1] Starting Analysis (5s Lite Mode + Real Image) ---")
        
        # 1. MANUAL FFMPEG (5 SECONDS LIMIT)
        # 5s is the "Sweet Spot": Full breath cycle, safe RAM usage.
        command = [
            'ffmpeg', '-y', '-i', file_path,
            '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-t', '5', 
            '-loglevel', 'error', '-'
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg Error: {err.decode()}")

        y = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        sr = 16000
        print(f"--- [STEP 2] Decoded {len(y)} samples ---")

        # 2. LIGHTWEIGHT DSP (No HPSS)
        # Calculates stats on the whole file to save RAM vs splitting it
        rms_energy = float(np.mean(librosa.feature.rms(y=y)))
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        tonality_score = float(1.0 - np.mean(spectral_flatness)) 
        
        print("--- [STEP 3] DSP Complete ---")

        # 3. SPECTROGRAM (Data Calculation)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Free audio memory immediately
        del y 
        gc.collect()

        # 4. GENERATE REAL SPECTROGRAM IMAGE (Memory Safe Way)
        # We use PIL instead of Matplotlib to avoid server crashes.
        
        # Normalize S_dB to 0-255 range for image conversion
        s_min, s_max = S_dB.min(), S_dB.max()
        s_norm = 255 * (S_dB - s_min) / (s_max - s_min)
        s_norm = s_norm.astype(np.uint8)
        
        # Flip Y-axis so low freq is at the bottom (standard view)
        img_data = np.flipud(s_norm)
        img = Image.fromarray(img_data).convert('RGB')
        
        # Save image to Base64 String in memory
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        spectrogram_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        print("--- [STEP 4] Real Spectrogram Generated ---")

        # 5. AI INFERENCE
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
                
        print(f"--- [STEP 5] AI Result: {ai_diagnosis} ---")

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
                # ✅ REAL IMAGE: Sending the actual generated spectrogram
                "spectrogram_image": f"data:image/png;base64,{spectrogram_b64}" 
            },
            "preliminary_assessment": f"{ai_diagnosis} Pattern",
            "risk_level_output": "High" if ai_diagnosis != "Normal" else "Low"
        }

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"status": "error", "message": str(e)}