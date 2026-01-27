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

print("🔄 Loading Lite Brain...")
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None) 
CLASSES = ['Asthma', 'Normal', 'Pneumonia']
# Map diagnosis to severity score for comparison
SEVERITY_SCORE = {'Pneumonia': 3, 'Asthma': 2, 'Normal': 1, 'Unknown': 0}

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
        print(f"--- [START] Analysis Job (Smart Slicing 30s) ---")
        
        # 1. EXTENDED RECORDING TIME (30 Seconds)
        command = [
            'ffmpeg', '-y', '-i', file_path,
            '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-t', '30', # 🕒 NOW 30 SECONDS
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
        # We need 5-second chunks (16000 * 5 = 80000 samples)
        CHUNK_SIZE = 80000
        total_samples = len(y_full)
        
        # If recording is short (<5s), just take what we have (padded by model resizing)
        # If recording is long, split it.
        chunks = []
        if total_samples <= CHUNK_SIZE:
            chunks.append(y_full)
        else:
            # Create overlapping chunks if possible, or just sequential
            for i in range(0, total_samples, CHUNK_SIZE):
                chunk = y_full[i : i + CHUNK_SIZE]
                if len(chunk) > 16000: # Only process if chunk is > 1 second
                    chunks.append(chunk)

        print(f"--- [STEP 2] Sliced into {len(chunks)} chunks for analysis ---")

        # 3. MULTI-PASS INFERENCE
        final_diagnosis = "Normal"
        highest_severity = 0
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
                    
                    # Determine winner for this chunk
                    winner_idx = torch.argmax(probs).item()
                    chunk_diagnosis = CLASSES[winner_idx]
                    chunk_severity = SEVERITY_SCORE.get(chunk_diagnosis, 0)

                    print(f"   🔹 Chunk {idx+1}: {chunk_diagnosis} (Pn: {p_pneumonia:.2f})")

                    #LOGIC: Taking the HIGHEST RISK chunk
                    if chunk_severity > highest_severity:
                        highest_severity = chunk_severity
                        final_diagnosis = chunk_diagnosis
                        # Save the probabilities of the "Worst Case" chunk
                        averaged_probs = {
                            "Asthma": p_asthma, 
                            "Normal": p_normal, 
                            "Pneumonia": p_pneumonia
                        }
            
            # Cleanup per chunk to save RAM
            del img
            gc.collect()

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