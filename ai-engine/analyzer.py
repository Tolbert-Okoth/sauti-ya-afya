import matplotlib
matplotlib.use('Agg') # Force Headless Mode
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

import librosa
import librosa.display
import numpy as np
import base64
import io
import torch
import torch.quantization
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import gc # Garbage Collection

# ==========================================
# 🧠 AI MODEL LOADER
# ==========================================
print("🔄 Loading Phase 5 Native Brain...")
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
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    print(f"✅ AI Brain Loaded & Quantized")
    ai_available = True
except Exception as e:
    print(f"⚠️ AI Model missing: {e}")

preprocess_ai = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 🔬 ANALYZER FUNCTION (ULTRA-LIGHT)
# ==========================================
def analyze_audio(file_path, sensitivity_threshold=0.75):
    try:
        print("--- [STEP 1] Starting Analysis ---")
        
        # 1. LOAD AUDIO (Strict 5 Seconds Limit)
        # ⚠️ OPTIMIZATION: Reducing to 5s drastically lowers RAM usage
        y, sr = librosa.load(file_path, sr=16000, duration=5.0) 
        duration = float(librosa.get_duration(y=y, sr=sr))
        print(f"--- [STEP 2] Audio Loaded ({duration}s) ---")

        # 2. DSP CALCULATIONS
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        rms_h = float(np.mean(librosa.feature.rms(y=y_harmonic)))
        p_rms = librosa.feature.rms(y=y_percussive)[0]
        rms_variance = float(np.var(p_rms))
        
        total_energy = rms_h + float(np.mean(p_rms)) + 1e-9
        harmonic_ratio = rms_h / total_energy 
        spectral_flatness = librosa.feature.spectral_flatness(y=y_harmonic)[0]
        tonality_score = float(1.0 - np.mean(spectral_flatness)) 
        
        print("--- [STEP 3] DSP Complete ---")

        # 3. GENERATE SPECTROGRAM DATA (No Image)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 🧹 AGGRESSIVE CLEANUP
        del y, y_harmonic, y_percussive
        gc.collect()

        # ⚠️ DISABLED VISUALIZER TO PREVENT CRASH ⚠️
        # We send an empty string instead of the image.
        spectrogram_b64 = "" 
        
        print("--- [STEP 4] Spectrogram Data Ready (Image Skipped) ---")

        # 4. AI INFERENCE (Optimized)
        ai_diagnosis = "Unknown"
        ai_probs = {"Asthma": 0.0, "Normal": 0.0, "Pneumonia": 0.0}
        
        if ai_available:
            # Generate tiny temp image just for AI logic
            import matplotlib.pyplot as plt
            plt.figure(figsize=(2.24, 2.24), dpi=100) 
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            librosa.display.specshow(S_dB, sr=sr, fmax=4000)
            
            buf_ai = io.BytesIO()
            plt.savefig(buf_ai, format='png', bbox_inches='tight', pad_inches=0)
            buf_ai.seek(0)
            plt.close('all') # Force close
            
            # Predict
            img = Image.open(buf_ai).convert('RGB')
            input_tensor = preprocess_ai(img).unsqueeze(0)
            buf_ai.close()
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                ai_probs["Asthma"] = float(probs[0])
                ai_probs["Normal"] = float(probs[1])
                ai_probs["Pneumonia"] = float(probs[2])
                winner_idx = torch.argmax(probs).item()
                ai_diagnosis = CLASSES[winner_idx]
                
        print(f"--- [STEP 5] AI Complete: {ai_diagnosis} ---")

        # 5. DECISION LOGIC
        strictness = float(sensitivity_threshold)
        result_tag = "Normal / Vesicular"
        final_risk = "Low"
        detail = "Clear breath sounds detected."

        if tonality_score > (0.4 * strictness) and harmonic_ratio > 0.3:
            result_tag = "Wheeze (DSP)"
            final_risk = "Medium"
            detail = "Musical sounds detected (Narrow Airways)."
        elif rms_variance > (0.005 / strictness): 
            result_tag = "Crackles (DSP)"
            final_risk = "High"
            detail = "Fluid sounds detected."

        if ai_available:
            if ai_diagnosis == "Pneumonia" and ai_probs["Pneumonia"] > 0.7:
                result_tag = "Pneumonia Pattern (AI)"
                final_risk = "High"
                detail = f"AI identified pneumonia ({int(ai_probs['Pneumonia']*100)}%)."
            elif ai_diagnosis == "Asthma" and ai_probs["Asthma"] > 0.7:
                result_tag = "Asthma Pattern (AI)"
                if final_risk == "Low": final_risk = "Medium"
                detail = f"AI identified asthma ({int(ai_probs['Asthma']*100)}%)."

        return {
            "status": "success",
            "duration_seconds": round(duration, 2),
            "biomarkers": {
                "harmonic_ratio": round(harmonic_ratio, 3),
                "tonality_score": round(tonality_score, 3),
                "burstiness_variance": round(rms_variance, 5),
                "ai_diagnosis": ai_diagnosis,
                "prob_pneumonia": round(ai_probs["Pneumonia"], 3),
                "prob_asthma": round(ai_probs["Asthma"], 3),
                "prob_normal": round(ai_probs["Normal"], 3)
            },
            "visualizer": {
                "spectrogram_image": "" # Empty for now
            },
            "preliminary_assessment": result_tag,
            "risk_level_output": final_risk,
            "ai_explanation_override": detail
        }

    except Exception as e:
        print(f"❌ ANALYZER CRASHED: {e}")
        return {"status": "error", "message": str(e)}