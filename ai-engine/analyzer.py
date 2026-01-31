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
    """
    try:
        if np.max(np.abs(y_chunk)) < 0.02: return 0

        y_sq = y_chunk[1:-1] ** 2
        y_cross = y_chunk[:-2] * y_chunk[2:]
        tkeo_energy = y_sq - y_cross
        tkeo_abs = np.abs(tkeo_energy)

        # 🛡️ HEAVY BREATH FILTER (8.0x)
        avg_energy = np.mean(tkeo_abs)
        thresh = max(avg_energy * 8.0, 0.0005) 
        
        block_size = 320 
        n_blocks = len(tkeo_abs) // block_size
        count = 0
        
        for i in range(n_blocks):
            if np.max(tkeo_abs[i*block_size : (i+1)*block_size]) > thresh:
                count += 1
        
        # 6. Artifact Guard (Machine Gun fire)
        if count > 35: return 0 
        return count
    except:
        return 0

def calculate_spectral_flux(y_chunk, sr=16000):
    """
    Calculates the rate of change of the power spectrum (Simulates Deltas).
    High Flux = Chaotic Change (Pneumonia/Crackles)
    Low Flux = Steady State (Asthma/Normal)
    """
    try:
        n_fft = 512
        hop_length = 256
        window = np.hanning(n_fft)
        
        n_frames = (len(y_chunk) - n_fft) // hop_length
        if n_frames < 2: return 0.0
        
        flux_sum = 0.0
        prev_spectrum = None
        
        for i in range(n_frames):
            start = i * hop_length
            frame = y_chunk[start : start + n_fft] * window
            spectrum = np.abs(np.fft.rfft(frame))
            spectrum = spectrum / (np.linalg.norm(spectrum) + 1e-9)
            
            if prev_spectrum is not None:
                flux = np.linalg.norm(spectrum - prev_spectrum)
                flux_sum += flux
            prev_spectrum = spectrum
            
        return flux_sum / n_frames
    except:
        return 0.0

def extract_physics_features_lite(y_chunk, sr=16000):
    try:
        zero_crossings = np.nonzero(np.diff(y_chunk > 0))[0]
        zcr = len(zero_crossings) / len(y_chunk)
        
        # FFT for Spectrum
        spectrum = np.abs(np.fft.rfft(y_chunk)) + 1e-10
        freqs = np.fft.rfftfreq(len(y_chunk), 1/sr)
        
        # ✨ Spectral Centroid
        spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum)

        # ✨ Spectral Bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * spectrum) / np.sum(spectrum))
        
        # ✨ Crest Factor
        rms = np.sqrt(np.mean(y_chunk**2))
        peak = np.max(np.abs(y_chunk))
        crest_factor = peak / (rms + 1e-9)
        
        # ✨ Spectral Flux
        spectral_flux = calculate_spectral_flux(y_chunk, sr)

        log_spectrum = np.log(spectrum)
        geom_mean = np.exp(np.mean(log_spectrum))
        arith_mean = np.mean(spectrum)
        spectral_flatness = geom_mean / arith_mean
        
        # Harmonic Ratio (Approximate as 1 - Flatness)
        # High Ratio (>0.7) = Tonal (Asthma)
        # Low Ratio (<0.5) = Noise (Normal/Pneumonia)
        harmonic_ratio = 1.0 - spectral_flatness
        
        if SCIPY_AVAILABLE:
            kurt = kurtosis(y_chunk)
            ent = entropy(np.abs(y_chunk) + 1e-10)
        else:
            kurt = 0.0; ent = 0.0
        
        mad = np.mean(np.abs(y_chunk - np.mean(y_chunk)))
        transients = count_transients_tkeo(y_chunk)
        
        return zcr, harmonic_ratio, spectral_flatness, kurt, ent, mad, transients, spectral_centroid, spectral_bandwidth, crest_factor, spectral_flux
    except:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0

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
        print(f"--- [START] Analysis Job (Vesicular Shield 2.0) ---")
        
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
        asthma_chunks_detected = 0
        detected_wheezes = 0

        for idx, chunk in enumerate(chunks):
            # 🛡️ SAFETY INIT (Prevents UnboundLocalError)
            chunk_severity = 1 
            
            rms = calculate_rms(chunk)
            if rms < 0.005: continue 

            img = generate_spectrogram(chunk)
            if ai_available and img:
                input_tensor = preprocess_ai(img).unsqueeze(0)
                probs = predict_with_tta(model, input_tensor)
                probs_list.append(probs) 
                
                zcr, harmonic_ratio, spectral_flatness, kurt, ent, mad, transients, centroid, bandwidth, crest_factor, flux = extract_physics_features_lite(chunk)
                
                winner_idx = torch.argmax(probs).item()
                chunk_diagnosis = CLASSES[winner_idx]
                winner_prob = float(probs[winner_idx])

                if chunk_diagnosis == "Normal" and winner_prob > 0.85:
                    pneumonia_pop_threshold = 8 
                else:
                    pneumonia_pop_threshold = 5 

                force_pneumonia = False
                force_asthma = False 
                
                # Guards
                is_friction_noise = (zcr > 0.18) 
                not_too_flat = (spectral_flatness < 0.42)
                
                # Crackle Definitions
                is_coarse_crackle = (0.10 < zcr <= 0.15) and (centroid > 1000) and (crest_factor > 12)
                is_fine_crackle = (zcr > 0.15) 

                # ✨ ASTHMA DEFINITION (STRICT)
                # Must be Tonal (High Harmonic Ratio / Low Flatness) and Steady (Low Flux)
                is_wheeze = (600 < centroid < 2500) and (bandwidth < 1000) and (spectral_flatness < 0.25) and (zcr > 0.08) and (transients < 2) and (flux < 0.5)

                # ✨ VESICULAR SHIELD 2.0 (STRICT NORMAL)
                # RMS < 0.04 (User Data: 0.03 + safety margin)
                # ZCR < 0.08 (User Data: Very smooth)
                # Harmonic Ratio < 0.5 (Non-musical, just air noise)
                is_vesicular = (zcr < 0.08) and (centroid < 650) and (rms < 0.04) and (transients == 0) and (harmonic_ratio < 0.5) and not is_wheeze
                
                # ✨ STABILITY CHECK (Silence/Calm)
                is_stable_normal = (rms < 0.03) and (transients == 0) and (not is_friction_noise)
                
                # ✨ GOLDEN ZONE (General Healthy)
                is_golden_normal = (0.05 < zcr < 0.15) and (0.30 < spectral_flatness < 0.60) and (transients < 4) and not is_wheeze

                if is_vesicular:
                     print(f"   ✨ VESICULAR SHIELD: Perfect Healthy Lung Pattern (ZCR={zcr:.2f}, RMS={rms:.3f}). Forcing Normal.")
                     chunk_diagnosis = "Normal"
                     chunk_severity = 1
                     probs_list[-1] = torch.tensor([0.01, 0.98, 0.01]) 

                elif is_stable_normal:
                     print(f"   ✨ STABILITY CHECK: Non-impulsive (RMS={rms:.3f}). Forcing Normal.")
                     chunk_diagnosis = "Normal"
                     chunk_severity = 1
                     probs_list[-1] = torch.tensor([0.02, 0.96, 0.02]) 

                elif is_wheeze and not is_friction_noise:
                     print(f"   🌬️ WHEEZE DETECTED: Tonal Sound (Flux={flux:.2f}, Flatness={spectral_flatness:.2f}). Forcing Asthma.")
                     force_asthma = True
                     detected_wheezes += 1

                elif is_golden_normal and not is_coarse_crackle:
                     print(f"   ✨ GOLDEN ZONE: Physics matches Healthy Breath. Forcing Normal.")
                     chunk_diagnosis = "Normal"
                     chunk_severity = 1
                     probs_list[-1] = torch.tensor([0.05, 0.90, 0.05]) 
                     total_transients += transients
                     
                elif is_friction_noise:
                     print(f"   ⚠️ ARTIFACT: Extreme Friction (ZCR={zcr:.2f}). Ignoring Pops.")

                else:
                    total_transients += transients

                    if transients >= pneumonia_pop_threshold and not_too_flat:
                        if is_fine_crackle:
                             print(f"   ⚠️ HIERARCHY: Fine Crackles ({transients} bursts) -> Forcing Pneumonia.")
                             force_pneumonia = True
                        elif is_coarse_crackle:
                             print(f"   ⚠️ HIERARCHY: Coarse Crackles ({transients} bursts, Flux={flux:.2f}) -> Forcing Pneumonia.")
                             force_pneumonia = True
                    
                    elif transients >= 4 and not_too_flat:
                         if is_fine_crackle or is_coarse_crackle:
                             print(f"   ⚠️ HIERARCHY: Moderate Crackles ({transients} bursts) -> Forcing Pneumonia.")
                             force_pneumonia = True

                if force_pneumonia:
                    chunk_diagnosis = "Pneumonia"
                    chunk_severity = 3
                    winner_prob = 0.90 
                    probs_list[-1] = torch.tensor([0.05, 0.05, 0.90]) 
                    physics_override = True
                    pneumonia_chunks_detected += 1
                
                elif force_asthma:
                    chunk_diagnosis = "Asthma"
                    chunk_severity = 2
                    winner_prob = 0.85
                    probs_list[-1] = torch.tensor([0.85, 0.05, 0.10])
                    physics_override = True
                    asthma_chunks_detected += 1

                else:
                    if chunk_diagnosis == "Asthma":
                        if is_wheeze: 
                             asthma_chunks_detected += 1
                             chunk_severity = 2 
                        
                        elif (transients > 4 or crest_factor > 15 or flux > 1.5) and centroid > 800 and not is_friction_noise:
                             print(f"   🛡️ VETO: AI=Asthma, but Sound is Chaotic (Flux={flux:.2f}, Crest={crest_factor:.1f}). Overriding to Pneumonia.")
                             chunk_diagnosis = "Pneumonia"
                             chunk_severity = 3
                             winner_prob = 0.85 
                             probs_list[-1] = torch.tensor([0.10, 0.05, 0.85]) 
                             physics_override = True
                             pneumonia_chunks_detected += 1

                        elif spectral_flatness > 0.35:
                             print(f"   ℹ️ INFO: AI=Asthma, but Sound is Flat. Likely Normal Breath.")
                             chunk_diagnosis = "Normal"
                             chunk_severity = 1
                             winner_prob = 0.60
                             probs_list[-1] = torch.tensor([0.20, 0.60, 0.20]) 
                        else:
                             if chunk_diagnosis != "Normal": chunk_severity = 2 
                    
                    elif chunk_diagnosis == "Pneumonia" and winner_prob > 0.60:
                        if spectral_flatness > 0.42:
                            print(f"   ℹ️ INFO: AI=Pneumonia, but Sound is Flat. Downgrading to Normal.")
                            chunk_diagnosis = "Normal"
                            chunk_severity = 1
                            probs_list[-1] = torch.tensor([0.10, 0.80, 0.10])
                        else:
                            pneumonia_chunks_detected += 1

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
                
                # HYBRID CONSENSUS
                p_pneumonia = averaged_probs["Pneumonia"]
                p_asthma = averaged_probs["Asthma"]
                
                if pneumonia_chunks_detected > asthma_chunks_detected:
                    final_diagnosis = "Pneumonia"
                    averaged_probs["Pneumonia"] = max(p_pneumonia, 0.80)
                
                elif asthma_chunks_detected > pneumonia_chunks_detected:
                    final_diagnosis = "Asthma"
                    averaged_probs["Asthma"] = max(p_asthma, 0.80)
                
                elif p_asthma > 0.5 or p_pneumonia > 0.5:
                     if abs(p_asthma - p_pneumonia) < 0.2:
                         print(f"   ⚖️ TIE BREAKER: Asthma ({p_asthma:.2f}) vs Pneumonia ({p_pneumonia:.2f})")
                         if detected_wheezes > 0:
                             print("   -> Wheezes found. Winner: Asthma.")
                             final_diagnosis = "Asthma"
                         elif total_transients > 5:
                             print("   -> Transients found. Winner: Pneumonia.")
                             final_diagnosis = "Pneumonia"
                         else:
                             final_diagnosis = max(averaged_probs, key=averaged_probs.get)
                     else:
                         final_diagnosis = max(averaged_probs, key=averaged_probs.get)

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
            
            # GLOBAL CHECK
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