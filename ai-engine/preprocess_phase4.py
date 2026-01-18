import os
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# CONFIG (Pointing to the new folders)
SOURCE_DIR = '../raw_data_phase4'
OUTPUT_DIR = '../dataset_phase4' # New image folder
CATEGORIES = ['pneumonia', 'normal', 'asthma_wheeze'] # 3 Classes
SAMPLE_RATE = 16000
DURATION = 5

def add_clinic_noise(y):
    noise_amp = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y.astype('float64') + noise_amp * np.random.normal(size=y.shape[0])
    return y_noise

def save_spectrogram(y, sr, save_path):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(2.24, 2.24), dpi=100) 
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    librosa.display.specshow(S_dB, sr=sr, fmax=4000)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    print("🚀 Starting Phase 4 Pre-Processing (3 Classes)...")
    
    if not os.path.exists(SOURCE_DIR):
        print("❌ Error: Run setup_phase4.py first!")
        exit()

    for category in CATEGORIES:
        path = os.path.join(SOURCE_DIR, category)
        target_path = os.path.join(OUTPUT_DIR, category)
        os.makedirs(target_path, exist_ok=True)
        
        files = [f for f in os.listdir(path) if f.endswith('.wav')]
        print(f"📸 Processing {category}: {len(files)} files...")
        
        for filename in files:
            try:
                file_path = os.path.join(path, filename)
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                if len(y) < SAMPLE_RATE * DURATION:
                    padding = (SAMPLE_RATE * DURATION) - len(y)
                    y = np.pad(y, (0, padding), 'constant')
                
                y_noisy = add_clinic_noise(y)
                image_name = filename.replace('.wav', '.png')
                save_spectrogram(y_noisy, sr, os.path.join(target_path, image_name))
            except:
                pass

    print("✅ PHASE 4 IMAGES READY")