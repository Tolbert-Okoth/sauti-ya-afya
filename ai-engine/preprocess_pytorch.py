import os
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg') # 🔴 CRITICAL: Run in "Headless Mode" (No Window Popup)
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ CONFIG
# ==========================================
SOURCE_DIR = '../raw_data'       # Where setup_data.py is putting files
OUTPUT_DIR = '../dataset_pytorch' # Where we save the images
CATEGORIES = ['pneumonia', 'normal']
SAMPLE_RATE = 16000
DURATION = 5 # seconds (We crop audio to fixed length)

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def add_clinic_noise(y):
    """
    Simulates a cheap Android microphone in a busy clinic.
    Adds white noise (static) relative to the audio volume.
    """
    noise_amp = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y.astype('float64') + noise_amp * np.random.normal(size=y.shape[0])
    return y_noise

def save_spectrogram(y, sr, save_path):
    """Generates and saves the visual pattern of the lung sound"""
    # 1. Generate Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 2. Plotting Setup (No axes, no white borders)
    # 2.24 inches * 100 dpi = 224 pixels (Standard AI size)
    plt.figure(figsize=(2.24, 2.24), dpi=100) 
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    # 3. Render
    librosa.display.specshow(S_dB, sr=sr, fmax=4000) # We focus on 0-4kHz range
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting Data Pre-Processing...")
    
    # Verify Source Exists
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: Source folder '{SOURCE_DIR}' not found.")
        print("   Did the download finish?")
        exit()

    for category in CATEGORIES:
        path = os.path.join(SOURCE_DIR, category)
        target_path = os.path.join(OUTPUT_DIR, category)
        os.makedirs(target_path, exist_ok=True)
        
        # Check if category folder exists (e.g. if setup_data failed)
        if not os.path.exists(path):
            print(f"⚠️ Skipping {category}: Folder not found.")
            continue

        files = [f for f in os.listdir(path) if f.endswith('.wav')]
        print(f"📸 Processing {category}: {len(files)} audio files...")
        
        count = 0
        for filename in files:
            try:
                # 1. Load Audio (Force 5 seconds)
                file_path = os.path.join(path, filename)
                
                # Check file length first
                duration = librosa.get_duration(path=file_path)
                
                # Logic: If file is long, chop it into 5s chunks to get MORE data
                # For now, we just take the first 5 seconds to keep it simple
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                
                # Pad if too short (less than 5s)
                if len(y) < SAMPLE_RATE * DURATION:
                    padding = (SAMPLE_RATE * DURATION) - len(y)
                    y = np.pad(y, (0, padding), 'constant')
                
                # 2. Augment (Add Noise)
                y_noisy = add_clinic_noise(y)
                
                # 3. Save as Image
                image_name = filename.replace('.wav', '.png')
                save_spectrogram(y_noisy, sr, os.path.join(target_path, image_name))
                count += 1
                
                if count % 50 == 0:
                    print(f"   ... converted {count} images")

            except Exception as e:
                print(f"   ⚠️ Failed {filename}: {e}")

    print("------------------------------------------------")
    print(f"✅ PRE-PROCESSING COMPLETE")
    print(f"📂 Location: {os.path.abspath(OUTPUT_DIR)}")
    print("👉 Next Step: Run train_pytorch.py")