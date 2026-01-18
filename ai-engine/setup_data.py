import kagglehub
import shutil
import os
import pandas as pd
import glob

# ==========================================
# ⚙️ CONFIG
# ==========================================
TARGET_ROOT = '../raw_data'  # We want this folder in the project root
DATASET_NAME = "vbookshelf/respiratory-sound-database"

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def setup_folders():
    """Creates the clean directory structure we need for training"""
    if os.path.exists(TARGET_ROOT):
        print(f"🧹 Cleaning existing folder: {TARGET_ROOT}...")
        shutil.rmtree(TARGET_ROOT)
    
    os.makedirs(os.path.join(TARGET_ROOT, 'pneumonia'))
    os.makedirs(os.path.join(TARGET_ROOT, 'normal'))
    print(f"📂 Created clean structure: {TARGET_ROOT}/[pneumonia, normal]")

def find_file(root_dir, filename):
    """Recursively searches for a file in the downloaded cache"""
    for dirpath, _, filenames in os.walk(root_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

def organize_dataset(download_path):
    print("------------------------------------------------")
    print("🔍 Organizing Dataset...")
    
    # 1. LOCATE DIAGNOSIS FILE
    # The dataset structure is messy (nested folders). We find the CSV first.
    csv_path = find_file(download_path, 'patient_diagnosis.csv')
    if not csv_path:
        raise FileNotFoundError("Could not find 'patient_diagnosis.csv' in the download.")
    
    # 2. READ DIAGNOSES
    # The CSV has no headers. Col 0 = Patient ID, Col 1 = Diagnosis
    print(f"📖 Reading diagnosis map from: {csv_path}")
    df = pd.read_csv(csv_path, header=None, names=['pid', 'diagnosis'])
    
    # Create a fast lookup dictionary: { 101: 'URTI', 102: 'Healthy', ... }
    diag_map = pd.Series(df.diagnosis.values, index=df.pid).to_dict()
    
    # 3. LOCATE AUDIO FILES
    # Audio files are usually in 'audio_and_txt_files' folder
    audio_dir = find_file(download_path, '101_1b1_Al_sc_Meditron.wav') # Search for first known file
    if audio_dir:
        audio_dir = os.path.dirname(audio_dir)
    else:
        raise FileNotFoundError("Could not find audio files folder.")

    print(f"🎧 Found audio files in: {audio_dir}")
    
    # 4. MOVE FILES
    files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    count_p = 0
    count_n = 0
    
    print(f"🚚 Processing {len(files)} audio files...")
    
    for filename in files:
        try:
            # Filename format: 101_1b1_Al_sc_Meditron.wav
            # Extract Patient ID (101)
            pid = int(filename.split('_')[0])
            condition = diag_map.get(pid, 'Unknown')
            
            src_path = os.path.join(audio_dir, filename)
            
            # Logic: We only want Pneumonia vs Healthy
            # We skip COPD, Asthma, etc. for this specific binary model
            if condition == 'Pneumonia':
                dst_path = os.path.join(TARGET_ROOT, 'pneumonia', filename)
                shutil.copy2(src_path, dst_path)
                count_p += 1
            elif condition == 'Healthy':
                dst_path = os.path.join(TARGET_ROOT, 'normal', filename)
                shutil.copy2(src_path, dst_path)
                count_n += 1
                
        except Exception as e:
            print(f"⚠️ Error moving {filename}: {e}")

    print("------------------------------------------------")
    print("✅ DATA SETUP COMPLETE")
    print(f"🦠 Pneumonia Samples: {count_p}")
    print(f"🍃 Normal Samples:    {count_n}")
    print(f"📍 Location:          {os.path.abspath(TARGET_ROOT)}")

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"⬇️  Downloading dataset: {DATASET_NAME}...")
    try:
        # kagglehub handles the caching automatically
        path = kagglehub.dataset_download(DATASET_NAME)
        print(f"📦 Downloaded to cache: {path}")
        
        setup_folders()
        organize_dataset(path)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 TIP: Did you set your KAGGLE_API_TOKEN?")