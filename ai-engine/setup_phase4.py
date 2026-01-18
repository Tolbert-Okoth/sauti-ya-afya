import kagglehub
import shutil
import os
import pandas as pd

# ==========================================
# ⚙️ CONFIG (Phase 4)
# ==========================================
# We create a NEW folder so we don't mess up your Phase 3 demo
TARGET_ROOT = '../raw_data_phase4' 
DATASET_NAME = "vbookshelf/respiratory-sound-database"

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def setup_folders():
    if os.path.exists(TARGET_ROOT):
        shutil.rmtree(TARGET_ROOT)
    
    # Create 3 Classes
    os.makedirs(os.path.join(TARGET_ROOT, 'pneumonia'))
    os.makedirs(os.path.join(TARGET_ROOT, 'normal'))
    os.makedirs(os.path.join(TARGET_ROOT, 'asthma_wheeze')) # New Class
    
    print(f"📂 Created Phase 4 structure: {TARGET_ROOT}/[pneumonia, normal, asthma_wheeze]")

def find_file(root_dir, filename):
    for dirpath, _, filenames in os.walk(root_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

def organize_dataset_phase4(download_path):
    print("------------------------------------------------")
    print("🔍 Extracting Multi-Class Data...")
    
    # 1. Locate Diagnosis File
    csv_path = find_file(download_path, 'patient_diagnosis.csv')
    df = pd.read_csv(csv_path, header=None, names=['pid', 'diagnosis'])
    diag_map = pd.Series(df.diagnosis.values, index=df.pid).to_dict()
    
    # 2. Locate Audio
    audio_dir = find_file(download_path, '101_1b1_Al_sc_Meditron.wav')
    if audio_dir:
        audio_dir = os.path.dirname(audio_dir)
    
    files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    counts = {'p': 0, 'n': 0, 'a': 0}
    
    print(f"🚚 Sorting {len(files)} files into 3 buckets...")
    
    for filename in files:
        try:
            pid = int(filename.split('_')[0])
            condition = diag_map.get(pid, 'Unknown')
            src = os.path.join(audio_dir, filename)
            
            # --- PHASE 4 LOGIC ---
            if condition == 'Pneumonia':
                dst = os.path.join(TARGET_ROOT, 'pneumonia', filename)
                shutil.copy2(src, dst)
                counts['p'] += 1
                
            elif condition == 'Healthy':
                dst = os.path.join(TARGET_ROOT, 'normal', filename)
                shutil.copy2(src, dst)
                counts['n'] += 1
                
            elif condition in ['Asthma', 'Bronchiolitis']:
                # Group Obstructive diseases together
                dst = os.path.join(TARGET_ROOT, 'asthma_wheeze', filename)
                shutil.copy2(src, dst)
                counts['a'] += 1
                
        except Exception as e:
            pass

    print("------------------------------------------------")
    print("✅ PHASE 4 DATA READY")
    print(f"🦠 Pneumonia:      {counts['p']} files")
    print(f"🍃 Normal:         {counts['n']} files")
    print(f"🌬️ Asthma/Wheeze:  {counts['a']} files (New!)")
    print(f"📂 Location:       {os.path.abspath(TARGET_ROOT)}")

if __name__ == "__main__":
    print(f"⬇️  Checking Cache for: {DATASET_NAME}...")
    # This just returns the path to the already downloaded files
    path = kagglehub.dataset_download(DATASET_NAME)
    setup_folders()
    organize_dataset_phase4(path)