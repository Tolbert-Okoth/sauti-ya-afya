import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchaudio
import torchaudio.transforms as T
from PIL import Image
import numpy as np
import random
import soundfile as sf  

# 1. SETUP
DATA_DIR = '../raw_data_phase4' 
CLASSES = ['asthma_wheeze', 'normal', 'pneumonia'] 
IMG_SIZE = (224, 224)
BATCH_SIZE = 8 # ⚠️ Reduced Batch Size to update weights more often
EPOCHS = 25    # ⚠️ Increased Epochs to force learning

device = torch.device("cpu")

# 2. PURE TORCH AUGMENTATION
class AudioAugmentor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, waveform):
        # A. Add Gaussian Noise (Reduced probability)
        if random.random() < 0.3: 
            noise = torch.randn_like(waveform)
            waveform = waveform + noise * random.uniform(0.001, 0.01)

        # B. Change Volume
        if random.random() < 0.5:
            gain = random.uniform(0.7, 1.3) # Tighter range
            waveform = waveform * gain

        return waveform

class RobustAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.samples = []
        self.augmentor = AudioAugmentor()
        self.class_counts = [0] * len(CLASSES) # Track counts for weighting
        
        print(f"🔍 Scanning directory: {os.path.abspath(root_dir)}")

        for label, class_name in enumerate(CLASSES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir): 
                continue
            
            files = os.listdir(class_dir)
            count_added = 0
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.webm', '.m4a', '.flac', '.ogg', '.aac')):
                    self.samples.append((os.path.join(class_dir, file), label))
                    count_added += 1
            
            self.class_counts[label] = count_added
            print(f"✅ {class_name}: {count_added} files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            data, sr = sf.read(path)
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
            else: waveform = waveform.t()

            if sr != 16000:
                resampler = T.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            target_len = 80000
            if waveform.shape[1] < target_len:
                pad_amount = target_len - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            else:
                waveform = waveform[:, :target_len]

            if self.augment:
                waveform = self.augmentor(waveform)
            
            mel_transform = T.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512)
            spectrogram = mel_transform(waveform)
            spectrogram_db = T.AmplitudeToDB(stype='power', top_db=80)(spectrogram)
            
            s_min, s_max = spectrogram_db.min(), spectrogram_db.max()
            s_norm = 255 * (spectrogram_db - s_min) / (s_max - s_min)
            s_norm = s_norm.byte().squeeze(0).numpy()
            s_norm = np.flipud(s_norm)
            img = Image.fromarray(s_norm).convert('RGB')

            if self.transform:
                img = self.transform(img)

            return img, label
        except Exception as e:
            return torch.zeros((3, 224, 224)), label

# Transforms
data_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Data
print(f"📂 Loading Dataset from {DATA_DIR}...")
full_dataset = RobustAudioDataset(DATA_DIR, transform=data_transform, augment=True)

if len(full_dataset) == 0:
    print("❌ Error: No data found.")
    exit()

# ⚖️ CALCULATE CLASS WEIGHTS (The Logic Fix)
# Weights = Total / (Num_Classes * Count)
# This makes rare classes (Asthma) heavier.
total_samples = len(full_dataset)
class_counts = full_dataset.class_counts
class_weights = []

print("\n⚖️ CALCULATING FAIRNESS WEIGHTS:")
for i, count in enumerate(class_counts):
    if count > 0:
        weight = total_samples / (len(CLASSES) * count)
        class_weights.append(weight)
        print(f"   🔹 {CLASSES[i]}: Weight = {weight:.2f}x multiplier")
    else:
        class_weights.append(1.0)

weights_tensor = torch.FloatTensor(class_weights).to(device)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# Model Setup
print("\n🧠 Initializing Weighted Brain...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
model = model.to(device)

# 🛠️ APPLY WEIGHTS TO LOSS FUNCTION
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
print(f"🚀 Starting Weighted Training for {EPOCHS} Epochs...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Acc: {acc:.2f}%")

print("💾 Saving Balanced Model...")
torch.save(model.state_dict(), 'sauti_mobilenet_v2_robust.pth')
print("✅ Done!")