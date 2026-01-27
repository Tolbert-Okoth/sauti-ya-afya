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
import soundfile as sf  # 🛠️ USING DIRECT SOUNDFILE LIBRARY

# 1. SETUP
DATA_DIR = '../raw_data_phase4' 
CLASSES = ['asthma_wheeze', 'normal', 'pneumonia'] 
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20 

device = torch.device("cpu")

# 2. PURE TORCH AUGMENTATION
class AudioAugmentor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, waveform):
        # A. Add Gaussian Noise
        if random.random() < 0.5:
            noise = torch.randn_like(waveform)
            waveform = waveform + noise * random.uniform(0.001, 0.015)

        # B. Change Volume
        if random.random() < 0.5:
            gain = random.uniform(0.5, 2.0)
            waveform = waveform * gain

        # C. Time Masking
        if random.random() < 0.3:
            mask_size = int(random.uniform(0.05, 0.1) * waveform.shape[1])
            start = random.randint(0, waveform.shape[1] - mask_size)
            waveform[:, start:start+mask_size] = 0

        return waveform

class RobustAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.samples = []
        self.augmentor = AudioAugmentor()
        
        print(f"🔍 Scanning directory: {os.path.abspath(root_dir)}")

        for label, class_name in enumerate(CLASSES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir): 
                print(f"⚠️ Warning: Folder not found: {class_dir}")
                continue
            
            files = os.listdir(class_dir)
            print(f"✅ Scanning {class_name} ({len(files)} items)...")
            
            count_added = 0
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.webm', '.m4a', '.flac', '.ogg', '.aac')):
                    self.samples.append((os.path.join(class_dir, file), label))
                    count_added += 1

            print(f"   👉 Added {count_added} valid audio files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            # 🛠️ FIX: Use SoundFile directly (Bypasses TorchCodec error)
            data, sr = sf.read(path)
            
            # Convert Numpy array to Torch Tensor
            waveform = torch.from_numpy(data).float()
            
            # Handle Mono vs Stereo (We need: Channels, Samples)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0) # (1, samples)
            else:
                waveform = waveform.t() # (channels, samples)

            # Resample if needed (to 16000Hz)
            if sr != 16000:
                resampler = T.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # Pad/Cut to 5 seconds
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
            print(f"Error loading {path}: {e}")
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
    print("\n❌ CRITICAL ERROR: No valid audio files were added.")
    exit()

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# Model Setup
print("🧠 Initializing Robust Brain (MobileNetV2)...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
print(f"🚀 Starting Training for {EPOCHS} Epochs...")
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

print("💾 Saving Robust Model...")
torch.save(model.state_dict(), 'sauti_mobilenet_v2_robust.pth')
print("✅ Done! New brain file: sauti_mobilenet_v2_robust.pth")