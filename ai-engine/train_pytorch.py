import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import time
import copy

# ==========================================
# ⚙️ CONFIG
# ==========================================
# Note: We look one folder up (..) because dataset is outside ai-engine
DATA_DIR = '../dataset_pytorch' 
MODEL_SAVE_PATH = 'sauti_mobilenet_v1.pth'
BATCH_SIZE = 32
IMG_SIZE = 224   # Standard input size for MobileNet
EPOCHS = 10      # How many times to study the whole dataset
LEARNING_RATE = 0.001

# Detect if we have a GPU (Graphics Card) to speed this up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training Device: {device}")

# ==========================================
# 🛠️ DATA PIPELINE
# ==========================================
def load_data():
    print("📂 Loading Images...")
    
    # 1. Define Transforms (Augmentation)
    # We add random flips/rotations to make the AI robust
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # Normalize using ImageNet standards (required for MobileNet)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Folder
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Dataset folder '{DATA_DIR}' not found.")
        print("   Did you run preprocess_pytorch.py?")
        exit()

    full_dataset = datasets.ImageFolder(DATA_DIR)
    
    # 3. Split 80% Training / 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms separately
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✅ Data Loaded: {len(train_dataset)} Training | {len(val_dataset)} Validation")
    print(f"   Classes: {full_dataset.classes}")
    
    return train_loader, val_loader

# ==========================================
# 🧠 MODEL SETUP
# ==========================================
def build_model():
    print("🏗️  Building MobileNetV2...")
    
    # Load Pre-trained weights (Knowledge Transfer)
    model = models.mobilenet_v2(weights='DEFAULT')
    
    # Freeze the "Feature Extractor" layers
    # We only want to train the final decision layer, not retrain the whole brain
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace the Classifier Head
    # MobileNet outputs 1280 features. We condense that to 1 output (Pneumonia Probability)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 1),
        nn.Sigmoid() # Squishes output between 0 (Normal) and 1 (Pneumonia)
    )
    
    return model.to(device)

# ==========================================
# 🏋️ TRAINING LOOP
# ==========================================
def train_model(model, train_loader, val_loader):
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print("------------------------------------------------")
    print(f"🔥 Starting Training for {EPOCHS} Epochs...")
    
    start_time = time.time()

    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            # Reshape labels to match output shape [Batch, 1]
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        epoch_loss = running_loss / train_total
        epoch_acc = train_correct / train_total

        # --- VALIDATION PHASE ---
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                
                outputs = model(inputs)
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Save the best model so far
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print("------------------------------------------------")
    print(f"🏁 Training Complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 Best Validation Accuracy: {best_acc:.4f}")
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    train_dl, val_dl = load_data()
    
    # 2. Build Brain
    model = build_model()
    
    # 3. Train
    model = train_model(model, train_dl, val_dl)
    
    # 4. Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Model Saved to: {os.path.abspath(MODEL_SAVE_PATH)}")