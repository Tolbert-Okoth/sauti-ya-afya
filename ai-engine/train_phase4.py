import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import copy

# CONFIG
DATA_DIR = '../dataset_phase4'
MODEL_SAVE_PATH = 'sauti_mobilenet_v2_multiclass.pth'
NUM_CLASSES = 3 # Pneumonia, Normal, Asthma
BATCH_SIZE = 32
EPOCHS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_phase4():
    print(f"🚀 Training Phase 4 Model ({NUM_CLASSES} Classes) on {device}...")
    
    # 1. Load Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    # Important: Print which index belongs to which class!
    print(f"🏷️  Class Mapping: {dataset.class_to_idx}")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # 2. Build Model
    model = models.mobilenet_v2(weights='DEFAULT')
    
    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Multi-Class Head
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, NUM_CLASSES) 
        # Note: No Softmax here! CrossEntropyLoss expects raw logits.
    )
    
    model = model.to(device)
    
    # 3. Setup Training
    criterion = nn.CrossEntropyLoss() # Optimized for Multi-Class
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # 4. Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) # Returns 3 numbers per image
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1) # Pick the highest number
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {epoch_acc:.4f}")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        print(f"   Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"🏆 Best Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Saved: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_phase4()