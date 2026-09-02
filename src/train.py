import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Config
DATA_DIR = "data/Indian Food Images/Indian Food Images"
MODEL_SAVE_PATH = "models/best_model.pth"
CLASSES_SAVE_PATH = "models/classes.json"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

def get_data_loaders():
    # Data Augmentation for training
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Transforms for validation
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load entire dataset
    full_dataset = datasets.ImageFolder(DATA_DIR)
    
    # Split sizes (80% train, 20% val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply different transforms
    train_dataset.dataset.transform = train_transforms
    
    # It's a bit tricky to apply different transforms to random_split subsets natively,
    # so we'll wrap it or just use the val_transform for the whole thing if we want to be strictly correct.
    # For a simple demo, we'll override the transform dynamically in a wrapper or just use a simpler approach.
    # Actually, a better way is to use a custom Dataset wrapper, but let's keep it simple:
    # We will just load it twice and use the indices from random_split.
    
    train_data = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    val_data = datasets.ImageFolder(DATA_DIR, transform=val_transforms)
    
    train_subset = torch.utils.data.Subset(train_data, train_dataset.indices)
    val_subset = torch.utils.data.Subset(val_data, val_dataset.indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Save classes
    classes = full_dataset.classes
    os.makedirs("models", exist_ok=True)
    with open(CLASSES_SAVE_PATH, 'w') as f:
        json.dump(classes, f)
        
    return train_loader, val_loader, classes

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    train_loader, val_loader, classes = get_data_loaders()
    num_classes = len(classes)
    
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze initial layers, unfreeze the last few blocks for fine-tuning
    for name, param in model.named_parameters():
        if "features." in name:
            layer_idx = int(name.split(".")[1])
            if layer_idx < 14:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            param.requires_grad = True
            
    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        all_preds = []
        all_labels = []
        all_outputs = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Calculate top-3 accuracy
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)
        top3_preds = np.argsort(all_outputs, axis=1)[:, -3:]
        top3_correct = sum([all_labels[i] in top3_preds[i] for i in range(len(all_labels))])
        top3_acc = top3_correct / len(all_labels)
        print(f"Val Top-3 Acc: {top3_acc:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("=> Saved new best model!")
            
    print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train_model()
