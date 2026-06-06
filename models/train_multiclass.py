"""
Multi-class Deepfake & AI-Generated Image Detector
Detects: Real (0), Deepfake (1), AI-Generated (2)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm
import json
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# Model Architecture
# ============================================================================

class SqueezeExcitationBlock(nn.Module):
    """Channel Attention mechanism"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class ResidualBlock(nn.Module):
    """Residual block with SE attention"""
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitationBlock(out_channels, reduction)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class MultiClassDetector(nn.Module):
    """Multi-class detector: Real (0), Deepfake (1), AI-Generated (2)"""
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Multi-scale pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classification head - adapted for 3 classes
        self.fc1 = nn.Linear(512 * 2, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn_fc3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)
        
        # Output: 3 classes (no sigmoid - use softmax in loss)
        self.fc_out = nn.Linear(256, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)
        x = torch.cat([avg_feat, max_feat], dim=1)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc_out(x)
        return x  # Returns logits for 3 classes

# ============================================================================
# Dataset
# ============================================================================

class MultiClassDataset(Dataset):
    """Multi-class dataset loader"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Class labels: Real=0, Deepfake=1, AIGenerated=2
        class_mapping = {
            'Real': 0,
            'Deepfake': 1,
            'AIGenerated': 2
        }
        
        for class_name, class_label in class_mapping.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        self.samples.append((os.path.join(class_dir, img_file), class_label, class_name))
        
        print(f"Loaded {len(self.samples)} samples")
        for class_name, class_label in class_mapping.items():
            count = sum(1 for _, label, _ in self.samples if label == class_label)
            print(f"  {class_name}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (299, 299))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

class SubsetDataset(Dataset):
    """Wrapper for PyTorch Subset to apply split-specific transforms safely"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        # Resolve index in the original dataset
        img_path, label, _ = self.subset.dataset.samples[self.subset.indices[idx]]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (299, 299))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)
        
    def __len__(self):
        return len(self.subset)

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # For accuracy calculation
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'preds': all_preds,
        'labels': all_labels
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--data_dir', default='../DATA', help='Path to dataset directory')
    args_parsed = parser.parse_args()

    print("=" * 70)
    print("=== Multi-Class Detector Training: Real vs Deepfake vs AI-Generated ===")
    print("=" * 70)
    
    # Data paths
    data_path = args_parsed.data_dir
    model_path = 'multiclass_detector.pth'
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load data
    if os.path.exists(data_path):
        print(f"\n[Data] Loading data from {data_path}")
        dataset = MultiClassDataset(data_path, transform=val_transform)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_sub, val_sub = random_split(dataset, [train_size, val_size])
        
        train_dataset = SubsetDataset(train_sub, transform=train_transform)
        val_dataset = SubsetDataset(val_sub, transform=val_transform)
        
        batch_size = args_parsed.batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
    else:
        print(f"[Error] Data directory not found: {data_path}")
        return
    # Model
    num_classes = 3
    model = MultiClassDetector(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[OK] Model created with {total_params:,} parameters")
    
    # Load existing checkpoint to resume training if it exists
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"[OK] Loaded existing model weights from {model_path} to resume training")
        except Exception as e:
            print(f"[Warning] Could not load existing weights: {e}. Starting from scratch.")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop configurations
    total_epochs = args_parsed.epochs
    patience = args_parsed.epochs  # Set patience to epochs to effectively disable early stopping
    patience_counter = 0
    best_accuracy = 0
    start_epoch = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    # Load training history and best accuracy to resume cleanly
    history_path = 'multiclass_training_history.json'
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            start_epoch = len(history.get('train_loss', []))
            print(f"[OK] Loaded training history. Resuming from epoch {start_epoch + 1}")
        except Exception as e:
            print(f"[Warning] Could not load history: {e}")
            
    info_path = 'multiclass_model_info.json'
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
            best_accuracy = info.get('best_accuracy', 0.0)
            print(f"[OK] Loaded best accuracy checkpoint: {best_accuracy:.4f}")
        except Exception as e:
            pass
            
    print(f"\n[Start] Starting training ({total_epochs} epochs, early stopping disabled, start_epoch={start_epoch})...\n")
    
    for epoch in range(start_epoch, total_epochs):
        print(f"--- Epoch {epoch+1}/{total_epochs} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        scheduler.step(val_metrics['accuracy'])
        
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"[OK] Best model saved (Acc: {best_accuracy:.4f})")
        else:
            patience_counter += 1
            print(f"[Info] No improvement ({patience_counter}/{patience})")
        
        # Save training history and info at each epoch to enable resumeability
        try:
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            model_info = {
                "architecture": "Multi-Class Detector (Real/Deepfake/AI-Generated)",
                "total_parameters": total_params,
                "input_shape": [3, 299, 299],
                "output": "3-class (0=Real, 1=Deepfake, 2=AIGenerated)",
                "framework": "PyTorch",
                "device_trained": str(device),
                "best_accuracy": float(best_accuracy),
                "timestamp": datetime.now().isoformat()
            }
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
        except Exception as se:
            print(f"[Warning] Could not auto-save epoch stats: {se}")
        
        # Early stopping check disabled to train full 50 epochs
        if patience_counter >= patience:
            print(f"\n[Stop] Early stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Save results
    print("\n" + "=" * 70)
    print(f"[OK] Training completed! Best Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("=" * 70)
    
    # Save training history
    with open('multiclass_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("[OK] Training history saved")
    
    # Model info
    model_info = {
        "architecture": "Multi-Class Detector (Real/Deepfake/AI-Generated)",
        "total_parameters": total_params,
        "input_shape": [3, 299, 299],
        "output": "3-class (0=Real, 1=Deepfake, 2=AIGenerated)",
        "framework": "PyTorch",
        "device_trained": str(device),
        "best_accuracy": float(best_accuracy),
        "timestamp": datetime.now().isoformat()
    }
    
    with open('multiclass_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("[OK] Model info saved")
    
    print("\n[OK] All training artifacts saved to models/")

if __name__ == "__main__":
    main()
