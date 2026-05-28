"""
Verifixia - Video Dataset Local Frame-Extraction & Model Training Pipeline
========================================================================
1. Scans DATA/Video for local .mp4 video files.
2. Identifies Real (single actor prefix) vs. Fake (double actor face swap prefix).
3. Slices and extracts face-cropped frames from all local videos dynamically.
4. Performs 30-epoch training on the extracted facial frames under heavy augmentations.
5. Saves model weights to models/xception_deepfake.pth and serializes history.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import cv2
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ──────────────────────────── Model Architecture ─────────────────────────────
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

class DeepfakeDetector(nn.Module):
    """Deepfake Detection Model with Residual Blocks & SE Attention"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Linear(512 * 2, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(512, 256)
        self.bn_fc3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
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
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
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
        return self.sigmoid(x)

# ──────────────────────────── In-Memory Dataset Loader ─────────────────────────
class LocalProxyDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_pil, label = self.samples[idx]
        if self.transform:
            try:
                img_tensor = self.transform(img_pil)
            except Exception:
                img_tensor = torch.zeros(3, 299, 299)
        else:
            img_tensor = transforms.ToTensor()(img_pil)
        return img_tensor, torch.tensor(label, dtype=torch.float32)

# ──────────────────────────── Video Processor ─────────────────────────────────
def process_local_videos(video_dir):
    """Classifies local videos and extracts face cropped frames."""
    if not _OPENCV_AVAILABLE:
        print("[FAIL] OpenCV is not installed!")
        sys.exit(1)
        
    vdir = Path(video_dir)
    if not vdir.exists():
        print(f"[FAIL] DATA/Video directory not found at: {vdir.resolve()}")
        sys.exit(1)
        
    # Load Haar Cascade Face Detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    print("\nScanning DATA/Video folder ...")
    real_files = []
    fake_files = []
    
    for f in os.listdir(vdir):
        if f.lower().endswith(".mp4"):
            parts = f.split("__")
            prefix = parts[0]
            path = vdir / f
            if "_" in prefix:
                fake_files.append(path)
            else:
                real_files.append(path)
                
    print(f"  Found: {len(real_files)} Real videos | {len(fake_files)} Fake (deepfake) videos")
    
    # We will sample from all videos to keep extraction fast but highly representative
    # Slices up to 40 Real and 40 Fake videos (gives plenty of frames!)
    sampled_real = random.sample(real_files, min(len(real_files), 40))
    sampled_fake = random.sample(fake_files, min(len(fake_files), 40))
    
    all_samples = []
    
    print("\nExtracting face crops from Real videos ...")
    for i, path in enumerate(sampled_real):
        crops = extract_crops_from_video(path, face_cascade, num_crops=3)
        for c in crops:
            all_samples.append((c, 1)) # 1: Real
        print(f"  [{i+1}/{len(sampled_real)}] Real processed. Extracted crops: {len(crops)}", flush=True)
        
    print("\nExtracting face crops from Fake videos ...")
    for i, path in enumerate(sampled_fake):
        crops = extract_crops_from_video(path, face_cascade, num_crops=3)
        for c in crops:
            all_samples.append((c, 0)) # 0: Fake
        print(f"  [{i+1}/{len(sampled_fake)}] Fake processed. Extracted crops: {len(crops)}", flush=True)
        
    print(f"\n[OK] Face crop extraction finished. Total dataset size: {len(all_samples)} frames.")
    return all_samples

def extract_crops_from_video(video_path, face_cascade, num_crops=3):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
        
    # Pick evenly-spaced indices
    ratios = [0.2, 0.5, 0.8] if num_crops == 3 else [0.5]
    indices = [int(total * r) for r in ratios]
    
    crops = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            pad_h, pad_w = int(h * 0.1), int(w * 0.1)
            y1 = max(0, y - pad_h)
            y2 = min(frame.shape[0], y + h + pad_h)
            x1 = max(0, x - pad_w)
            x2 = min(frame.shape[1], x + w + pad_w)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, (299, 299), interpolation=cv2.INTER_CUBIC)
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                crops.append(Image.fromarray(face_rgb))
                break # keep 1 face crop per index
                
    cap.release()
    return crops

# ──────────────────────────── Training Epochs ─────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.unsqueeze(1).to(device)
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

# ──────────────────────────── Main Script ─────────────────────────────────────
def main():
    print("=" * 65)
    print("  Verifixia - Video Dataset Pipeline & 30-Epoch Classifier Training")
    print("=" * 65)
    
    root_path = Path(__file__).resolve().parent.parent
    video_dir = root_path / "DATA" / "Video"
    model_dir = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "xception_deepfake.pth"
    history_path = model_dir / "deeperforensics_history.json"
    info_path = model_dir / "deeperforensics_info.json"
    
    # 1. Process local videos
    raw_samples = process_local_videos(video_dir)
    random.shuffle(raw_samples)
    
    # Split
    split_idx = int(0.8 * len(raw_samples))
    train_raw = raw_samples[:split_idx]
    val_raw = raw_samples[split_idx:]
    
    # Transforms representing real-world face perturbations
    train_tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = LocalProxyDataset(train_raw, transform=train_tf)
    val_dataset = LocalProxyDataset(val_raw, transform=val_tf)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = DeepfakeDetector().to(device)
    
    # Load base weights to fine-tune
    if model_path.exists():
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"[OK] Loaded base model weights from {model_path} for fine-tuning")
        except Exception as e:
            print(f"[Warning] Could not load base weights: {e}")
            
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    epochs = 80
    start_epoch = 1
    best_val_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": []
    }
    
    if history_path.exists():
        try:
            with open(history_path, 'r') as f:
                loaded_history = json.load(f)
            if all(k in loaded_history for k in history.keys()):
                history = loaded_history
                start_epoch = len(history["train_loss"]) + 1
                if len(history["val_acc"]) > 0:
                    best_val_acc = max(history["val_acc"])
                print(f"[OK] Resuming training from Epoch {start_epoch}. Best Val Accuracy so far: {best_val_acc*100:.2f}%")
        except Exception as e:
            print(f"[Warning] Could not load training history: {e}. Starting fresh.")
            
    # Align scheduler step if resuming
    for _ in range(1, start_epoch):
        scheduler.step()
        
    print(f"\n[Start] Continuing training from Epoch {start_epoch} to {epochs} on extracted video frames ...\n")
    print(f"{'Epoch':>5} | {'Tr Loss':>8} | {'Tr Acc':>7} | {'Val Loss':>8} | {'Val Acc':>7} | {'Val F1':>6}")
    print("-" * 65)
    
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        # Save metrics history
        history["train_loss"].append(float(tr_loss))
        history["train_acc"].append(float(tr_acc))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(float(val_metrics["accuracy"]))
        history["val_precision"].append(float(val_metrics["precision"]))
        history["val_recall"].append(float(val_metrics["recall"]))
        history["val_f1"].append(float(val_metrics["f1"]))
        
        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), model_path)
            
        elapsed = time.time() - t0
        flag = " *" if is_best else ""
        print(f"{epoch:>5d} | {tr_loss:>8.4f} | {tr_acc*100:>6.1f}% | {val_metrics['loss']:>8.4f} | {val_metrics['accuracy']*100:>6.1f}% | {val_metrics['f1']:>6.3f}{flag} ({elapsed:.1f}s)")
        
    print("-" * 65)
    print(f"[OK] Training complete! Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"[OK] Saved best model weights to {model_path}")
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Training history saved to {history_path}")
    
    model_info = {
        "model_name": "Verifixia Video-Extracted Deepfake Detector",
        "epochs_trained": epochs,
        "dataset": "DeeperForensics local videos",
        "best_accuracy": float(best_val_acc),
        "timestamp": datetime.now().isoformat(),
        "framework": "PyTorch",
        "device": str(device)
    }
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"[OK] Model metadata saved to {info_path}")
    
    print("\n[OK] All training artifacts successfully created and saved.")

if __name__ == "__main__":
    main()
