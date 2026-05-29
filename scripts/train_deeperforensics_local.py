"""
Verifixia - Video Dataset Local Frame-Extraction & Model Training Pipeline
========================================================================
1. Scans DATA/Video for local .mp4 video files.
2. Groups videos by scene and actor pair base prefix to prevent identity/leakage.
3. Partitions groups into Train (70%), Validation (15%), and Test (15%) sets.
4. Slices and extracts 25 face-cropped frames per video dynamically.
5. Uses advanced torchvision data augmentations (affine, erasing cutout) for robustness.
6. Employs enhanced regularization: increased Dropout (0.6/0.5/0.4), AdamW weight decay (1e-3).
7. Trains with lower learning rate (5e-5) and Cosine Annealing scheduler.
8. Evaluates on the isolated Test Set with confusion matrix and optimal F1 threshold sweep.
9. Plots and saves a professional ROC Curve to models/deeperforensics_roc.png.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

try:
    import cv2
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg') # Non-interactive backend
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
        
        # Increased Dropout rates slightly (0.6, 0.5, 0.4) to combat pixel memorization
        self.fc1 = nn.Linear(512 * 2, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.6)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn_fc3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.4)
        
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

# ============================================================================
# Dataset
# ============================================================================

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

# ============================================================================
# Dynamic Frame Extraction & Group Splitting
# ============================================================================

def get_group_key(filename):
    """
    Extracts base scene and actor pair key to keep different deepfake methods 
    and matching original real video segments together in the same split fold.
    E.g. '01_02__exit_phone_room__YVGY8LOK.mp4' -> '01_02__exit_phone_room'
    """
    parts = filename.split("__")
    if len(parts) >= 2:
        return parts[0] + "__" + parts[1]
    return filename.replace(".mp4", "")

def process_local_videos(video_dir, num_crops=25):
    """
    Groups local videos, segments them into Train (70%), Val (15%), and Test (15%) splits
    by video group, and dynamically extracts 25 face crops per video.
    """
    if not _OPENCV_AVAILABLE:
        print("[FAIL] OpenCV is not installed!")
        sys.exit(1)
        
    vdir = Path(video_dir)
    if not vdir.exists():
        print(f"[FAIL] Video folder not found: {vdir.resolve()}")
        sys.exit(1)
        
    # Scans video folder and compiles video files
    all_videos = [f for f in os.listdir(vdir) if f.lower().endswith(".mp4")]
    
    # Group videos by their core subject/scene signature
    video_groups = {}
    for f in all_videos:
        key = get_group_key(f)
        if key not in video_groups:
            video_groups[key] = []
        video_groups[key].append(vdir / f)
        
    group_keys = list(video_groups.keys())
    random.shuffle(group_keys)
    
    # 70% Train, 15% Val, 15% Test Split
    n_groups = len(group_keys)
    train_end = int(0.70 * n_groups)
    val_end = int(0.85 * n_groups)
    
    train_groups = set(group_keys[:train_end])
    val_groups = set(group_keys[train_end:val_end])
    test_groups = set(group_keys[val_end:])
    
    print("\n" + "=" * 65)
    print("  Dataset Split Breakdown (Group Splitting by Video)")
    print("=" * 65)
    print(f"  Total Video Groups    : {n_groups}")
    print(f"  Train Groups (70%)    : {len(train_groups)}")
    print(f"  Val Groups (15%)      : {len(val_groups)}")
    print(f"  Test Groups (15%)     : {len(test_groups)}")
    print("=" * 65)
    
    # Load Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Limit number of active groups processed on CPU to keep it responsive and fast (e.g. 20 Train, 6 Val, 6 Test)
    max_train_gps = 15
    max_val_gps = 5
    max_test_gps = 5
    
    train_raw = extract_frames_from_groups(train_groups, video_groups, face_cascade, num_crops, limit=max_train_gps, desc="Train")
    val_raw = extract_frames_from_groups(val_groups, video_groups, face_cascade, num_crops, limit=max_val_gps, desc="Val")
    test_raw = extract_frames_from_groups(test_groups, video_groups, face_cascade, num_crops, limit=max_test_gps, desc="Test")
    
    return train_raw, val_raw, test_raw

def extract_frames_from_groups(group_set, video_groups, face_cascade, num_crops, limit, desc):
    print(f"\n[{desc}] Extracting {num_crops} frames per video (Limit: {limit} groups) ...")
    sampled_groups = random.sample(list(group_set), min(len(group_set), limit))
    samples = []
    
    for idx, gkey in enumerate(sampled_groups):
        files = video_groups[gkey]
        for fpath in files:
            # Determine label based on naming convention
            # Real has single actor prefix (e.g., "01__"), Fake has double actor face swap prefix (e.g., "01_02__")
            parts = fpath.name.split("__")
            prefix = parts[0]
            label = 0 if "_" in prefix else 1 # 0: Fake, 1: Real
            
            crops = extract_crops_from_video(fpath, face_cascade, num_crops)
            for c in crops:
                samples.append((c, label))
                
        print(f"  [{idx+1}/{len(sampled_groups)}] Group '{gkey}' processed. Total frames: {len(samples)}", flush=True)
        
    return samples

def extract_crops_from_video(video_path, face_cascade, num_crops=25):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
        
    # Generate evenly-spaced frame indexes across the video duration
    ratios = np.linspace(0.05, 0.95, num_crops)
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
                break # Extract 1 face crop per selected frame
                
    cap.release()
    return crops

# ============================================================================
# Training Loops
# ============================================================================

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
    all_probs = []
    all_labels = []
    
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.unsqueeze(1).to(device)
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        all_probs.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = running_loss / len(loader)
    return avg_loss, np.array(all_probs), np.array(all_labels)

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    print("=" * 70)
    print("  Verifixia - Video Dataset Pipeline & Professional 30-Epoch Training")
    print("=" * 70)
    
    root_path = Path(__file__).resolve().parent.parent
    video_dir = root_path / "DATA" / "Video"
    model_dir = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "xception_deepfake.pth"
    history_path = model_dir / "deeperforensics_history.json"
    info_path = model_dir / "deeperforensics_info.json"
    
    # 1. Process local videos with 25 crops per video
    train_raw, val_raw, test_raw = process_local_videos(video_dir, num_crops=25)
    
    if len(train_raw) == 0 or len(val_raw) == 0 or len(test_raw) == 0:
        print("[FAIL] Extracted frame splits are empty! Ensure DATA/Video contains .mp4 files.")
        return
        
    # Shuffling splits
    random.shuffle(train_raw)
    random.shuffle(val_raw)
    random.shuffle(test_raw)
    
    # 2. Define data transforms
    # Added affine translation/scaling and cutout (RandomErasing) transforms to prevent memorization
    train_tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.08),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = LocalProxyDataset(train_raw, transform=train_tf)
    val_dataset = LocalProxyDataset(val_raw, transform=val_tf)
    test_dataset = LocalProxyDataset(test_raw, transform=val_tf)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print("\n" + "=" * 65)
    print("  Split Size (Total Frames extracted)")
    print("=" * 65)
    print(f"  Train Frames   : {len(train_dataset)}")
    print(f"  Val Frames     : {len(val_dataset)}")
    print(f"  Test Frames    : {len(test_dataset)}")
    print("=" * 65)
    
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
    
    # Lowered LR (5e-5) and increased AdamW weight decay (1e-3) for regularization
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    
    # Total epochs is 30 for deeperforensics local training
    epochs = 30
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
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
                print(f"[OK] Resuming training from Epoch {start_epoch}. Best Val Accuracy: {best_val_acc*100:.2f}%")
        except Exception as e:
            print(f"[Warning] Could not load training history: {e}. Starting fresh.")
            
    # Align scheduler step if resuming
    for _ in range(1, start_epoch):
        scheduler.step()
        
    print(f"\n[Start] Training from Epoch {start_epoch} to {epochs} on grouped video frames ...\n")
    print(f"{'Epoch':>5} | {'Tr Loss':>8} | {'Tr Acc':>7} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 65)
    
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs, val_labels = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        # Validation evaluation at standard 0.5 threshold
        val_preds = (val_probs > 0.5).astype(float)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, zero_division=0)
        val_rec = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        # Save metrics history
        history["train_loss"].append(float(tr_loss))
        history["train_acc"].append(float(tr_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_precision"].append(float(val_prec))
        history["val_recall"].append(float(val_rec))
        history["val_f1"].append(float(val_f1))
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            
        elapsed = time.time() - t0
        flag = " *" if is_best else ""
        print(f"{epoch:>5d} | {tr_loss:>8.4f} | {tr_acc*100:>6.1f}% | {val_loss:>8.4f} | {val_acc*100:>6.1f}%{flag} ({elapsed:.1f}s)")
        
    print("-" * 65)
    print(f"[OK] Training complete! Best validation accuracy: {best_val_acc*100:.2f}%")
    
    # Save training history
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Training history saved to {history_path}")
    
    # ============================================================================
    # Post-Training Independent Test Set & Optimal Threshold Evaluation
    # ============================================================================
    print("\n" + "=" * 65)
    print("  Optimal Threshold Search & Isolated Test Set Evaluation")
    print("=" * 65)
    
    # Load the best model weights
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("[OK] Loaded best model checkpoint for final Test set evaluation")
        
    # Get predictions on Validation set to find the optimal decision threshold
    val_loss, val_probs, val_labels = val_epoch(model, val_loader, criterion, device)
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    # Sweep threshold values to maximize F1 Score
    for th in np.arange(0.05, 0.95, 0.01):
        th_preds = (val_probs > th).astype(float)
        th_f1 = f1_score(val_labels, th_preds, zero_division=0)
        if th_f1 > best_f1:
            best_f1 = th_f1
            best_threshold = float(th)
            
    print(f"[INFO] Optimal Validation Threshold discovered: {best_threshold:.2f} (Val F1: {best_f1*100:.2f}%)")
    
    # Now evaluate on the isolated TEST set using the discovered optimal threshold
    test_loss, test_probs, test_labels = val_epoch(model, test_loader, criterion, device)
    
    test_preds = (test_probs > best_threshold).astype(float)
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, zero_division=0)
    test_rec = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    # Extract confusion elements
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    print("\n" + "=" * 65)
    print("  Final Isolated Test Set Performance")
    print("=" * 65)
    print(f"  Test Loss       : {test_loss:.4f}")
    print(f"  Test Accuracy   : {test_acc*100:.2f}%")
    print(f"  Test Precision  : {test_prec*100:.2f}%")
    print(f"  Test Recall     : {test_rec*100:.2f}%")
    print(f"  Test F1 Score   : {test_f1:.4f}")
    print("=" * 65)
    
    print("\n=== CONFUSION MATRIX (TEST SET) ===")
    print(f"  True Positives  (Fake correctly detected): {tp}")
    print(f"  False Positives (Real misflagged as Fake): {fp}")
    print(f"  True Negatives  (Real correctly detected): {tn}")
    print(f"  False Negatives (Fake missed by model)   : {fn}")
    print("=" * 65)
    
    # ============================================================================
    # ROC Curves & Threshold Analysis Plotted Professionally
    # ============================================================================
    fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
    roc_auc = auc(fpr, tpr)
    print(f"[INFO] Area Under the Curve (Test ROC-AUC): {roc_auc:.4f}")
    
    if _MATPLOTLIB_AVAILABLE:
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            # Plot the optimal validation threshold point
            # Find index of closest threshold in list
            closest_idx = np.argmin(np.abs(thresholds - best_threshold))
            plt.scatter(fpr[closest_idx], tpr[closest_idx], color='red', s=100, zorder=5, 
                        label=f'Optimal Threshold ({best_threshold:.2f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('Receiver Operating Characteristic (ROC) - Test Set')
            plt.legend(loc="lower right")
            plt.grid(True, linestyle=':', alpha=0.6)
            
            roc_img_path = model_dir / "deeperforensics_roc.png"
            plt.savefig(roc_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] ROC Curve successfully plotted and saved to {roc_img_path}")
        except Exception as e:
            print(f"[Warning] Failed to generate ROC Curve plot: {e}")
    else:
        print("[Warning] Matplotlib is not available. Skipping ROC Curve plotting.")
        
    model_info = {
        "model_name": "Verifixia Video-Extracted Deepfake Detector",
        "epochs_trained": epochs,
        "dataset": "DeeperForensics local videos",
        "best_validation_accuracy": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "test_roc_auc": float(roc_auc),
        "optimal_threshold": float(best_threshold),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "timestamp": datetime.now().isoformat(),
        "framework": "PyTorch",
        "device": str(device)
    }
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"[OK] Complete model metadata and test stats saved to {info_path}")
    print("\n[OK] All advanced training pipeline requirements fully resolved!")

if __name__ == "__main__":
    main()
