"""
Verifixia - Advanced 3-Tier Multi-Modal Deepfake Detector Training Script
========================================================================
1. Scans DATA/Video for local .mp4 videos.
2. Splits them into Train (70%), Val (15%), and Test (15%) groups.
3. Dynamically extracts 16 face crops per video as a temporal sequence.
4. Trains the AdvancedCNNLSTMDetector using CrossEntropyLoss.
5. Saves model weights to models/advanced_cnn_lstm.pth.
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

# Import backend classes
sys.path.append(os.path.join(os.path.dirname(__file__), '../Backend'))
from utils.model_utils import AdvancedCNNLSTMDetector

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# Dataset
# ============================================================================

class VideoSequenceDataset(Dataset):
    """Wrapper to return a sequence of 16 face crops and its binary label"""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        crops, label = self.samples[idx]
        tensors = []
        for img_pil in crops:
            if self.transform:
                try:
                    tensor = self.transform(img_pil)
                except Exception:
                    tensor = torch.zeros(3, 299, 299)
            else:
                tensor = transforms.ToTensor()(img_pil)
            tensors.append(tensor)
            
        # Stacks sequence of crops: (SequenceLength, Channels, Height, Width) -> (16, 3, 299, 299)
        sequence_tensor = torch.stack(tensors, dim=0)
        return sequence_tensor, torch.tensor(label, dtype=torch.long)

# ============================================================================
# Dynamic Frame Extraction & Group Splitting
# ============================================================================

def get_group_key(filename):
    parts = filename.split("__")
    if len(parts) >= 2:
        return parts[0] + "__" + parts[1]
    return filename.replace(".mp4", "")

def extract_crops_from_video(video_path, face_cascade, num_crops=16):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
        
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
        
        extracted = False
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
                extracted = True
                break
                
        # Pad with blank dummy image if face detection failed
        if not extracted:
            crops.append(Image.new("RGB", (299, 299), 0))
            
    cap.release()
    return crops

def extract_frames_from_groups(group_set, video_groups, face_cascade, num_crops, limit, desc):
    print(f"\n[{desc}] Extracting frame sequences (Limit: {limit} groups) ...")
    sampled_groups = random.sample(list(group_set), min(len(group_set), limit))
    samples = []
    
    for idx, gkey in enumerate(sampled_groups):
        files = video_groups[gkey]
        for fpath in files:
            parts = fpath.name.split("__")
            prefix = parts[0]
            # Mapping: 0 = Real, 1 = Fake (Consistent with new architecture)
            label = 1 if "_" in prefix else 0
            
            crops = extract_crops_from_video(fpath, face_cascade, num_crops)
            if len(crops) == num_crops:
                samples.append((crops, label))
                
        print(f"  [{idx+1}/{len(sampled_groups)}] Group '{gkey}' processed. Total sequences: {len(samples)}", flush=True)
        
    return samples

def process_local_videos(video_dir, num_crops=16):
    if not _OPENCV_AVAILABLE:
        print("[FAIL] OpenCV is not installed!")
        sys.exit(1)
        
    vdir = Path(video_dir)
    if not vdir.exists():
        print(f"[FAIL] Video folder not found: {vdir.resolve()}")
        sys.exit(1)
        
    all_videos = [f for f in os.listdir(vdir) if f.lower().endswith(".mp4")]
    
    video_groups = {}
    for f in all_videos:
        key = get_group_key(f)
        if key not in video_groups:
            video_groups[key] = []
        video_groups[key].append(vdir / f)
        
    group_keys = list(video_groups.keys())
    random.shuffle(group_keys)
    
    n_groups = len(group_keys)
    train_end = int(0.70 * n_groups)
    val_end = int(0.85 * n_groups)
    
    train_groups = set(group_keys[:train_end])
    val_groups = set(group_keys[train_end:val_end])
    test_groups = set(group_keys[val_end:])
    
    print("\n" + "=" * 65)
    print("  Dataset Group Splits for Advanced Model")
    print("=" * 65)
    print(f"  Total Video Groups    : {n_groups}")
    print(f"  Train Groups (70%)    : {len(train_groups)}")
    print(f"  Val Groups (15%)      : {len(val_groups)}")
    print(f"  Test Groups (15%)     : {len(test_groups)}")
    print("=" * 65)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Restrict group size for local CPU/GPU limits
    max_train_gps = 15
    max_val_gps = 5
    max_test_gps = 5
    
    train_raw = extract_frames_from_groups(train_groups, video_groups, face_cascade, num_crops, limit=max_train_gps, desc="Train")
    val_raw = extract_frames_from_groups(val_groups, video_groups, face_cascade, num_crops, limit=max_val_gps, desc="Val")
    test_raw = extract_frames_from_groups(test_groups, video_groups, face_cascade, num_crops, limit=max_test_gps, desc="Test")
    
    return train_raw, val_raw, test_raw

# ============================================================================
# Training Loops
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for seqs, labels in loader:
        seqs = seqs.to(device)
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        logits, rppg_score, spike_flags, deltas = model(seqs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
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
    
    for seqs, labels in loader:
        seqs = seqs.to(device)
        labels = labels.to(device).long()
        
        logits, rppg_score, spike_flags, deltas = model(seqs)
        loss = criterion(logits, labels)
        running_loss += loss.item()
        
        probs = torch.softmax(logits, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = running_loss / len(loader)
    return avg_loss, np.array(all_probs), np.array(all_labels)

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    print("=" * 70)
    print("  Verifixia - Advanced 3-Tier Model 30-Epoch Video Training Pipeline")
    print("=" * 70)
    
    root_path = Path(__file__).resolve().parent.parent
    video_dir = root_path / "DATA" / "Video"
    model_dir = root_path / "models"
    
    model_path = model_dir / "advanced_cnn_lstm.pth"
    history_path = model_dir / "advanced_cnn_lstm_history.json"
    info_path = model_dir / "advanced_cnn_lstm_info.json"
    
    train_raw, val_raw, test_raw = process_local_videos(video_dir, num_crops=16)
    
    if len(train_raw) == 0 or len(val_raw) == 0 or len(test_raw) == 0:
        print("[FAIL] Extracted splits are empty! Ensure DATA/Video contains .mp4 files.")
        return
        
    random.shuffle(train_raw)
    random.shuffle(val_raw)
    random.shuffle(test_raw)
    
    train_tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = VideoSequenceDataset(train_raw, transform=train_tf)
    val_dataset = VideoSequenceDataset(val_raw, transform=val_tf)
    test_dataset = VideoSequenceDataset(test_raw, transform=val_tf)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    print("\n" + "=" * 65)
    print("  Split Size (Total Sequences of 16 frames)")
    print("=" * 65)
    print(f"  Train Sequences: {len(train_dataset)}")
    print(f"  Val Sequences  : {len(val_dataset)}")
    print(f"  Test Sequences : {len(test_dataset)}")
    print("=" * 65)
    
    model = AdvancedCNNLSTMDetector(use_pretrained=False).to(device)
    
    # Freeze CNN backbone parameters to speed up training on CPU and prevent overfitting
    # We freeze self.stage1, self.stage2, and self.stage3
    for param in model.stage1.parameters():
        param.requires_grad = False
    for param in model.stage2.parameters():
        param.requires_grad = False
    for param in model.stage3.parameters():
        param.requires_grad = False
    print("[OK] Froze 2D CNN backbone stages to accelerate training and reduce overfitting")
    
    criterion = nn.CrossEntropyLoss()
    # Optimize only parameters with requires_grad=True (FAG, BPM, LSTMs, Cross-Attn, Verdict Head)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-3)
    
    epochs = 30
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": []
    }
    
    print(f"\n[Start] Training Advanced model on video sequences ...\n", flush=True)
    print(f"{'Epoch':>5} | {'Tr Loss':>8} | {'Tr Acc':>7} | {'Val Loss':>8} | {'Val Acc':>7}", flush=True)
    print("-" * 65, flush=True)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs, val_labels = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, zero_division=0)
        val_rec = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
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
        print(f"{epoch:>5d} | {tr_loss:>8.4f} | {tr_acc*100:>6.1f}% | {val_loss:>8.4f} | {val_acc*100:>6.1f}%{flag} ({elapsed:.1f}s)", flush=True)
        
    print("-" * 65, flush=True)
    print(f"[OK] Training complete! Best validation accuracy: {best_val_acc*100:.2f}%", flush=True)
    
    # Save training history
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    # Evaluate on the isolated Test set
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    test_loss, test_probs, test_labels = val_epoch(model, test_loader, criterion, device)
    test_preds = np.argmax(test_probs, axis=1)
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, zero_division=0)
    test_rec = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    # Save model metadata
    model_info = {
        "model_name": "Verifixia Advanced 3-Tier Video Deepfake Detector",
        "epochs_trained": epochs,
        "test_accuracy": float(test_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "timestamp": datetime.now().isoformat(),
        "framework": "PyTorch",
        "device": str(device)
    }
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
        
    print(f"[OK] Saved Advanced model artifacts to models/", flush=True)

if __name__ == "__main__":
    main()
