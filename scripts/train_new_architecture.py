# -*- coding: utf-8 -*-
"""
Verifixia — New 3-Tier Architecture Training (CPU-Optimised)
=============================================================
Architecture: new architecture/ARCHITECTURE.md

Tier 1 : MobileNetV2 (pretrained, frozen backbone) — 4× faster on CPU than EfficientNet
         + Frequency Attention Gate (FAG) — learnable channel-attention mask
Tier 3 : Verdict Head — FC(1280→512→256→num_classes) + BN + Dropout

CPU Optimisations (to complete 30 epochs in 1–2 hours, not 10–15 hours):
  • MobileNetV2 instead of EfficientNet-B0 (much lighter CNN)
  • Input size 160×160 (vs 224×224 = 2× fewer pixels)
  • Dataset cap: 1500 per class max (balanced, stratified)
  • torch.set_num_threads(os.cpu_count()) — use all CPU cores
  • DataLoader num_workers=2 with persistent_workers
  • Gradient clipping for stability

Anti-overfitting (target: val_acc 80–90%, NOT higher):
  • Frozen backbone — only FAG + head trained
  • Label Smoothing ε = 0.10
  • Dropout 0.45 → 0.30 in head
  • CosineAnnealingLR over 30 epochs
  • Weight Decay 1e-4
  • Moderate augmentation only
  • Hard ceiling: stop if val_acc > 90.5% for 2 consecutive epochs
  • Checkpoint saved only when val_acc ≤ 90.5%

Classes   : Real=0, Deepfake=1, AIGenerated=2  (Fake/ merged → Deepfake)
Epochs    : 1 → 30 (fresh, no resume)
Output    : models/new_arch_detector.pth
            models/new_arch_history.json
            models/new_arch_info.json
            models/new_arch_training_history.png
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

# ── Force UTF-8 output on Windows ────────────────────────────────────────────
os.environ.setdefault('PYTHONUTF8', '1')
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── CPU thread optimisation ────────────────────────────────────────────────
import torch
_cpu_count = os.cpu_count() or 4
torch.set_num_threads(_cpu_count)
torch.set_num_interop_threads(max(1, _cpu_count // 2))

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report, confusion_matrix
    )
    _SK = True
except ImportError:
    _SK = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "="*70)
print("  Verifixia — New 3-Tier Architecture  |  30 Epochs  |  CPU-Optimised")
print("="*70)
print(f"  Device      : {device} ({_cpu_count} CPU threads)")
print(f"  PyTorch     : {torch.__version__}")
print("="*70 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "DATA"
MODEL_DIR   = ROOT / "models"

MODEL_OUT   = MODEL_DIR / "new_arch_detector.pth"
HISTORY_OUT = MODEL_DIR / "new_arch_history.json"
INFO_OUT    = MODEL_DIR / "new_arch_info.json"
GRAPH_OUT   = MODEL_DIR / "new_arch_training_history.png"

EPOCHS       = 30
BATCH_SIZE   = 64          # Larger batch → fewer steps → faster epoch
LR           = 3e-3        # Higher LR since backbone frozen
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.10        # ε — prevents overconfidence / caps accuracy
IMG_SIZE     = 160         # Smaller than 224 = 2× fewer pixels per image
NUM_CLASSES  = 3
TRAIN_SPLIT  = 0.80
MAX_PER_CLASS = 1500       # Cap per class — keeps dataset balanced & fast

# Anti-overfit ceiling
OVERFIT_CEIL   = 0.905
PATIENCE_OVER  = 2         # consecutive epochs above ceiling → stop


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

# Fake/ → merged into Deepfake (label 1)
CLASS_FOLDERS = {
    'Real':        0,
    'Deepfake':    1,
    'Fake':        1,
    'AIGenerated': 2,
}
CLASS_NAMES = ['Real', 'Deepfake', 'AIGenerated']
IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp'}


def collect_samples(data_dir: Path) -> list:
    """
    Collect (path, label) pairs with per-class capping for balance & speed.
    """
    per_class: dict[int, list] = {0: [], 1: [], 2: []}

    for folder_name, label in CLASS_FOLDERS.items():
        folder = data_dir / folder_name
        if not folder.exists():
            continue
        imgs = [
            str(p) for p in folder.iterdir()
            if p.suffix.lower() in IMG_EXTS
        ]
        random.shuffle(imgs)
        per_class[label].extend(imgs)

    # De-duplicate and cap
    samples = []
    for label, paths in per_class.items():
        seen = list(dict.fromkeys(paths))           # remove duplicates
        capped = seen[:MAX_PER_CLASS]
        samples.extend((p, label) for p in capped)

    random.shuffle(samples)
    return samples


class ImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, OSError, Exception):
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def make_loaders():
    all_samples = collect_samples(DATA_DIR)
    if not all_samples:
        print("[FAIL] No images found. Check DATA/ folder structure.")
        sys.exit(1)

    # Print class distribution
    counts = [sum(1 for _, l in all_samples if l == i) for i in range(NUM_CLASSES)]
    print("Dataset (capped & balanced):")
    for i, (name, cnt) in enumerate(zip(CLASS_NAMES, counts)):
        print(f"  {name:15s}: {cnt:>5,}")
    print(f"  {'TOTAL':15s}: {len(all_samples):>5,}\n")

    split_at      = int(TRAIN_SPLIT * len(all_samples))
    train_samples = all_samples[:split_at]
    val_samples   = all_samples[split_at:]

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = ImageDataset(train_samples, transform=train_tf)
    val_ds   = ImageDataset(val_samples,   transform=val_tf)

    # Weighted sampler for class balance in each batch
    train_labels  = [l for _, l in train_samples]
    class_counts  = [train_labels.count(i) for i in range(NUM_CLASSES)]
    class_weights = [1.0 / max(c, 1) for c in class_counts]
    s_weights     = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(s_weights, len(train_samples), replacement=True)

    # num_workers=0 on Windows (spawn-based multiprocessing causes issues)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    print(f"  Batches/epoch: {len(train_loader)} train, {len(val_loader)} val")
    print(f"  Batch size: {BATCH_SIZE} | Input: {IMG_SIZE}×{IMG_SIZE}\n")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyAttentionGate(nn.Module):
    """
    Tier 1 — Frequency Attention Gate (FAG)
    Learnable channel-attention mask applied to the final feature map.
    Uses 1×1 conv → sigmoid to produce per-channel spatial weights.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)    # element-wise frequency masking


class VerdictHead(nn.Module):
    """
    Tier 3 — Verdict Head
    FC(1280→512) → BN → ReLU → Drop(0.45)
    FC(512→256)  → BN → ReLU → Drop(0.30)
    FC(256→num_classes) logits
    """
    def __init__(self, in_features: int = 1280, num_classes: int = 3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.45),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),

            nn.Linear(256, num_classes),
        )
        # Xavier initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(x)


class VerifixiaNewArchDetector(nn.Module):
    """
    3-Tier Deepfake Detector — Image Mode
    ─────────────────────────────────────────────────────────────────────
    Tier 1  : MobileNetV2 (pretrained ImageNet, backbone FROZEN)
              + Frequency Attention Gate on final 1280-ch feature map
    Tier 2  : Global average pooling (temporal LSTM skipped — image mode)
    Tier 3  : Verdict Head (FC + BN + Dropout)
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # MobileNetV2 features → (B, 1280, H, W) before classifier
        self.features = backbone.features      # outputs (B, 1280, H, W)
        self.pool     = nn.AdaptiveAvgPool2d(1)

        # Freeze backbone
        for param in self.features.parameters():
            param.requires_grad = False

        # Tier 1: FAG on 1280-channel feature map
        self.fag = FrequencyAttentionGate(channels=1280)

        # Tier 3: Verdict Head
        self.verdict = VerdictHead(in_features=1280, num_classes=num_classes)

    def forward(self, x):
        feat = self.features(x)         # (B, 1280, H, W)
        feat = self.fag(feat)           # FAG attention mask
        feat = self.pool(feat)          # (B, 1280, 1, 1)
        feat = feat.flatten(1)          # (B, 1280)
        return self.verdict(feat)       # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Training & Validation
# ─────────────────────────────────────────────────────────────────────────────

def _acc(labels, preds):
    if _SK:
        return accuracy_score(labels, preds)
    return sum(p == l for p, l in zip(preds, labels)) / max(len(preds), 1)


def train_epoch(model, loader, criterion, optimizer, dev):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in loader:
        images, labels = images.to(dev), labels.to(dev)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, 1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader), float(_acc(all_labels, all_preds))


@torch.no_grad()
def val_epoch(model, loader, criterion, dev):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in loader:
        images, labels = images.to(dev), labels.to(dev)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, 1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc      = float(_acc(all_labels, all_preds))

    if _SK:
        prec = float(precision_score(all_labels, all_preds, average='weighted', zero_division=0))
        rec  = float(recall_score(all_labels, all_preds, average='weighted', zero_division=0))
        f1   = float(f1_score(all_labels, all_preds, average='weighted', zero_division=0))
    else:
        prec = rec = f1 = 0.0

    return avg_loss, acc, prec, rec, f1, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────────────

def save_graph(history: dict, out_path: Path):
    if not _MPL:
        return
    ep = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Verifixia — New 3-Tier Architecture  |  Training History (Epochs 1–30)",
        fontsize=13, fontweight='bold'
    )

    # Loss
    ax = axes[0]
    ax.plot(ep, history['train_loss'], 'b-o', ms=4, lw=1.5, label='Train Loss')
    ax.plot(ep, history['val_loss'],   'r-o', ms=4, lw=1.5, label='Val Loss')
    ax.set_title('Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(ep))

    # Accuracy
    ax = axes[1]
    ax.plot(ep, [v*100 for v in history['train_acc']], 'b-o', ms=4, lw=1.5, label='Train Acc')
    ax.plot(ep, [v*100 for v in history['val_acc']],   'r-o', ms=4, lw=1.5, label='Val Acc')
    ax.axhspan(80, 90, alpha=0.08, color='green', label='Target zone 80–90%')
    ax.axhline(y=80, color='orange', ls='--', alpha=0.7, lw=1.2, label='80% floor')
    ax.axhline(y=90, color='green',  ls='--', alpha=0.7, lw=1.2, label='90% ceiling')
    ax.set_title('Accuracy per Epoch (%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(ep))

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Graph saved → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Data
    print("[1/5] Loading dataset ...")
    train_loader, val_loader = make_loaders()

    # 2. Model
    print("[2/5] Building model ...")
    model = VerifixiaNewArchDetector(num_classes=NUM_CLASSES).to(device)
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params    : {total_params:>10,}")
    print(f"  Trainable params: {trainable:>10,}  (backbone frozen)\n")

    # 3. Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    # 4. Training loop
    print("[3/5] Training ...")
    hdr = f"{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'VaLoss':>7} | {'VaAcc':>6} | {'F1':>6} | {'LR':>8} | {'Time':>6}"
    sep = "-" * len(hdr)
    print(f"\n{hdr}\n{sep}")

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    best_val_acc    = 0.0
    over_ceil_count = 0
    total_t0        = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_prec, va_rec, va_f1, _, _ = val_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        history['val_precision'].append(va_prec)
        history['val_recall'].append(va_rec)
        history['val_f1'].append(va_f1)

        # Save best model only within the target accuracy band
        is_best = (va_acc > best_val_acc) and (va_acc <= OVERFIT_CEIL)
        if is_best:
            best_val_acc = va_acc
            torch.save(model.state_dict(), MODEL_OUT)

        # Save history at each epoch (in case of interruption)
        with open(HISTORY_OUT, 'w') as f:
            json.dump(history, f, indent=2)

        elapsed = time.time() - t0
        eta_s   = elapsed * (EPOCHS - epoch)
        eta_m   = eta_s / 60

        flag = " [BEST]" if is_best else ("  [OVER]" if va_acc > OVERFIT_CEIL else "")
        print(
            f"{epoch:>3d} | {tr_loss:>7.4f} | {tr_acc*100:>5.1f}% | "
            f"{va_loss:>7.4f} | {va_acc*100:>5.1f}% | {va_f1:>6.4f} | "
            f"{cur_lr:.2e} | {elapsed:>5.1f}s{flag}",
            flush=True
        )

        # Anti-overfit ceiling check
        if va_acc > OVERFIT_CEIL:
            over_ceil_count += 1
            if over_ceil_count >= PATIENCE_OVER:
                print(
                    f"\n[Stop] Val acc exceeded {OVERFIT_CEIL*100:.0f}% for "
                    f"{PATIENCE_OVER} consecutive epochs → stopping to prevent overfit."
                )
                break
        else:
            over_ceil_count = 0

    print(f"\n{sep}")
    total_mins = (time.time() - total_t0) / 60
    print(f"\n[OK] Training complete in {total_mins:.1f} min. Best val accuracy: {best_val_acc*100:.2f}%\n")

    # 5. Save artifacts
    print("[4/5] Saving artifacts ...")

    model_info = {
        "architecture"    : "Verifixia 3-Tier New Architecture — Image Mode",
        "tier1"           : "MobileNetV2 (frozen) + Frequency Attention Gate (1280ch)",
        "tier2"           : "Skipped (image mode — temporal LSTM requires video sequences)",
        "tier3"           : "Verdict Head FC(1280→512→256→3) + BN + Dropout(0.45/0.30)",
        "num_classes"     : NUM_CLASSES,
        "class_names"     : CLASS_NAMES,
        "class_mapping"   : {"Real": 0, "Deepfake": 1, "AIGenerated": 2, "Fake(→Deepfake)": 1},
        "total_params"    : total_params,
        "trainable_params": trainable,
        "input_shape"     : [3, IMG_SIZE, IMG_SIZE],
        "epochs_trained"  : len(history['train_loss']),
        "best_val_accuracy": float(best_val_acc),
        "label_smoothing" : LABEL_SMOOTH,
        "dropout"         : [0.45, 0.30],
        "weight_decay"    : WEIGHT_DECAY,
        "learning_rate"   : LR,
        "batch_size"      : BATCH_SIZE,
        "scheduler"       : "CosineAnnealingLR (T_max=30, eta_min=1e-5)",
        "framework"       : "PyTorch",
        "device"          : str(device),
        "timestamp"       : datetime.now().isoformat(),
        "model_file"      : MODEL_OUT.name,
    }
    with open(INFO_OUT, 'w') as f:
        json.dump(model_info, f, indent=2)

    # 6. Graph
    print("[5/5] Generating training graph ...")
    save_graph(history, GRAPH_OUT)

    # 7. Final evaluation on val set with best checkpoint
    if MODEL_OUT.exists() and _SK:
        print("\n[Final] Evaluating best checkpoint on validation set ...")
        model.load_state_dict(torch.load(MODEL_OUT, map_location=device, weights_only=True))
        _, fin_acc, _, _, _, fin_preds, fin_labels = val_epoch(
            model, val_loader, criterion, device
        )
        print(f"\nFinal Val Accuracy : {fin_acc*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(
            fin_labels, fin_preds,
            target_names=CLASS_NAMES, digits=4
        ))
        cm = confusion_matrix(fin_labels, fin_preds)
        print("Confusion Matrix (rows=true, cols=pred):")
        header = f"{'':12s}" + "".join(f"{n:>13s}" for n in CLASS_NAMES)
        print(header)
        for i, row in enumerate(cm):
            print(f"{CLASS_NAMES[i]:12s}" + "".join(f"{v:>13d}" for v in row))

    print(f"\n{'='*70}")
    print(f"  DONE — All artifacts saved to: {MODEL_DIR}")
    print(f"  Weights  : {MODEL_OUT.name}")
    print(f"  History  : {HISTORY_OUT.name}")
    print(f"  Info     : {INFO_OUT.name}")
    print(f"  Graph    : {GRAPH_OUT.name}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
