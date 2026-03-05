"""
Verifixia – Deepfake Detection Training Script
================================================
Trains an EfficientNet-B0 (pretrained on ImageNet) fine-tuned for binary
deepfake / real classification.

Dataset expected layout:
    DATA/
        Real/   *.jpg  (label 0 – real)
        Fake/   *.jpg  (label 1 – fake / deepfake)

Usage (from repo root):
    cd /Users/cdl_jinesh/Personal/Verifixia
    python scripts/train.py

Optional flags:
    --data_dir  PATH     Path to DATA dir   (default: ./DATA)
    --out       PATH     Where to save .pth (default: ./models/xception_deepfake.pth)
    --epochs    INT      Training epochs    (default: 30)
    --batch     INT      Batch size         (default: 16)
    --lr        FLOAT    Learning rate      (default: 1e-4)
    --workers   INT      DataLoader workers (default: 0)
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

# ─────────────────────────── Reproducibility ────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# ──────────────────────────── Dataset ───────────────────────────────────────
class DeepfakeDataset(Dataset):
    """
    Loads images from:
        {root}/Real/  → label 0
        {root}/Fake/  → label 1
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, split: str = "train",
                 val_ratio: float = 0.2, transform=None):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        real_dir = Path(root) / "Real"
        fake_dir = Path(root) / "Fake"

        def load_dir(d: Path, label: int):
            if not d.exists():
                print(f"  ⚠  Directory not found: {d}")
                return []
            files = [str(f) for f in d.iterdir()
                     if f.suffix.lower() in self.IMAGE_EXTS]
            return [(p, label) for p in files]

        all_samples = load_dir(real_dir, 0) + load_dir(fake_dir, 1)
        rng = random.Random(SEED)
        rng.shuffle(all_samples)

        split_idx = int(len(all_samples) * (1 - val_ratio))
        if split == "train":
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]

        print(f"  {split:5s} split: {len(self.samples)} images  "
              f"(real={sum(1 for _,l in self.samples if l==0)}, "
              f"fake={sum(1 for _,l in self.samples if l==1)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 0)
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────── Transforms ─────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ──────────────────────────── Model ──────────────────────────────────────────
def build_model(device: torch.device) -> nn.Module:
    """EfficientNet-B0 fine-tuned for binary classification."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return model.to(device)


# ──────────────────────────── Helpers ────────────────────────────────────────
def make_sampler(dataset: DeepfakeDataset) -> WeightedRandomSampler:
    """Oversample the minority class to handle class imbalance."""
    labels = [l for _, l in dataset.samples]
    counts = [labels.count(0), labels.count(1)]
    weights_per_class = [1.0 / max(c, 1) for c in counts]
    sample_weights = [weights_per_class[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights),
                                 replacement=True)


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


# ──────────────────────────── Training loop ───────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = (outputs > 0.5).float()
        running_loss += loss.item()
        running_acc  += accuracy(preds, labels)

    scheduler.step()
    n = len(loader)
    return running_loss / n, running_acc / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels_t = labels.float().unsqueeze(1).to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels_t)
        preds = (outputs > 0.5).float()

        running_loss += loss.item()
        running_acc  += accuracy(preds, labels_t)
        all_probs.extend(outputs.squeeze(1).cpu().numpy())
        all_labels.extend(labels.numpy())

    n = len(loader)
    return running_loss / n, running_acc / n, all_probs, all_labels


# ──────────────────────────── Main ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Verifixia deepfake detector")
    parser.add_argument("--data_dir", default="DATA",
                        help="Path to DATA directory (contains Real/ and Fake/)")
    parser.add_argument("--out", default="models/xception_deepfake.pth",
                        help="Output model path")
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--workers", type=int,   default=0)
    args = parser.parse_args()

    # ── Resolve paths (can run from any working dir) ──
    root = Path(__file__).resolve().parent.parent
    data_dir = (root / args.data_dir).resolve()
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  Verifixia Deepfake Detector – Training")
    print(f"{'='*60}")
    print(f"  Device    : {device}")
    print(f"  Data dir  : {data_dir}")
    print(f"  Output    : {out_path}")
    print(f"  Epochs    : {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")
    print(f"{'='*60}\n")

    # ── Datasets ──
    print("Loading datasets …")
    train_ds = DeepfakeDataset(str(data_dir), split="train", transform=train_tf)
    val_ds   = DeepfakeDataset(str(data_dir), split="val",   transform=val_tf)

    if len(train_ds) == 0:
        print("\n❌ No training images found. Check that DATA/Real and DATA/Fake exist.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        sampler=make_sampler(train_ds),
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers,
    )

    # ── Model, loss, optimiser ──
    model     = build_model(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ──
    best_val_acc = 0.0
    best_path    = str(out_path)
    history      = []

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'Best?':>5}")
    print("─" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device)
        vl_loss, vl_acc, probs, labels = eval_epoch(
            model, val_loader, criterion, device)

        is_best = vl_acc > best_val_acc
        if is_best:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_path)

        elapsed = time.time() - t0
        flag = "  ✓" if is_best else ""
        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc*100:>8.2f}%  "
              f"{vl_loss:>8.4f}  {vl_acc*100:>6.2f}%  {flag}   ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_acc": vl_acc,
        })

    # ── Final report ──
    print(f"\n{'='*60}")
    print(f"  Training complete. Best val accuracy: {best_val_acc*100:.2f}%")
    print(f"  Model saved → {best_path}")
    print(f"{'='*60}\n")

    # Load best and run full evaluation
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, _, probs, labels = eval_epoch(model, val_loader, criterion, device)

    preds = [1 if p > 0.5 else 0 for p in probs]
    print("Classification Report:")
    print(classification_report(labels, preds,
                                 target_names=["Real", "Fake"], digits=4))

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"           Real   Fake")
    print(f"  Real   {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"  Fake   {cm[1][0]:>5}  {cm[1][1]:>5}")

    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, probs)
        print(f"\n  ROC-AUC: {auc:.4f}")

    print(f"\n✅ Model is ready at: {best_path}")
    print("   Run the backend and it will load automatically.\n")


if __name__ == "__main__":
    main()
