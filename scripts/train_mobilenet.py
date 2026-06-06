"""
============================================================
  Deepfake Detection Pipeline — MobileNetV3-Small (CPU)
============================================================
  Stages:
    1. Frame extraction from videos (with face cropping)
    2. Dataset class (split by video to prevent data leakage)
    3. Model definition (custom head)
    4. Two-phase training
    5. Evaluation + metrics
    6. Save & load model

  Requirements:
    pip install torch torchvision opencv-python scikit-learn tqdm matplotlib
============================================================
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights


# ─────────────────────────────────────────────
#  CONFIG  — edit these paths before running
# ─────────────────────────────────────────────
CFG = {
    # Root dataset folder structure expected:
    #   dataset_root/
    #     real/   <- real face videos (.mp4 / .avi)
    #     fake/   <- deepfake videos  (.mp4 / .avi)
    "dataset_root"   : "dataset",
    "frames_root"    : "frames",        # extracted frames saved here
    "frames_per_vid" : 20,              # frames to extract per video
    "img_size"       : 224,
    "batch_size"     : 8,               # keep small for CPU
    "num_workers"    : 2,
    "phase1_epochs"  : 8,               # train head only
    "phase2_epochs"  : 12,              # fine-tune last 2 blocks + head
    "phase1_lr"      : 1e-3,
    "phase2_lr"      : 1e-4,
    "weight_decay"   : 1e-4,
    "val_split"      : 0.2,
    "seed"           : 42,
    "save_path"      : "deepfake_mobilenetv3.pth",
    "device"         : "cpu",
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.set_num_threads(os.cpu_count())       # use all CPU cores


# ─────────────────────────────────────────────
#  STAGE 1 — FRAME EXTRACTION
# ─────────────────────────────────────────────

def extract_frames(video_path: str, out_dir: str, n_frames: int = 20) -> int:
    """
    Uniformly samples n_frames from a video, detects faces using Haar Cascades,
    crops them, and saves them as JPEGs.
    Returns the number of frames actually saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return 0

    # Load Haar Cascade face detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect faces in frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        face_found = False
        for (x, y, w, h) in faces:
            pad_h, pad_w = int(h * 0.1), int(w * 0.1)
            y1 = max(0, y - pad_h)
            y2 = min(frame.shape[0], y + h + pad_h)
            x1 = max(0, x - pad_w)
            x2 = min(frame.shape[1], x + w + pad_w)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                out_path = os.path.join(out_dir, f"frame_{idx:05d}.jpg")
                cv2.imwrite(out_path, face_crop)
                saved += 1
                face_found = True
                break  # Save the primary face in the frame
        
        # Fallback: if no face is detected, save the full frame to avoid gaps
        if not face_found:
            out_path = os.path.join(out_dir, f"frame_{idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1

    cap.release()
    return saved


def extract_all_frames(dataset_root: str, frames_root: str, n_frames: int = 20):
    """
    Walks dataset_root/real and dataset_root/fake, extracts frames
    into frames_root/real/<video_name>/ and frames_root/fake/<video_name>/
    """
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    for label in ["real", "fake"]:
        video_dir = Path(dataset_root) / label
        if not video_dir.exists():
            print(f"[WARNING] {video_dir} not found — skipping.")
            continue
        videos = [v for v in video_dir.iterdir() if v.suffix.lower() in video_exts]
        print(f"\nExtracting frames from {len(videos)} {label} videos...")
        for video in tqdm(videos, desc=label):
            out_dir = Path(frames_root) / label / video.stem
            if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) >= n_frames:
                continue    # already extracted
            extract_frames(str(video), str(out_dir), n_frames)

    print("\nFrame extraction complete.")


# ─────────────────────────────────────────────
#  STAGE 2 — DATASET CLASS
# ─────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Loads individual frames from a pre-defined list of image samples.
    Label: 0 = real, 1 = fake
    """
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (img_path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.05),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), value='random'),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_dataloaders(frames_root, img_size, batch_size, num_workers, val_split, seed):
    # 1. Scan folders to group frames by video folder (preventing frame-level leakage)
    real_vids = []
    fake_vids = []
    
    real_dir = Path(frames_root) / "real"
    if real_dir.exists():
        real_vids = [d for d in real_dir.iterdir() if d.is_dir()]
        
    fake_dir = Path(frames_root) / "fake"
    if fake_dir.exists():
        fake_vids = [d for d in fake_dir.iterdir() if d.is_dir()]
        
    # Compile list of (video_folder_path, label)
    all_videos = [(str(d), 0) for d in real_vids] + [(str(d), 1) for d in fake_vids]
    
    if len(all_videos) == 0:
        raise RuntimeError(
            f"No video frame folders found in {frames_root}. "
            "Run extract_all_frames() first."
        )

    # 2. Shuffle and split at the VIDEO level (so frames from one video are never mixed)
    import random
    rng = random.Random(seed)
    rng.shuffle(all_videos)
    
    n_val = int(len(all_videos) * val_split)
    val_vids = all_videos[:n_val]
    train_vids = all_videos[n_val:]
    
    # 3. Compile individual frame paths for train and val splits
    train_samples = []
    for v_path, label in train_vids:
        for img_path in Path(v_path).glob("*.jpg"):
            train_samples.append((str(img_path), label))
            
    val_samples = []
    for v_path, label in val_vids:
        for img_path in Path(v_path).glob("*.jpg"):
            val_samples.append((str(img_path), label))
            
    print(f"\nDataset loaded and split by video folders:")
    print(f"  Total Video Folders   : {len(all_videos)}")
    print(f"  Train Split           : {len(train_vids)} videos ({len(train_samples)} frames)")
    print(f"  Val Split             : {len(val_vids)} videos ({len(val_samples)} frames)")

    train_tf, val_tf = get_transforms(img_size)
    
    train_ds = DeepfakeDataset(train_samples, transform=train_tf)
    val_ds = DeepfakeDataset(val_samples, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# ─────────────────────────────────────────────
#  STAGE 3 — MODEL
# ─────────────────────────────────────────────

def build_model():
    """
    MobileNetV3-Small with ImageNet pretrained weights.
    Replaces the classifier with a custom 2-class head.
    Backbone is frozen initially (Phase 1).
    """
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Linear(576, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),
    )

    # Phase 1: freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel built:")
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable (P1)  : {trainable_params:,}")

    return model


def unfreeze_for_phase2(model):
    """Unfreeze last 2 bottleneck blocks + classifier for fine-tuning."""
    # MobileNetV3-Small features: indices 0-12
    # Unfreeze blocks 11 and 12
    for i, layer in enumerate(model.features):
        if i >= 11:
            for param in layer.parameters():
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPhase 2 — unfrozen. Trainable params: {trainable_params:,}")
    return model


# ─────────────────────────────────────────────
#  STAGE 4 — TRAINING
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / total, correct / total, auc


def run_training(model, train_loader, val_loader, cfg):
    device    = cfg["device"]
    criterion = nn.CrossEntropyLoss()
    history   = {"train_loss": [], "val_loss": [],
                 "train_acc":  [], "val_acc":  [], "val_auc": []}
    best_val_acc = 0.0

    # ── Phase 1 ──────────────────────────────
    print("\n" + "="*50)
    print("PHASE 1 — Training classifier head only")
    print("="*50)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["phase1_lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(cfg["phase1_epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_auc"].append(vl_auc)

        print(f"[P1] Epoch {epoch+1:02d}/{cfg['phase1_epochs']} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.1f}% | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc*100:.1f}% AUC: {vl_auc:.3f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), cfg["save_path"])
            print(f"  ✓ Saved best model (val acc: {best_val_acc*100:.1f}%)")

    # ── Phase 2 ──────────────────────────────
    print("\n" + "="*50)
    print("PHASE 2 — Fine-tuning last 2 blocks + head")
    print("="*50)
    model = unfreeze_for_phase2(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["phase2_lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["phase2_epochs"]
    )

    for epoch in range(cfg["phase2_epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_auc"].append(vl_auc)

        print(f"[P2] Epoch {epoch+1:02d}/{cfg['phase2_epochs']} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.1f}% | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc*100:.1f}% AUC: {vl_auc:.3f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), cfg["save_path"])
            print(f"  ✓ Saved best model (val acc: {best_val_acc*100:.1f}%)")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc*100:.1f}%")
    return model, history


# ─────────────────────────────────────────────
#  STAGE 5 — EVALUATION & PLOTS
# ─────────────────────────────────────────────

def plot_history(history, save_path="training_curves.png"):
    total_epochs = len(history["train_loss"])
    epochs = range(1, total_epochs + 1)
    phase_split = CFG["phase1_epochs"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MobileNetV3-Small — Deepfake Detection Training", fontsize=13)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#534AB7")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#E8593C")
    axes[0].axvline(phase_split + 0.5, color="gray", linestyle="--", linewidth=1, label="Phase 1→2")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy + AUC
    axes[1].plot(epochs, [a*100 for a in history["train_acc"]], label="Train Acc %", color="#534AB7")
    axes[1].plot(epochs, [a*100 for a in history["val_acc"]],   label="Val Acc %",   color="#E8593C")
    axes[1].plot(epochs, [a*100 for a in history["val_auc"]],   label="Val AUC×100", color="#0F6E56", linestyle="--")
    axes[1].axvline(phase_split + 0.5, color="gray", linestyle="--", linewidth=1, label="Phase 1→2")
    axes[1].set_title("Accuracy & AUC"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved → {save_path}")
    plt.show()


def full_evaluation(model, val_loader, device):
    """Prints classification report and confusion matrix."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT")
    print("="*50)
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  AUC-ROC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=["Real", "Fake"]))

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Deepfake Detection")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Confusion matrix saved → confusion_matrix.png")
    plt.show()

    return acc, auc


# ─────────────────────────────────────────────
#  STAGE 6 — INFERENCE ON SINGLE VIDEO
# ─────────────────────────────────────────────

def predict_video(video_path: str, model_path: str, cfg: dict) -> dict:
    """
    Predicts whether a video is real or fake.
    Extracts frames, runs model on each, returns majority vote + confidence.
    """
    device = cfg["device"]
    _, val_tf = get_transforms(cfg["img_size"])

    # Load model
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Extract temp frames (will be face-cropped using the updated extract_frames)
    tmp_dir = "tmp_predict_frames"
    n = extract_frames(video_path, tmp_dir, n_frames=cfg["frames_per_vid"])
    if n == 0:
        return {"error": "Could not extract frames from video."}

    from PIL import Image
    fake_scores = []
    for frame_path in sorted(Path(tmp_dir).glob("*.jpg")):
        img = Image.open(str(frame_path)).convert("RGB")
        tensor = val_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            prob_fake = torch.softmax(out, dim=1)[0, 1].item()
        fake_scores.append(prob_fake)

    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    avg_fake_prob = np.mean(fake_scores)
    prediction    = "FAKE" if avg_fake_prob >= 0.5 else "REAL"
    confidence    = avg_fake_prob if prediction == "FAKE" else 1 - avg_fake_prob

    result = {
        "prediction"      : prediction,
        "confidence"      : f"{confidence*100:.1f}%",
        "avg_fake_prob"   : round(avg_fake_prob, 4),
        "frames_analyzed" : n,
    }
    print(f"\nVideo: {video_path}")
    print(f"  Prediction : {result['prediction']}")
    print(f"  Confidence : {result['confidence']}")
    return result


# ─────────────────────────────────────────────
#  MAIN — run the full pipeline
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Deepfake Detection — MobileNetV3-Small (CPU)")
    print("=" * 50)

    # Step 1: Extract frames
    print("\nStep 1: Extracting frames from videos...")
    extract_all_frames(
        dataset_root=CFG["dataset_root"],
        frames_root=CFG["frames_root"],
        n_frames=CFG["frames_per_vid"],
    )

    # Step 2: Build dataloaders
    print("\nStep 2: Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        frames_root=CFG["frames_root"],
        img_size=CFG["img_size"],
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        val_split=CFG["val_split"],
        seed=CFG["seed"],
    )

    # Step 3: Build model
    print("\nStep 3: Building model...")
    model = build_model().to(CFG["device"])

    # Step 4: Train
    print("\nStep 4: Training...")
    model, history = run_training(model, train_loader, val_loader, CFG)

    # Step 5: Plot & evaluate
    print("\nStep 5: Evaluating...")
    plot_history(history)

    # Load best saved model for final eval
    model.load_state_dict(torch.load(CFG["save_path"], map_location=CFG["device"]))
    full_evaluation(model, val_loader, CFG["device"])

    print("\nPipeline complete!")
    print(f"  Model saved  → {CFG['save_path']}")
    print("  Plots saved  → training_curves.png, confusion_matrix.png")
    print("\nTo predict a new video:")
    print('  result = predict_video("path/to/video.mp4", "deepfake_mobilenetv3.pth", CFG)')


if __name__ == "__main__":
    main()
