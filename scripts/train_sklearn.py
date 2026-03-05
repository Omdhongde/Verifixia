"""
Verifixia – Deepfake Detector Training (scikit-learn, runs on any Python)
=========================================================================
Uses HOG + colour-histogram features with an SVM + calibration.
Works WITHOUT PyTorch/TensorFlow – just scikit-learn + Pillow + numpy.

Produces:  models/deepfake_sklearn.pkl   (used by the backend automatically)
           models/xception_deepfake.pth  is NOT produced by this script –
           see scripts/train_pytorch.py for the full deep-learning version
           (requires Python ≤ 3.12 or a Linux/Colab environment).

Usage (from repo root):
    python scripts/train_sklearn.py

Optional flags:
    --data_dir  PATH     (default: ./DATA)
    --out       PATH     (default: ./models/deepfake_sklearn.pkl)
    --n_aug     INT      Augmentation copies per image (default: 8)
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# ── Feature extractor ──────────────────────────────────────────────────────
IMG_SIZE = (128, 128)   # resize before feature extraction

def _hog_features(gray_arr: np.ndarray, cell=8, block=2, bins=9) -> np.ndarray:
    """Minimal HOG (no external dependencies)."""
    h, w = gray_arr.shape
    cx, cy = cell, cell
    gx = np.zeros_like(gray_arr, dtype=np.float32)
    gy = np.zeros_like(gray_arr, dtype=np.float32)
    gx[:, 1:-1] = gray_arr[:, 2:].astype(np.float32) - gray_arr[:, :-2].astype(np.float32)
    gy[1:-1, :] = gray_arr[2:, :].astype(np.float32) - gray_arr[:-2, :].astype(np.float32)
    mag = np.sqrt(gx**2 + gy**2)
    ang = (np.arctan2(gy, gx) * 180 / np.pi) % 180

    cells_y, cells_x = h // cy, w // cx
    hist = np.zeros((cells_y, cells_x, bins), dtype=np.float32)
    bin_w = 180.0 / bins
    for bi in range(bins):
        lo, hi = bi * bin_w, (bi + 1) * bin_w
        mask = (ang >= lo) & (ang < hi)
        for r in range(cells_y):
            for c in range(cells_x):
                patch_mask = mask[r*cy:(r+1)*cy, c*cx:(c+1)*cx]
                patch_mag  = mag [r*cy:(r+1)*cy, c*cx:(c+1)*cx]
                hist[r, c, bi] = patch_mag[patch_mask].sum()

    # Block normalise
    feats = []
    for r in range(cells_y - block + 1):
        for c in range(cells_x - block + 1):
            block_hist = hist[r:r+block, c:c+block, :].ravel()
            norm = np.sqrt((block_hist**2).sum() + 1e-6)
            feats.append(block_hist / norm)
    return np.concatenate(feats)


def extract_features(img: Image.Image) -> np.ndarray:
    """
    Concatenate:
      • HOG on grayscale
      • RGB colour histogram (32 bins × 3 channels)
      • LAB colour histogram (16 bins × 3 channels)
      • LBP-like texture (difference sign histogram, 256 bins)
      • Statistical moments (mean, std, skew, kurt) × 3 RGB channels
    """
    img_resized = img.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img_resized, dtype=np.uint8)

    # 1. HOG on grayscale
    gray = np.array(img_resized.convert("L"), dtype=np.uint8)
    hog = _hog_features(gray)

    # 2. RGB histogram
    rgb_hist = []
    for ch in range(3):
        h, _ = np.histogram(arr[:, :, ch], bins=32, range=(0, 256))
        rgb_hist.append(h / (h.sum() + 1e-6))
    rgb_hist = np.concatenate(rgb_hist)

    # 3. LAB histogram (approximate via PIL)
    lab = np.array(img_resized.convert("LAB")) if hasattr(Image, "LAB") else \
          np.array(img_resized)   # fallback to RGB if LAB not supported
    lab_hist = []
    for ch in range(3):
        h, _ = np.histogram(lab[:, :, ch], bins=16, range=(0, 256))
        lab_hist.append(h / (h.sum() + 1e-6))
    lab_hist = np.concatenate(lab_hist)

    # 4. Statistical moments per channel
    stats = []
    for ch in range(3):
        ch_data = arr[:, :, ch].astype(np.float32) / 255.0
        mu  = ch_data.mean()
        std = ch_data.std()
        skew = float(np.mean(((ch_data - mu) / (std + 1e-6))**3))
        kurt = float(np.mean(((ch_data - mu) / (std + 1e-6))**4))
        stats.extend([mu, std, skew, kurt])
    stats = np.array(stats, dtype=np.float32)

    # 5. Frequency domain – DCT-like (absolute FFT coefficients, top-64)
    f = np.fft.rfft2(gray.astype(np.float32))
    fabs = np.abs(f).ravel()
    fabs_sorted = np.sort(fabs)[::-1][:64]
    fabs_feat = fabs_sorted / (fabs_sorted.max() + 1e-6)

    return np.concatenate([hog, rgb_hist, lab_hist, stats, fabs_feat]).astype(np.float32)


# ── Augmentation ───────────────────────────────────────────────────────────
def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
    if rng.random() < 0.4:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.4:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.4))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
    if rng.random() < 0.2:
        img = ImageOps.autocontrast(img)
    return img


# ── Dataset loader ─────────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_images(data_dir: Path) -> tuple[list, list]:
    """Returns (paths, labels) for Real(0) and Fake(1)."""
    paths, labels = [], []
    for label, subdir in [(0, "Real"), (1, "Fake")]:
        d = data_dir / subdir
        if not d.exists():
            print(f"  ⚠  {d} not found – skipping")
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                paths.append(str(f))
                labels.append(label)
    return paths, labels


def build_dataset(paths, labels, n_aug: int, rng: random.Random, desc: str):
    X, y = [], []
    total = len(paths)
    print(f"  {desc}: extracting features + {n_aug}× augmentation …")
    for i, (p, label) in enumerate(zip(paths, labels), 1):
        print(f"\r    [{i}/{total}] {Path(p).name[:40]:40s}", end="", flush=True)
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"\n    ⚠  Could not open {p}: {e}")
            continue
        # Original
        X.append(extract_features(img))
        y.append(label)
        # Augmented copies
        for _ in range(n_aug):
            aug = augment(img, rng)
            X.append(extract_features(aug))
            y.append(label)
    print()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="DATA")
    parser.add_argument("--out",      default="models/deepfake_sklearn.pkl")
    parser.add_argument("--n_aug",    type=int, default=8,
                        help="Augmentation copies per image (more = better generalization)")
    args = parser.parse_args()

    root     = Path(__file__).resolve().parent.parent
    data_dir = (root / args.data_dir).resolve()
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Verifixia – Sklearn Deepfake Detector Training")
    print(f"{'='*60}")
    print(f"  Data dir : {data_dir}")
    print(f"  Output   : {out_path}")
    print(f"  Aug×     : {args.n_aug}")
    print(f"{'='*60}\n")

    # ── Load paths ──
    paths, labels = load_images(data_dir)
    if not paths:
        print("❌ No images found. Check DATA/Real and DATA/Fake exist.")
        sys.exit(1)

    real_n = labels.count(0)
    fake_n = labels.count(1)
    print(f"  Found {len(paths)} images  (real={real_n}, fake={fake_n})\n")

    # ── Stratified train/val split ──
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"  Train: {len(train_paths)}  |  Val: {len(val_paths)}\n")

    rng = random.Random(42)

    # ── Build feature arrays ──
    X_train, y_train = build_dataset(train_paths, train_labels, args.n_aug, rng, "Train")
    X_val,   y_val   = build_dataset(val_paths,   val_labels,   0,          rng, "Val  ")

    print(f"\n  Feature shape: {X_train.shape}  ({X_train.shape[1]} features per image)")

    # ── Normalise ──
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # ── Train SVM + calibration ──
    print("\n  Training SVM (this may take ~1-2 min) …")
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_auc_score, accuracy_score)

    svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
              probability=False, random_state=42)
    clf = CalibratedClassifierCV(svm, cv=3, method="isotonic")
    clf.fit(X_train, y_train)

    # ── Evaluate ──
    print("\n  Evaluating …")
    preds  = clf.predict(X_val)
    probs  = clf.predict_proba(X_val)[:, 1]
    acc    = accuracy_score(y_val, preds)

    print(f"\n{'='*60}")
    print(f"  Validation Accuracy : {acc*100:.2f}%")
    print(f"{'='*60}")
    print("\n  Classification Report:")
    print(classification_report(y_val, preds, target_names=["Real", "Fake"], digits=4))

    cm = confusion_matrix(y_val, preds)
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"             Real   Fake")
    print(f"    Real   {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"    Fake   {cm[1][0]:>5}  {cm[1][1]:>5}")

    if len(set(y_val)) > 1:
        auc = roc_auc_score(y_val, probs)
        print(f"\n  ROC-AUC : {auc:.4f}")

    # ── Save ──
    import pickle
    bundle = {"scaler": scaler, "classifier": clf, "img_size": IMG_SIZE}
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n✅  Model saved → {out_path}")
    print("   The backend will load it automatically at startup.\n")


if __name__ == "__main__":
    main()
