"""
Verifixia – Deepfake Detector Training  (scikit-learn, any Python version)
===========================================================================
Trains an ensemble (SVM + GradientBoosting) on rich image features:
  • HOG (scikit-image if available, else pure-numpy fallback)
  • Multi-scale RGB + LAB colour histograms
  • DCT frequency features (top-128 coefficients)
  • Statistical moments per channel (mean/std/skew/kurt)
  • Pixel-difference texture (LBP-lite)

Works on Python 3.13/3.14 without PyTorch or TensorFlow.

Usage (from repo root):
    python scripts/train_sklearn.py                 # defaults
    python scripts/train_sklearn.py --n_aug 5       # less aug, faster
    python scripts/train_sklearn.py --n_aug 10      # more aug, better model

Output:  models/deepfake_sklearn.pkl
"""

import argparse
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# ── Optional scikit-image HOG (faster & better than pure-numpy) ───────────
try:
    from skimage.feature import hog as ski_hog  # type: ignore
    from skimage.color import rgb2gray as ski_rgb2gray  # type: ignore
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

# ── Constants ────────────────────────────────────────────────────────────
IMG_SIZE   = (160, 160)   # larger than before → richer HOG
SEED       = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ════════════════════════════════════════════════════════════════════════════

def _numpy_hog(gray: np.ndarray, cell: int = 8, block: int = 2,
               bins: int = 9) -> np.ndarray:
    """Pure-numpy HOG fallback."""
    h, w = gray.shape
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)
    gy[1:-1, :] = gray[2:, :].astype(np.float32) - gray[:-2, :].astype(np.float32)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) * 180 / np.pi) % 180
    cy, cx = h // cell, w // cell
    hist = np.zeros((cy, cx, bins), dtype=np.float32)
    bw = 180.0 / bins
    for bi in range(bins):
        mask = (ang >= bi * bw) & (ang < (bi + 1) * bw)
        for r in range(cy):
            for c in range(cx):
                sl_m = mag[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
                sl_k = mask[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
                hist[r, c, bi] = sl_m[sl_k].sum()
    feats = []
    for r in range(cy - block + 1):
        for c in range(cx - block + 1):
            bh = hist[r:r + block, c:c + block, :].ravel()
            feats.append(bh / (np.sqrt((bh ** 2).sum()) + 1e-6))
    return np.concatenate(feats)


def _hog_features(img_rgb: np.ndarray) -> np.ndarray:
    if _SKIMAGE:
        gray = ski_rgb2gray(img_rgb)
        return ski_hog(
            gray, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)
    else:
        gray = (0.299 * img_rgb[:, :, 0] +
                0.587 * img_rgb[:, :, 1] +
                0.114 * img_rgb[:, :, 2]).astype(np.uint8)
        return _numpy_hog(gray)


def extract_features(img: Image.Image) -> np.ndarray:
    """
    Returns a 1-D float32 feature vector combining:
      HOG | RGB hist | LAB hist | stats | DCT | LBP-lite
    """
    img_r = img.resize(IMG_SIZE).convert("RGB")
    arr   = np.array(img_r, dtype=np.uint8)

    # 1. HOG
    hog_f = _hog_features(arr)

    # 2. RGB colour histograms (64 bins per channel)
    rgb_h = []
    for ch in range(3):
        h, _ = np.histogram(arr[:, :, ch], bins=64, range=(0, 256))
        rgb_h.append(h / (h.sum() + 1e-6))
    rgb_h = np.concatenate(rgb_h).astype(np.float32)

    # 3. LAB colour histograms (32 bins per channel)
    try:
        lab_arr = np.array(img_r.convert("LAB"), dtype=np.uint8)
    except Exception:
        lab_arr = arr
    lab_h = []
    for ch in range(3):
        h, _ = np.histogram(lab_arr[:, :, ch], bins=32, range=(0, 256))
        lab_h.append(h / (h.sum() + 1e-6))
    lab_h = np.concatenate(lab_h).astype(np.float32)

    # 4. Per-channel statistical moments (mean, std, skew, kurt, p5, p95)
    stats = []
    for ch in range(3):
        cd  = arr[:, :, ch].astype(np.float64) / 255.0
        mu  = cd.mean()
        std = cd.std() + 1e-9
        stats += [
            mu, std,
            float(np.mean(((cd - mu) / std) ** 3)),   # skew
            float(np.mean(((cd - mu) / std) ** 4)),   # kurt
            float(np.percentile(cd, 5)),
            float(np.percentile(cd, 95)),
        ]
    stats_f = np.array(stats, dtype=np.float32)

    # 5. DCT-frequency fingerprint (top-128 absolute coefficients)
    gray_f32 = arr.mean(axis=2).astype(np.float32)
    fft_abs  = np.abs(np.fft.rfft2(gray_f32)).ravel()
    top128   = np.sort(fft_abs)[::-1][:128]
    dct_f    = (top128 / (top128.max() + 1e-6)).astype(np.float32)

    # 6. LBP-lite: sign of difference between pixel and 8-neighbours
    #    Gives a 256-bin texture histogram
    pad = np.pad(gray_f32, 1, mode="edge")
    centre = gray_f32
    lbp_code = np.zeros_like(centre, dtype=np.uint8)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for bit, (dr, dc) in enumerate(offsets):
        neighbour = pad[1+dr:1+dr+IMG_SIZE[0], 1+dc:1+dc+IMG_SIZE[1]]
        lbp_code += ((neighbour >= centre).astype(np.uint8) << bit)
    lbp_h, _ = np.histogram(lbp_code.ravel(), bins=256, range=(0, 256))
    lbp_f    = (lbp_h / (lbp_h.sum() + 1e-6)).astype(np.float32)

    return np.concatenate([hog_f, rgb_h, lab_h, stats_f, dct_f, lbp_f])


# ════════════════════════════════════════════════════════════════════════════
# Augmentation
# ════════════════════════════════════════════════════════════════════════════

def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    ops = [
        lambda i: i.transpose(Image.FLIP_LEFT_RIGHT),
        lambda i: i.transpose(Image.FLIP_TOP_BOTTOM),
        lambda i: i.rotate(rng.uniform(-20, 20), fillcolor=(128, 128, 128)),
        lambda i: ImageEnhance.Brightness(i).enhance(rng.uniform(0.6, 1.5)),
        lambda i: ImageEnhance.Contrast(i).enhance(rng.uniform(0.6, 1.5)),
        lambda i: ImageEnhance.Sharpness(i).enhance(rng.uniform(0.5, 2.0)),
        lambda i: ImageEnhance.Color(i).enhance(rng.uniform(0.7, 1.4)),
        lambda i: i.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.0))),
        lambda i: ImageOps.autocontrast(i),
        lambda i: i.crop((
            rng.randint(0, 16), rng.randint(0, 16),
            i.width - rng.randint(0, 16), i.height - rng.randint(0, 16)
        )).resize(i.size),
    ]
    # Apply 2–4 random ops
    for op in rng.sample(ops, k=rng.randint(2, 4)):
        try:
            img = op(img)
        except Exception:
            pass
    return img


# ════════════════════════════════════════════════════════════════════════════
# Dataset loading & feature building
# ════════════════════════════════════════════════════════════════════════════

def load_image_paths(data_dir: Path):
    paths, labels = [], []
    for label, subdir in [(0, "Real"), (1, "Fake")]:
        d = data_dir / subdir
        if not d.exists():
            print(f"  ⚠  Not found: {d}")
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                paths.append(str(f))
                labels.append(label)
    return paths, labels


def build_features(paths, labels, n_aug: int, rng: random.Random,
                   tag: str) -> tuple:
    X, y = [], []
    t0   = time.time()
    n    = len(paths)
    for i, (p, lbl) in enumerate(zip(paths, labels), 1):
        elapsed = time.time() - t0
        eta     = (elapsed / i) * (n - i) if i > 1 else 0
        print(f"\r  {tag} [{i:>4}/{n}]  ETA {eta:>4.0f}s  {Path(p).name[:35]:35s}",
              end="", flush=True)
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"\n  ⚠  Skip {p}: {e}")
            continue
        X.append(extract_features(img))
        y.append(lbl)
        for _ in range(n_aug):
            X.append(extract_features(augment(img, rng)))
            y.append(lbl)
    elapsed = time.time() - t0
    print(f"\r  {tag} done – {len(X)} samples in {elapsed:.1f}s" + " " * 40)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="DATA")
    ap.add_argument("--out",      default="models/deepfake_sklearn.pkl")
    ap.add_argument("--n_aug",    type=int, default=4,
                    help="Augmentation copies per training image (default 4)")
    ap.add_argument("--no_ensemble", action="store_true",
                    help="Train SVM only (faster, no GBM)")
    args = ap.parse_args()

    root     = Path(__file__).resolve().parent.parent
    data_dir = (root / args.data_dir).resolve()
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print("  Verifixia – Deepfake Detector  (re-train with full dataset)")
    print(f"{'='*62}")
    print(f"  Data      : {data_dir}")
    print(f"  Output    : {out_path}")
    print(f"  Img size  : {IMG_SIZE}")
    print(f"  Aug×      : {args.n_aug}  |  scikit-image HOG: {_SKIMAGE}")
    print(f"{'='*62}\n")

    # ── Load & split ──────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    paths, labels = load_image_paths(data_dir)
    if not paths:
        print("❌  No images found. Check DATA/Real and DATA/Fake.")
        sys.exit(1)

    real_n = labels.count(0)
    fake_n = labels.count(1)
    print(f"  Dataset   : {len(paths)} images  (real={real_n}, fake={fake_n})")

    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=SEED
    )
    print(f"  Split     : train={len(tr_p)}  val={len(va_p)}\n")

    rng = random.Random(SEED)

    # ── Feature extraction ────────────────────────────────────────────────
    X_tr, y_tr = build_features(tr_p, tr_l, args.n_aug, rng, "Train")
    X_va, y_va = build_features(va_p, va_l, 0,          rng, "Val  ")

    print(f"\n  Features  : {X_tr.shape[1]} dims per image")
    print(f"  Train set : {X_tr.shape[0]} samples  "
          f"(real={int((y_tr==0).sum())}, fake={int((y_tr==1).sum())})")

    # ── Normalise ─────────────────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)

    # ── Models ────────────────────────────────────────────────────────────
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (accuracy_score, classification_report,
                                  confusion_matrix, roc_auc_score)

    print("\n  Training SVM …")
    t0  = time.time()
    svm = SVC(kernel="rbf", C=20, gamma="scale",
              class_weight="balanced", probability=False, random_state=SEED)
    svm_cal = CalibratedClassifierCV(svm, cv=5, method="isotonic")
    svm_cal.fit(X_tr_sc, y_tr)
    print(f"  SVM done  ({time.time()-t0:.1f}s)")

    if not args.no_ensemble:
        print("\n  Training Gradient Boosting …")
        t0  = time.time()
        gbm = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, min_samples_leaf=5, random_state=SEED,
        )
        gbm.fit(X_tr_sc, y_tr)
        print(f"  GBM done  ({time.time()-t0:.1f}s)")

        # Soft-voting ensemble  (SVM prob + GBM prob averaged)
        print("\n  Building ensemble …")
        ensemble_probs = (
            svm_cal.predict_proba(X_va_sc)[:, 1] * 0.5 +
            gbm.predict_proba(X_va_sc)[:, 1]     * 0.5
        )
        ens_preds = (ensemble_probs > 0.5).astype(int)
        ens_acc   = accuracy_score(y_va, ens_preds)
        svm_acc   = accuracy_score(y_va, svm_cal.predict(X_va_sc))
        gbm_acc   = accuracy_score(y_va, gbm.predict(X_va_sc))

        print(f"\n  Accuracy  → SVM: {svm_acc*100:.2f}%  "
              f"GBM: {gbm_acc*100:.2f}%  Ensemble: {ens_acc*100:.2f}%")

        # Pick best classifier for saving
        if ens_acc >= max(svm_acc, gbm_acc):
            final_probs  = ensemble_probs
            final_preds  = ens_preds
            model_label  = "Ensemble (SVM+GBM)"
            saved_bundle = {
                "type":       "ensemble",
                "svm":        svm_cal,
                "gbm":        gbm,
                "scaler":     scaler,
                "img_size":   IMG_SIZE,
            }
        elif svm_acc >= gbm_acc:
            final_probs  = svm_cal.predict_proba(X_va_sc)[:, 1]
            final_preds  = svm_cal.predict(X_va_sc)
            model_label  = "SVM"
            saved_bundle = {
                "type":       "svm",
                "classifier": svm_cal,
                "scaler":     scaler,
                "img_size":   IMG_SIZE,
            }
        else:
            final_probs  = gbm.predict_proba(X_va_sc)[:, 1]
            final_preds  = gbm.predict(X_va_sc)
            model_label  = "GradientBoosting"
            saved_bundle = {
                "type":       "gbm",
                "classifier": gbm,
                "scaler":     scaler,
                "img_size":   IMG_SIZE,
            }
    else:
        final_probs  = svm_cal.predict_proba(X_va_sc)[:, 1]
        final_preds  = svm_cal.predict(X_va_sc)
        model_label  = "SVM"
        saved_bundle = {
            "type":       "svm",
            "classifier": svm_cal,
            "scaler":     scaler,
            "img_size":   IMG_SIZE,
        }

    # ── Final evaluation ──────────────────────────────────────────────────
    acc = accuracy_score(y_va, final_preds)
    print(f"\n{'='*62}")
    print(f"  Best model  : {model_label}")
    print(f"  Val accuracy: {acc*100:.2f}%")
    print(f"{'='*62}")
    print("\n  Classification Report:")
    print(classification_report(y_va, final_preds,
                                target_names=["Real", "Fake"], digits=4))

    cm = confusion_matrix(y_va, final_preds)
    print("  Confusion Matrix  (rows=actual, cols=predicted):")
    print(f"             Real   Fake")
    print(f"    Real   {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"    Fake   {cm[1][0]:>5}  {cm[1][1]:>5}")
    if len(set(y_va)) > 1:
        auc = roc_auc_score(y_va, final_probs)
        print(f"\n  ROC-AUC     : {auc:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    with open(out_path, "wb") as f:
        pickle.dump(saved_bundle, f)

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"\n✅  Saved {model_label} → {out_path}  ({size_mb:.1f} MB)")
    print("   Backend will load it automatically on next start.\n")


if __name__ == "__main__":
    main()
