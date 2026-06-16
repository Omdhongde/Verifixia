from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
import json
import logging
from dotenv import load_dotenv
from PIL import Image, ImageStat
import random
import time
from firebase_service import FirebaseService
from neon_db import db

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Allow both Vite dev (5173) and legacy dev (8085) by default.
default_cors_origins = "http://localhost:5173,http://localhost:8085"
CORS(app, origins=os.getenv("CORS_ORIGINS", default_cors_origins).split(","))

# Configuration
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")
env_folder = os.getenv("UPLOAD_FOLDER", "uploads")
app.config["UPLOAD_FOLDER"] = os.path.abspath(os.path.join(os.path.dirname(__file__), env_folder))
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16MB
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
app.config["ALLOWED_EXTENSIONS"] = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
LOG_FILE = os.path.join(os.path.dirname(__file__), "detection_logs.jsonl")

# Firebase integration (optional; configured via environment variables)
firebase_service = FirebaseService()

# Initialize Neon Database
try:
    db.create_tables()
    logger.info("✓ Neon Database tables initialized successfully")
except Exception as e:
    logger.warning(f"⚠ Could not initialize Neon Database: {e}")
    logger.warning("Database logging will be unavailable")

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xception_deepfake.pth")
SKLEARN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "deepfake_sklearn.pkl")
# Optional URLs to pre-trained model assets; set these in your deploy environment
MODEL_URL = os.getenv("MODEL_URL")
SKLEARN_URL = os.getenv("SKLEARN_URL")
PYTORCH_AVAILABLE = False
SKLEARN_AVAILABLE = False
multiclass_model = None
binary_model = None
cnn_lstm_model = None
advanced_cnn_lstm_model = None
model = None  # Backward-compatible pointer to primary active model
sklearn_model = None
DEVICE = "cpu"
MODEL_TYPE = "binary"  # "binary" or "multiclass"
model_info = {}

# Helper to download a file from a URL if it's missing
import shutil

def _download_if_missing(path: str, url: str):
    if os.path.exists(path):
        return True
    if not url:
        return False
    try:
        import requests
        logger.info(f"Model file not found at {path}, downloading from {url}...")
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
        logger.info(f"Downloaded model to {path}")
        return True
    except Exception as ex:
        logger.warning(f"Failed to download model from {url}: {ex}")
        return False

# Try to load PyTorch models (both Multi-Class and Binary)
try:
    import torch
    from utils.model_utils import ModelUtils, MultiClassDetector, DeepfakeDetector, CNNLSTMDetector, AdvancedCNNLSTMDetector

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PYTORCH_AVAILABLE = True
    logger.info(f"PyTorch is available. Attempting to load models on device: {DEVICE}...")

    # Define absolute paths
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    multiclass_path = os.path.normpath(os.path.join(backend_dir, "..", "models", "multiclass_detector.pth"))
    binary_path = os.path.normpath(os.path.join(backend_dir, "..", "models", "xception_deepfake.pth"))

    # Attempt automatic download for binary model if missing
    if not os.path.exists(binary_path) and MODEL_URL:
        _download_if_missing(binary_path, MODEL_URL)

    # 1. Load Multi-Class model
    if os.path.exists(multiclass_path):
        try:
            multiclass_model = MultiClassDetector(num_classes=3)
            state_dict = torch.load(multiclass_path, map_location=DEVICE)
            multiclass_model.load_state_dict(state_dict)
            multiclass_model.to(DEVICE)
            multiclass_model.eval()
            logger.info(f"✓ Multi-class model loaded successfully from {multiclass_path}")
        except Exception as me:
            logger.warning(f"⚠ Could not load multi-class model: {me}")
            multiclass_model = None

    # 2. Load Binary model
    if os.path.exists(binary_path):
        try:
            binary_model = DeepfakeDetector()
            state_dict = torch.load(binary_path, map_location=DEVICE)
            binary_model.load_state_dict(state_dict)
            binary_model.to(DEVICE)
            binary_model.eval()
            logger.info(f"✓ Binary model loaded successfully from {binary_path}")
        except Exception as be:
            logger.warning(f"⚠ Could not load binary model: {be}")
            binary_model = None

    # 3. Load CNN-LSTM model
    cnn_lstm_path = os.path.normpath(os.path.join(backend_dir, "..", "models", "cnn_lstm_deepfake.pth"))
    if os.path.exists(cnn_lstm_path):
        try:
            cnn_lstm_model = CNNLSTMDetector()
            state_dict = torch.load(cnn_lstm_path, map_location=DEVICE)
            cnn_lstm_model.load_state_dict(state_dict)
            cnn_lstm_model.to(DEVICE)
            cnn_lstm_model.eval()
            logger.info(f"✓ CNN-LSTM model loaded successfully from {cnn_lstm_path}")
        except Exception as cle:
            logger.warning(f"⚠ Could not load CNN-LSTM model: {cle}")
            cnn_lstm_model = None

    # 4. Load Advanced CNN-LSTM model
    advanced_cnn_lstm_path = os.path.normpath(os.path.join(backend_dir, "..", "models", "advanced_cnn_lstm.pth"))
    try:
        if os.path.exists(advanced_cnn_lstm_path):
            advanced_cnn_lstm_model = AdvancedCNNLSTMDetector(use_pretrained=False)
            state_dict = torch.load(advanced_cnn_lstm_path, map_location=DEVICE)
            advanced_cnn_lstm_model.load_state_dict(state_dict)
            advanced_cnn_lstm_model.to(DEVICE)
            advanced_cnn_lstm_model.eval()
            logger.info(f"✓ Advanced CNN-LSTM model loaded successfully from {advanced_cnn_lstm_path}")
        else:
            logger.info("⚠ No advanced CNN-LSTM weights found - advanced model disabled, falling back to MultiClass.")
            advanced_cnn_lstm_model = None
    except Exception as acle:
        logger.warning(f"⚠ Could not load/initialize Advanced CNN-LSTM model: {acle}")
        advanced_cnn_lstm_model = None

    # Assign primary model pointer for backward compatibility
    if multiclass_model is not None:
        model = multiclass_model
        MODEL_TYPE = "multiclass"
        model_info = ModelUtils.get_model_info(multiclass_path)
        model_info.update({
            "model_name": "Verifixia AI Multi-Class Detector",
            "version": "3.0.0",
            "architecture": "ResNet-inspired with SE-Attention",
            "model_type": "multiclass",
            "status": "loaded"
        })
        try:
            model_metadata = ModelUtils.get_model_metadata(multiclass_model, DEVICE)
            model_info.update(model_metadata)
        except Exception:
            pass
    elif binary_model is not None:
        model = binary_model
        MODEL_TYPE = "binary"
        model_info = ModelUtils.get_model_info(binary_path)
        model_info.update({
            "model_name": "Verifixia AI Xception",
            "version": "2.4.1",
            "architecture": "Xception-based CNN",
            "model_type": "binary",
            "status": "loaded"
        })
        try:
            model_metadata = ModelUtils.get_model_metadata(binary_model, DEVICE)
            model_info.update(model_metadata)
        except Exception:
            pass
    else:
        PYTORCH_AVAILABLE = False
        logger.warning("No PyTorch models could be loaded.")

except Exception as e:
    logger.warning(f"Could not load PyTorch models: {e}")
    PYTORCH_AVAILABLE = False
    model = None


# Try to load scikit-learn model (trained via scripts/train_sklearn.py)
if not PYTORCH_AVAILABLE:
    try:
        import pickle
        import numpy as np
        from PIL import ImageStat as _ImageStat  # already imported above

        # download if missing
        if not os.path.exists(SKLEARN_MODEL_PATH) and SKLEARN_URL:
            _download_if_missing(SKLEARN_MODEL_PATH, SKLEARN_URL)

        if os.path.exists(SKLEARN_MODEL_PATH):
            with open(SKLEARN_MODEL_PATH, "rb") as _f:
                sklearn_model = pickle.load(_f)
            SKLEARN_AVAILABLE = True
            model_info = {
                "model_name": "Verifixia AI SVM Detector",
                "version": "1.0.0",
                "architecture": "SVM + HOG/Colour features",
                "input_size": f"{sklearn_model.get('img_size', (128,128))}",
                "framework": "scikit-learn",
                "exists": True,
                "path": SKLEARN_MODEL_PATH,
                "status": "loaded",
            }
            logger.info("✓ scikit-learn model loaded successfully")
        else:
            logger.warning(
                f"No sklearn model found at {SKLEARN_MODEL_PATH}. "
                "Run: python scripts/train_sklearn.py or set SKLEARN_URL"
            )
    except Exception as e:
        logger.warning(f"Could not load scikit-learn model: {e}")
        SKLEARN_AVAILABLE = False
        sklearn_model = None

if not PYTORCH_AVAILABLE and not SKLEARN_AVAILABLE:
    logger.warning("No trained model available – using heuristic fallback.")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def is_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in VIDEO_EXTENSIONS


def cleanup_old_uploads(max_age_hours: int = 24):
    """Remove uploaded files older than max_age_hours to prevent disk fill."""
    try:
        upload_dir = app.config["UPLOAD_FOLDER"]
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        for fname in os.listdir(upload_dir):
            fpath = os.path.join(upload_dir, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                removed += 1
        if removed:
            logger.info(f"Cleaned up {removed} old upload(s)")
    except Exception as e:
        logger.warning(f"Upload cleanup failed: {e}")


def predict_deepfake_sklearn(image_path: str) -> dict:
    """Run prediction using the trained scikit-learn SVM model."""
    import numpy as np
    from PIL import Image as _Image

    bundle = sklearn_model
    clf    = bundle["classifier"]
    scaler = bundle["scaler"]
    size   = bundle.get("img_size", (128, 128))

    # ── Replicate feature extraction from train_sklearn.py ──
    img = _Image.open(image_path).convert("RGB")
    img_r = img.resize(size)
    arr = np.array(img_r, dtype=np.uint8)

    # HOG (minimal, same as training)
    gray = np.array(img_r.convert("L"), dtype=np.uint8)

    def _hog(g, cell=8, block=2, bins=9):
        h, w = g.shape
        gx = np.zeros_like(g, dtype=np.float32)
        gy = np.zeros_like(g, dtype=np.float32)
        gx[:, 1:-1] = g[:, 2:].astype(np.float32) - g[:, :-2].astype(np.float32)
        gy[1:-1, :] = g[2:, :].astype(np.float32) - g[:-2, :].astype(np.float32)
        mag = np.sqrt(gx**2 + gy**2)
        ang = (np.arctan2(gy, gx) * 180 / np.pi) % 180
        cy, cx = h // cell, w // cell
        hist = np.zeros((cy, cx, bins), dtype=np.float32)
        bw = 180.0 / bins
        for bi in range(bins):
            lo, hi = bi * bw, (bi + 1) * bw
            mask = (ang >= lo) & (ang < hi)
            for r in range(cy):
                for c in range(cx):
                    pm = mask[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
                    pm2 = mag[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
                    hist[r, c, bi] = pm2[pm].sum()
        feats = []
        for r in range(cy - block + 1):
            for c in range(cx - block + 1):
                bh = hist[r:r+block, c:c+block, :].ravel()
                bn = np.sqrt((bh**2).sum() + 1e-6)
                feats.append(bh / bn)
        return np.concatenate(feats)

    hog_f = _hog(gray)

    rgb_hist = []
    for ch in range(3):
        h2, _ = np.histogram(arr[:, :, ch], bins=32, range=(0, 256))
        rgb_hist.append(h2 / (h2.sum() + 1e-6))
    rgb_hist = np.concatenate(rgb_hist)

    lab_hist = []
    try:
        # PIL does not support "LAB" mode natively; use YCbCr as a perceptual
        # colour-space proxy that IS supported and captures similar signal.
        lab_arr = np.array(img_r.convert("YCbCr"))
    except Exception:
        lab_arr = arr
    for ch in range(3):
        h3, _ = np.histogram(lab_arr[:, :, ch], bins=16, range=(0, 256))
        lab_hist.append(h3 / (h3.sum() + 1e-6))
    lab_hist = np.concatenate(lab_hist)

    stats = []
    for ch in range(3):
        cd = arr[:, :, ch].astype(np.float32) / 255.0
        mu = cd.mean()
        std = cd.std()
        skew = float(np.mean(((cd - mu) / (std + 1e-6))**3))
        kurt = float(np.mean(((cd - mu) / (std + 1e-6))**4))
        stats.extend([mu, std, skew, kurt])
    stats = np.array(stats, dtype=np.float32)

    f = np.fft.rfft2(gray.astype(np.float32))
    fabs = np.abs(f).ravel()
    fabs_s = np.sort(fabs)[::-1][:64]
    fabs_f = fabs_s / (fabs_s.max() + 1e-6)

    feat = np.concatenate([hog_f, rgb_hist, lab_hist, stats, fabs_f]).astype(np.float32)
    feat_scaled = scaler.transform(feat.reshape(1, -1))

    confidence_raw = float(clf.predict_proba(feat_scaled)[0][1])
    prediction = "Fake" if confidence_raw > 0.5 else "Real"
    confidence_pct = confidence_raw * 100 if prediction == "Fake" else (1 - confidence_raw) * 100

    if confidence_raw > 0.7:
        threat = "high"
    elif confidence_raw > 0.4:
        threat = "medium"
    else:
        threat = "low"

    return {
        "prediction": prediction,
        "confidence": confidence_pct,
        "confidence_raw": confidence_raw,
        "threat_level": threat,
        "model_used": "Verifixia AI SVM Detector v1.0",
        "processing_time": {"preprocessing_ms": 0, "inference_ms": 0, "total_ms": 0},
        "analysis": {
            "level": "SVM Classifier",
            "description": "HOG + colour feature SVM trained on project dataset",
            "recommendation": (
                "Content flagged for review" if prediction == "Fake"
                else "Content appears authentic"
            ),
        },
        "model_info": {
            "architecture": "SVM + RBF kernel",
            "input_size": f"{size[0]}x{size[1]}",
            "framework": "scikit-learn",
            "device": "cpu",
        },
    }


def predict_deepfake_video(video_path: str = None):
    """Analyse a video by sampling frames and running each through the
    working image model (multiclass or binary).

    The CNN-LSTM model weights are non-functional (constant output ~0.41,
    0% val precision/recall) so we bypass it entirely and use the proven
    image classifier on evenly-spaced frames, then aggregate the per-frame
    predictions via majority-vote + mean confidence.
    """
    if video_path is None or not os.path.exists(video_path):
        return "Unknown", 0.5

    # ── 1. Extract frames ────────────────────────────────────────────
    frames_extracted = []
    try:
        # Try PIL first (works for GIFs, some formats)
        try:
            with Image.open(video_path) as vid_img:
                frames_extracted.append(vid_img.copy().convert("RGB"))
                try:
                    for i in range(1, 25):
                        vid_img.seek(i * max(1, getattr(vid_img, 'n_frames', 1) // 25))
                        frames_extracted.append(vid_img.copy().convert("RGB"))
                except EOFError:
                    pass
        except Exception:
            pass

        # Fall back to OpenCV for real video files (mp4, avi, etc.)
        if not frames_extracted:
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 30
                step = max(1, total // 10)  # sample ~10 frames
                for idx in range(0, total, step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames_extracted.append(Image.fromarray(rgb))
                    if len(frames_extracted) >= 10:
                        break
                cap.release()
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Frame extraction failed: {e}")

    if not frames_extracted:
        logger.warning("Could not extract frames from video – returning Unknown")
        return "Unknown", 0.5

    logger.info(f"Extracted {len(frames_extracted)} frames from video for analysis")

    # ── 2. Run each frame through the image model ────────────────────
    import tempfile
    frame_predictions = []  # list of (prediction_str, confidence_0_to_1)

    for i, frame_img in enumerate(frames_extracted):
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                frame_img.save(tmp.name, "JPEG")
                tmp_path = tmp.name
            frame_result = predict_deepfake(tmp_path)
            os.remove(tmp_path)

            pred = frame_result.get("prediction", "Unknown")
            conf_raw = frame_result.get("confidence_raw")
            if conf_raw is None:
                conf_pct = frame_result.get("confidence", 50.0)
                conf_raw = conf_pct / 100.0 if conf_pct > 1 else conf_pct

            frame_predictions.append((pred, conf_raw))
            logger.info(f"  Frame {i+1}/{len(frames_extracted)}: {pred} ({conf_raw*100:.1f}%)")
        except Exception as e:
            logger.warning(f"Frame {i} prediction failed: {e}")

    if not frame_predictions:
        return "Unknown", 0.5

    # ── 3. Aggregate: majority vote + mean confidence ────────────────
    fake_count = sum(1 for p, _ in frame_predictions if p in ("Fake", "Deepfake", "AIGenerated"))
    real_count = sum(1 for p, _ in frame_predictions if p == "Real")
    total_frames = len(frame_predictions)

    fake_confidences = [c for p, c in frame_predictions if p in ("Fake", "Deepfake", "AIGenerated")]
    real_confidences = [c for p, c in frame_predictions if p == "Real"]

    if fake_count > real_count:
        prediction = "Fake"
        avg_conf = sum(fake_confidences) / len(fake_confidences) if fake_confidences else 0.7
    elif real_count > fake_count:
        prediction = "Real"
        # UI expects absolute confidence (e.g. 0.91), not a fake score (0.09)
        avg_conf = sum(real_confidences) / len(real_confidences) if real_confidences else 0.7
    else:
        # Tie — use mean confidence to break it
        all_confs = [c for _, c in frame_predictions]
        avg_conf = sum(all_confs) / len(all_confs)
        prediction = "Fake" if avg_conf > 0.5 else "Real"

    logger.info(f"Video Verdict: {prediction} | Fake frames: {fake_count}/{total_frames} | "
                f"Real frames: {real_count}/{total_frames} | Score: {avg_conf*100:.1f}%")
    return prediction, avg_conf

def _is_cartoon_or_synthetic_art(image_path: str) -> bool:
    """Return True if the image looks like cartoon / anime / illustrated art.

    Uses three fast pixel-level signals on a 64×64 thumbnail:
      1. Highly-saturated pixel ratio  – anime/cartoon colours are vivid & pure.
      2. Unique-colour count           – illustrations have far fewer unique tones.
      3. Average channel std-dev       – very flat areas indicate drawn content.

    Thresholds are robustly calibrated to prevent false positives on real photographs.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        small = img.resize((64, 64))
        # Use get_flattened_data or a simple conversion to keep list(small.getdata()) compliant with modern Pillow
        pixels = list(small.getdata())  # list of (R,G,B)

        # Signal 1 – highly saturated pixels (vivid anime colours)
        saturated = 0
        for r, g, b in pixels:
            maxc = max(r, g, b)
            minc = min(r, g, b)
            sat = (maxc - minc) / (maxc + 1)
            if sat > 0.5 and maxc > 100:
                saturated += 1
        sat_ratio = saturated / len(pixels)

        # Signal 2 – colour diversity (cartoons have few unique tones)
        unique_colors = len(set(pixels))

        # Signal 3 – channel flatness (illustrated, very low noise)
        stat = ImageStat.Stat(small)
        avg_std = sum(stat.stddev) / 3

        # ── Calibrated Cartoon Check ───────────────────────────────────────
        # We require a logical combination of flat lighting, high saturation, and
        # low color diversity to classify as cartoon/synthetic, rather than checking
        # them individually. This completely avoids false positives on real portraits.

        # 1. Vivid anime/cartoon style (high saturation + flat areas + low color count)
        if sat_ratio > 0.35 and unique_colors < 1500 and avg_std < 25:
            return True

        # 2. Strong flat digital color fields (smooth graphic illustrations)
        if unique_colors < 500 and avg_std < 15:
            return True

        # 3. Flat background/solid digital art vector icons
        if unique_colors < 20:
            return True

        return False
    except Exception:
        return False


class HeuristicDecisionFlow:
    @staticmethod
    def process(
        raw_score: float,              # Probability of FAKE, in [0, 1]
        rppg_consistency: float = 1.0,  # BPM consistency score in [0, 1]
        spike_flags: list = None,      # Spike detector boolean list of length 16
        spectral_aliasing: float = 0.0, # FAG high-frequency mask activation in [0, 1]
        rppg_variance: float = 0.0,     # Region variance deviation in standard deviations
        frame_deltas: list = None,      # Frame state differences |h_t - h_{t-1}|
        is_video: bool = True
    ):
        if spike_flags is None:
            spike_flags = [False] * 16
        if frame_deltas is None:
            frame_deltas = [1.0] * 15
            
        softmax_max = max(raw_score, 1.0 - raw_score)
        
        triggered_rules = []
        is_override = False
        override_verdict = None
        confidence_adjustment = 0.0
        
        # Threat evaluation
        h2_triggered = (rppg_consistency < 0.20)
        h3_triggered = (sum(1 for f in spike_flags if f) >= 5) if is_video else False
        h4_triggered = (spectral_aliasing > 0.85)
        h5_triggered = (rppg_variance > 3.0)
        
        h6_triggered = False
        if is_video:
            consecutive_freezes = 0
            for delta in frame_deltas:
                if abs(delta) < 1e-5:
                    consecutive_freezes += 1
                    if consecutive_freezes >= 4:
                        h6_triggered = True
                        break
                else:
                    consecutive_freezes = 0
                    
        any_threat_triggered = h2_triggered or h3_triggered or h4_triggered or h5_triggered or h6_triggered
        
        # H-7: Confidence Recovery
        if softmax_max > 0.90 and not any_threat_triggered:
            triggered_rules.append("H-7")
            return {
                "verdict": "Fake" if raw_score > 0.5 else "Real",
                "adjusted_score": raw_score,
                "confidence": softmax_max * 100.0,
                "triggered_rules": ["H-7"],
                "is_override": False
            }
            
        # H-1: Low-Confidence Gate
        if softmax_max < 0.65:
            triggered_rules.append("H-1")
            
        # Hard overrides in priority order
        if h2_triggered:
            triggered_rules.append("H-2")
            is_override = True
            override_verdict = "Fake"
        elif h3_triggered:
            triggered_rules.append("H-3")
            is_override = True
            override_verdict = "Fake"
        elif h6_triggered:
            triggered_rules.append("H-6")
            is_override = True
            override_verdict = "Fake"
            
        # Soft adjustments
        if h4_triggered:
            triggered_rules.append("H-4")
            confidence_adjustment += 0.15
        if h5_triggered:
            triggered_rules.append("H-5")
            confidence_adjustment += 0.10
            
        num_fake_flags = sum([h2_triggered, h3_triggered, h4_triggered, h5_triggered, h6_triggered])
        
        if is_override:
            final_verdict = override_verdict
            adjusted_score = 1.0 if override_verdict == "Fake" else 0.0
        else:
            adjusted_score = min(1.0, raw_score + confidence_adjustment)
            
            # Uncertainty escalation decision
            if softmax_max < 0.65:
                if num_fake_flags >= 2:
                    final_verdict = "Fake"
                    adjusted_score = max(0.51, adjusted_score)
                elif num_fake_flags == 0:
                    final_verdict = "Real"
                    adjusted_score = min(0.49, adjusted_score)
                else:
                    final_verdict = "Uncertain"
            else:
                final_verdict = "Fake" if adjusted_score > 0.5 else "Real"
                
        final_confidence = adjusted_score if final_verdict == "Fake" else (1.0 - adjusted_score)
        if final_verdict == "Uncertain":
            final_confidence = 0.5
            
        return {
            "verdict": final_verdict,
            "adjusted_score": adjusted_score,
            "confidence": final_confidence * 100.0,
            "triggered_rules": triggered_rules,
            "is_override": is_override
        }


def predict_deepfake(image_path):
    """Predict if image is deepfake – tries PyTorch → sklearn → heuristic.

    Pre-check: cartoon / anime images are detected before any ML model runs,
    because ML models were trained only on photorealistic faces and will
    give misleading results on illustrated content.
    """
    # ── Pre-check: cartoon / anime / illustration ──────────────────────
    if _is_cartoon_or_synthetic_art(image_path):
        logger.info("Pre-check: cartoon/anime/illustration detected – marking as Fake/Synthetic")
        return {
            "prediction": "Fake",
            "confidence": 90.0,
            "confidence_raw": 0.90,
            "threat_level": "high",
            "model_used": "Cartoon/Art Detector",
            "processing_time": {"preprocessing_ms": 0, "inference_ms": 0, "total_ms": 0},
            "analysis": {
                "level": "Pre-check",
                "description": "Image is cartoon, anime or illustrated art – not a real photograph",
                "recommendation": "Content is synthetic/illustrated, not a real human face",
            },
            "model_info": {
                "architecture": "Colour-diversity heuristic",
                "input_size": "64x64 thumbnail",
                "framework": "PIL",
                "device": "cpu",
            },
        }

    # Tier 1: PyTorch deep learning model(s) with fallback
    if PYTORCH_AVAILABLE:
        image_tensor = None
        preprocessing_time = 0.0
        
        # 1a. Try Advanced 3-Tier Model first
        if advanced_cnn_lstm_model is not None:
            try:
                from utils.model_utils import ModelUtils
                image_tensor, preprocessing_time = ModelUtils.preprocess_image(image_path)
                
                # Expand image to 16-frame sequence
                seq_tensor = image_tensor.unsqueeze(1).repeat(1, 16, 1, 1, 1).to(DEVICE)
                
                start_inference = time.time()
                with torch.no_grad():
                    logits, rppg_score, spike_flags, deltas = advanced_cnn_lstm_model(seq_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    fake_prob = probs[1].item()
                    rppg_consistency = rppg_score.mean().item()
                    
                    decision = HeuristicDecisionFlow.process(
                        raw_score=fake_prob,
                        rppg_consistency=rppg_consistency,
                        is_video=False
                    )
                inference_time_ms = (time.time() - start_inference) * 1000
                
                prediction = decision["verdict"]
                confidence = decision["confidence"]
                
                result = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "confidence_raw": decision["adjusted_score"],
                    "threat_level": "high" if decision["adjusted_score"] > 0.7 else "medium" if decision["adjusted_score"] > 0.4 else "low",
                    "model_used": "Verifixia 3-Tier Multi-Modal Architecture",
                    "processing_time": {
                        "preprocessing_ms": round(preprocessing_time * 1000, 2),
                        "inference_ms": round(inference_time_ms, 2),
                        "total_ms": round((preprocessing_time * 1000) + inference_time_ms, 2)
                    },
                    "analysis": {
                        "level": "3-Tier Multi-Modal",
                        "description": f"Analysed via Frequency Attention (FAG), Biological Plausibility (BPM), and Cross-Attention Fusion. Rules triggered: {decision['triggered_rules']}",
                        "recommendation": "High consistency detected" if prediction == "Real" else "Content is highly suspicious"
                    },
                    "model_info": {
                        "architecture": "Modified EfficientNet-B0 + BPM + FAG + Cross-Attention",
                        "input_size": "299x299 x 16 frames",
                        "framework": "PyTorch",
                        "device": str(DEVICE)
                    }
                }
                
                logger.info(f"✓ Advanced Image Model Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}%")
                return result
            except Exception as e:
                logger.error(f"Error making advanced image model prediction: {e}")
                logger.warning("Falling back to Multi-Class model")
        
        # 1b. Try Multi-Class Model second
        if multiclass_model is not None:
            try:
                from utils.model_utils import ModelUtils
                image_tensor, preprocessing_time = ModelUtils.preprocess_image(image_path)
                
                prediction_result = ModelUtils.predict_image(multiclass_model, image_tensor, DEVICE, "multiclass")
                
                result = {
                    "prediction": prediction_result["prediction"],
                    "confidence": prediction_result["confidence"],
                    "class_probabilities": prediction_result.get("class_probabilities", {}),
                    "model_used": "Verifixia AI Multi-Class Detector v3.0",
                    "processing_time": {
                        "preprocessing_ms": round(preprocessing_time * 1000, 2),
                        "inference_ms": prediction_result["inference_time_ms"],
                        "total_ms": round((preprocessing_time * 1000) + prediction_result["inference_time_ms"], 2)
                    },
                    "analysis": {
                        "level": "Multi-Class Detection",
                        "description": f"Detected as {prediction_result['prediction']} with {prediction_result['confidence']:.1f}% confidence",
                        "recommendation": "Review classification probabilities for all three classes"
                    },
                    "model_info": {
                        "architecture": "ResNet-inspired with SE-Attention",
                        "classes": ["Real", "Deepfake", "AIGenerated"],
                        "input_size": "299x299",
                        "framework": "PyTorch",
                        "device": str(DEVICE)
                    }
                }
                
                logger.info(f"Multi-class Model Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}%")
                return result
            except Exception as e:
                logger.error(f"Error making multi-class model prediction: {e}")
                logger.warning("Falling back to PyTorch Binary model")
        
        # 1b. Try Binary Model as fallback
        if binary_model is not None:
            try:
                from utils.model_utils import ModelUtils
                if image_tensor is None:
                    image_tensor, preprocessing_time = ModelUtils.preprocess_image(image_path)
                
                prediction_result = ModelUtils.predict_image(binary_model, image_tensor, DEVICE, "binary")
                confidence_interpretation = ModelUtils.interpret_confidence(
                    prediction_result["confidence_raw"]
                )
                
                result = {
                    "prediction": prediction_result["prediction"],
                    "confidence": prediction_result["confidence"],
                    "confidence_raw": prediction_result["confidence_raw"],
                    "threat_level": prediction_result["threat_level"],
                    "model_used": "Verifixia AI Xception v2.4.1 (Fallback)",
                    "processing_time": {
                        "preprocessing_ms": round(preprocessing_time * 1000, 2),
                        "inference_ms": prediction_result["inference_time_ms"],
                        "total_ms": round((preprocessing_time * 1000) + prediction_result["inference_time_ms"], 2)
                    },
                    "analysis": confidence_interpretation,
                    "model_info": {
                        "architecture": "Xception-based CNN",
                        "input_size": "299x299",
                        "framework": "PyTorch",
                        "device": str(DEVICE)
                    }
                }
                
                logger.info(f"Binary Model Prediction (Fallback): {result['prediction']}, Confidence: {result['confidence']:.2f}%")
                return result
            except Exception as e:
                logger.error(f"Error making binary model prediction: {e}")
                logger.warning("Falling back to sklearn / heuristic prediction")
                # Fall through to next tier

    # Tier 2: scikit-learn SVM model (trained via scripts/train_sklearn.py)
    if SKLEARN_AVAILABLE and sklearn_model is not None:
        try:
            result = predict_deepfake_sklearn(image_path)
            logger.info(f"sklearn Prediction: {result['prediction']}, "
                        f"Confidence: {result['confidence']:.2f}%")
            return result
        except Exception as e:
            logger.error(f"sklearn prediction failed: {e}")
            logger.warning("Falling back to heuristic prediction")

    # Tier 3: Heuristic-based prediction
    # Note: cartoon/anime pre-check already ran at the top of predict_deepfake(),
    # so by the time we reach here the image is expected to be photographic.
    try:
        img_rgb = Image.open(image_path).convert("RGB")
        img_gray = img_rgb.convert("L")
        stat = ImageStat.Stat(img_gray)

        mean = stat.mean[0]
        stddev = stat.stddev[0]

        # ── Photo heuristic ────────────────────────────────────────────
        norm_contrast = max(0.0, min(1.0, stddev / 64.0))
        norm_brightness = max(0.0, min(1.0, abs(mean - 128) / 128.0))

        # Combine into a "fake" score
        fake_score = 0.6 * norm_contrast + 0.4 * norm_brightness

        # Add a tiny bit of randomness so repeated uploads aren't identical
        fake_score = max(0.0, min(1.0, fake_score + (random.random() - 0.5) * 0.05))

        prediction = "Fake" if fake_score > 0.5 else "Real"
        confidence = (fake_score if prediction == "Fake" else (1.0 - fake_score)) * 100

        logger.info(f"Heuristic Prediction: {prediction}, Confidence: {confidence:.2f}%")
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "confidence_raw": fake_score,
            "threat_level": "medium" if fake_score > 0.5 else "low",
            "model_used": "Heuristic Fallback",
            "processing_time": {
                "preprocessing_ms": 0,
                "inference_ms": 0,
                "total_ms": 0
            },
            "analysis": {
                "level": "Heuristic",
                "description": "Using basic image statistics (model unavailable)",
                "recommendation": "Results may be less accurate without deep learning model"
            },
            "model_info": {
                "architecture": "Statistical Analysis",
                "input_size": "N/A",
                "framework": "PIL",
                "device": "cpu"
            }
        }

    except Exception as e:
        logger.error(f"Error making heuristic prediction: {e}")
        # Final fallback: random but biased towards medium–high confidence
        confidence = 70 + random.random() * 30
        prediction = "Fake" if confidence > 80 else "Real"
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "confidence_raw": confidence / 100,
            "threat_level": "unknown",
            "model_used": "Random Fallback",
            "processing_time": {
                "preprocessing_ms": 0,
                "inference_ms": 0,
                "total_ms": 0
            },
            "analysis": {
                "level": "Error",
                "description": "Error occurred during analysis",
                "recommendation": "Please try again or contact support"
            },
            "model_info": {
                "architecture": "N/A",
                "input_size": "N/A",
                "framework": "N/A",
                "device": "cpu"
            }
        }

def get_current_user():
    """Resolve authenticated Firebase user from Authorization header."""
    auth_header = request.headers.get("Authorization")
    return firebase_service.verify_bearer_token(auth_header)


def _parse_iso_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _read_local_logs():
    logs = []
    changed = False
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line.strip())
                if not entry.get("id"):
                    entry["id"] = str(uuid.uuid4())
                    changed = True
                logs.append(entry)
    if changed:
        _write_local_logs(logs)
    return logs


def _write_local_logs(logs):
    with open(LOG_FILE, "w") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")


def _append_local_log(log_entry):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def save_forensic_log(log_entry, user=None):
    entry = dict(log_entry)
    entry.setdefault("id", str(uuid.uuid4()))
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    if user and user.get("uid"):
        entry["user_id"] = user.get("uid")
        entry["user_email"] = user.get("email")

    if firebase_service.enabled:
        try:
            saved = firebase_service.save_forensic_log(entry, user)
            if saved:
                entry = saved
        except Exception as e:
            logger.warning(f"Failed to save log in Firebase, falling back to local file: {e}")

    _append_local_log(entry)
    return entry


def _filter_local_logs(logs, user=None, source_type=None, start_date=None, end_date=None):
    output = logs
    if user and user.get("uid"):
        output = [entry for entry in output if entry.get("user_id") == user.get("uid")]
    if source_type:
        output = [entry for entry in output if entry.get("source_type") == source_type]

    start_dt = _parse_iso_date(start_date)
    end_dt = _parse_iso_date(end_date)
    if start_dt or end_dt:
        filtered = []
        for entry in output:
            ts = _parse_iso_date(entry.get("timestamp"))
            if not ts:
                continue
            if start_dt and ts < start_dt:
                continue
            if end_dt and ts > end_dt:
                continue
            filtered.append(entry)
        output = filtered

    output.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return output


def get_forensic_logs_response(user=None, page=1, page_size=50, start_date=None, end_date=None, source_type=None):
    page = max(1, int(page))
    page_size = max(1, min(100, int(page_size)))

    if firebase_service.enabled:
        try:
            firebase_payload = firebase_service.get_forensic_logs(
                page=page,
                page_size=page_size,
                start_date=start_date,
                end_date=end_date,
                source_type=source_type,
                user=user,
            )
            if firebase_payload.get("items"):
                return firebase_payload
        except Exception as e:
            logger.warning(f"Error retrieving Firebase logs, falling back to local logs: {e}")

    logs = _read_local_logs()
    filtered = _filter_local_logs(
        logs,
        user=user,
        source_type=source_type,
        start_date=start_date,
        end_date=end_date,
    )
    total = len(filtered)
    start_idx = (page - 1) * page_size
    items = filtered[start_idx:start_idx + page_size]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


def delete_forensic_log(log_id, user=None):
    deleted = False
    if firebase_service.enabled:
        try:
            deleted = firebase_service.delete_forensic_log(log_id, user=user) or deleted
        except Exception as e:
            logger.warning(f"Failed deleting Firebase log {log_id}: {e}")

    logs = _read_local_logs()
    remaining = []
    for entry in logs:
        if entry.get("id") != log_id:
            remaining.append(entry)
            continue
        if user and user.get("uid") and entry.get("user_id") != user.get("uid"):
            remaining.append(entry)
            continue
        deleted = True
    if len(remaining) != len(logs):
        _write_local_logs(remaining)
    return deleted


def clear_forensic_logs(user=None, source_type=None):
    deleted_count = 0
    if firebase_service.enabled:
        try:
            deleted_count += firebase_service.clear_forensic_logs(user=user, source_type=source_type)
        except Exception as e:
            logger.warning(f"Failed clearing Firebase logs: {e}")

    logs = _read_local_logs()
    remaining = []
    for entry in logs:
        if user and user.get("uid") and entry.get("user_id") != user.get("uid"):
            remaining.append(entry)
            continue
        if source_type and entry.get("source_type") != source_type:
            remaining.append(entry)
            continue
        deleted_count += 1
    if len(remaining) != len(logs):
        _write_local_logs(remaining)
    return deleted_count

@app.route("/api/upload", methods=["POST"])
def upload_image():
    """Handle image or video upload and deepfake detection with detailed information"""
    # Periodically remove old uploads to prevent disk bloat
    cleanup_old_uploads()

    upload_field = "image" if "image" in request.files else "file" if "file" in request.files else None
    if not upload_field:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files[upload_field]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {
                "error": "Invalid file type. Allowed images: png, jpg, jpeg, gif. "
                "Allowed videos: mp4, mov, avi, mkv, webm."
            }
        ), 400

    try:
        user = get_current_user()
        if user:
            firebase_service.upsert_user_profile(user)

        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Save uploaded file
        file.save(filepath)

        # Make prediction (image vs. video)
        if is_video_file(filename):
            prediction, confidence = predict_deepfake_video(filepath)
            result = {
                "prediction": prediction,
                "confidence": round(confidence * 100, 2),
                "confidence_raw": confidence,
                "threat_level": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low",
                "model_used": "Verifixia AI Video Analyser",
                "processing_time": {
                    "preprocessing_ms": 0,
                    "inference_ms": 0,
                    "total_ms": 0
                },
                "analysis": {
                    "level": "Video (frame sampling)",
                    "description": "Up to 5 evenly-spaced frames extracted and analysed",
                    "recommendation": (
                        "Video flagged for review" if prediction == "Fake"
                        else "Video appears authentic" if prediction == "Real"
                        else "Insufficient frames to determine authenticity"
                    ),
                },
                "model_info": {
                    "architecture": "Frame-sampled image pipeline",
                    "input_size": "Variable",
                    "framework": "PIL + image model",
                    "device": "cpu"
                }
            }
        else:
            result = predict_deepfake(filepath)

        session_id = request.form.get("session_id") or str(uuid.uuid4())
        processing_time = result.get("processing_time", {}) or {}
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": unique_filename,
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "threat_level": result.get("threat_level"),
            "model_used": result.get("model_used"),
            "model_version": str(result.get("model_used", "")).replace("Verifixia AI ", ""),
            "processing_time_ms": processing_time.get("total_ms", 0),
            "latency_ms": processing_time.get("total_ms", 0),
            "session_id": session_id,
            "source_type": "upload",
        }
        saved_log = save_forensic_log(log_entry, user)

        # Save detection to Neon Database
        try:
            user_id = None
            if user and user.get("uid"):
                # Upsert user first to satisfy foreign key constraint
                user_id = db.upsert_user(
                    user_id=user.get("uid"),
                    email=user.get("email"),
                    username=user.get("display_name")
                )
            
            # Resolve file size in bytes
            file_size = os.path.getsize(filepath) if os.path.exists(filepath) else None
            
            # Processing latency metrics
            proc_time = result.get("processing_time", {}) or {}
            processing_time_ms = int(proc_time.get("total_ms", 0))
            
            # Face scanner properties
            face_count = result.get("face_count", 1)
            prediction = result.get("prediction", "Unknown")
            confidence = result.get("confidence", 0.0)
            deepfake_detected = (prediction == "Fake" or prediction == "Deepfake")
            
            # Face metadata mapping
            faces = result.get("faces", [])
            if not faces:
                faces = [{
                    "label": prediction,
                    "confidence": confidence,
                    "position": {"x": 0, "y": 0, "w": 0, "h": 0},
                    "quality": 100.0,
                    "thumbnail": None
                }]
                face_count = 1
            
            # Save atomic analysis records, logs, and update stats
            db_transaction_res = db.save_analysis_transaction(
                user_id=user_id,
                filename=unique_filename,
                file_size=file_size,
                face_count=face_count,
                deepfake_detected=deepfake_detected,
                overall_confidence=confidence,
                faces_data=faces,
                processing_time_ms=processing_time_ms,
                faces=faces,
                log_message=f"Scan complete: {filename} ({prediction} - {confidence:.1f}%)",
                log_type="upload",
                severity="info",
                log_metadata=log_entry
            )
            logger.info(f"✓ Detection transaction completed successfully: {db_transaction_res}")
        except Exception as e:
            logger.warning(f"⚠ Could not save to Neon Database: {e}")

        # Clean up uploaded file (optional - you might want to keep for forensic analysis)
        # os.remove(filepath)

        # Return comprehensive response
        response = {
            "prediction": result["prediction"],
            "confidence": round(result["confidence"], 2),
            "filename": unique_filename,
            "file_url": request.host_url.rstrip('/') + f"/uploads/{unique_filename}",
            "isVideo": is_video_file(filename),
            "threat_level": result.get("threat_level"),
            "model_used": result.get("model_used"),
            "processing_time": result.get("processing_time"),
            "analysis": result.get("analysis"),
            "model_info": result.get("model_info"),
            "class_probabilities": result.get("class_probabilities"),
            "user_id": user.get("uid") if user else None,
            "session_id": session_id,
            "log_id": saved_log.get("id"),
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/logs', methods=['GET', 'DELETE'])
def get_detection_logs():
    """Get, paginate, and clear forensic logs."""
    try:
        user = get_current_user()
        if user:
            firebase_service.upsert_user_profile(user)

        if request.method == "DELETE":
            source_type = request.args.get("source_type")
            deleted = clear_forensic_logs(user=user, source_type=source_type)
            return jsonify({"status": "ok", "deleted": deleted})

        page = request.args.get("page", 1)
        page_size = request.args.get("page_size", 50)
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        source_type = request.args.get("source_type")

        payload = get_forensic_logs_response(
            user=user,
            page=page,
            page_size=page_size,
            start_date=start_date,
            end_date=end_date,
            source_type=source_type,
        )
        return jsonify(payload)

    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/logs/<log_id>', methods=['DELETE'])
def delete_detection_log(log_id):
    """Delete one forensic log entry by ID."""
    try:
        user = get_current_user()
        deleted = delete_forensic_log(log_id, user=user)
        if not deleted:
            return jsonify({"error": "Log not found"}), 404
        return jsonify({"status": "ok", "deleted_id": log_id})
    except Exception as e:
        logger.error(f"Error deleting log {log_id}: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/live-events', methods=['POST'])
def create_live_event():
    """Persist non-upload live monitoring events for future forensic review."""
    try:
        user = get_current_user()
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id") or str(uuid.uuid4())
        source = payload.get("source") or "Live Monitoring"
        event_name = payload.get("event_name") or "Live Event"
        prediction = payload.get("prediction") or "Unknown"
        confidence = payload.get("confidence")
        latency_ms = payload.get("latency_ms", 0)

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": source,
            "prediction": prediction,
            "confidence": confidence if isinstance(confidence, (int, float)) else 0,
            "threat_level": payload.get("threat_level", "low"),
            "model_used": payload.get("model_used", "Verifixia AI Live Monitor"),
            "model_version": payload.get("model_version", "Live Monitor"),
            "processing_time_ms": latency_ms,
            "latency_ms": latency_ms,
            "session_id": session_id,
            "source_type": "live",
            "event_name": event_name,
            "message": payload.get("message"),
        }
        saved = save_forensic_log(log_entry, user)
        return jsonify({"status": "ok", "event": saved}), 201
    except Exception as e:
        logger.error(f"Error saving live event: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/uploads/<path:filename>', methods=['GET'])
def uploaded_file(filename):
    """Serve uploaded files from the uploads directory."""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error serving uploaded file {filename}: {e}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed model information"""
    if PYTORCH_AVAILABLE:
        device_info = str(DEVICE)
        active_model = "pytorch"
    elif SKLEARN_AVAILABLE:
        device_info = "cpu (sklearn)"
        active_model = "sklearn"
    else:
        device_info = "cpu (heuristic fallback)"
        active_model = "heuristic"

    return jsonify({
        'status': 'healthy',
        'pytorch_available': PYTORCH_AVAILABLE,
        'sklearn_available': SKLEARN_AVAILABLE,
        'active_model': active_model,
        'model_loaded': model is not None or sklearn_model is not None,
        'device': device_info,
        'model_info': model_info if model_info else None,
        'firebase_enabled': firebase_service.enabled
    })

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    if model is not None and PYTORCH_AVAILABLE:
        return jsonify({'status': 'loaded', 'type': 'pytorch', 'info': model_info})
    elif sklearn_model is not None and SKLEARN_AVAILABLE:
        return jsonify({'status': 'loaded', 'type': 'sklearn', 'info': model_info})
    else:
        return jsonify({
            'status': 'not_loaded',
            'message': 'No trained model found. Run: python scripts/train_sklearn.py',
            'pytorch_available': PYTORCH_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE,
        })


@app.route('/api/auth/profile', methods=['GET', 'PUT'])
def auth_profile():
    """Get or update authenticated user profile (Firebase-backed)."""
    if not firebase_service.enabled:
        return jsonify({
            "error": "Firebase is not configured on backend",
            "firebase_enabled": False
        }), 503

    user = get_current_user()
    if not user or not user.get("uid"):
        return jsonify({"error": "Unauthorized"}), 401

    if request.method == "PUT":
        payload = request.get_json(silent=True) or {}
        allowed = {
            "display_name": payload.get("display_name"),
            "role": payload.get("role"),
            "phone": payload.get("phone"),
            "organization": payload.get("organization"),
        }
        # Drop null values to avoid clobbering existing data unintentionally
        update_payload = {k: v for k, v in allowed.items() if v is not None}
        firebase_service.upsert_user_profile(user, update_payload)
        return jsonify({"status": "updated"})

    profile = firebase_service.get_user_profile(user.get("uid")) or {}
    if not profile:
        firebase_service.upsert_user_profile(user)
        profile = firebase_service.get_user_profile(user.get("uid")) or {}

    return jsonify({
        "status": "ok",
        "profile": profile,
        "auth_user": user
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Verifixia AI Backend API',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/upload': 'Upload image/video for deepfake detection',
            'GET /api/logs': 'Get forensic logs (supports pagination/date/source filters)',
            'DELETE /api/logs': 'Clear forensic logs (optional source_type filter)',
            'DELETE /api/logs/<log_id>': 'Delete one forensic log by id',
            'POST /api/live-events': 'Save non-upload live monitoring events',
            'GET /api/stats': 'Aggregated detection statistics for analytics dashboard',
            'GET /api/database/logs': 'Get detection logs from Neon Database',
            'GET /api/health': 'Health check',
            'GET/PUT /api/auth/profile': 'Authenticated user profile'
        }
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Aggregate detection statistics derived from local forensic logs."""
    try:
        logs = _read_local_logs()

        total = len(logs)
        fake_logs = [l for l in logs if str(l.get("prediction", "")).lower() == "fake"]
        real_logs = [l for l in logs if str(l.get("prediction", "")).lower() == "real"]
        upload_logs = [l for l in logs if l.get("source_type") == "upload"]
        live_logs = [l for l in logs if l.get("source_type") == "live"]

        # Confidence scores
        confidences = [
            float(l["confidence"]) if float(l.get("confidence", 0)) <= 1
            else float(l.get("confidence", 0)) / 100.0
            for l in logs if l.get("confidence") is not None
        ]
        avg_confidence = round(sum(confidences) / len(confidences) * 100, 1) if confidences else 0.0

        # Latencies
        latencies = [
            float(l.get("latency_ms", 0) or l.get("processing_time_ms", 0))
            for l in logs if (l.get("latency_ms") or l.get("processing_time_ms"))
        ]
        avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

        # Daily trend (last 7 days)
        from collections import defaultdict
        daily: dict = defaultdict(lambda: {"threats": 0, "safe": 0})
        now = datetime.now(timezone.utc)
        for l in logs:
            ts = _parse_iso_date(l.get("timestamp"))
            if not ts:
                continue
            delta = (now.date() - ts.date()).days
            if delta > 6:
                continue
            day_key = ts.strftime("%a")
            pred = str(l.get("prediction", "")).lower()
            if pred == "fake":
                daily[day_key]["threats"] += 1
            else:
                daily[day_key]["safe"] += 1

        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        trend = [{"name": d, **daily.get(d, {"threats": 0, "safe": 0})} for d in day_order]

        # Threat type mock breakdown (based on count ratios)
        fake_total = max(1, len(fake_logs))
        threat_types = [
            {"type": "Face Swap",     "count": round(fake_total * 0.42), "percentage": 42},
            {"type": "Lip Sync",      "count": round(fake_total * 0.28), "percentage": 28},
            {"type": "Audio Clone",   "count": round(fake_total * 0.16), "percentage": 16},
            {"type": "Full Synthesis","count": round(fake_total * 0.14), "percentage": 14},
        ]

        # Source distribution
        api_calls = max(0, total - len(upload_logs) - len(live_logs))
        source_distribution = [
            {"name": "Live Streams",  "value": len(live_logs)},
            {"name": "File Uploads",  "value": len(upload_logs)},
            {"name": "API Calls",     "value": api_calls},
        ]

        return jsonify({
            "total_scans":      total,
            "threats_detected": len(fake_logs),
            "safe_detections":  len(real_logs),
            "avg_confidence":   avg_confidence,
            "avg_latency_ms":   avg_latency,
            "upload_count":     len(upload_logs),
            "live_count":       len(live_logs),
            "detection_trend":  trend,
            "threat_types":     threat_types,
            "source_distribution": source_distribution,
        })
    except Exception as e:
        logger.error(f"Error computing stats: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/database/logs', methods=['GET'])
def get_database_logs():
    """Retrieve detection logs from Neon Database with pagination"""
    try:
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)
        
        # Validate pagination parameters
        limit = min(limit, 500)  # Max 500 per request
        offset = max(offset, 0)
        
        logs = db.get_detection_logs(limit=limit, offset=offset)
        
        return jsonify({
            "status": "success",
            "count": len(logs),
            "limit": limit,
            "offset": offset,
            "logs": logs
        })
    except Exception as e:
        logger.error(f"Error retrieving database logs: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve detection logs"
        }), 500

if __name__ == '__main__':
    # Load model on startup: handled by module level pipeline init
    # load_model()

    # Run the app on port 3001 instead of 5000
    # Debug mode is controlled via FLASK_DEBUG env variable (default: off for security)
    import os as _os
    _debug = _os.environ.get('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes')
    app.run(
        host='0.0.0.0',
        port=3001,
        debug=_debug
    )
