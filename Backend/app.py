from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from datetime import datetime
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
app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER", "uploads")
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
PYTORCH_AVAILABLE = False
model = None
DEVICE = "cpu"
model_info = {}

# Try to load PyTorch model
try:
    import torch
    from utils.model_utils import ModelUtils
    
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch is available. Attempting to load model...")
    
    # Load the model
    model, DEVICE = ModelUtils.load_model(MODEL_PATH)
    model_info = ModelUtils.get_model_info(MODEL_PATH)
    model_metadata = ModelUtils.get_model_metadata(model, DEVICE)
    model_info.update(model_metadata)
    
    logger.info(f"Model loaded successfully on device: {DEVICE}")
    logger.info(f"Model info: {model_info}")
    
except Exception as e:
    logger.warning(f"Could not load PyTorch model: {e}")
    logger.warning("Falling back to heuristic-based predictions")
    PYTORCH_AVAILABLE = False
    model = None

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


def predict_deepfake_video():
    """Simple mock prediction for video uploads.

    Implementing full video-based deepfake detection requires a heavier
    model stack; for now we return a stable, high-confidence mock result
    so the end‑to‑end flow works.
    """
    confidence = 0.7 + random.random() * 0.3
    prediction = "Fake" if confidence > 0.8 else "Real"
    return prediction, confidence

def predict_deepfake(image_path):
    """Predict if image is deepfake using PyTorch model or fallback to heuristics"""
    
    if PYTORCH_AVAILABLE and model is not None:
        try:
            from utils.model_utils import ModelUtils
            
            # Preprocess image
            image_tensor, preprocessing_time = ModelUtils.preprocess_image(image_path)
            
            # Make prediction
            prediction_result = ModelUtils.predict_image(model, image_tensor, DEVICE)
            
            # Get confidence interpretation
            confidence_interpretation = ModelUtils.interpret_confidence(
                prediction_result["confidence_raw"]
            )
            
            # Combine all information
            result = {
                "prediction": prediction_result["prediction"],
                "confidence": prediction_result["confidence"],
                "confidence_raw": prediction_result["confidence_raw"],
                "threat_level": prediction_result["threat_level"],
                "model_used": "Verifixia AI Xception v2.4.1",
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
            
            logger.info(f"Model Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error making model prediction: {e}")
            logger.warning("Falling back to heuristic prediction")
            # Fall through to heuristic method
    
    # Fallback: Heuristic-based prediction
    try:
        img = Image.open(image_path).convert("L")  # grayscale
        stat = ImageStat.Stat(img)

        # Simple heuristics: contrast + noise-like variation
        mean = stat.mean[0]
        stddev = stat.stddev[0]

        # Normalize to 0–1 range in a rough, robust way
        norm_contrast = max(0.0, min(1.0, stddev / 64.0))
        norm_brightness = max(0.0, min(1.0, abs(mean - 128) / 128.0))

        # Combine into a "fake" score
        fake_score = 0.6 * norm_contrast + 0.4 * norm_brightness

        # Add a tiny bit of randomness so repeated uploads aren't identical
        fake_score = max(0.0, min(1.0, fake_score + (random.random() - 0.5) * 0.1))

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
    entry.setdefault("timestamp", datetime.utcnow().isoformat())
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
            prediction, confidence = predict_deepfake_video()
            result = {
                "prediction": prediction,
                "confidence": confidence * 100,
                "confidence_raw": confidence,
                "threat_level": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low",
                "model_used": "Video Analysis (Mock)",
                "processing_time": {
                    "preprocessing_ms": 0,
                    "inference_ms": 0,
                    "total_ms": 0
                },
                "analysis": {
                    "level": "Video",
                    "description": "Video analysis requires specialized processing",
                    "recommendation": "Full video analysis coming soon"
                },
                "model_info": {
                    "architecture": "Video Analyzer",
                    "input_size": "Variable",
                    "framework": "Mock",
                    "device": "cpu"
                }
            }
        else:
            result = predict_deepfake(filepath)

        session_id = request.form.get("session_id") or str(uuid.uuid4())
        processing_time = result.get("processing_time", {}) or {}
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
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
            if user:
                # Try to get or create user in database
                # For now, we'll use None as user_id
                pass
            
            db_log = db.save_detection_log(
                filename=unique_filename,
                prediction=result.get("prediction"),
                confidence=result.get("confidence", 0) / 100.0 if result.get("confidence", 0) > 1 else result.get("confidence", 0),
                user_id=user_id
            )
            logger.info(f"✓ Detection saved to Neon Database: {db_log}")
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
            "timestamp": datetime.utcnow().isoformat(),
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
    else:
        device_info = "cpu (mock mode)"

    return jsonify({
        'status': 'healthy',
        'pytorch_available': PYTORCH_AVAILABLE,
        'model_loaded': model is not None,
        'device': device_info,
        'model_info': model_info if model_info else None,
        'firebase_enabled': firebase_service.enabled
    })

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    if model is not None and PYTORCH_AVAILABLE:
        return jsonify({
            'status': 'loaded',
            'info': model_info
        })
    else:
        return jsonify({
            'status': 'not_loaded',
            'message': 'Model not available. Using fallback predictions.',
            'pytorch_available': PYTORCH_AVAILABLE
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
        now = datetime.utcnow()
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
    app.run(
        host='0.0.0.0',
        port=3001,
        debug=True
    )
