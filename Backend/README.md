# Verifixia AI Backend API

A Flask-based REST API for deepfake detection using PyTorch models.

## Features

- **Image Upload & Analysis**: Upload images for deepfake detection
- **PyTorch Model Integration**: Uses trained deepfake detection models
- **RESTful API**: Clean endpoints for frontend integration
- **Detection Logging**: Logs all detection results for analysis
- **CORS Support**: Ready for frontend integration

## API Endpoints

### POST /api/upload
Upload an image for deepfake detection.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
```json
{
  "prediction": "Fake",
  "confidence": 0.87,
  "filename": "uuid_filename.jpg"
}
```

### GET /api/logs
Get recent detection logs.

**Response:**
```json
[
  {
    "timestamp": "2024-01-27T10:30:00",
    "filename": "uuid_filename.jpg",
    "prediction": "Fake",
    "confidence": 0.87
  }
]
```

### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "pytorch_available": true,
  "sklearn_available": false,
  "active_model": "pytorch",
  "model_loaded": true,
  "device": "cpu",
  "model_info": { /* ... */ }
}
```

## Setup

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
# install a CPU build of torch so the server can load weights
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

2. **Obtain a pretrained model** (you do not have to train locally):
   - copy `models/xception_deepfake.pth` and/or
     `models/deepfake_sklearn.pkl` into the repository (Git LFS is
     recommended for large binaries), **or**
   - run the helper script from the repo root:
     ```bash
     python ../scripts/download_pretrained_models.py \
         --pytorch-url <URL> --sklearn-url <URL>
     ```
   - you may also set the `MODEL_URL` / `SKLEARN_URL` environment
     variables and the backend will fetch missing weights on startup.

3. **Run the API:**
```bash
python app.py
```

The API will be available at `http://localhost:3001` (or whatever port
is set by `PORT`/`FLASK_RUN_PORT`).

## Project Structure

```
Backend/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── pytorch/
│   ├── train_improved.py  # Model training script
│   └── config.yaml        # Training configuration
├── utils/
│   ├── __init__.py
│   └── model_utils.py     # Model utility functions
└── uploads/               # Temporary uploaded files
```

## Model Training

The system uses an Xception-based architecture for deepfake detection. To train your own model:

1. Prepare your dataset in the expected format
2. Update `pytorch/config.yaml` with your settings
3. Run the training script

## Integration with Frontend

Update the `API_BASE` in `frontend/api.js` to point to the backend:

```javascript
const API_BASE = "http://localhost:5000";
```

Then uncomment the real API calls and comment out the mock responses.