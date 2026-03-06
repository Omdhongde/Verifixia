# Verifixia – Setup Guide

A deepfake-detection web app with a **Python/Flask** backend and a **React/Vite/TypeScript** frontend.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.14 | `python3 --version` |
| Node.js | 18 + | `node --version` |
| npm | 9 + | `npm --version` |
| Git | any | to clone the repo |
| PostgreSQL | optional | Neon cloud DB is supported out-of-the-box |

---

## 1 – Clone the Repository

```bash
git clone <your-repo-url>
cd Verifixia
```

---

## 2 – Backend Setup (Python / Flask)

### 2a – Create & activate a virtual environment

```bash
cd Backend
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 2b – Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **PyTorch** (optional – only needed for the Xception deep-learning model):
> ```bash
> # CPU (macOS / Linux)
> pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
>
> # Apple Silicon (MPS)
> pip install torch torchvision
>
> # CUDA 12.x (Linux GPU)
> pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
> ```
>
> **HuggingFace training** (optional – only for `Backend/pytorch/train_hf.py`):
> ```bash
> pip install -r requirements-ml.txt
> ```

### 2c – Configure environment variables

Copy the example file and fill in your values:

```bash
cp .env.example .env   # or create .env manually
```

Minimum `.env` for running without Firebase/DB:

```dotenv
SECRET_KEY=change-me
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216   # 16 MB

# PostgreSQL / Neon (optional – leave empty to disable)
DATABASE_URL=

# Firebase (optional – leave empty to disable)
FIREBASE_CREDENTIALS_PATH=
FIREBASE_CREDENTIALS_JSON=

# CORS – add your frontend URL if deploying
CORS_ORIGINS=http://localhost:5173,http://localhost:8085
```

### 2d – Train a model (skip if you already have one in `models/`)

**Fast path – scikit-learn SVM (no GPU needed, Python 3.10-3.14):**

```bash
cd ..                          # back to repo root
python scripts/train_sklearn.py
# Saves: models/deepfake_sklearn.pkl
```

**Optional – PyTorch EfficientNet (requires PyTorch):**

```bash
python scripts/train.py
# Saves: models/xception_deepfake.pth
```

### 2e – Start the backend server

```bash
cd Backend
source .venv/bin/activate
python app.py
# Listening on http://localhost:5001
```

---

## 3 – Frontend Setup (React / Vite / TypeScript)

```bash
cd Frontend
npm install
npm run dev
# Opens http://localhost:5173
```

### Available frontend scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Start Vite dev server with HMR |
| `npm run build` | Production build → `dist/` |
| `npm run preview` | Preview production build |
| `npm run lint` | ESLint check |
| `npm run test` | Run Vitest tests |

---

## 4 – Project Structure

```
Verifixia/
├── Backend/
│   ├── app.py               # Flask API entry-point
│   ├── firebase_service.py  # Optional Firebase integration
│   ├── neon_db.py           # PostgreSQL / Neon DB layer
│   ├── requirements.txt     # Core Python deps (install this)
│   ├── requirements-ml.txt  # Optional HuggingFace / PyTorch deps
│   ├── utils/
│   │   └── model_utils.py   # PyTorch model loader helpers
│   └── pytorch/
│       └── train_hf.py      # HuggingFace fine-tuning script
├── Frontend/
│   ├── src/                 # React + TypeScript source
│   ├── package.json         # Node dependencies
│   └── vite.config.ts
├── scripts/
│   ├── train_sklearn.py     # Train SVM model (no GPU)
│   ├── train.py             # Train EfficientNet model (PyTorch)
│   └── download_data.py     # Download dataset from HuggingFace Hub
├── models/                  # Saved model files (.pth / .pkl)
└── DATA/
    ├── Real/                # Real images for training
    └── Fake/                # Deepfake images for training
```

---

## 5 – API Endpoints (Backend)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/detect` | Upload image/video for deepfake detection |
| `GET` | `/api/stats` | Detection statistics |
| `GET` | `/api/model-info` | Loaded model information |

---

## 6 – Troubleshooting

| Problem | Fix |
|---|---|
| `psycopg2` install fails on Python 3.14 | Use `psycopg[binary]` (already in `requirements.txt`) |
| `No module named torch` | Install PyTorch separately (see Step 2b) |
| Backend port conflict | Change port in `app.py` or set `PORT` env var |
| CORS errors | Add your frontend URL to `CORS_ORIGINS` in `.env` |
| `No trained model available` | Run `python scripts/train_sklearn.py` from repo root |
