# Verifixia – AI Deepfake Detection Platform

A full-stack deepfake detection system with a **React/TypeScript frontend**, a **Flask backend**, optional **Firebase Auth**, and a **Neon (PostgreSQL)** database.

---

## ✨ Features

- **Three-tier detection pipeline** — PyTorch Xception model → scikit-learn SVM → statistical heuristic fallback
- **Image & video upload** — supports PNG/JPG/GIF and MP4/MOV/AVI/MKV/WebM
- **Cartoon / synthetic art pre-check** — flags illustrated or AI-generated images before the ML model runs
- **Live monitoring dashboard** — simulated live stream analysis with real-time confidence gauge
- **Forensic logs** — paginated, filterable history of every detection stored locally and optionally in Firebase Firestore and Neon DB
- **Analytics page** — weekly trend charts, threat type breakdown, source distribution
- **Firebase Auth** — email/password and Google Sign-In (optional; app works without it)
- **Neon PostgreSQL** — persistent structured log storage (optional)

---

## 🗂 Project Structure

```text
Verifixia/
├── Backend/
│   ├── app.py                 # Flask API (main entry point)
│   ├── firebase_service.py    # Optional Firebase Admin SDK integration
│   ├── neon_db.py             # Neon/PostgreSQL connection & helpers
│   ├── create_model.py        # Script to create a fresh .pth weights file
│   ├── verify_integration.py  # Quick smoke-test for backend integrations
│   ├── requirements.txt       # Python dependencies
│   ├── requirements-ml.txt    # Extra ML deps (HuggingFace fine-tuning)
│   ├── utils/
│   │   └── model_utils.py     # PyTorch model definition & inference helpers
│   ├── pytorch/               # HuggingFace / PyTorch training scripts
│   │   ├── train_hf.py
│   │   ├── train_improved.py
│   │   └── config.yaml
│   └── uploads/               # Uploaded files (auto-cleaned after 24 h)
│
├── Frontend/
│   ├── api.js                 # Thin API client (fetch wrappers)
│   ├── src/
│   │   ├── App.tsx            # Router + auth context
│   │   ├── pages/             # Dashboard, Analytics, ForensicLogs, …
│   │   ├── components/        # UI components (shadcn/ui + custom)
│   │   └── lib/
│   │       ├── firebase.ts    # Firebase app initialisation
│   │       └── auth.ts        # Auth helpers (register, login, logout, …)
│   └── .env.example           # Copy to .env and fill in values
│
├── models/
│   └── xception_deepfake.pth  # Trained PyTorch model weights
│
├── DATA/
│   ├── Fake/                  # Training data – fake images
│   └── Real/                  # Training data – real images
│
├── scripts/
│   ├── train_sklearn.py       # Train the SVM fallback model
│   ├── train.py               # Train the PyTorch model on local data
│   ├── download_data.py       # Download datasets via HuggingFace
│   └── inspect_datasets.py    # Explore dataset contents
│
└── netlify.toml               # Netlify deployment config (frontend)
```

---

## 🚀 Quick Start
> **Automation**: we provide a `Makefile` in the repo root plus a
> convenient `Backend/run_backend.sh` helper.  The Makefile wraps the usual
> tasks (environment setup, training, downloading models, starting the server)
> so you don't need to remember exact commands.  Example:
>
> ```sh
> # create or update virtualenv & install deps
> make env
>
> # start backend (uses venv automatically)
> make backend
>
> # train an sklearn detector
> make train-sklearn
>
> # download weights (pass --pytorch-url/--sklearn-url args)
> make download-models -- --pytorch-url=https://.../xception_deepfake.pth
> ```
>
> The `run_backend.sh` script already selects `python3.11` if available and
> sets up the venv for you.
> **Docker**: a `Backend/Dockerfile` is provided for building a containerized backend
> that includes the pretrained model(s).  Place your model files in `models/` or set
> `MODEL_URL` / `SKLEARN_URL` and the image will fetch them on start-up.
>
> **Pre-trained weights**: we publish known-good model weights on the project's
> GitHub releases (look for `xception_deepfake.pth` and
> `deepfake_sklearn.pkl`).  Download those assets and store them under the
> `models/` folder before deploying, or set the `MODEL_URL`/`SKLEARN_URL`
> environment variables to the raw asset URLs so the backend can pull them on
> first run.  This way you never have to retrain on the deployment host,
> ensuring portability across platforms.


### Prerequisites

- Python 3.10 – 3.14
- Node.js 18+
- (Optional) Firebase project — for authentication
- (Optional) Neon / PostgreSQL database URL

---

### 1 · Backend

```bash
cd Backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (optional) install ML dependencies for local training
# pip install -r requirements-ml.txt

# Configure environment variables
cp .env.example .env           # then edit .env with your values

# If you are deploying the application, make sure the pretrained model
# files are available in `../models/`.  You can either commit them to your
# repository (e.g. using Git LFS) or fetch them during CI using the
# `scripts/download_pretrained_models.py` helper:
#
#     python ../scripts/download_pretrained_models.py \
#         --pytorch-url <URL> --sklearn-url <URL>
#
# Alternatively, set the MODEL_URL and SKLEARN_URL environment variables in
# your deployment container; `app.py` will automatically download missing
# weights on startup.

# Start the server (listens on http://localhost:3001)
python app.py
```

**Optional – train the SVM fallback model** (runs without a GPU):

```bash
python ../scripts/train_sklearn.py
```

**Optional – train the PyTorch model** (CPU or GPU):

```bash
# CPU-only PyTorch
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

python ../scripts/train.py
# or use the HuggingFace trainer:
python pytorch/train_hf.py
```

#### Backend environment variables (`.env`)

| Variable | Description | Required |
| --- | --- | --- |
| `SECRET_KEY` | Flask secret key | No (dev default used) |
| `UPLOAD_FOLDER` | Path for uploaded files | No (default: `uploads`) |
| `CORS_ORIGINS` | Comma-separated allowed origins | No (default: localhost:5173,localhost:8085) |
| `DATABASE_URL` | Neon/PostgreSQL connection string | No |
| `FIREBASE_CREDENTIALS_PATH` | Path to Firebase service-account JSON | No |
| `FIREBASE_CREDENTIALS_JSON` | JSON string of service-account credentials | No |
| `MODEL_URL` | HTTP(S) URL pointing to a pretrained PyTorch `.pth` file; if
| | the model path is missing the server will try to download it | No |
| `SKLEARN_URL` | HTTP(S) URL pointing to pretrained scikit-learn `.pkl` file | No |

> **Note:** shipping the pre-trained model as a static file in the repo or
> embedding it in your container image is the recommended approach for
> production.  Use the download script/URLs only when you cannot commit the
> binary weights directly.

---

### 2 · Frontend

```bash
cd Frontend

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env           # then edit .env

# Start dev server (http://localhost:5173)
npm run dev
```

#### Frontend environment variables (`Frontend/.env`)

| Variable | Description | Required |
| --- | --- | --- |
| `VITE_API_BASE_URL` | Backend URL | No (default: `http://localhost:3001`) |
| `VITE_USE_MOCK_API` | Use mock responses if backend is down | No (default: `false`) |
| `VITE_FIREBASE_API_KEY` | Firebase web API key | For auth only |
| `VITE_FIREBASE_AUTH_DOMAIN` | Firebase auth domain | For auth only |
| `VITE_FIREBASE_PROJECT_ID` | Firebase project ID | For auth only |
| `VITE_FIREBASE_STORAGE_BUCKET` | Firebase storage bucket | For auth only |
| `VITE_FIREBASE_MESSAGING_SENDER_ID` | Firebase messaging sender | For auth only |
| `VITE_FIREBASE_APP_ID` | Firebase app ID | For auth only |
| `VITE_FIREBASE_MEASUREMENT_ID` | Firebase measurement ID (Analytics) | No |
| `VITE_FIREBASE_DATABASE_URL` | Firebase Realtime DB URL | No |

> **Without Firebase**: The app will bypass login and let all routes through. Set `VITE_USE_MOCK_API=true` to also bypass the backend.

---

## 🔌 Backend API Reference

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/upload` | Upload image or video for detection (field: `image` or `file`) |
| `GET` | `/api/logs` | Paginated forensic logs (`page`, `page_size`, `start_date`, `end_date`, `source_type`) |
| `DELETE` | `/api/logs` | Clear all logs (optional `source_type` query param) |
| `DELETE` | `/api/logs/<id>` | Delete one log entry |
| `POST` | `/api/live-events` | Save a live monitoring event |
| `GET` | `/api/stats` | Aggregated analytics statistics |
| `GET` | `/api/health` | Health check with model status |
| `GET` | `/api/model-info` | Detailed model information |
| `GET/PUT` | `/api/auth/profile` | Authenticated user profile (Firebase required) |
| `GET` | `/api/database/logs` | Detection logs from Neon DB (`limit`, `offset`) |

---

## 🧠 Detection Pipeline

```text
Upload
  │
  ▼
Pre-check: cartoon / synthetic art?  ──YES──▶ Fake (90 %)
  │ NO
  ▼
Tier 1: PyTorch Xception model  ──available?──▶ result
  │ unavailable / error
  ▼
Tier 2: scikit-learn SVM (HOG + colour features)  ──trained?──▶ result
  │ unavailable / error
  ▼
Tier 3: Statistical heuristic (contrast + brightness)  ──▶ result
```

---

## 🗄 Database Setup (Neon / PostgreSQL)

1. Create a free database at [neon.tech](https://neon.tech)
2. Copy the connection string
3. Set `DATABASE_URL=<connection-string>` in `Backend/.env`
4. Tables are created automatically on first startup

Schema:

```sql
-- users           (id, username, email, password_hash, created_at)
-- detection_logs  (id, filename, prediction, confidence, timestamp, user_id)
```

---

## 🔥 Firebase Setup (optional)

1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com)
2. Enable **Authentication** (Email/Password + Google)
3. Enable **Firestore** (for log/profile storage)
4. **Frontend**: copy Web app SDK config values into `Frontend/.env`
5. **Backend**: download a service-account JSON and set `FIREBASE_CREDENTIALS_PATH` in `Backend/.env`

---

## 🚢 Deployment

### Netlify (frontend)

`netlify.toml` is pre-configured:

- Build command: `cd Frontend && npm install && npm run build`
- Publish directory: `Frontend/dist`
- All routes redirect to `index.html` (SPA support)

Set `VITE_API_BASE_URL` as a Netlify environment variable pointing to your deployed backend.

### Backend (any Python host)

```bash
# Example – Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:3001 app:app
```

---

## 🧪 Tests

```bash
# Frontend unit tests
cd Frontend && npm test

# Backend smoke test
cd Backend && python verify_integration.py
```
