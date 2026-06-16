# System Architecture & Data Flow

---

## ✅ Active Model — MultiClassDetector (Old Architecture)

> **Currently in production use as of June 2026**

| Property | Value |
|---|---|
| **Model file** | `models/multiclass_detector.pth` (49.6 MB) |
| **Architecture** | Custom ResNet-style CNN with Squeeze-Excitation (SE) blocks |
| **Classes** | Real (0) / Deepfake (1) / AIGenerated (2) |
| **Val Accuracy** | ~72–74% — honest, no overfit (train ≈ val) |
| **Trained epochs** | 50 full epochs from scratch |
| **Input size** | 299×299 |
| **Loaded by** | `Backend/utils/model_utils.py` → `ModelUtils.load_model()` |

**Why this model is preferred:** Train accuracy and val accuracy stayed within 1% of each other across all 50 epochs — genuine learning, not memorisation. This is reliable for real-world unseen inputs.

---

## 🔮 New Architecture — Reserved for Future Use

> **DO NOT use in production yet — kept in `models/new_arch_detector.pth`**

| Property | Value |
|---|---|
| **Model file** | `models/new_arch_detector.pth` (18 MB) |
| **Architecture** | MobileNetV2 (frozen) + Frequency Attention Gate + Verdict Head |
| **Design doc** | `new architecture/ARCHITECTURE.md` |
| **Issue** | Train accuracy reached 98% while val was 90% (7–8% gap = overfit) |
| **Future fix** | Partial unfreezing + stronger regularisation + larger dataset |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND LAYER                            │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Dashboard   │  │  Analytics   │  │   Settings   │  (React)  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                    │
│  ┌──────▼────────────────▼────────────────▼─────────────────┐   │
│  │           API Client Module (api.js)                      │   │
│  │  - uploadImage()      - getWebcamFrame()                 │   │
│  │  - startWebcam()      - getAnalysisHistory()            │   │
│  │  - getStatistics()    - getDetectionLogs()              │   │
│  └──────┬───────────────────────────────────────────────────┘   │
│         │                                                        │
└─────────┼────────────────────────────────────────────────────────┘
          │
          │  HTTP/REST (8 endpoints)
          │  Port: 5173 → 3000
          │
┌─────────▼────────────────────────────────────────────────────────┐
│                      BACKEND LAYER (Flask)                        │
│  Port: 3000                                                       │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  API Routes                                                │  │
│  │  • POST   /api/upload                                     │  │
│  │  • GET    /api/webcam/frame                               │  │
│  │  • GET    /api/history                                    │  │
│  │  • GET    /api/stats                                      │  │
│  └────────────┬────────────────────────────────────────────┬─┘  │
│               │                                            │     │
│  ┌────────────▼──────────────┐       ┌────────────────────▼──┐  │
│  │  Image Processing Layer   │       │  Logging & Metrics     │  │
│  │                           │       │                        │  │
│  │  • Face Detection         │       │  • log_detection()     │  │
│  │  • Preprocessing          │       │  • save_analysis_to_db()│ │
│  │  • PyTorch Inference      │       │  • Process tracking    │  │
│  │  • Quality Scoring        │       │                        │  │
│  └────────────┬──────────────┘       └─────────────┬──────────┘  │
│               │                                     │             │
│  ┌────────────▼─────────────────────────────────────▼──────────┐ │
│  │        SQLAlchemy ORM Layer                                 │ │
│  │  Models: User, Analysis, FaceDetection, Log, Statistics    │ │
│  └────────────┬─────────────────────────────────────────────┬──┘ │
│               │                                             │    │
└───────────────┼─────────────────────────────────────────────┼────┘
                │                                             │
                │  psycopg2 (PostgreSQL driver)             │
                │  SQLAlchemy → SQL                         │
                │                                             │
┌───────────────▼──────────────────────────────────────────────▼───┐
│                   DATABASE LAYER (PostgreSQL)                     │
│  Port: 5432                                                       │
│  Database: deepfake_db                                            │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  TABLES (5)                    INDEXES (8+)               │   │
│  │                                                            │   │
│  │  users                         idx_user_id                │   │
│  │  ├─ id (UUID)                 idx_created_at             │   │
│  │  ├─ username                  idx_user_created           │   │
│  │  └─ email                      idx_analyses_deepfake      │   │
│  │                                                            │   │
│  │  analyses                      CONSTRAINTS                │   │
│  │  ├─ user_id (FK)              FOREIGN KEYS               │   │
│  │  ├─ face_count                CHECK constraints           │   │
│  │  ├─ deepfake_detected         UNIQUE constraints          │   │
│  │  ├─ faces_data (JSON)                                     │   │
│  │  └─ processing_time_ms                                    │   │
│  │                                                            │   │
│  │  face_detections                                           │   │
│  │  ├─ analysis_id (FK)                                      │   │
│  │  ├─ label                                                  │   │
│  │  ├─ confidence                                             │   │
│  │  ├─ position_*                                             │   │
│  │  └─ thumbnail_base64                                       │   │
│  │                                                            │   │
│  │  detection_logs                                            │   │
│  │  ├─ user_id (FK)                                           │   │
│  │  ├─ message                                                │   │
│  │  ├─ severity                                               │   │
│  │  └─ metadata (JSON)                                        │   │
│  │                                                            │   │
│  │  statistics_snapshots                                      │   │
│  │  ├─ period_type (daily/weekly/monthly)                     │   │
│  │  ├─ total_analyses                                         │   │
│  │  └─ average_confidence                                     │   │
│  │                                                            │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                   │
│  🔑 Key Features:                                                │
│  • ACID compliance                                              │
│  • Connection pooling (10 connections)                          │
│  • Automatic index maintenance                                  │
│  • JSON support for flexible data                               │
│  • Transaction support for data integrity                       │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Diagram

### Image Upload Flow
```
User Interface
    │
    ├─ Selects image file
    │
    ▼
uploadImage(file)  ─────────────┐
    │                           │
    │                      HTTP POST
    │                      /api/upload
    │                           │
    ├────────────────────────────►│
    │                            │
    │                       Backend Processing
    │                            │
    │                       ┌────▼──────────┐
    │                       │ Save File     │
    │                       │ Detect Faces  │
    │                       │ Analyze Each  │
    │                       │ Create Log    │
    │                       └────┬──────────┘
    │                            │
    │                       ┌────▼──────────────────┐
    │                       │ Save to Database      │
    │                       │ • analyses            │
    │                       │ • face_detections     │
    │                       │ • detection_logs      │
    │                       └────┬─────────────────┘
    │                            │
    │                   HTTP Response (JSON)
    │◄───────────────────────────┤
    │                            │
    ├─ Display Results
    │  • Face count
    │  • Confidence
    │  • Deepfake/Real
    │
    ▼
Dashboard Updated
```

### Real-time Webcam Flow
```
startWebcam()
    │
    ▼
POST /api/webcam/start
    │
    ▼
Backend: Initialize camera
    │
    ├─ Connect to camera (0)
    ├─ Set resolution
    └─ Create detection session
    │
    ▼
Poll Loop (Browser)
    │
    ├─ getWebcamFrame()
    │
    ▼
    GET /api/webcam/frame
    │
    ▼
    Backend:
    ├─ Capture frame
    ├─ Detect faces
    ├─ Analyze detections
    └─ Encode to base64
    │
    ▼
    Response: {
        "frame": "data:image/jpeg;base64,...",
        "detections": [
            {"label": "REAL", "confidence": 95.2}
        ]
    }
    │
    ▼
    Display frame in VideoFeed component
    │
    ├─ Draw bounding boxes
    ├─ Show confidence
    └─ Update metrics
    │
    ▼
    Wait 100ms → Repeat Poll
    │
    ▼
stopWebcam()
    │
    ▼
POST /api/webcam/stop
    │
    ▼
Release camera & resources
```

### Analytics Data Flow
```
User Views Analytics Page
    │
    ▼
getStatistics()  ─────────────┐
    │                    HTTP GET /api/stats
    │                         │
    └─────────────────────────►│
                               │
                         Backend:
                         SELECT * FROM analyses
                         WHERE user_id = current_user
                         AND created_at >= TODAY
                               │
                         Calculate:
                         ├─ Total analyses
                         ├─ Deepfakes count
                         ├─ Real count
                         └─ Average confidence
                               │
                         Response: {
                             "total_analyses": 45,
                             "deepfakes_detected": 8,
                             "real_faces": 37,
                             "average_confidence": 92.3
                         }
                               │
    ◄──────────────────────────┤
    │
    ▼
Update Dashboard:
├─ Confidence Gauge
├─ Metric Cards
└─ Detection Logs
```

---

## 🔄 Request/Response Flow

### Example: Upload and Analyze Image

**Request:**
```javascript
POST /api/upload
Content-Type: multipart/form-data

file: <binary image data>
```

**Processing Chain:**
```
1. Receive file
   ↓
2. Validate file type/size
   ↓
3. Save to disk
   ↓
4. Detect faces (OpenCV)
   ↓
5. For each face:
   ├─ Preprocess (224×224, normalize)
   ├─ Run through PyTorch model
   ├─ Get label (REAL/DEEPFAKE)
   ├─ Get confidence score
   └─ Create thumbnail
   ↓
6. Calculate overall statistics
   ├─ Face count
   ├─ Deepfake detected (any == 1?)
   └─ Average confidence
   ↓
7. Save to database
   ├─ INSERT analyses row
   ├─ INSERT face_detections rows
   └─ INSERT detection_log row
   ↓
8. Return response
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "Analysis complete",
  "face_count": 2,
  "deepfake_detected": true,
  "overall_confidence": 87.5,
  "faces": [
    {
      "label": "DEEPFAKE",
      "confidence": 92.3,
      "position": {"x": 100, "y": 50, "w": 150, "h": 150},
      "quality": 85.2,
      "thumbnail": "data:image/png;base64,..."
    },
    {
      "label": "REAL",
      "confidence": 82.7,
      "position": {"x": 300, "y": 100, "w": 140, "h": 140},
      "quality": 88.1,
      "thumbnail": "data:image/png;base64,..."
    }
  ],
  "processing_time_ms": 245,
  "mode": "pytorch",
  "filename": "20240128_154530_image.jpg"
}
```

---

## 💾 Database Transaction Example

```sql
BEGIN TRANSACTION;

-- 1. Create analysis record
INSERT INTO analyses (
    user_id, filename, file_size, face_count,
    deepfake_detected, overall_confidence, processing_time_ms
) VALUES (
    'uuid-1', 'image.jpg', 2048576, 2,
    true, 87.5, 245
) RETURNING id;
-- Result: analysis_id = 'uuid-analysis-1'

-- 2. Create face detection records
INSERT INTO face_detections (
    analysis_id, label, confidence, position_x, position_y,
    position_width, position_height, quality_score
) VALUES
    ('uuid-analysis-1', 'DEEPFAKE', 92.3, 100, 50, 150, 150, 85.2),
    ('uuid-analysis-1', 'REAL', 82.7, 300, 100, 140, 140, 88.1);

-- 3. Create log entry
INSERT INTO detection_logs (
    user_id, message, log_type, severity, analysis_id
) VALUES (
    'uuid-1',
    'Analysis completed: image.jpg (2 faces, 1 deepfake)',
    'analysis',
    'info',
    'uuid-analysis-1'
);

-- 4. Update statistics (optional)
INSERT INTO statistics_snapshots (
    user_id, period_type, period_date,
    total_analyses, total_deepfakes, total_real, average_confidence
) VALUES (
    'uuid-1', 'daily', '2024-01-28', 1, 1, 1, 87.5
) ON CONFLICT (user_id, period_type, period_date)
DO UPDATE SET
    total_analyses = total_analyses + 1,
    total_deepfakes = total_deepfakes + 1,
    average_confidence = (average_confidence + 87.5) / 2;

COMMIT;
```

---

## 📈 Query Performance

### Index Strategy

```sql
-- Most common query: Get user's recent analyses
SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT 50;
→ Uses: idx_user_created (user_id, created_at DESC)

-- Trending query: Deep
SELECT * FROM analyses WHERE deepfake_detected = true;
→ Uses: idx_analyses_deepfake

-- Label analysis: Get detection breakdown
SELECT label FROM face_detections WHERE analysis_id = ?;
→ Uses: idx_face_detections_analysis

-- Logging query: Get logs by severity
SELECT * FROM detection_logs WHERE severity IN ('error', 'critical');
→ Uses: idx_detection_logs_severity
```

---

## 🔒 Data Integrity

### Foreign Key Relationships
```
users (1) ──────┐
                ├─ (N) analyses
                │        │
                │        ├─ (N) face_detections
                │        │
                │        └─ (N) detection_logs
                │
                └─ (N) statistics_snapshots
```

**CASCADE Deletes**: Deleting user removes:
- All their analyses
- All face detections from those analyses
- All detection logs
- All statistics snapshots

**SET NULL**: If analysis deleted:
- detection_log.analysis_id becomes NULL (preserves log)
- But analysis still referenced in statistics

---

## ⚡ Performance Characteristics

### Query Performance Estimates

| Query | Complexity | Time | Rows |
|-------|-----------|------|------|
| Get user history | O(n log n) | <100ms | 50 |
| Get daily stats | O(n) | <50ms | 1 |
| Get face detections | O(n) | <30ms | 5-20 |
| Get logs by severity | O(n log n) | <100ms | 50 |
| Get weekly trends | O(n log n) | <200ms | 8 |

### Scalability

**Current Setup**:
- Connection pool: 10 connections
- Max concurrent: 10 users
- Database size: ~1GB per 100k analyses

**For 1M analyses**:
- Database: ~10GB
- Archival needed for older data
- Need for partitioning by date

---

## 🎯 Summary

**Architecture**: 3-Layer (Frontend → Backend → Database)  
**Technology**: React, Flask, PostgreSQL  
**Data**: 5 tables, 8+ indexes, ACID compliance  
**Performance**: <500ms per request (typical)  
**Scalability**: Single server → distributed (sharding)  
**Reliability**: Transaction support, foreign keys, constraints  

✅ **Production Ready!**
