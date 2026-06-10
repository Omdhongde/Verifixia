# System Architecture & Data Flow

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

## 🧠 3-Tier Multi-Modal Deepfake Detection Architecture

The system incorporates a highly sophisticated **3-Tier Multi-Modal Deepfake Detection Pipeline** designed to achieve robust generalization and prevent shortcut learning (such as composition, framing, and double-interpolation artifacts).

```mermaid
graph TD
    classDef stageStyle fill:#1a1c23,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef moduleStyle fill:#1e293b,stroke:#60a5fa,stroke-width:2px,color:#fff;
    classDef temporalStyle fill:#1e293b,stroke:#a78bfa,stroke-width:2px,color:#fff;
    classDef heuristicStyle fill:#2e1065,stroke:#f472b6,stroke-width:2px,color:#fff;
    classDef inputStyle fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#fff;

    subgraph Inputs ["Input Layer"]
        InSeq["Input Sequence (16 frames × 3 × 299 × 299)"]:::inputStyle
    end

    subgraph Tier1 ["Tier 1: Feature Extraction Layer (Spatial & Physiological)"]
        Stage1["Stage 1 (EfficientNet B0)"]:::stageStyle
        FAG["Frequency Attention Gate (FAG)"]:::moduleStyle
        Stage2["Stage 2 (EfficientNet B3-B5)"]:::stageStyle
        BPM["Biological Plausibility Module (BPM)"]:::moduleStyle
        Stage3["Stage 3 (EfficientNet B6-B7)"]:::stageStyle
        GRL["Gradient Reversal Anomaly Branch (GRL)"]:::moduleStyle
        HeadConv["Head Conv (1×1 compression)"]:::stageStyle
        Concat1["Feature Concat (672-dim)"]:::stageStyle

        InSeq --> Stage1
        Stage1 --> FAG
        FAG --> Stage2
        
        Stage2 --> BPM
        Stage2 --> Stage3
        
        Stage3 --> GRL
        Stage3 --> HeadConv
        
        HeadConv --> Concat1
        BPM -->|Bio Embedding 32-dim| Concat1
        BPM -->|Consistency Score| Tier3Gate
    end

    subgraph Tier2 ["Tier 2: Temporal Modeling & Cross-Attention Fusion"]
        Concat1 -->|Sequence of 16 × 672-dim| LSTMStack["2-Layer LSTM Stack"]:::temporalStyle
        LSTMStack --> TempAttn["Temporal Self-Attention (512-dim Query)"]:::temporalStyle
        LSTMStack --> SpikeDet["Inconsistency Spike Detector"]:::temporalStyle
        
        TempAttn --> CAF["Cross-Attention Fusion (CAF)"]:::temporalStyle
        GRL -->|Average Anomaly (320-dim)| CAF
        BPM -->|Average Bio-embed (32-dim)| CAF
    end

    subgraph OutputLayer ["Classification & Safety Net"]
        CAF --> VerdictFC["Verdict Head (FC Layers)"]:::stageStyle
        VerdictFC -->|Raw Softmax Score p| Tier3Gate["Tier 3: Heuristic Fallback Engine"]:::heuristicStyle
        SpikeDet -->|Spike Flags| Tier3Gate
        LSTMStack -->|Frame Deltas| Tier3Gate
        
        Tier3Gate -->|H-1 to H-7 Rules applied| FinalOut["Final Output (REAL / FAKE)"]:::heuristicStyle
    end
```

### 📊 Parameter Budget Across Tiers

The unified sequence detector operates with a total budget of **~11.85M parameters**, divided as follows:

| Layer / Component | Type | Parameter Count | Primary Role / Output |
| :--- | :---: | :---: | :--- |
| **Tier 1** - Modified EfficientNet-B0 | Trainable / Frozen | 5.5M | 2D CNN spatial feature extraction |
| **Tier 1** - Biological Plausibility Module | Trainable | 2.1M | 5-region facial rPPG extraction & MLP consistency scorer |
| **Tier 1** - Frequency Attention Gate | Trainable | ~1.2K | Learnable frequency attention mask ($56 \times 56 \times 1$) |
| **Tier 2** - LSTM Temporal Engine | Trainable | 3.8M | 2-layer temporal sequence processing |
| **Tier 2** - Cross-Attention Fusion | Trainable | ~400K | Fusion of query, spatial anomaly, and biological embeddings |
| **Tier 3** - Heuristic Engine | Fixed | 0 | Rule-based safety net and override logic |
| Verdict Layer | Trainable | ~50K | Multi-layer fully-connected classification head |
| **TOTAL LEARNED PARAMETERS** | | **≈ 11.85M** | **End-to-End Multimodal Detection Pipeline** |

---

### 1. Tier 1: Spatial & Physiological Feature Extraction

#### 1.1 Stage 1 & Frequency Attention Gate (FAG)
* **Inputs:** A sequence of 16 cropped face frames ($16 \times 3 \times 299 \times 299$).
* **Operation:** Stage 1 of the EfficientNet-B0 backbone extracts basic low-level spatial features resulting in a $56 \times 56 \times 24$ tensor.
* **FAG Mechanism:**
  * Learns a static spatial frequency mask $\sigma$ of shape $56 \times 56 \times 1$.
  * Performs element-wise multiplication with the stage output to isolate high-frequency anomalies (e.g., GAN noise or compression boundaries).
  * Fuses features via concatenation ($56 \times 56 \times 48$) and compresses them back to $56 \times 56 \times 24$ via a $1 \times 1$ convolution.

#### 1.2 Stage 2 & Biological Plausibility Module (BPM)
* **Operation:** Extracts mid-level features through MBConv blocks B3 ($28 \times 28 \times 40$), B4 ($14 \times 14 \times 80$), and B5 ($14 \times 14 \times 112$) utilizing a novel **Dual-Domain Squeeze** mechanism.
* **BPM Architecture:**
  * **5-Region RoI Align:** Dynamically extracts features from 5 facial regions containing key vascular flow information: Forehead, Left Cheek, Right Cheek, Nose Bridge, and Chin. Each crop is mapped to a $7 \times 7 \times 64$ tensor.
  * **Bio Correlation Matrix:** The 5 regional tensors are flattened into 3136-dimensional vectors. A cross-product correlation matrix ($5 \times 5 = 25$) is calculated alongside regional rPPG variances.
  * **MLP Scorer:** An MLP ($25 \to 64 \to 32 \to 1$) outputs a **Biological Consistency Score** $C_{bio} \in [0, 1]$ indicating whether the temporal blood-flow patterns match a real human face.
  * **Feature Merge:** The primary spatial head vector (640-dim) is concatenated with the biological embedding (32-dim) to output a unified **672-dim Spatial+Bio vector**.

#### 1.3 Stage 3 & Anomaly Branch (GRL)
* **Operation:** MBConv blocks B6 ($7 \times 7 \times 192$) and B7 ($7 \times 7 \times 320$) extract high-level semantic features.
* **Gradient Reversal Layer (GRL):** A fork splits off into an anomaly branch (320-dim). During backpropagation, GRL reverses the gradients to prevent the model from learning composition-based dataset shortcuts, forcing it to extract domain-invariant representation.
* **Head Conv:** A final $1 \times 1$ Conv compresses features to 640-dim before spatial pooling.

---

### 2. Tier 2: Temporal Modeling & Cross-Attention Fusion

#### 2.1 LSTM Stack & Temporal Self-Attention
* **LSTM Stack:** Receives the sequence of 16 frames represented by the 672-dim Spatial+Bio vectors.
  * **Layer 1 (Unidirectional):** 672-dim $\to$ 512 hidden units (Dropout $p = 0.3$).
  * **Layer 2 (Bidirectional):** 512-dim $\to$ $256 \times 2 = 512$ hidden units (Dropout $p = 0.3$).
* **Temporal Self-Attention:** Fuses temporal information across frames using an 8-head self-attention layer (dimension 512) followed by LayerNorm to output a 512-dim final sequence representation (Query vector).
* **Inconsistency Spike Detector:**
  * Computes frame-to-frame delta $|h_t - h_{t-1}|$ of LSTM hidden states.
  * Compares it against a **learned threshold** $\theta$ (trainable scalar).
  * Flagged anomalies per frame are tracked as a boolean vector of size 16. Regularized via auxiliary frame-level loss supervision during training.

#### 2.2 Cross-Attention Fusion Layer (CAF)
Fuses different feature modalities by projecting them into a common 256-dimensional space:
* **Query (A):** LSTM temporal query state ($512 \to 256$).
* **Key/Value (B):** Average spatial anomaly features from GRL branch ($320 \to 256$).
* **Key/Value (C):** Average biological embeddings from BPM ($32 \to 256$).
* **Multi-Head Cross-Attention:** Query (A) queries key/value sequences of (B + C) using 4 heads. The output is added back to Query (A) as a residual connection, normalized via LayerNorm, and passed to the Verdict Head.

---

### 3. Verdict Head & Heuristic Fallback Engine

#### 3.1 Verdict Head (FC Classifier)
The final classification layer maps the 256-dim fused representation to binary logit values:
* **FC Layer 1:** $256 \to 128$ (ReLU, Dropout 0.4)
* **FC Layer 2:** $128 \to 64$ (ReLU)
* **FC Output:** $64 \to 2$ logits (Real vs. Fake classification)

#### 3.2 Heuristic Fallback Engine (Tier 3 Safety Net)
Tier 3 operates as a **rule-based safety net** (0 learned parameters) that triggers when the model's confidence is low or when physical/biological rules are violated. It either overrides the verdict to "Fake" or adjusts the confidence.

| Rule ID | Rule Name | Trigger Condition | Signal Source | Action / Impact |
| :---: | :--- | :--- | :--- | :--- |
| **H-1** | Low-Confidence Gate | Model softmax max $< 0.65$ | Verdict layer output | Flag as **UNCERTAIN**; escalate to H-2 through H-6 sequential checks. |
| **H-2** | Bio-Consistency Hard Veto | rPPG consistency score $< 0.20$ | Tier 1 - BPM Scorer | Hard override to **FAKE** regardless of model prediction. |
| **H-3** | Spike Density Rule | $\ge 5$ of 16 frames flagged as spike anomaly | Tier 2 - Spike Detector | Hard override to **FAKE** (detects abrupt splicing artifacts). |
| **H-4** | Spectral Aliasing Check | DCT frequency mask activation $> 0.85$ in high-frequency | Tier 1 - FAG Output | Boost **FAKE** confidence by $+0.15$. |
| **H-5** | Regional rPPG Incoherence | Cross-region rPPG variance $> 3\sigma$ from real-face distribution | Tier 1 - Bio Correlation Matrix | Boost **FAKE** confidence by $+0.10$. |
| **H-6** | Temporal Freeze Rule | Frame delta $|h_t - h_{t-1}| = 0$ for $\ge 4$ consecutive frames | Tier 2 - LSTM hidden states | Hard override to **FAKE** (detects video looping/freezing). |
| **H-7** | Confidence Recovery | Model confidence $> 0.90$ AND no heuristic flags | All tiers | Pass through directly; skip all overrides. |

#### 3.3 Uncertainty Escalation Decision Flow
When **H-1** triggers, the engine evaluates H-2 through H-6 checks sequentially:
* **If $\ge 2$ heuristics flag FAKE:** The final verdict is escalated to **FAKE**.
* **If $0$ heuristics flag FAKE:** The final verdict is escalated to **REAL**.
* **If exactly 1 heuristic flags FAKE:** The final verdict becomes **UNCERTAIN**.

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
