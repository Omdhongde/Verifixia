import os
import logging
import threading
import uuid
import json
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter: support both psycopg2 (v2) and psycopg (v3).
# psycopg2-binary has no wheel for Python 3.14+; psycopg v3 does.
# ---------------------------------------------------------------------------
_DB_DRIVER = None  # "psycopg2" | "psycopg3" | None

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    _DB_DRIVER = "psycopg2"
except ImportError:
    pass

if _DB_DRIVER is None:
    try:
        import psycopg  # psycopg v3  # noqa: F401
        _DB_DRIVER = "psycopg3"
    except ImportError:
        pass

if _DB_DRIVER is None:
    logger.warning(
        "No PostgreSQL driver found (psycopg2-binary / psycopg[binary]). "
        "Database operations will be unavailable."
    )


def _open_connection(database_url: str):
    """Open a raw connection using whichever driver is available."""
    if _DB_DRIVER == "psycopg2":
        return psycopg2.connect(database_url)
    elif _DB_DRIVER == "psycopg3":
        conn = psycopg.connect(database_url)
        conn.autocommit = False
        return conn
    raise RuntimeError("No PostgreSQL driver available")


def _cursor_as_dict(conn):
    """Return a dict-row cursor."""
    if _DB_DRIVER == "psycopg2":
        return conn.cursor(cursor_factory=RealDictCursor)
    else:  # psycopg3
        from psycopg.rows import dict_row  # type: ignore[import]
        return conn.cursor(row_factory=dict_row)


# ---------------------------------------------------------------------------
# Minimal thread-safe connection pool
# ---------------------------------------------------------------------------
class _SimplePool:
    def __init__(self, database_url: str, maxconn: int = 5):
        self._url = database_url
        self._maxconn = maxconn
        self._pool: list = []
        self._lock = threading.Lock()

    def getconn(self):
        with self._lock:
            if self._pool:
                return self._pool.pop()
        return _open_connection(self._url)

    def putconn(self, conn):
        try:
            conn.rollback()
            with self._lock:
                if len(self._pool) < self._maxconn:
                    self._pool.append(conn)
                    return
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    def closeall(self):
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except Exception:
                    pass
            self._pool.clear()


# ---------------------------------------------------------------------------
# NeonDB class
# ---------------------------------------------------------------------------
class NeonDB:
    def __init__(self):
        self.database_url = os.getenv("NETLIFY_DATABASE_URL") or os.getenv("DATABASE_URL")
        self.pool: _SimplePool | None = None

        if _DB_DRIVER is None:
            logger.warning("No DB driver – database disabled.")
        elif not self.database_url:
            logger.warning("DATABASE_URL / NETLIFY_DATABASE_URL not set – database disabled.")
        else:
            try:
                self.pool = _SimplePool(self.database_url)
                logger.info(f"DB pool ready (driver={_DB_DRIVER})")
            except Exception as e:
                logger.error(f"Failed to create connection pool: {e}")
                self.pool = None

    # ------------------------------------------------------------------
    def _get_conn(self):
        if not self.pool:
            raise RuntimeError("Database not configured or driver unavailable.")
        return self.pool.getconn()

    def _put_conn(self, conn):
        if self.pool:
            self.pool.putconn(conn)

    # ------------------------------------------------------------------
    def execute_query(self, query: str, params=None) -> list:
        """Run a SELECT; return list of dicts."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = _cursor_as_dict(conn)
            cursor.execute(query, params or ())
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            if cursor:
                cursor.close()
            self._put_conn(conn)

    def execute_update(self, query: str, params=None) -> int:
        """Run INSERT / UPDATE / DELETE; return rowcount."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            conn.rollback()
            logger.error(f"DB update error: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            self._put_conn(conn)

    def execute_query_single(self, query: str, params=None):
        """Return first row or None."""
        rows = self.execute_query(query, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    def create_tables(self):
        """Create schema if it doesn't exist."""
        ddl = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(255) PRIMARY KEY,
                username VARCHAR(100) UNIQUE,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) REFERENCES users(id) ON DELETE CASCADE,
                filename VARCHAR(255) NOT NULL,
                file_size INTEGER NULL,
                face_count INTEGER NOT NULL CHECK (face_count >= 0),
                deepfake_detected BOOLEAN NOT NULL,
                overall_confidence FLOAT NOT NULL,
                faces_data JSON NULL,
                processing_time_ms INTEGER NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS face_detections (
                id SERIAL PRIMARY KEY,
                analysis_id VARCHAR(255) REFERENCES analyses(id) ON DELETE CASCADE,
                label VARCHAR(50) NOT NULL,
                confidence FLOAT NOT NULL,
                position_x INTEGER NULL,
                position_y INTEGER NULL,
                position_width INTEGER NULL,
                position_height INTEGER NULL,
                quality_score FLOAT NULL,
                thumbnail_base64 TEXT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS detection_logs (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) REFERENCES users(id) ON DELETE CASCADE,
                message TEXT NOT NULL,
                log_type VARCHAR(50) DEFAULT 'analysis',
                severity VARCHAR(20) DEFAULT 'info',
                analysis_id VARCHAR(255) REFERENCES analyses(id) ON DELETE SET NULL,
                metadata JSON NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS statistics_snapshots (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) REFERENCES users(id) ON DELETE CASCADE,
                period_type VARCHAR(20) NOT NULL,
                period_date DATE NOT NULL,
                total_analyses INTEGER DEFAULT 0,
                total_deepfakes INTEGER DEFAULT 0,
                total_real INTEGER DEFAULT 0,
                average_confidence FLOAT DEFAULT 0.0,
                UNIQUE (user_id, period_type, period_date)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_user_created ON analyses(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_analyses_deepfake ON analyses(deepfake_detected)",
            "CREATE INDEX IF NOT EXISTS idx_face_detections_analysis ON face_detections(analysis_id)",
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_severity ON detection_logs(severity)",
            "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_timestamp ON detection_logs(timestamp)"
        ]
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            for stmt in ddl:
                cursor.execute(stmt)
            conn.commit()
            logger.info("Database tables ready.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating tables: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            self._put_conn(conn)

    def save_analysis_transaction(self, user_id, filename, file_size, face_count,
                                  deepfake_detected, overall_confidence, faces_data,
                                  processing_time_ms, faces, log_message=None,
                                  log_type='analysis', severity='info', log_metadata=None):
        """Atomic transaction for saving all analysis details and updating statistics."""
        analysis_id = str(uuid.uuid4())
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            
            # 1. Insert analysis record
            analysis_query = """
                INSERT INTO analyses (
                    id, user_id, filename, file_size, face_count,
                    deepfake_detected, overall_confidence, faces_data, processing_time_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(analysis_query, (
                analysis_id, user_id, filename, file_size, face_count,
                deepfake_detected, overall_confidence, json.dumps(faces_data), processing_time_ms
            ))
            
            # 2. Insert face detections
            if faces:
                face_query = """
                    INSERT INTO face_detections (
                        analysis_id, label, confidence, position_x, position_y,
                        position_width, position_height, quality_score, thumbnail_base64
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                for face in faces:
                    pos = face.get("position", {})
                    cursor.execute(face_query, (
                        analysis_id,
                        face.get("label"),
                        face.get("confidence"),
                        pos.get("x"),
                        pos.get("y"),
                        pos.get("w"),
                        pos.get("h"),
                        face.get("quality"),
                        face.get("thumbnail")
                    ))
            
            # 3. Create log entry
            log_msg = log_message or f"Analysis completed: {filename} ({face_count} faces, {'deepfake' if deepfake_detected else 'real'})"
            log_query = """
                INSERT INTO detection_logs (
                    user_id, message, log_type, severity, analysis_id, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(log_query, (
                user_id,
                log_msg,
                log_type,
                severity,
                analysis_id,
                json.dumps(log_metadata or {})
            ))
            
            # 4. Update statistics (only if user_id is not None)
            if user_id:
                tf = 1 if deepfake_detected else 0
                tr = 0 if deepfake_detected else 1
                
                stats_query = """
                    INSERT INTO statistics_snapshots (
                        user_id, period_type, period_date,
                        total_analyses, total_deepfakes, total_real, average_confidence
                    ) VALUES (%s, %s, CURRENT_DATE, 1, %s, %s, %s)
                    ON CONFLICT (user_id, period_type, period_date)
                    DO UPDATE SET
                        total_analyses = statistics_snapshots.total_analyses + 1,
                        total_deepfakes = statistics_snapshots.total_deepfakes + EXCLUDED.total_deepfakes,
                        total_real = statistics_snapshots.total_real + EXCLUDED.total_real,
                        average_confidence = (statistics_snapshots.average_confidence * statistics_snapshots.total_analyses + EXCLUDED.average_confidence) / (statistics_snapshots.total_analyses + 1)
                """
                for period in ['daily', 'weekly', 'monthly']:
                    cursor.execute(stats_query, (
                        user_id,
                        period,
                        tf,
                        tr,
                        overall_confidence
                    ))
                    
            conn.commit()
            logger.info(f"✓ DB transaction completed successfully for analysis ID: {analysis_id}")
            return {
                "analysis_id": analysis_id,
                "status": "success"
            }
        except Exception as e:
            conn.rollback()
            logger.error(f"Error in DB transaction: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            self._put_conn(conn)

    def upsert_user(self, user_id, email, username=None):
        """Insert a user or update their email/username on conflict."""
        query = """
            INSERT INTO users (id, email, username)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                email = EXCLUDED.email,
                username = COALESCE(EXCLUDED.username, users.username)
            RETURNING id
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(query, (user_id, email, username))
            conn.commit()
            return user_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error upserting user: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            self._put_conn(conn)

    # ------------------------------------------------------------------
    def get_detection_logs(self, limit: int = 100, offset: int = 0) -> list:
        return self.execute_query(
            "SELECT * FROM detection_logs ORDER BY timestamp DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )

    def save_detection_log(self, filename: str, prediction: str,
                           confidence: float, user_id=None):
        query = """
            INSERT INTO detection_logs (filename, prediction, confidence, user_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id, timestamp
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = _cursor_as_dict(conn)
            cursor.execute(query, (filename, prediction, confidence, user_id))
            conn.commit()
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving detection log: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            self._put_conn(conn)

    def close(self):
        if self.pool:
            self.pool.closeall()


# Module-level singleton
db = NeonDB()
