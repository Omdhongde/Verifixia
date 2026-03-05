import os
import logging
import threading
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
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS detection_logs (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                prediction VARCHAR(10) NOT NULL,
                confidence FLOAT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_detection_logs_timestamp
                ON detection_logs(timestamp)
            """,
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
