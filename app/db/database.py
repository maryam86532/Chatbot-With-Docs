import os
import sqlite3
import threading
from contextlib import contextmanager

from app.core.config import config, ensure_data_dir
from app.core.logger import get_logger

logger = get_logger(__name__)

# SQLite is connections-per-thread; use a lock to keep it simple for a single
# writer/reader app like this.
_db_lock = threading.Lock()
_db_path = ensure_data_dir(config.SQLITE_DB_PATH)


def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row access by name."""
    conn = sqlite3.connect(_db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_cursor():
    """Context manager yielding a cursor, committing on success."""
    conn = get_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def initialize_database():
    """Create SQLite tables if they do not exist."""
    logger.info("Initializing SQLite database at %s", _db_path)
    ensure_data_dir(config.SQLITE_DB_PATH)
    with db_cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """
        )
    logger.info("SQLite database initialized successfully.")
