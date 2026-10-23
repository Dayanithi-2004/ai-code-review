import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
import typing

logger = logging.getLogger(__name__)


class SessionLocal(threading.local):
    conn: typing.Optional[sqlite3.Connection] = None

class SessionManager:
    """
    Manages the SQLite database for the scan session.
    Provides thread-safe connections and initializes the schema.

    Concurrency model
    -----------------
    Revelyt runs Wave 1, Wave 2, and PM summarisation concurrently via
    asyncio.gather() + asyncio.to_thread(). Each to_thread() call runs in a
    real OS thread, so multiple threads can attempt DB writes simultaneously.

    SQLite in-memory databases (journal_mode=memory) do NOT support WAL —
    PRAGMA journal_mode=WAL is silently ignored and returns 'memory', leaving
    the default exclusive-locking behaviour in place. With fast models
    (Gemini Flash, GPT-4o) completing in ~1s, concurrent writes land in the
    same millisecond window and trigger "database table is locked" errors.

    Fix: a single threading.RLock (_db_lock) that all callers acquire via
    get_connection(). This serialises writers at the Python level before SQLite
    ever sees the contention. DB writes take ~5ms, so the serialisation adds
    negligible latency even with 10+ concurrent agents.

    busy_timeout=5000 is kept as a secondary safety net for any code path
    that accesses SQLite outside get_connection() (e.g. direct connections
    in tests or third-party tooling).
    """

    def __init__(self, db_path=":memory:"):
        # Unique ID for this scan session — used to attribute findings to a specific run
        self.session_id = str(uuid.uuid4())

        # Named shared memory URI so all threads see the same in-memory database
        self.db_path = (
            "file:revelyt_memdb?mode=memory&cache=shared"
            if db_path == ":memory:"
            else db_path
        )

        # Single process-wide write lock.
        # Acquired by get_connection() for the entire duration of each DB operation.
        # RLock (re-entrant) allows the same thread to acquire it multiple times,
        # which is needed when _init_schema() calls get_connection() from __init__.
        self._db_lock = threading.RLock()

        self.local = SessionLocal()
        self._init_schema()
        logger.info(
            "SessionManager created: session_id=%s, db=%s",
            self.session_id, self.db_path,
        )

    @contextmanager
    def get_connection(self):
        """
        Yields a thread-local database connection inside a write lock.

        The _db_lock is acquired before yielding and released after commit
        or rollback. This prevents concurrent writes from different threads
        (Wave 1, Wave 2, PM agent) from colliding inside SQLite.

        Commits on clean exit, rolls back on exception.
        """
        # Acquire the write lock before touching the DB.
        # RLock means the same thread can re-enter (e.g. nested get_connection calls).
        with self._db_lock:
            if not hasattr(self.local, "conn") or self.local.conn is None:
                self.local.conn = sqlite3.connect(
                    self.db_path, uri=True, check_same_thread=False
                )
                self.local.conn.row_factory = sqlite3.Row
                # busy_timeout: secondary safety net — if anything bypasses the
                # RLock, SQLite will retry for 5s before raising "table is locked".
                # Note: journal_mode=WAL is NOT set here because SQLite silently
                # ignores it for in-memory databases (returns 'memory', not 'wal').
                self.local.conn.execute("PRAGMA busy_timeout=30000")
                logger.debug("New thread-local DB connection (busy_timeout=30000ms)")

            try:
                yield self.local.conn
                self.local.conn.commit()
            except Exception as e:
                logger.error("DB transaction rolled back: %s", e, exc_info=True)
                self.local.conn.rollback()
                raise

    def _init_schema(self):
        """Creates all tables required for the 6-phase pipeline."""
        with self.get_connection() as conn:
            conn.executescript("""
                -- Phase 2: Dependency Graph Edges
                CREATE TABLE IF NOT EXISTS dep_graph (
                    source_file TEXT,
                    target_file TEXT,
                    UNIQUE(source_file, target_file)
                );

                -- Phase 2 & 3: File Summaries and Blast Radius
                CREATE TABLE IF NOT EXISTS module_summaries (
                    file_path        TEXT PRIMARY KEY,
                    purpose          TEXT,
                    key_functions    TEXT,
                    dependant_count  INTEGER DEFAULT 0
                );

                -- Phase 3: AI Review Findings
                -- session_id links every finding back to the scan run that produced it
                -- UNIQUE(session_id, file_path, title, agent_name) allows the same
                -- issue to be stored once per agent (needed for corroboration)
                CREATE TABLE IF NOT EXISTS findings (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT,
                    file_path    TEXT,
                    agent_name   TEXT DEFAULT '',
                    severity     TEXT,
                    category     TEXT,
                    title        TEXT,
                    description  TEXT,
                    suggestion   TEXT,
                    line_number INTEGER,
                    class_or_method TEXT,
                    confidence   REAL,
                    impact_scope TEXT,
                    UNIQUE(session_id, file_path, title, agent_name)
                );

                -- Phase 4: BRD Verification
                CREATE TABLE IF NOT EXISTS brd_requirements (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    requirement TEXT,
                    status      TEXT,
                    evidence    TEXT
                );
            """)