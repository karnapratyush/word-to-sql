"""SQLite engine implementation.

All SQLite-specific code is contained in this single file:
- sqlite3 module usage
- PRAGMA calls (WAL mode, foreign keys)
- sqlite_master queries for schema introspection
- PRAGMA table_info / foreign_key_list for column/FK metadata
- executescript for schema management
- Thread-local connection pooling

No other file in the repository layer imports sqlite3. This isolation
means switching to a different database engine requires no changes to
business logic — only a new engine file needs to be created.
"""

import os
import sqlite3
import threading
from typing import Optional

# ── Thread-Local Connection Pool ─────────────────────────────────────
# Each thread gets its own connection to avoid SQLite's threading issues.
# Connections are stored as {db_path: sqlite3.Connection} per thread.
_local = threading.local()

# ── Path Constants ───────────────────────────────────────────────────
# Navigate from src/repositories/engines/ up three levels to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "db", "logistics.db")


class SQLiteEngine:
    """SQLite database engine with thread-local connection management.

    Each thread gets its own connection, reused across calls within
    that thread. PRAGMAs (WAL mode, foreign keys) are set once per
    connection creation, not per query, for performance.

    Connection health is verified on each access — if the connection
    was externally closed, it is automatically recreated.

    Args:
        db_path: Path to the SQLite database file.
                 Defaults to db/logistics.db in the project root.
        **kwargs: Ignored; accepted for compatibility with the engine
                  factory's generic **kwargs passing.
    """

    def __init__(self, db_path: Optional[str] = None, **kwargs):
        self._db_path = db_path or DEFAULT_DB_PATH

    # ── Connection lifecycle ─────────────────────────────────────────

    def connect(self) -> sqlite3.Connection:
        """Return a thread-local connection, creating one if needed.

        First call per thread:
          - Opens a new connection to the database file
          - Enables WAL journal mode (allows concurrent reads during writes)
          - Enables foreign key enforcement (OFF by default in SQLite)
          - Sets row_factory to sqlite3.Row (rows behave like dicts)

        Subsequent calls from the same thread:
          - Returns the cached connection if still alive
          - Recreates the connection if it was externally closed

        Returns:
            A configured sqlite3.Connection instance.
        """
        # Initialize the thread-local connections dict if this is a new thread
        if not hasattr(_local, "connections"):
            _local.connections = {}

        conn = _local.connections.get(self._db_path)
        if conn is not None:
            try:
                # Health check: verify connection is still usable
                conn.execute("SELECT 1")
                return conn
            except sqlite3.ProgrammingError:
                # Connection was closed externally; remove stale reference
                del _local.connections[self._db_path]

        # Create a fresh connection with SQLite-specific configuration
        conn = self._create_connection()
        _local.connections[self._db_path] = conn
        return conn

    def close(self) -> None:
        """Close the thread-local connection for this database path.

        Safe to call even if no connection exists. Silently ignores
        errors during close (e.g., connection already closed).
        """
        if hasattr(_local, "connections") and self._db_path in _local.connections:
            try:
                _local.connections[self._db_path].close()
            except Exception:
                pass  # Best-effort close; don't crash on cleanup
            del _local.connections[self._db_path]

    def health_check(self) -> bool:
        """Return True if the engine can execute a simple query.

        Returns:
            True if SELECT 1 succeeds, False otherwise.
        """
        try:
            self.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ── Query execution ──────────────────────────────────────────────

    def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a read query and return results as list of dicts.

        Args:
            sql: SQL query string.
            params: Query parameters for parameterized queries.

        Returns:
            List of dicts, one per result row.
        """
        conn = self.connect()
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        # Convert sqlite3.Row objects to plain dicts for serialization
        return [dict(row) for row in rows]

    def execute_write(self, sql: str, params: tuple = ()) -> int:
        """Execute a write query, commit the transaction, and return lastrowid.

        Args:
            sql: SQL statement (INSERT, UPDATE, DELETE).
            params: Query parameters for parameterized queries.

        Returns:
            The rowid of the last inserted row.
        """
        conn = self.connect()
        cursor = conn.execute(sql, params)
        conn.commit()  # Auto-commit each write operation
        return cursor.lastrowid

    # ── Schema introspection ─────────────────────────────────────────

    def get_table_names(self) -> list[str]:
        """Return user table names by querying sqlite_master.

        Excludes SQLite's internal tables (those starting with 'sqlite_').

        Returns:
            Sorted list of table name strings.
        """
        rows = self.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        return [row["name"] for row in rows]

    def get_table_columns(self, table: str) -> list[dict]:
        """Return column metadata using SQLite's PRAGMA table_info().

        Translates SQLite's native column info format into the standardized
        format expected by BaseRepository and the schema description builder.

        Args:
            table: The table name to introspect.

        Returns:
            List of dicts with keys: name, type, notnull, default_value, is_pk.
        """
        conn = self.connect()
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        return [
            {
                "name": col["name"],
                "type": col["type"] or "TEXT",       # SQLite allows empty type; default to TEXT
                "notnull": bool(col["notnull"]),
                "default_value": col["dflt_value"],   # None if no default
                "is_pk": bool(col["pk"]),             # 1-based PK index, 0 = not PK
            }
            for col in columns
        ]

    def get_foreign_keys(self, table: str) -> list[dict]:
        """Return foreign key metadata using SQLite's PRAGMA foreign_key_list().

        Translates SQLite's FK info into the standardized format.

        Args:
            table: The table name to introspect.

        Returns:
            List of dicts with keys: from_column, to_table, to_column.
        """
        conn = self.connect()
        cursor = conn.execute(f"PRAGMA foreign_key_list({table})")
        fks = cursor.fetchall()
        return [
            {
                "from_column": fk["from"],    # Local column name
                "to_table": fk["table"],       # Referenced table
                "to_column": fk["to"],         # Referenced column
            }
            for fk in fks
        ]

    def get_row_count(self, table: str) -> int:
        """Return the number of rows in a table using standard SQL COUNT(*).

        Args:
            table: The table name to count rows for.

        Returns:
            Integer row count.
        """
        rows = self.execute(f"SELECT COUNT(*) as cnt FROM {table}")
        return rows[0]["cnt"]

    # ── Schema management ────────────────────────────────────────────

    def run_schema_script(self, script_path: str) -> None:
        """Execute a SQL script file using SQLite's executescript().

        executescript() handles multiple statements separated by semicolons,
        making it ideal for running schema.sql which contains CREATE TABLE,
        CREATE INDEX, and INSERT statements.

        Args:
            script_path: Absolute path to the .sql script file.
        """
        conn = self.connect()
        with open(script_path) as f:
            conn.executescript(f.read())

    # ── Internal ─────────────────────────────────────────────────────

    def _create_connection(self) -> sqlite3.Connection:
        """Create a fresh SQLite connection with optimized configuration.

        Configuration applied:
        - check_same_thread=False: Allow connection sharing within our
          thread-local pool (we manage thread safety ourselves)
        - Row factory: sqlite3.Row enables dict-like row access
        - WAL mode: Write-Ahead Logging for concurrent read performance
        - Foreign keys: Enforce FK constraints (OFF by default in SQLite)

        Returns:
            A fully configured sqlite3.Connection instance.
        """
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")     # Concurrent reads during writes
        conn.execute("PRAGMA foreign_keys=ON")       # Enforce FK constraints
        return conn
