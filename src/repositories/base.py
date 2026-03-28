"""Base repository — engine-agnostic.

This file contains ZERO database-specific code. No sqlite3 imports,
no PRAGMAs, no sqlite_master. All engine-specific logic lives in
src/repositories/engines/<engine>.py.

The engine is selected based on config/settings.yaml:
    database:
      engine: "sqlite"     # Change to "mysql" or "postgresql"
      path: "db/logistics.db"

Architecture:
    AnalyticsRepository / DocumentRepository (business repos)
        | inherits
    BaseRepository (this file — delegates to engine)
        | uses
    DatabaseEngine (interface in engines/interface.py)
        | implemented by
    SQLiteEngine / MySQLEngine / PostgresEngine (in engines/)
"""

import os
from typing import Any, Optional

from src.repositories.engines import get_engine
from src.repositories.engines.interface import DatabaseEngine

# ── Path Constants ───────────────────────────────────────────────────
# Navigate from src/repositories/ up two levels to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Default database file location within the project
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "db", "logistics.db")
# SQL schema script used for database initialization
SCHEMA_PATH = os.path.join(BASE_DIR, "db", "schema.sql")

# Default engine name — can be overridden via settings or constructor
DEFAULT_ENGINE = "sqlite"


class BaseRepository:
    """Engine-agnostic base class for all repositories.

    This class acts as a thin delegation layer between business-logic
    repositories (AnalyticsRepository, DocumentRepository) and the
    actual database engine. Subclasses call methods like self._execute()
    and self._get_table_names() without knowing which database engine
    is being used underneath.

    The engine can be provided in three ways (in priority order):
    1. Direct injection via the `engine` parameter (for testing/DI)
    2. Explicit engine name via `engine_name` (e.g., "sqlite", "mysql")
    3. Default engine (SQLite) with default database path

    Args:
        db_path: Database path (for SQLite) or connection string.
        engine_name: Override engine selection ("sqlite", "mysql", etc.)
        engine: Directly inject a pre-configured engine instance (for testing).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        engine_name: Optional[str] = None,
        engine: Optional[DatabaseEngine] = None,
    ):
        if engine is not None:
            # Direct injection — used in testing and FastAPI DI
            self._engine = engine
        else:
            # Create engine from config, falling back to defaults
            self._engine = get_engine(
                engine_name or DEFAULT_ENGINE,
                db_path=db_path or DEFAULT_DB_PATH,
            )

    # ── Connection (for callers that need raw access) ────────────────

    def _get_connection(self) -> Any:
        """Return the underlying database connection object.

        Prefer _execute() over this for queries. Only use when you need
        raw connection access (e.g., the database.py facade's get_db()).

        Returns:
            Engine-specific connection object (e.g., sqlite3.Connection).
        """
        return self._engine.connect()

    # ── Query execution ──────────────────────────────────────────────

    def _execute(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a read query and return results as list of dicts.

        Args:
            sql: SQL query string.
            params: Query parameters for parameterized queries.

        Returns:
            List of dicts, one per result row.
        """
        return self._engine.execute(sql, params)

    def _execute_write(self, sql: str, params: tuple = ()) -> int:
        """Execute a write query (INSERT/UPDATE) and return lastrowid.

        Args:
            sql: SQL statement (INSERT, UPDATE, DELETE).
            params: Query parameters for parameterized queries.

        Returns:
            Last inserted row ID (engine-specific behavior).
        """
        return self._engine.execute_write(sql, params)

    # ── Schema introspection ─────────────────────────────────────────

    def _get_table_names(self) -> list[str]:
        """Return all user-created table names, sorted alphabetically.

        Returns:
            List of table name strings (excludes system/internal tables).
        """
        return self._engine.get_table_names()

    def _get_table_columns(self, table: str) -> list[dict]:
        """Return column metadata for a given table.

        Args:
            table: The table name to introspect.

        Returns:
            List of dicts with keys: name, type, notnull, default_value, is_pk.
        """
        return self._engine.get_table_columns(table)

    def _get_foreign_keys(self, table: str) -> list[dict]:
        """Return foreign key metadata for a given table.

        Args:
            table: The table name to introspect.

        Returns:
            List of dicts with keys: from_column, to_table, to_column.
        """
        return self._engine.get_foreign_keys(table)

    def _get_row_count(self, table: str) -> int:
        """Return the number of rows in a table.

        Args:
            table: The table name to count rows for.

        Returns:
            Integer row count.
        """
        return self._engine.get_row_count(table)

    def _run_schema_script(self, script_path: str) -> None:
        """Execute a SQL schema script file.

        Used for database initialization from db/schema.sql.

        Args:
            script_path: Absolute path to the .sql script file.
        """
        return self._engine.run_schema_script(script_path)
