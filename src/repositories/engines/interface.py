"""Database engine interface — the contract every engine must implement.

This is a Protocol (structural typing). Any class that has these methods
is a valid DatabaseEngine, no inheritance required. But inheriting from it
gives you IDE autocomplete and type checking.

To implement a new engine (MySQL, PostgreSQL, etc.):
1. Create a new file in this directory (e.g., mysql.py)
2. Implement all methods defined below
3. Register the class in __init__.py's ENGINES dict

The return formats are standardized across all engines:
- execute() always returns list[dict] regardless of engine
- get_table_columns() returns list of {"name", "type", "notnull", "default_value", "is_pk"}
- get_foreign_keys() returns list of {"from_column", "to_table", "to_column"}

This standardization allows BaseRepository and all business repositories
to work identically across SQLite, MySQL, PostgreSQL, etc.
"""

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class DatabaseEngine(Protocol):
    """Protocol defining the interface every database engine must implement.

    Using @runtime_checkable allows isinstance() checks at runtime, which
    is useful for dependency injection validation in tests.

    Methods are grouped into four categories:
    - Connection lifecycle: connect, close, health_check
    - Query execution: execute (read), execute_write (write)
    - Schema introspection: get_table_names, get_table_columns, etc.
    - Schema management: run_schema_script
    """

    # ── Connection lifecycle ─────────────────────────────────────────

    def connect(self) -> Any:
        """Return a connection object.

        The engine manages connection pooling/caching internally.
        Callers should not close the returned connection directly.

        Returns:
            Engine-specific connection object.
        """
        ...

    def close(self) -> None:
        """Close all connections held by this engine instance.

        Called during application shutdown to release database resources.
        """
        ...

    def health_check(self) -> bool:
        """Return True if the engine can successfully execute queries.

        Used by the /api/health endpoint to verify database connectivity.

        Returns:
            True if a simple query succeeds, False otherwise.
        """
        ...

    # ── Query execution ──────────────────────────────────────────────

    def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a read query and return results as list of dicts.

        Args:
            sql: SQL query string.
            params: Query parameters for parameterized queries.

        Returns:
            List of dicts, one dict per result row.
        """
        ...

    def execute_write(self, sql: str, params: tuple = ()) -> int:
        """Execute a write query (INSERT/UPDATE) and return last inserted row ID.

        Must auto-commit the transaction.

        Args:
            sql: SQL statement (INSERT, UPDATE, DELETE).
            params: Query parameters for parameterized queries.

        Returns:
            The ID of the last inserted row (engine-specific behavior).
        """
        ...

    # ── Schema introspection ─────────────────────────────────────────

    def get_table_names(self) -> list[str]:
        """Return all user-created table names, sorted alphabetically.

        Must exclude internal/system tables (e.g., sqlite_* tables).

        Returns:
            Sorted list of table name strings.
        """
        ...

    def get_table_columns(self, table: str) -> list[dict]:
        """Return column metadata for a table.

        Args:
            table: The table name to introspect.

        Returns:
            List of dicts, each with keys:
                name (str), type (str), notnull (bool),
                default_value (str|None), is_pk (bool)
        """
        ...

    def get_foreign_keys(self, table: str) -> list[dict]:
        """Return foreign key metadata for a table.

        Args:
            table: The table name to introspect.

        Returns:
            List of dicts, each with keys:
                from_column (str), to_table (str), to_column (str)
        """
        ...

    def get_row_count(self, table: str) -> int:
        """Return the number of rows in a table.

        Args:
            table: The table name to count.

        Returns:
            Integer row count.
        """
        ...

    # ── Schema management ────────────────────────────────────────────

    def run_schema_script(self, script_path: str) -> None:
        """Execute a SQL script file to create or update the database schema.

        Used for initial database setup from db/schema.sql.

        Args:
            script_path: Absolute path to the .sql script file.
        """
        ...
