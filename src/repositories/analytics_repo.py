"""Repository for analytics read-only data access.

This module contains ZERO engine-specific code. All database introspection
(table listing, column info, foreign keys) is delegated to BaseRepository's
engine-specific methods. To switch from SQLite to MySQL/PostgreSQL, only
the engine implementation needs to change — this file stays untouched.

Key safety feature: all queries are validated to be SELECT-only.
DML/DDL statements (INSERT, UPDATE, DELETE, DROP, etc.) are blocked
with a regex check before execution.
"""

import re
from typing import Optional

from src.repositories.base import BaseRepository, SCHEMA_PATH


# Regex pattern to detect non-SELECT statements (engine-agnostic).
# Matches DML/DDL keywords at the start of the query or after a semicolon.
_DML_PATTERN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

# Default row limit appended to queries missing a LIMIT clause,
# preventing unbounded result sets from overwhelming the LLM or UI.
DEFAULT_LIMIT = 100


class AnalyticsRepository(BaseRepository):
    """Read-only data access for the analytics agent.

    This repository is used by the analytics pipeline to execute
    user-driven SQL queries against the logistics database. All queries
    are validated to be SELECT-only, and a LIMIT clause is auto-appended
    if missing to prevent runaway queries.

    Inherits from BaseRepository, which provides engine-agnostic
    database operations via the DatabaseEngine interface.
    """

    def execute_readonly(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a SELECT query after validating it is read-only.

        Safety checks (in order):
        1. Block DML/DDL keywords (INSERT, UPDATE, DELETE, DROP, etc.)
        2. Ensure query starts with SELECT (blocks PRAGMA, ATTACH, etc.)
        3. Auto-append LIMIT if not present

        Args:
            sql: The SQL query string (must be SELECT).
            params: Optional query parameters for parameterized queries.

        Returns:
            List of dicts, one per row.

        Raises:
            ValueError: If sql contains DML/DDL or does not start with SELECT.
        """
        stripped = sql.strip()

        # Block DML/DDL — these would modify data
        if _DML_PATTERN.match(stripped):
            raise ValueError(
                f"Only SELECT queries are allowed. Got: {stripped[:50]}..."
            )

        # Also block non-SELECT statements (e.g., PRAGMA, ATTACH, EXPLAIN)
        if not stripped.upper().startswith("SELECT"):
            raise ValueError(
                f"Only SELECT queries are allowed. Got: {stripped[:50]}..."
            )

        # Auto-add LIMIT to prevent unbounded result sets
        if "LIMIT" not in stripped.upper():
            # Strip trailing semicolons before appending LIMIT
            cleaned = stripped.rstrip(";").rstrip()
            sql = f"{cleaned} LIMIT {DEFAULT_LIMIT}"

        return self._execute(sql, params)

    def get_schema_description(self) -> str:
        """Return a human-readable schema description for LLM context.

        Builds a comprehensive description including:
        - Table names with row counts
        - Column names, types, and constraints (PK, NOT NULL, DEFAULT)
        - Foreign key relationships between tables

        This string is injected into LLM prompts so the model understands
        the database structure when generating SQL.

        Returns:
            Multi-line string describing all tables, columns, and relationships.
        """
        tables = self.get_table_names()
        parts = []

        for table in tables:
            # Get columns via engine-agnostic introspection method
            columns = self._get_table_columns(table)
            col_lines = []
            for col in columns:
                # Build constraint annotations for each column
                pk = " PRIMARY KEY" if col["is_pk"] else ""
                notnull = " NOT NULL" if col["notnull"] and not col["is_pk"] else ""
                default = f" DEFAULT {col['default_value']}" if col["default_value"] else ""
                col_lines.append(f"    {col['name']} {col['type']}{pk}{notnull}{default}")

            # Get foreign keys via engine-agnostic introspection method
            fks = self._get_foreign_keys(table)
            fk_lines = [
                f"    FOREIGN KEY ({fk['from_column']}) REFERENCES {fk['to_table']}({fk['to_column']})"
                for fk in fks
            ]

            # Get row count for context (helps LLM understand data volume)
            row_count = self._get_row_count(table)

            # Assemble the table description block
            header = f"TABLE: {table} ({row_count:,} rows)"
            body = "\n".join(col_lines)
            part = f"{header}\n{body}"
            if fk_lines:
                part += "\n" + "\n".join(fk_lines)

            parts.append(part)

        # Note: extracted_documents JSON field names are injected by the
        # vector store's retrieve_context() — always appended regardless
        # of similarity score. See vector_store._build_extracted_docs_context().
        # No need to duplicate that here.

        return "\n\n".join(parts)

    def get_table_names(self) -> list[str]:
        """Return list of all user-created table names.

        Returns:
            Sorted list of table name strings.
        """
        return self._get_table_names()

    def get_row_count(self, table: str) -> int:
        """Return row count for a given table.

        Validates the table name against the list of known tables to
        prevent SQL injection via crafted table names.

        Args:
            table: Table name (must exist in the database).

        Returns:
            Number of rows in the table.

        Raises:
            ValueError: If the table name is not found in the database.
        """
        # Validate table name to prevent SQL injection
        valid_tables = self.get_table_names()
        if table not in valid_tables:
            raise ValueError(f"Unknown table: {table}. Valid tables: {valid_tables}")

        return self._get_row_count(table)

    def init_db(self) -> None:
        """Initialize the database from db/schema.sql.

        Creates all tables, indexes, and constraints. Safe to call on an
        already-initialized database because schema.sql uses
        CREATE TABLE IF NOT EXISTS throughout.
        """
        self._run_schema_script(SCHEMA_PATH)
