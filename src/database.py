"""Database access facade — convenience functions for the rest of the app.

This module provides simple module-level functions that delegate to
AnalyticsRepository. It exists so callers don't need to instantiate
a repository class just to run a quick query or check the schema.

All functions accept an optional db_path for testing with alternate databases.
In production, the default path (db/logistics.db) is used automatically.

Usage:
    from src.database import execute_readonly, get_schema_description
    rows = execute_readonly("SELECT COUNT(*) FROM shipments")
    schema = get_schema_description()
"""

import sqlite3
from typing import Optional

from src.repositories.analytics_repo import AnalyticsRepository


# ── Connection Access ────────────────────────────────────────────────

def get_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a raw SQLite connection. Uses WAL mode and returns rows as dicts.

    Prefer execute_readonly() over this for queries. Only use get_db() when
    you need the raw connection object (e.g., for custom cursor operations).

    Args:
        db_path: Optional path to a SQLite database file.

    Returns:
        A configured sqlite3.Connection with Row factory enabled.
    """
    repo = AnalyticsRepository(db_path=db_path)
    return repo._get_connection()


# ── Query Execution ──────────────────────────────────────────────────

def execute_readonly(sql: str, params: tuple = (), db_path: Optional[str] = None) -> list[dict]:
    """Execute a SELECT query in readonly mode.

    Blocks DML/DDL statements and auto-appends LIMIT if missing.
    This is the primary way the analytics pipeline reads data.

    Args:
        sql: SQL query string (must be a SELECT statement).
        params: Optional query parameters for parameterized queries.
        db_path: Optional database path override.

    Returns:
        List of dicts, one dict per result row.

    Raises:
        ValueError: If the SQL contains DML/DDL or is not a SELECT.
    """
    repo = AnalyticsRepository(db_path=db_path)
    return repo.execute_readonly(sql, params)


# ── Schema Introspection ─────────────────────────────────────────────

def get_schema_description(db_path: Optional[str] = None) -> str:
    """Return human-readable schema description for LLM context.

    The description includes table names, column types, foreign keys,
    and row counts. This is injected into LLM prompts so the model
    understands the database structure.

    Args:
        db_path: Optional database path override.

    Returns:
        Multi-line string describing all tables and their columns.
    """
    repo = AnalyticsRepository(db_path=db_path)
    return repo.get_schema_description()


def get_table_names(db_path: Optional[str] = None) -> list[str]:
    """Return list of all user-created table names in the database.

    Args:
        db_path: Optional database path override.

    Returns:
        Sorted list of table name strings.
    """
    repo = AnalyticsRepository(db_path=db_path)
    return repo.get_table_names()


def get_row_count(table: str, db_path: Optional[str] = None) -> int:
    """Return the number of rows in a given table.

    Args:
        table: Table name (validated against known tables to prevent injection).
        db_path: Optional database path override.

    Returns:
        Integer row count.

    Raises:
        ValueError: If the table name is not found in the database.
    """
    repo = AnalyticsRepository(db_path=db_path)
    return repo.get_row_count(table)


# ── Schema Initialization ────────────────────────────────────────────

def init_db(db_path: Optional[str] = None) -> None:
    """Initialize the database from db/schema.sql.

    Creates all tables and indexes. Safe to call on an already-initialized
    database because schema.sql uses CREATE TABLE IF NOT EXISTS.

    Args:
        db_path: Optional database path override.
    """
    repo = AnalyticsRepository(db_path=db_path)
    repo.init_db()
