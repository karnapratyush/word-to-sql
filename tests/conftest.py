"""Shared fixtures for all tests."""

import os
import sqlite3
import tempfile

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "logistics.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "db", "schema.sql")


@pytest.fixture
def seeded_db_path():
    """Return path to the seeded logistics.db (read-only tests should use this)."""
    assert os.path.exists(DB_PATH), f"Seeded DB not found at {DB_PATH}. Run db/seed_data.py first."
    return DB_PATH


@pytest.fixture
def empty_db_path(tmp_path):
    """Create a fresh empty database with schema applied. For write tests."""
    db_file = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_file)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.close()
    return db_file


@pytest.fixture
def db_connection(seeded_db_path):
    """Return a connection to the seeded database."""
    conn = sqlite3.connect(seeded_db_path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
