"""Tests for src/database.py

Tests database connection, readonly execution, schema description,
and safety guards against DML.
"""

import sqlite3

import pytest

from src.database import (
    execute_readonly,
    get_db,
    get_row_count,
    get_schema_description,
    get_table_names,
    init_db,
)
from src.common.exceptions import SQLExecutionError


# ── get_db ──────────────────────────────────────────────────────

class TestGetDb:
    def test_returns_connection(self, seeded_db_path):
        conn = get_db(db_path=seeded_db_path)
        assert conn is not None
        assert isinstance(conn, sqlite3.Connection)

    def test_returns_dict_rows(self, seeded_db_path):
        conn = get_db(db_path=seeded_db_path)
        cursor = conn.execute("SELECT 1 as val")
        row = cursor.fetchone()
        # Should be accessible by column name
        assert row["val"] == 1 or row[0] == 1

    def test_same_path_returns_working_connection(self, seeded_db_path):
        conn = get_db(db_path=seeded_db_path)
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM shipments")
        row = cursor.fetchone()
        assert row[0] > 0


# ── execute_readonly ────────────────────────────────────────────

class TestExecuteReadonly:
    def test_select_returns_list_of_dicts(self, seeded_db_path):
        result = execute_readonly("SELECT * FROM carriers LIMIT 5", db_path=seeded_db_path)
        assert isinstance(result, list)
        assert len(result) == 5
        assert isinstance(result[0], dict)
        assert "carrier_name" in result[0]

    def test_select_count(self, seeded_db_path):
        result = execute_readonly(
            "SELECT COUNT(*) as cnt FROM shipments", db_path=seeded_db_path
        )
        assert result[0]["cnt"] == 10000

    def test_blocks_insert(self, seeded_db_path):
        with pytest.raises((SQLExecutionError, Exception)):
            execute_readonly(
                "INSERT INTO carriers (carrier_code, carrier_name, carrier_type) VALUES ('X','X','X')",
                db_path=seeded_db_path,
            )

    def test_blocks_delete(self, seeded_db_path):
        with pytest.raises((SQLExecutionError, Exception)):
            execute_readonly("DELETE FROM carriers WHERE id = 1", db_path=seeded_db_path)

    def test_blocks_drop(self, seeded_db_path):
        with pytest.raises((SQLExecutionError, Exception)):
            execute_readonly("DROP TABLE carriers", db_path=seeded_db_path)

    def test_blocks_update(self, seeded_db_path):
        with pytest.raises((SQLExecutionError, Exception)):
            execute_readonly(
                "UPDATE carriers SET carrier_name = 'X' WHERE id = 1",
                db_path=seeded_db_path,
            )

    def test_adds_limit_if_missing(self, seeded_db_path):
        result = execute_readonly("SELECT * FROM shipments", db_path=seeded_db_path)
        # Should not return all 10000 rows — limit should be auto-applied
        assert len(result) <= 100

    def test_respects_existing_limit(self, seeded_db_path):
        result = execute_readonly(
            "SELECT * FROM shipments LIMIT 3", db_path=seeded_db_path
        )
        assert len(result) == 3

    def test_handles_empty_result(self, seeded_db_path):
        result = execute_readonly(
            "SELECT * FROM shipments WHERE shipment_id = 'NONEXISTENT'",
            db_path=seeded_db_path,
        )
        assert result == []

    def test_join_query_works(self, seeded_db_path):
        result = execute_readonly(
            "SELECT s.shipment_id, c.carrier_name FROM shipments s "
            "JOIN carriers c ON s.carrier_id = c.id LIMIT 3",
            db_path=seeded_db_path,
        )
        assert len(result) == 3
        assert "carrier_name" in result[0]


# ── get_schema_description ──────────────────────────────────────

class TestGetSchemaDescription:
    def test_returns_string(self, seeded_db_path):
        desc = get_schema_description(db_path=seeded_db_path)
        assert isinstance(desc, str)
        assert len(desc) > 100

    def test_includes_table_names(self, seeded_db_path):
        desc = get_schema_description(db_path=seeded_db_path)
        for table in ["shipments", "carriers", "customers", "invoices", "tracking_events"]:
            assert table in desc

    def test_includes_column_info(self, seeded_db_path):
        desc = get_schema_description(db_path=seeded_db_path)
        assert "shipment_id" in desc
        assert "carrier_name" in desc

    def test_includes_foreign_keys(self, seeded_db_path):
        desc = get_schema_description(db_path=seeded_db_path)
        # Should mention relationships
        assert "customer_id" in desc


# ── get_table_names ─────────────────────────────────────────────

class TestGetTableNames:
    def test_returns_all_tables(self, seeded_db_path):
        tables = get_table_names(db_path=seeded_db_path)
        assert isinstance(tables, list)
        expected = {"carriers", "customers", "shipments", "shipment_charges",
                    "tracking_events", "invoices", "extracted_documents"}
        assert expected.issubset(set(tables))

    def test_does_not_include_sqlite_internals(self, seeded_db_path):
        tables = get_table_names(db_path=seeded_db_path)
        for t in tables:
            assert not t.startswith("sqlite_")


# ── get_row_count ───────────────────────────────────────────────

class TestGetRowCount:
    def test_shipments_count(self, seeded_db_path):
        assert get_row_count("shipments", db_path=seeded_db_path) == 10000

    def test_carriers_count(self, seeded_db_path):
        assert get_row_count("carriers", db_path=seeded_db_path) == 30

    def test_extracted_documents_exists(self, seeded_db_path):
        # Table exists and is queryable (may have data from vision pipeline usage)
        count = get_row_count("extracted_documents", db_path=seeded_db_path)
        assert count >= 0


# ── init_db ─────────────────────────────────────────────────────

class TestInitDb:
    def test_creates_all_tables(self, tmp_path):
        db_file = str(tmp_path / "new.db")
        init_db(db_path=db_file)
        conn = sqlite3.connect(db_file)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        expected = {"carriers", "customers", "shipments", "shipment_charges",
                    "tracking_events", "invoices", "extracted_documents"}
        assert expected.issubset(tables)

    def test_idempotent(self, tmp_path):
        db_file = str(tmp_path / "new.db")
        init_db(db_path=db_file)
        init_db(db_path=db_file)  # should not fail
