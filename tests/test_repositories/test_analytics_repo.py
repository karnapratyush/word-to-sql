"""Tests for AnalyticsRepository."""

import pytest
from src.repositories.analytics_repo import AnalyticsRepository


class TestExecuteReadonly:
    """Test read-only query execution."""

    def test_simple_select(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        rows = repo.execute_readonly("SELECT COUNT(*) as cnt FROM shipments")
        assert rows[0]["cnt"] == 10_000

    def test_select_with_params(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        rows = repo.execute_readonly(
            "SELECT COUNT(*) as cnt FROM shipments WHERE mode = ?", ("ocean",)
        )
        assert rows[0]["cnt"] > 0

    def test_returns_list_of_dicts(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        rows = repo.execute_readonly("SELECT carrier_code FROM carriers LIMIT 3")
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)
        assert "carrier_code" in rows[0]

    def test_auto_adds_limit(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        rows = repo.execute_readonly("SELECT * FROM shipments")
        assert len(rows) == 100  # DEFAULT_LIMIT

    def test_respects_existing_limit(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        rows = repo.execute_readonly("SELECT * FROM shipments LIMIT 5")
        assert len(rows) == 5

    def test_blocks_insert(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        with pytest.raises(ValueError, match="Only SELECT"):
            repo.execute_readonly("INSERT INTO shipments (shipment_id) VALUES ('test')")

    def test_blocks_update(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        with pytest.raises(ValueError, match="Only SELECT"):
            repo.execute_readonly("UPDATE shipments SET status='test'")

    def test_blocks_delete(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        with pytest.raises(ValueError, match="Only SELECT"):
            repo.execute_readonly("DELETE FROM shipments")

    def test_blocks_drop(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        with pytest.raises(ValueError, match="Only SELECT"):
            repo.execute_readonly("DROP TABLE shipments")

    def test_blocks_pragma(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        with pytest.raises(ValueError, match="Only SELECT"):
            repo.execute_readonly("PRAGMA table_info(shipments)")

    def test_handles_join_query(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        rows = repo.execute_readonly(
            """SELECT c.carrier_name, COUNT(*) as cnt
            FROM shipments s
            JOIN carriers c ON s.carrier_id = c.id
            GROUP BY c.carrier_name
            ORDER BY cnt DESC LIMIT 5"""
        )
        assert len(rows) == 5
        assert "carrier_name" in rows[0]
        assert "cnt" in rows[0]


class TestGetSchemaDescription:
    """Test schema introspection."""

    def test_returns_string(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        schema = repo.get_schema_description()
        assert isinstance(schema, str)
        assert len(schema) > 100

    def test_includes_all_tables(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        schema = repo.get_schema_description()
        for table in ["carriers", "customers", "shipments", "shipment_charges",
                       "tracking_events", "invoices", "extracted_documents"]:
            assert table in schema

    def test_includes_column_names(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        schema = repo.get_schema_description()
        assert "shipment_id" in schema
        assert "carrier_name" in schema
        assert "booking_date" in schema

    def test_includes_row_counts(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        schema = repo.get_schema_description()
        assert "10,000" in schema  # shipments or customers

    def test_includes_foreign_keys(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        schema = repo.get_schema_description()
        assert "FOREIGN KEY" in schema


class TestGetTableNames:
    """Test table name listing."""

    def test_returns_all_seven_tables(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        tables = repo.get_table_names()
        assert len(tables) == 7
        assert "shipments" in tables
        assert "extracted_documents" in tables

    def test_sorted_alphabetically(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        tables = repo.get_table_names()
        assert tables == sorted(tables)


class TestGetRowCount:
    """Test row counting."""

    def test_shipments_count(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        assert repo.get_row_count("shipments") == 10_000

    def test_carriers_count(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        assert repo.get_row_count("carriers") == 30

    def test_invalid_table_raises(self, seeded_db_path):
        repo = AnalyticsRepository(db_path=seeded_db_path)
        with pytest.raises(ValueError, match="Unknown table"):
            repo.get_row_count("nonexistent_table")


class TestInitDb:
    """Test database initialization."""

    def test_creates_tables(self, empty_db_path):
        repo = AnalyticsRepository(db_path=empty_db_path)
        repo.init_db()
        tables = repo.get_table_names()
        assert "shipments" in tables
        assert "extracted_documents" in tables

    def test_idempotent(self, empty_db_path):
        repo = AnalyticsRepository(db_path=empty_db_path)
        repo.init_db()
        repo.init_db()  # Second call should not fail
        tables = repo.get_table_names()
        assert len(tables) == 7
