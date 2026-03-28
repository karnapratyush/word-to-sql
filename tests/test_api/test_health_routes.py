"""Tests for health and schema API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture
def client(seeded_db_path):
    """TestClient with seeded database."""
    app = create_app(db_path=seeded_db_path)
    return TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_status_healthy(self, client):
        data = client.get("/api/health").json()
        assert data["status"] == "healthy"

    def test_includes_tables(self, client):
        data = client.get("/api/health").json()
        assert "shipments" in data["tables"]
        assert "carriers" in data["tables"]
        assert len(data["tables"]) == 7

    def test_includes_row_counts(self, client):
        data = client.get("/api/health").json()
        assert data["row_counts"]["shipments"] == 10000
        assert data["row_counts"]["carriers"] == 30


class TestSchemaEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/schema")
        assert resp.status_code == 200

    def test_includes_schema_description(self, client):
        data = client.get("/api/schema").json()
        assert "schema_description" in data
        assert "shipments" in data["schema_description"]
        assert len(data["schema_description"]) > 100

    def test_includes_table_list(self, client):
        data = client.get("/api/schema").json()
        assert "tables" in data
        assert "shipments" in data["tables"]
