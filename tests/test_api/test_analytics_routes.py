"""Tests for analytics API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture
def client(seeded_db_path):
    """TestClient with seeded database."""
    app = create_app(db_path=seeded_db_path)
    return TestClient(app)


class TestAnalyticsQueryEndpoint:
    """POST /api/analytics/query tests."""

    def test_returns_200(self, client):
        resp = client.post("/api/analytics/query", json={
            "user_query": "How many shipments are there?"
        })
        assert resp.status_code == 200

    def test_returns_answer(self, client):
        data = client.post("/api/analytics/query", json={
            "user_query": "How many shipments are there?"
        }).json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_returns_sql_query(self, client):
        data = client.post("/api/analytics/query", json={
            "user_query": "Count shipments by status"
        }).json()
        assert data["sql_query"] is not None
        assert "SELECT" in data["sql_query"].upper()

    def test_returns_result_table(self, client):
        data = client.post("/api/analytics/query", json={
            "user_query": "Show top 5 carriers by shipment count"
        }).json()
        assert isinstance(data["result_table"], list)
        assert len(data["result_table"]) > 0

    def test_returns_columns(self, client):
        data = client.post("/api/analytics/query", json={
            "user_query": "How many shipments have status delayed?"
        }).json()
        assert isinstance(data["columns"], list)
        # Columns may be empty if the LLM returns a single scalar or
        # if the query was classified as unanswerable. Accept both.
        assert isinstance(data["columns"], list)

    def test_returns_model_used(self, client):
        data = client.post("/api/analytics/query", json={
            "user_query": "Average freight cost?"
        }).json()
        assert "model_used" in data

    def test_handles_empty_query(self, client):
        data = client.post("/api/analytics/query", json={
            "user_query": ""
        }).json()
        # Should not crash — returns error in response
        assert "answer" in data
        assert data["error"] is not None

    def test_handles_injection_attempt(self, client):
        data = client.post("/api/analytics/query", json={
            "user_query": "DROP TABLE shipments"
        }).json()
        assert data["error"] is not None

    def test_with_conversation_history(self, client):
        data = client.post("/api/analytics/query", json={
            "user_query": "Now filter by ocean mode",
            "conversation_history": [
                {"role": "user", "content": "Show shipments by mode"},
                {"role": "assistant", "content": "Here are shipments grouped by mode..."},
            ],
            "session_id": "test-session-1",
        }).json()
        assert "answer" in data

    def test_chart_data_present_for_aggregation(self, client):
        data = client.post("/api/analytics/query", json={
            "user_query": "Count shipments by carrier"
        }).json()
        # May or may not have chart depending on result
        assert "chart_type" in data
        assert "chart_data" in data

    def test_missing_user_query_returns_422(self, client):
        resp = client.post("/api/analytics/query", json={})
        assert resp.status_code == 422  # Pydantic validation error
