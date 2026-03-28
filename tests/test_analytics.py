"""Tests for the analytics pipeline modules.

Covers: planner, sql_generator, verifier, visualizer, and the full agent.
These test the interfaces and contracts — LLM calls will be mocked later.
"""

import pytest

from src.analytics.planner import classify_intent
from src.analytics.sql_generator import generate_sql
from src.analytics.verifier import verify_and_execute
from src.analytics.visualizer import suggest_chart, create_chart
from src.analytics.agent import run_analytics_query
from src.common.schemas import (
    AnalyticsRequest,
    AnalyticsResponse,
    Intent,
    PlannerResult,
    SQLGenerationResult,
    VerificationResult,
)


# ═══════════════════════════════════════════════════════════════
# PLANNER
# ═══════════════════════════════════════════════════════════════

class TestClassifyIntent:
    def test_returns_planner_result(self, seeded_db_path):
        result = classify_intent(
            query="How many shipments are delayed?",
            conversation_history=[],
            schema_description="shipments table with status column",
        )
        assert isinstance(result, PlannerResult)

    def test_sql_query_intent(self, seeded_db_path):
        result = classify_intent(
            query="What is the average freight cost per carrier?",
            conversation_history=[],
            schema_description="shipments, carriers, shipment_charges tables",
        )
        assert result.intent == Intent.SQL_QUERY
        assert result.requires_sql is True

    def test_unanswerable_intent(self, seeded_db_path):
        result = classify_intent(
            query="What is the weather in Shanghai?",
            conversation_history=[],
            schema_description="shipments table",
        )
        assert result.intent == Intent.UNANSWERABLE


# ═══════════════════════════════════════════════════════════════
# SQL GENERATOR
# ═══════════════════════════════════════════════════════════════

class TestGenerateSQL:
    def test_returns_sql_generation_result(self, seeded_db_path):
        planner = PlannerResult(intent=Intent.SQL_QUERY, reasoning="ok", requires_sql=True)
        result = generate_sql(
            query="How many shipments are there?",
            schema_description="shipments table",
            conversation_history=[],
            planner_result=planner,
        )
        assert isinstance(result, SQLGenerationResult)
        assert "SELECT" in result.sql.upper()

    def test_generates_valid_sql(self, seeded_db_path):
        planner = PlannerResult(intent=Intent.SQL_QUERY, reasoning="ok", requires_sql=True)
        result = generate_sql(
            query="Count shipments by status",
            schema_description="shipments(id, status, ...)",
            conversation_history=[],
            planner_result=planner,
        )
        assert "GROUP BY" in result.sql.upper() or "COUNT" in result.sql.upper()

    def test_retry_with_error(self, seeded_db_path):
        planner = PlannerResult(intent=Intent.SQL_QUERY, reasoning="ok", requires_sql=True)
        result = generate_sql(
            query="Average cost by carrier",
            schema_description="shipments, shipment_charges, carriers",
            conversation_history=[],
            planner_result=planner,
            retry_error="no such column: cost",
        )
        assert isinstance(result, SQLGenerationResult)


# ═══════════════════════════════════════════════════════════════
# VERIFIER
# ═══════════════════════════════════════════════════════════════

class TestVerifyAndExecute:
    def test_safe_select_passes(self, seeded_db_path):
        """Indexed query on large table should pass both safety and cost checks."""
        sql = SQLGenerationResult(
            sql="SELECT COUNT(*) as cnt FROM shipments WHERE status = 'delivered'",
            explanation="count delivered shipments (uses index on status)",
            tables_used=["shipments"],
        )
        result = verify_and_execute(sql, db_path=seeded_db_path)
        assert isinstance(result, VerificationResult)
        assert result.is_safe is True
        assert result.is_valid is True
        assert result.row_count >= 1

    def test_dangerous_sql_blocked(self, seeded_db_path):
        sql = SQLGenerationResult(
            sql="DROP TABLE shipments",
            explanation="hack",
            tables_used=["shipments"],
        )
        result = verify_and_execute(sql, db_path=seeded_db_path)
        assert result.is_safe is False

    def test_invalid_sql_caught(self, seeded_db_path):
        sql = SQLGenerationResult(
            sql="SELECTT * FROMM shipments",
            explanation="typo",
            tables_used=["shipments"],
        )
        result = verify_and_execute(sql, db_path=seeded_db_path)
        assert result.is_valid is False
        assert result.error is not None

    def test_empty_result(self, seeded_db_path):
        sql = SQLGenerationResult(
            sql="SELECT * FROM shipments WHERE shipment_id = 'NONEXISTENT'",
            explanation="no match",
            tables_used=["shipments"],
        )
        result = verify_and_execute(sql, db_path=seeded_db_path)
        assert result.is_valid is True
        assert result.row_count == 0

    def test_join_query(self, seeded_db_path):
        sql = SQLGenerationResult(
            sql="SELECT c.carrier_name, COUNT(*) as cnt FROM shipments s "
                "JOIN carriers c ON s.carrier_id = c.id GROUP BY c.carrier_name "
                "ORDER BY cnt DESC LIMIT 5",
            explanation="top 5 carriers",
            tables_used=["shipments", "carriers"],
        )
        result = verify_and_execute(sql, db_path=seeded_db_path)
        assert result.is_safe is True
        assert result.is_valid is True
        assert result.row_count == 5

    def test_indexed_query_passes_cost_check(self, seeded_db_path):
        """Queries using indexes should pass cost estimation even on large tables."""
        sql = SQLGenerationResult(
            sql="SELECT * FROM shipments WHERE status = 'delayed' LIMIT 10",
            explanation="indexed lookup on status",
            tables_used=["shipments"],
        )
        result = verify_and_execute(sql, db_path=seeded_db_path)
        assert result.is_safe is True
        assert result.is_valid is True

    def test_cost_estimation_does_not_block_small_tables(self, seeded_db_path):
        """Full scans on small tables (< threshold) should still pass."""
        sql = SQLGenerationResult(
            sql="SELECT * FROM carriers",
            explanation="scan all carriers (only 30 rows)",
            tables_used=["carriers"],
        )
        result = verify_and_execute(sql, db_path=seeded_db_path)
        # carriers has 30 rows — well below 1K threshold
        assert result.is_safe is True
        assert result.is_valid is True

    def test_full_scan_large_table_blocked(self, seeded_db_path):
        """Full scan on shipments (10K rows, threshold 1K) should be blocked."""
        sql = SQLGenerationResult(
            sql="SELECT * FROM shipments",
            explanation="scan all shipments without filter",
            tables_used=["shipments"],
        )
        result = verify_and_execute(sql, db_path=seeded_db_path)
        assert result.is_safe is True   # SQL syntax is fine
        assert result.is_valid is False  # But too expensive
        assert "scan" in result.error.lower() or "filter" in result.error.lower()


# ═══════════════════════════════════════════════════════════════
# COST ESTIMATION (unit tests for _estimate_query_cost)
# ═══════════════════════════════════════════════════════════════

class TestEstimateQueryCost:
    """Tests for the EXPLAIN QUERY PLAN cost estimation step."""

    def test_full_scan_on_large_table_blocked(self, seeded_db_path):
        """Full scan on shipments (10K rows) should be blocked when threshold is 1K."""
        from src.analytics.verifier import _estimate_query_cost
        from src.repositories.analytics_repo import AnalyticsRepository
        repo = AnalyticsRepository(db_path=seeded_db_path)
        # SELECT * FROM shipments without WHERE → full SCAN on 10K rows → blocked
        result = _estimate_query_cost(repo, "SELECT * FROM shipments")
        assert result is not None  # Should return error message
        assert "full table scan" in result.lower() or "scan" in result.lower()

    def test_full_scan_on_small_table_passes(self, seeded_db_path):
        """Full scan on carriers (30 rows) should pass — below 1K threshold."""
        from src.analytics.verifier import _estimate_query_cost
        from src.repositories.analytics_repo import AnalyticsRepository
        repo = AnalyticsRepository(db_path=seeded_db_path)
        result = _estimate_query_cost(repo, "SELECT * FROM carriers")
        assert result is None  # 30 rows is well below 1K

    def test_indexed_where_no_cost_issue(self, seeded_db_path):
        """Queries with indexed WHERE clauses should use SEARCH, not SCAN."""
        from src.analytics.verifier import _estimate_query_cost
        from src.repositories.analytics_repo import AnalyticsRepository
        repo = AnalyticsRepository(db_path=seeded_db_path)
        result = _estimate_query_cost(
            repo,
            "SELECT * FROM shipments WHERE status = 'delayed' LIMIT 10"
        )
        assert result is None  # Index used, no full scan

    def test_count_star_blocked_on_large_table(self, seeded_db_path):
        """COUNT(*) without WHERE on a large table also does a full scan."""
        from src.analytics.verifier import _estimate_query_cost
        from src.repositories.analytics_repo import AnalyticsRepository
        repo = AnalyticsRepository(db_path=seeded_db_path)
        result = _estimate_query_cost(repo, "SELECT COUNT(*) FROM shipments")
        # COUNT(*) on 10K rows with threshold 1K → should be flagged
        assert result is not None

    def test_returns_none_on_explain_failure(self, seeded_db_path):
        """If EXPLAIN itself fails, should return None (non-blocking)."""
        from src.analytics.verifier import _estimate_query_cost
        from src.repositories.analytics_repo import AnalyticsRepository
        repo = AnalyticsRepository(db_path=seeded_db_path)
        # Intentionally bad SQL — EXPLAIN should fail gracefully
        result = _estimate_query_cost(repo, "INVALID SQL GIBBERISH")
        assert result is None  # Failed EXPLAIN is non-blocking

    def test_cost_error_message_suggests_filters(self, seeded_db_path):
        """Error message should suggest adding WHERE filters."""
        from src.analytics.verifier import _estimate_query_cost
        from src.repositories.analytics_repo import AnalyticsRepository
        repo = AnalyticsRepository(db_path=seeded_db_path)
        result = _estimate_query_cost(repo, "SELECT * FROM shipments")
        assert result is not None
        assert "WHERE" in result or "filter" in result.lower()


# ═══════════════════════════════════════════════════════════════
# VISUALIZER
# ═══════════════════════════════════════════════════════════════

class TestSuggestChart:
    def test_categorical_numeric_returns_bar(self):
        result = suggest_chart(
            query="shipments by carrier",
            columns=["carrier_name", "count"],
            row_count=10,
        )
        assert result == "bar"

    def test_date_numeric_returns_line(self):
        result = suggest_chart(
            query="shipments over time",
            columns=["booking_date", "count"],
            row_count=30,
        )
        assert result == "line"

    def test_few_categories_returns_pie(self):
        result = suggest_chart(
            query="shipments by mode",
            columns=["mode", "count"],
            row_count=4,
        )
        assert result == "pie"

    def test_single_row_returns_none(self):
        result = suggest_chart(
            query="total count",
            columns=["count"],
            row_count=1,
        )
        assert result is None

    def test_returns_string_or_none(self):
        result = suggest_chart("test", ["a", "b"], 5)
        assert result is None or isinstance(result, str)


class TestCreateChart:
    def test_bar_chart_returns_dict(self):
        data = [
            {"carrier": "Maersk", "count": 100},
            {"carrier": "DHL", "count": 80},
        ]
        result = create_chart("bar", data, ["carrier", "count"], "shipments by carrier")
        assert isinstance(result, dict)

    def test_line_chart_returns_dict(self):
        data = [
            {"month": "2024-01", "count": 50},
            {"month": "2024-02", "count": 60},
        ]
        result = create_chart("line", data, ["month", "count"], "monthly trend")
        assert isinstance(result, dict)

    def test_pie_chart_returns_dict(self):
        data = [
            {"mode": "ocean", "count": 5500},
            {"mode": "air", "count": 2500},
            {"mode": "road", "count": 1200},
        ]
        result = create_chart("pie", data, ["mode", "count"], "by mode")
        assert isinstance(result, dict)

    def test_empty_data_returns_none(self):
        result = create_chart("bar", [], ["a", "b"], "empty")
        assert result is None

    def test_unknown_chart_type_returns_none(self):
        result = create_chart("scatter3d", [{"a": 1}], ["a"], "test")
        assert result is None


# ═══════════════════════════════════════════════════════════════
# FULL ANALYTICS AGENT
# ═══════════════════════════════════════════════════════════════

class TestRunAnalyticsQuery:
    def test_returns_analytics_response(self, seeded_db_path):
        req = AnalyticsRequest(user_query="How many shipments are there?")
        resp = run_analytics_query(req)
        assert isinstance(resp, AnalyticsResponse)

    def test_includes_sql_for_data_question(self, seeded_db_path):
        req = AnalyticsRequest(user_query="Count shipments by status")
        resp = run_analytics_query(req)
        assert resp.sql_query is not None
        assert "SELECT" in resp.sql_query.upper()

    def test_includes_model_used(self, seeded_db_path):
        req = AnalyticsRequest(user_query="Average freight cost?")
        resp = run_analytics_query(req)
        assert resp.model_used != ""

    def test_handles_unanswerable(self, seeded_db_path):
        req = AnalyticsRequest(user_query="What is the meaning of life?")
        resp = run_analytics_query(req)
        # Should not crash. May be classified as UNANSWERABLE ("cannot answer")
        # or GENERAL (welcome message). Either is acceptable — the key is no SQL runs.
        assert resp.sql_query is None

    def test_handles_empty_input(self, seeded_db_path):
        req = AnalyticsRequest(user_query="")
        resp = run_analytics_query(req)
        assert resp.error is not None or resp.answer != ""
