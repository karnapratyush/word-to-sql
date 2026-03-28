"""Tests for src/common/schemas.py

Validates that all Pydantic models work correctly —
construction, validation, serialization, edge cases.
"""

import pytest

from src.common.schemas import (
    AnalyticsRequest,
    AnalyticsResponse,
    DocumentRecord,
    ExtractionRequest,
    ExtractionResult,
    FieldExtraction,
    GuardrailResult,
    Intent,
    PlannerResult,
    ReviewStatus,
    SQLGenerationResult,
    VerificationResult,
)


# ── AnalyticsRequest ────────────────────────────────────────────

class TestAnalyticsRequest:
    def test_minimal_construction(self):
        req = AnalyticsRequest(user_query="test")
        assert req.user_query == "test"
        assert req.conversation_history == []
        assert req.session_id == ""

    def test_with_history(self):
        req = AnalyticsRequest(
            user_query="follow up",
            conversation_history=[{"role": "user", "content": "first q"}],
        )
        assert len(req.conversation_history) == 1

    def test_empty_query_allowed(self):
        # Validation of empty queries is done by guardrails, not schema
        req = AnalyticsRequest(user_query="")
        assert req.user_query == ""


# ── PlannerResult ───────────────────────────────────────────────

class TestPlannerResult:
    def test_sql_query_intent(self):
        pr = PlannerResult(intent=Intent.SQL_QUERY, reasoning="has data", requires_sql=True)
        assert pr.intent == Intent.SQL_QUERY
        assert pr.requires_sql is True

    def test_unanswerable_intent(self):
        pr = PlannerResult(
            intent=Intent.UNANSWERABLE,
            reasoning="no weather data",
            suggested_questions=["Try asking about shipments"],
        )
        assert pr.intent == Intent.UNANSWERABLE
        assert len(pr.suggested_questions) == 1

    def test_invalid_intent_rejected(self):
        with pytest.raises(Exception):
            PlannerResult(intent="invalid_intent", reasoning="x")


# ── SQLGenerationResult ────────────────────────────────────────

class TestSQLGenerationResult:
    def test_construction(self):
        r = SQLGenerationResult(
            sql="SELECT * FROM shipments LIMIT 10",
            explanation="Gets first 10 shipments",
            tables_used=["shipments"],
        )
        assert "SELECT" in r.sql

    def test_empty_tables_used_default(self):
        r = SQLGenerationResult(sql="SELECT 1", explanation="test")
        assert r.tables_used == []


# ── VerificationResult ──────────────────────────────────────────

class TestVerificationResult:
    def test_successful_result(self):
        r = VerificationResult(
            is_safe=True,
            is_valid=True,
            result_rows=[{"a": 1}],
            row_count=1,
            columns=["a"],
        )
        assert r.is_safe and r.is_valid

    def test_failed_result(self):
        r = VerificationResult(
            is_safe=False,
            is_valid=False,
            error="DML detected",
        )
        assert not r.is_safe


# ── FieldExtraction ─────────────────────────────────────────────

class TestFieldExtraction:
    def test_high_confidence(self):
        f = FieldExtraction(value="INV-001", confidence=0.95)
        assert f.needs_review is False

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            FieldExtraction(value="x", confidence=1.5)
        with pytest.raises(Exception):
            FieldExtraction(value="x", confidence=-0.1)

    def test_null_value_allowed(self):
        f = FieldExtraction(value=None, confidence=0.3, needs_review=True)
        assert f.value is None


# ── ExtractionResult ────────────────────────────────────────────

class TestExtractionResult:
    def test_construction(self):
        r = ExtractionResult(
            document_type="invoice",
            fields={"inv_num": FieldExtraction(value="INV-001", confidence=0.9)},
            overall_confidence=0.9,
        )
        assert r.document_type == "invoice"

    def test_empty_fields(self):
        r = ExtractionResult(
            document_type="unknown",
            fields={},
            overall_confidence=0.0,
        )
        assert len(r.fields) == 0


# ── DocumentRecord ──────────────────────────────────────────────

class TestDocumentRecord:
    def test_construction(self):
        r = DocumentRecord(
            document_id="doc-123",
            document_type="invoice",
            file_name="test.pdf",
            extraction_model="gemini-2.0-flash",
            overall_confidence=0.85,
            extracted_fields={"invoice_number": "INV-001"},
            confidence_scores={"invoice_number": 0.95},
        )
        assert r.review_status == ReviewStatus.PENDING

    def test_with_linked_shipment(self):
        r = DocumentRecord(
            document_id="doc-456",
            document_type="bill_of_lading",
            file_name="bol.pdf",
            extraction_model="gpt-4o-mini",
            overall_confidence=0.78,
            extracted_fields={},
            confidence_scores={},
            linked_shipment_id="SHP-2024-000001",
        )
        assert r.linked_shipment_id is not None


# ── GuardrailResult ─────────────────────────────────────────────

class TestGuardrailResult:
    def test_pass(self):
        r = GuardrailResult(passed=True, sanitized_input="clean input")
        assert r.passed

    def test_fail_with_reason(self):
        r = GuardrailResult(passed=False, reason="SQL injection detected", blocked_patterns=["DROP"])
        assert not r.passed
        assert len(r.blocked_patterns) == 1


# ── AnalyticsResponse ───────────────────────────────────────────

class TestAnalyticsResponse:
    def test_successful_response(self):
        r = AnalyticsResponse(
            answer="Maersk has 45 delays",
            sql_query="SELECT ...",
            result_table=[{"carrier": "Maersk", "count": 45}],
            columns=["carrier", "count"],
            chart_type="bar",
            model_used="groq/llama-3.3-70b",
        )
        assert r.answer
        assert r.sql_query

    def test_error_response(self):
        r = AnalyticsResponse(
            answer="",
            error="Could not generate valid SQL",
        )
        assert r.error
