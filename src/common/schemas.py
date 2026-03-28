"""Pydantic domain schemas shared across the entire application.

These schemas define the data contracts between pipeline stages.
They are NOT the API request/response schemas (those live in
src/api/schemas/). This separation allows the domain logic to
evolve independently from the HTTP interface.

Schema flow through the analytics pipeline:
    AnalyticsRequest → PlannerResult → SQLGenerationResult
        → VerificationResult → AnalyticsResponse

Schema flow through the vision pipeline:
    ExtractionRequest → ExtractionResult → DocumentRecord
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────

class Intent(str, Enum):
    """Possible intents classified by the Planner stage.

    The planner examines the user query and assigns one of these intents
    to determine which downstream pipeline path to follow.
    """
    SQL_QUERY = "sql_query"          # Needs SQL generation and execution
    CLARIFICATION = "clarification"  # Too vague; ask the user to clarify
    UNANSWERABLE = "unanswerable"    # Data needed is not in our database
    GENERAL = "general"              # Greeting, thanks, or general chat


class ReviewStatus(str, Enum):
    """Review lifecycle states for extracted documents.

    After vision extraction, a document starts as PENDING and moves
    through human review to one of the terminal states.
    """
    PENDING = "pending"      # Awaiting human review
    APPROVED = "approved"    # Human confirmed extraction is correct
    REJECTED = "rejected"    # Human rejected the extraction entirely
    CORRECTED = "corrected"  # Human fixed some extracted fields


# ── Analytics Pipeline Schemas ───────────────────────────────────────

class AnalyticsRequest(BaseModel):
    """Input to the analytics pipeline (from API or direct call).

    Attributes:
        user_query: The natural language question from the user.
        conversation_history: Previous turns for follow-up context.
            Each dict has "role" (user/assistant) and "content" keys.
        session_id: Unique session identifier for tracing and context.
    """
    user_query: str
    conversation_history: list[dict] = []
    session_id: str = ""


class PlannerResult(BaseModel):
    """Output of the Planner stage — intent classification.

    Attributes:
        intent: The classified intent (sql_query, clarification, etc.).
        reasoning: LLM's explanation of why it chose this intent.
        requires_sql: Whether downstream SQL generation is needed.
        clarification_question: Question to ask user (if intent=clarification).
        suggested_questions: Alternative questions the user could ask.
    """
    intent: Intent
    reasoning: str
    requires_sql: bool = False
    clarification_question: Optional[str] = None
    suggested_questions: list[str] = []


class SQLGenerationResult(BaseModel):
    """Output of the SQL Generator stage.

    Attributes:
        sql: The generated SQL query string.
        explanation: LLM's explanation of what the SQL does.
        tables_used: List of table names referenced in the query.
    """
    sql: str
    explanation: str
    tables_used: list[str] = []


class VerificationResult(BaseModel):
    """Output of the Verifier stage — SQL validation + execution result.

    Attributes:
        is_safe: Whether the SQL passed output guardrail checks (SELECT only).
        is_valid: Whether the SQL executed without database errors.
        error: Error message if either check failed.
        result_rows: Query results as list of dicts (one dict per row).
        row_count: Number of rows returned.
        columns: Column names from the result set.
    """
    is_safe: bool
    is_valid: bool
    error: Optional[str] = None
    result_rows: list[dict] = []
    row_count: int = 0
    columns: list[str] = []


class AnalyticsResponse(BaseModel):
    """Final output of the analytics pipeline, returned to the user.

    Attributes:
        answer: Natural language answer synthesized from query results.
        sql_query: The SQL that was executed (for transparency).
        result_table: Raw query results for display in a data table.
        columns: Column names for the result table.
        chart_type: Suggested visualization type (bar, line, pie, or None).
        chart_data: Serialized Plotly figure dict for rendering.
        model_used: Which LLM model produced the answer.
        error: Error message if the pipeline encountered issues.
        is_followup: Whether this was a follow-up question in a conversation.
    """
    answer: str
    sql_query: Optional[str] = None
    result_table: list[dict] = []
    columns: list[str] = []
    chart_type: Optional[str] = None
    chart_data: Optional[dict] = None
    model_used: str = ""
    error: Optional[str] = None
    is_followup: bool = False


# ── Vision Pipeline Schemas ──────────────────────────────────────────

class ExtractionRequest(BaseModel):
    """Input to the vision extraction pipeline.

    Attributes:
        file_bytes: Raw bytes of the uploaded PDF or image file.
        file_name: Original filename (used for logging and storage).
        document_type_hint: Optional hint from the user about the document
            type (e.g., "invoice", "bill_of_lading") to improve extraction.
    """
    file_bytes: bytes
    file_name: str
    document_type_hint: Optional[str] = None


class FieldExtraction(BaseModel):
    """A single extracted field with its confidence score.

    Attributes:
        value: The extracted value (can be string, number, list, or None).
        confidence: Confidence score from 0.0 (no confidence) to 1.0 (certain).
        needs_review: Whether this field should be flagged for human review
            (typically True when confidence < threshold from settings.yaml).
    """
    value: str | float | int | list | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False


class ExtractionResult(BaseModel):
    """Output of the vision extraction pipeline (before storage).

    Attributes:
        document_type: Detected document type (invoice, bill_of_lading, etc.).
        fields: Dict mapping field names to FieldExtraction objects.
        overall_confidence: Average confidence across all extracted fields.
        model_used: Which vision LLM model performed the extraction.
        notes: Any additional notes or warnings from the extraction.
        raw_text: The raw OCR text from the document (for debugging).
    """
    document_type: str
    fields: dict[str, FieldExtraction]
    overall_confidence: float
    model_used: str = ""
    notes: str = ""
    raw_text: Optional[str] = None


class DocumentRecord(BaseModel):
    """A stored document record in the extracted_documents table.

    This represents a fully persisted extraction, including the review
    status and optional link to a shipment.

    Attributes:
        document_id: Unique UUID assigned at storage time.
        document_type: Type of document (invoice, bill_of_lading, etc.).
        file_name: Original uploaded file name.
        extraction_model: LLM model that performed the extraction.
        overall_confidence: Aggregate confidence score.
        review_status: Current review state (pending, approved, etc.).
        extracted_fields: Flat dict of field_name → value.
        confidence_scores: Dict of field_name → confidence float.
        linked_shipment_id: Optional FK to shipments.shipment_id.
    """
    document_id: str
    document_type: str
    file_name: str
    extraction_model: str
    overall_confidence: float
    review_status: ReviewStatus = ReviewStatus.PENDING
    extracted_fields: dict
    confidence_scores: dict[str, float]
    linked_shipment_id: Optional[str] = None


# ── Guardrail Schemas ────────────────────────────────────────────────

class GuardrailResult(BaseModel):
    """Result from an input or output guardrail check.

    Attributes:
        passed: True if the content passed validation.
        reason: Human-readable explanation when passed=False.
        sanitized_input: Cleaned/safe version of the input (e.g., with
            LIMIT appended to SQL). None when validation fails.
        blocked_patterns: List of regex patterns that triggered the block,
            useful for debugging which guardrail rule fired.
    """
    passed: bool
    reason: Optional[str] = None
    sanitized_input: Optional[str] = None
    blocked_patterns: list[str] = []
