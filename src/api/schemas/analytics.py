"""API request/response schemas for analytics endpoints.

These Pydantic models define the HTTP API contract. They wrap the
domain schemas (from src/common/schemas.py) with API-specific concerns
like default values for optional fields.

This separation means:
- Domain schemas can change without breaking the API contract
- API schemas can add HTTP-specific fields (pagination, etc.)
- Validation rules can differ between domain and API layers

All schemas use Pydantic's BaseModel for automatic JSON serialization
and request validation in FastAPI.
"""

from pydantic import BaseModel
from typing import Optional


# ── Analytics Endpoint Schemas ───────────────────────────────────────

class AnalyticsQueryRequest(BaseModel):
    """POST /api/analytics/query — request body.

    Attributes:
        user_query: The natural language question to answer.
        conversation_history: Previous conversation turns for follow-up context.
            Each dict should have "role" and "content" keys.
        session_id: Unique session identifier for tracing and grouping.
    """
    user_query: str
    conversation_history: list[dict] = []
    session_id: str = ""


class AnalyticsQueryResponse(BaseModel):
    """POST /api/analytics/query — response body.

    Contains all outputs from the analytics pipeline, including the
    natural language answer, the SQL used, raw results, and chart data.

    Attributes:
        answer: Natural language answer to the user's question.
        sql_query: The SQL that was generated and executed (for transparency).
        result_table: Raw query results as list of row dicts.
        columns: Column names for rendering the result table.
        chart_type: Suggested chart type (bar, line, pie, or None).
        chart_data: Serialized Plotly figure dict for client-side rendering.
        model_used: Which LLM model produced the answer.
        error: Error message if the pipeline encountered issues.
        is_followup: Whether this was a follow-up question in conversation.
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


# ── Health Endpoint Schemas ──────────────────────────────────────────

class HealthResponse(BaseModel):
    """GET /api/health — response body.

    Reports database connectivity status and basic statistics.

    Attributes:
        status: "healthy" if database is accessible, "unhealthy" otherwise.
        tables: List of all table names in the database.
        row_counts: Dict mapping table name to row count.
        error: Error message if health check failed.
    """
    status: str
    tables: list[str] = []
    row_counts: dict[str, int] = {}
    error: Optional[str] = None


class SchemaResponse(BaseModel):
    """GET /api/schema — response body.

    Returns the database schema in the same format used by LLM prompts.

    Attributes:
        schema_description: Multi-line human-readable schema text.
        tables: List of table names extracted from the schema.
    """
    schema_description: str
    tables: list[str] = []
