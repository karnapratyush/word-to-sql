"""Analytics API endpoints.

This router handles the main analytics query endpoint. The full NL-to-SQL
pipeline runs synchronously within a single POST request.

Endpoints:
    POST /api/analytics/query — Full NL-to-SQL pipeline
        Input: user_query, conversation_history, session_id
        Output: answer, sql_query, result_table, chart_data, model_used

The endpoint never raises HTTP errors for pipeline failures — all errors
are returned in the response body with the "error" field populated.
"""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_analytics_service
from src.api.schemas.analytics import AnalyticsQueryRequest, AnalyticsQueryResponse
from src.common.schemas import AnalyticsRequest
from src.services.analytics_service import AnalyticsService

# Create a router instance for analytics endpoints
router = APIRouter()


@router.post("/query", response_model=AnalyticsQueryResponse)
def analytics_query(
    request: AnalyticsQueryRequest,
    service: AnalyticsService = Depends(get_analytics_service),
):
    """Process a natural language analytics query through the full pipeline.

    The pipeline runs these steps in order:
    1. Input guardrails (injection detection, length validation)
    2. Planner (classify intent: sql_query / clarification / etc.)
    3. SQL Generator (NL -> SQL with retry on failure)
    4. Verifier (validate SQL safety + execute against database)
    5. Answer Synthesizer (LLM generates natural language from results)
    6. Visualizer (rule-based chart suggestion + Plotly figure creation)

    Returns a complete response with answer, SQL, result table, and chart.
    Never throws HTTP exceptions — errors are returned in the response body.

    Args:
        request: AnalyticsQueryRequest with user_query, history, session_id.
        service: Injected AnalyticsService instance.

    Returns:
        AnalyticsQueryResponse with all pipeline outputs.
    """
    # Convert API request schema to domain schema
    # This separation allows the API contract to evolve independently
    domain_request = AnalyticsRequest(
        user_query=request.user_query,
        conversation_history=request.conversation_history,
        session_id=request.session_id,
    )

    # Run the full analytics pipeline
    result = service.query(domain_request)

    # Convert domain response back to API response schema
    return AnalyticsQueryResponse(
        answer=result.answer,
        sql_query=result.sql_query,
        result_table=result.result_table,
        columns=result.columns,
        chart_type=result.chart_type,
        chart_data=result.chart_data,
        model_used=result.model_used,
        error=result.error,
        is_followup=result.is_followup,
    )
