"""Analytics Agent — full pipeline orchestrator.

This is the main entry point for the analytics capability. It orchestrates
the complete flow from user question to final response:

    User Query
        -> Input Guardrails (block injection/invalid input)
        -> Planner (classify intent: sql_query / clarification / etc.)
        -> SQL Generator (NL -> SQL, with retry on failure)
        -> Verifier (validate + execute SQL readonly)
        -> Answer Synthesizer (LLM generates text answer from results)
        -> Visualizer (suggest + create Plotly chart)
        -> AnalyticsResponse

Retry logic: if SQL generation or execution fails, the agent retries
up to max_retries times (from settings.yaml). Each retry includes the
previous error message in the prompt so the LLM can self-correct.

Error handling: this module NEVER raises exceptions. All errors are
caught and returned as AnalyticsResponse objects with error messages.
"""

from src.common.schemas import (
    AnalyticsRequest,
    AnalyticsResponse,
    Intent,
)
from src.common.config_loader import load_prompts, load_settings
from src.guardrails.input_guards import validate_input
from src.analytics.planner import classify_intent
from src.analytics.sql_generator import generate_sql
from src.analytics.verifier import verify_and_execute
from src.analytics.visualizer import suggest_chart, create_chart
from src.database import get_schema_description
from src.models import get_model_with_fallback


def run_analytics_query(request: AnalyticsRequest) -> AnalyticsResponse:
    """Full analytics pipeline with retry and error handling.

    This is the top-level function called by the AnalyticsService.
    It wraps _run_pipeline() with a catch-all exception handler to
    ensure the API never returns a 500 error.

    Args:
        request: AnalyticsRequest with user_query, conversation_history, session_id.

    Returns:
        AnalyticsResponse with answer, sql, table, chart, and metadata.
        Never raises — all errors are caught and returned as error responses.
    """
    try:
        return _run_pipeline(request)
    except Exception as e:
        # Absolute last resort — pipeline should never crash
        return AnalyticsResponse(
            answer="An unexpected error occurred. Please try again.",
            error=f"Pipeline error: {type(e).__name__}: {str(e)[:200]}",
        )


def _run_pipeline(request: AnalyticsRequest) -> AnalyticsResponse:
    """Internal pipeline implementation with all 7 steps.

    Args:
        request: AnalyticsRequest with user_query, conversation_history, session_id.

    Returns:
        AnalyticsResponse with complete results.
    """

    settings = load_settings()
    max_retries = settings.get("analytics", {}).get("max_retries", 2)

    # ── Step 1: Input Guardrails ─────────────────────────────────
    # Block SQL injection, prompt injection, empty/too-long inputs
    guard_result = validate_input(request.user_query)
    if not guard_result.passed:
        return AnalyticsResponse(
            answer=f"I can't process that input: {guard_result.reason}",
            error=guard_result.reason,
        )

    # Use sanitized (trimmed) input from here on
    query = guard_result.sanitized_input or request.user_query.strip()

    # ── Step 2: Get Schema ───────────────────────────────────────
    # Fetch the human-readable schema for LLM context
    schema = get_schema_description()

    # ── Step 3: Planner (classify intent) ────────────────────────
    # Determine if this question needs SQL, clarification, etc.
    planner_result = classify_intent(
        query=query,
        conversation_history=request.conversation_history,
        schema_description=schema,
    )

    # Handle non-SQL intents — return early without running SQL pipeline
    if planner_result.intent == Intent.UNANSWERABLE:
        prompts = load_prompts()
        unanswerable_template = prompts["analytics"]["unanswerable"]
        # Provide suggested questions the user could ask instead
        suggested = "\n".join(f"  - {q}" for q in planner_result.suggested_questions[:5]) or "  - How many shipments are delayed?\n  - What is the average freight cost by carrier?"
        answer = unanswerable_template.format(
            reasoning=planner_result.reasoning,
            suggested_questions=suggested,
        )
        return AnalyticsResponse(answer=answer)

    if planner_result.intent == Intent.GENERAL:
        return AnalyticsResponse(
            answer="Hello! I'm the GoComet logistics analytics assistant. "
                   "Ask me questions about shipments, carriers, costs, delays, "
                   "invoices, or tracking events.",
        )

    if planner_result.intent == Intent.CLARIFICATION:
        return AnalyticsResponse(
            answer=planner_result.clarification_question
                   or "Could you provide more detail about what you'd like to know?",
        )

    # ── Step 4: SQL Generation + Verification (with retry) ──────
    # Try generating and executing SQL, retrying on failure with error feedback
    last_error = None
    verification = None

    for attempt in range(max_retries + 1):
        # Generate SQL (includes error feedback on retry attempts)
        sql_result = generate_sql(
            query=query,
            schema_description=schema,
            conversation_history=request.conversation_history,
            planner_result=planner_result,
            retry_error=last_error,
        )

        # Verify safety and execute against the database
        verification = verify_and_execute(sql_result)

        if verification.is_safe and verification.is_valid:
            break  # Success — exit retry loop

        # Failed — capture error for the next retry's prompt
        last_error = verification.error or "Unknown verification error"

    # If all retries exhausted without success
    if verification is None or not verification.is_valid:
        error_msg = last_error or "SQL generation failed after all retries."
        return AnalyticsResponse(
            answer=f"I couldn't generate a valid query for that question. {error_msg}",
            sql_query=sql_result.sql if sql_result else None,
            error=error_msg,
        )

    # ── Step 5: Answer Synthesis ─────────────────────────────────
    # Use LLM to generate a natural language answer from the raw results
    model_used = ""
    if verification.row_count == 0:
        answer = "No data matches that query. Try broadening your filters or asking a different question."
    else:
        answer, model_used = _synthesize_answer(
            query=query,
            sql=sql_result.sql,
            results=verification.result_rows,
            row_count=verification.row_count,
        )

    # ── Step 6: Visualization ────────────────────────────────────
    # Suggest and create a chart (rule-based, no LLM needed)
    chart_type = suggest_chart(
        query=query,
        columns=verification.columns,
        row_count=verification.row_count,
    )
    chart_data = None
    if chart_type and verification.result_rows:
        chart_data = create_chart(
            chart_type=chart_type,
            data=verification.result_rows,
            columns=verification.columns,
            query=query,
        )

    # ── Step 7: Build Response ───────────────────────────────────
    return AnalyticsResponse(
        answer=answer,
        sql_query=sql_result.sql,
        result_table=verification.result_rows,
        columns=verification.columns,
        chart_type=chart_type,
        chart_data=chart_data,
        model_used=model_used,
        is_followup=len(request.conversation_history) > 0,
    )


# ── Answer Synthesis ─────────────────────────────────────────────────

def _synthesize_answer(query: str, sql: str, results: list[dict], row_count: int) -> tuple[str, str]:
    """Use LLM to generate a natural language answer from query results.

    Sends the query, SQL, and result data to the answer_synthesis LLM task.
    Results are truncated to 20 rows and 3000 characters to stay within
    token limits.

    Falls back to a basic formatted answer if the LLM is unavailable.

    Args:
        query: The user's original question.
        sql: The SQL that was executed.
        results: Query results (list of row dicts).
        row_count: Total number of rows returned.

    Returns:
        Tuple of (answer_text, model_name_used).
    """
    prompts = load_prompts()
    template = prompts["analytics"]["answer_synthesizer"]

    # Truncate results to avoid sending too much data to the LLM
    display_results = results[:20]
    results_str = str(display_results)
    if len(results_str) > 3000:
        results_str = results_str[:3000] + "... (truncated)"

    prompt = template.format(
        query=query,
        sql=sql,
        results=results_str,
        row_count=row_count,
    )

    try:
        answer, model_name = get_model_with_fallback(
            "answer_synthesis",
            messages=[{"role": "user", "content": prompt}],
            trace_name="answer_synthesis",
        )
        return answer.strip(), model_name

    except Exception as e:
        # LLM failed — generate a basic answer from the raw data
        if row_count == 1 and results:
            first = results[0]
            parts = [f"{k}: {v}" for k, v in first.items()]
            return f"Result: {', '.join(parts)}", "fallback"
        return f"Query returned {row_count} rows. See the table below for details.", "fallback"
