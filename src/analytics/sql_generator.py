"""SQL Generator — convert natural language to SQL.

Uses a strong LLM (sql_generation task) with FOCUSED schema context
retrieved from the vector store (not the full schema). This focused
approach reduces token usage and improves accuracy because the LLM
only sees the 3-5 most relevant tables.

Pipeline within this module:
1. Query the knowledge store for relevant tables + few-shot examples
2. Build a focused prompt with only relevant schema + similar examples
3. Send to LLM for SQL generation
4. Run SQL pitfall checker on the result (catch common mistakes)
5. Return SQLGenerationResult

Supports retry with error feedback: if the first SQL attempt fails
execution, the error message is included in a retry prompt so the
LLM can learn from its mistake and correct the SQL.
"""

import json
from typing import Optional

from src.common.schemas import SQLGenerationResult, PlannerResult
from src.common.config_loader import load_prompts
from src.models import get_model_with_fallback
from src.knowledge.vector_store import get_knowledge_store
from src.knowledge.sql_pitfall_checker import check_sql_pitfalls, format_pitfall_warnings


def generate_sql(
    query: str,
    schema_description: str,
    conversation_history: list[dict],
    planner_result: PlannerResult,
    retry_error: Optional[str] = None,
) -> SQLGenerationResult:
    """Generate SQL from a natural language question.

    Uses the knowledge store to:
    - Retrieve only relevant table schemas (not all tables)
    - Find similar few-shot examples for accuracy
    - Check generated SQL for common pitfalls

    On retry (retry_error is set), uses a different prompt template
    that includes the previous error to guide the LLM toward a fix.

    Args:
        query: The user's natural language question.
        schema_description: Full database schema (used as fallback
            if the vector store is unavailable).
        conversation_history: Previous turns for follow-up support.
        planner_result: Intent classification from the planner.
        retry_error: If this is a retry, the error message from the
            previous failed attempt (fed back into the prompt).

    Returns:
        SQLGenerationResult with sql, explanation, and tables_used.
    """
    prompts = load_prompts()

    # ── Retrieve focused context from knowledge store ────────────
    try:
        store = get_knowledge_store()
        context = store.retrieve_context(query, n_tables=5, n_examples=3)
        focused_schema = context.schema_text
        few_shot_text = context.few_shot_text
    except Exception:
        # Fallback to full schema if vector store fails
        focused_schema = schema_description
        few_shot_text = ""

    # ── Build the prompt ─────────────────────────────────────────
    if retry_error:
        # Retry prompt includes the previous error for self-correction
        template = prompts["analytics"]["sql_retry"]
        prompt = template.format(
            error=retry_error,
            query=query,
            schema=focused_schema,
        )
    else:
        # Normal first-attempt prompt with schema and few-shot examples
        template = prompts["analytics"]["sql_generator"]
        history_str = _format_history(conversation_history)

        # Append few-shot examples after the schema for additional context
        schema_with_examples = focused_schema
        if few_shot_text:
            schema_with_examples += "\n\nSIMILAR QUERIES (use as reference):\n" + few_shot_text

        prompt = template.format(
            schema=schema_with_examples,
            history=history_str,
            query=query,
        )

    # ── Invoke the LLM ───────────────────────────────────────────
    try:
        response, model_name = get_model_with_fallback(
            "sql_generation",
            messages=[{"role": "user", "content": prompt}],
            trace_name="sql_generator" if not retry_error else "sql_retry",
        )

        result = _parse_sql_response(response)

        # Run SQL pitfall checker to catch common mistakes
        pitfall_warnings = check_sql_pitfalls(result.sql)
        if pitfall_warnings:
            # Append warnings to the explanation so users can see them
            warning_text = format_pitfall_warnings(pitfall_warnings)
            result.explanation += f"\n\n{warning_text}"

        return result

    except Exception as e:
        # LLM completely failed — return a safe fallback query
        return SQLGenerationResult(
            sql="SELECT 'LLM unavailable' as error",
            explanation=f"SQL generation failed: {str(e)[:100]}",
            tables_used=[],
        )


# ── Response Parsing ─────────────────────────────────────────────────

def _parse_sql_response(raw: str) -> SQLGenerationResult:
    """Parse LLM response into SQLGenerationResult.

    Handles three response formats (in order of preference):
    1. Clean JSON: {"sql": "...", "explanation": "...", "tables_used": [...]}
    2. Markdown-wrapped JSON: ```json { ... } ```
    3. Raw SQL: Just the SQL query with no JSON wrapper

    Args:
        raw: Raw string response from the LLM.

    Returns:
        SQLGenerationResult with sql, explanation, and tables_used.
    """
    cleaned = raw.strip()

    # Strip markdown code blocks if the LLM wrapped the response
    if "```" in cleaned:
        lines = cleaned.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        if json_lines:
            cleaned = "\n".join(json_lines).strip()

    # Try to parse as JSON (preferred format)
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(cleaned[start:end])
            return SQLGenerationResult(
                sql=data.get("sql", "").strip(),
                explanation=data.get("explanation", ""),
                tables_used=data.get("tables_used", []),
            )
        except json.JSONDecodeError:
            pass  # Fall through to raw SQL parsing

    # Fallback: treat the whole response as raw SQL
    sql = cleaned
    # Strip common non-SQL prefixes that LLMs sometimes add
    for prefix in ("sql:", "SQL:", "query:", "Query:"):
        if sql.lower().startswith(prefix.lower()):
            sql = sql[len(prefix):].strip()

    return SQLGenerationResult(
        sql=sql,
        explanation="(extracted raw SQL from response)",
        tables_used=[],
    )


# ── History Formatting ───────────────────────────────────────────────

def _format_history(history: list[dict]) -> str:
    """Format conversation history for inclusion in the LLM prompt.

    Limits to the last 10 turns and truncates long messages to keep
    the prompt within token limits.

    Args:
        history: List of message dicts with "role" and "content" keys.

    Returns:
        Formatted string with "ROLE: content" lines.
    """
    if not history:
        return "(no previous conversation)"

    lines = []
    for msg in history[-10:]:  # Last 10 turns max
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")[:200]  # Truncate long messages
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
