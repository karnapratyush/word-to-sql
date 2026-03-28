"""Planner — classify user intent before generating SQL.

The planner is the FIRST step in the analytics pipeline. It examines
the user's question and the database schema to decide the appropriate
response path:

- sql_query: Question needs SQL generation and execution
- clarification: Question is too vague; ask the user to clarify
- unanswerable: Question asks about data we don't have
- general: Greeting, thanks, or general chat (no data needed)

Uses a fast, cheap LLM (classification task from models.yaml) to
minimize cost and latency. Falls back to rule-based classification
if the LLM is unavailable or returns unparseable output.
"""

import json

from src.common.schemas import PlannerResult, Intent
from src.common.config_loader import load_prompts
from src.models import get_model_with_fallback


def classify_intent(
    query: str,
    conversation_history: list[dict],
    schema_description: str,
) -> PlannerResult:
    """Classify user intent using LLM with schema context.

    The LLM receives the database schema so it can determine whether
    the question is answerable with available data. Conversation history
    is included for follow-up context (e.g., "and for air shipments?").

    Args:
        query: The user's natural language question.
        conversation_history: Previous conversation turns for follow-up context.
            Each dict has "role" (user/assistant) and "content" keys.
        schema_description: Human-readable database schema description.

    Returns:
        PlannerResult with intent classification, reasoning, and optional
        clarification question or suggested alternatives.
    """
    prompts = load_prompts()
    prompt_template = prompts["analytics"]["planner"]

    # Format conversation history into a string for the prompt
    history_str = _format_history(conversation_history)

    # Fill in the prompt template with schema, history, and query
    prompt = prompt_template.format(
        schema=schema_description,
        history=history_str,
        query=query,
    )

    try:
        # Use the "classification" task — fast, cheap model from the fallback chain
        response, model_name = get_model_with_fallback(
            "classification",
            messages=[{"role": "user", "content": prompt}],
            trace_name="planner",
        )

        return _parse_planner_response(response)

    except Exception:
        # LLM failed entirely — fall back to rule-based classification
        # so the pipeline continues even without LLM access
        return _rule_based_classify(query)


# ── Response Parsing ─────────────────────────────────────────────────

def _parse_planner_response(raw: str) -> PlannerResult:
    """Parse LLM JSON response into PlannerResult.

    Handles common LLM output quirks:
    - Markdown-wrapped JSON (```json ... ```)
    - Extra text before/after the JSON object
    - Missing optional fields (clarification_question, suggested_questions)
    - Unknown intent strings (defaults to sql_query)

    Args:
        raw: Raw string response from the LLM.

    Returns:
        PlannerResult with parsed intent and metadata.
    """
    cleaned = raw.strip()

    # Strip markdown code blocks if the LLM wrapped the JSON
    if "```" in cleaned:
        lines = cleaned.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block or not cleaned.startswith("```"):
                json_lines.append(line)
        cleaned = "\n".join(json_lines).strip()

    # Try to find the JSON object within the response text
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        cleaned = cleaned[start:end]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Can't parse LLM output — default to SQL query intent so the
        # pipeline continues (better to try SQL than to give up)
        return PlannerResult(
            intent=Intent.SQL_QUERY,
            reasoning="Could not parse LLM response, defaulting to SQL query.",
            requires_sql=True,
        )

    # Map the intent string from the LLM to our Intent enum
    intent_str = data.get("intent", "sql_query").lower().strip()
    intent_map = {
        "sql_query": Intent.SQL_QUERY,
        "clarification": Intent.CLARIFICATION,
        "unanswerable": Intent.UNANSWERABLE,
        "general": Intent.GENERAL,
    }
    # Default to SQL_QUERY for unknown intent strings
    intent = intent_map.get(intent_str, Intent.SQL_QUERY)

    return PlannerResult(
        intent=intent,
        reasoning=data.get("reasoning", ""),
        requires_sql=data.get("requires_sql", intent == Intent.SQL_QUERY),
        clarification_question=data.get("clarification_question"),
        suggested_questions=data.get("suggested_questions", []),
    )


# ── Rule-Based Fallback ─────────────────────────────────────────────

def _rule_based_classify(query: str) -> PlannerResult:
    """Fallback rule-based classification when the LLM is unavailable.

    Simple heuristics based on query content:
    - Short greetings → GENERAL
    - Very short queries (< 5 chars) → CLARIFICATION
    - Everything else → SQL_QUERY (let the SQL generator try)

    Args:
        query: The user's natural language question.

    Returns:
        PlannerResult with rule-based intent classification.
    """
    lower = query.lower().strip()

    # Detect greetings and thanks
    if lower in ("hi", "hello", "hey", "thanks", "thank you", "bye"):
        return PlannerResult(
            intent=Intent.GENERAL,
            reasoning="Detected greeting/thanks pattern.",
            requires_sql=False,
        )

    # Very short queries are likely too vague
    if len(lower) < 5:
        return PlannerResult(
            intent=Intent.CLARIFICATION,
            reasoning="Query too short to determine intent.",
            requires_sql=False,
            clarification_question="Could you provide more detail about what you'd like to know?",
        )

    # Default: assume it is a data query and let the SQL generator handle it
    return PlannerResult(
        intent=Intent.SQL_QUERY,
        reasoning="Rule-based fallback: assuming data query.",
        requires_sql=True,
    )


# ── History Formatting ───────────────────────────────────────────────

def _format_history(history: list[dict]) -> str:
    """Format conversation history for inclusion in the LLM prompt.

    Limits to the last 10 turns and truncates long messages to 200 chars
    to keep the prompt within token limits.

    Args:
        history: List of message dicts with "role" and "content" keys.

    Returns:
        Formatted string with "ROLE: content" lines, or a placeholder
        string if there is no history.
    """
    if not history:
        return "(no previous conversation)"

    lines = []
    for msg in history[-10:]:  # Last 10 turns max to control prompt size
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")[:500]  # Truncate long messages
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
