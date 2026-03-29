"""Shared utilities used across multiple modules.

Centralizes duplicate code that was previously defined independently
in planner.py, sql_generator.py, output_guards.py, analytics_repo.py,
and extractor.py.
"""

import re


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM responses.
    Handles ```json ... ```, ``` ... ```, and bare fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|sql)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def format_history(history: list[dict], max_turns: int = 10, max_content_length: int = 300) -> str:
    """Format conversation history for LLM prompts."""
    if not history:
        return "(no prior conversation)"
    recent = history[-max_turns:]
    lines = []
    for msg in recent:
        role = msg.get("role", "user").upper()
        content = str(msg.get("content", ""))
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# Compiled regex for detecting DML/DDL statements -- shared across guardrails and repo
DML_PATTERN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|GRANT|REVOKE)\b",
    re.IGNORECASE,
)
