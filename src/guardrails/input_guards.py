"""Input guardrails — validate user input before sending to any LLM.

This is the first line of defense in the analytics pipeline. Every user
query passes through validate_input() before reaching the planner or
any LLM. The checks are ordered from cheapest to most expensive:

Checks (in order):
1. Empty/whitespace check — reject blank inputs
2. Minimum length (> 1 char) — reject single characters
3. Maximum length (configurable, default 1000) — prevent abuse
4. SQL injection patterns (DROP, DELETE, INSERT, etc.)
5. Prompt injection patterns (ignore instructions, you are now, etc.)

IMPORTANT: Normal natural language that happens to contain SQL keywords
(e.g., "Which carrier should I select from the options?") must PASS.
Only structured attack patterns are blocked. The regex patterns use
word boundaries and statement-start anchors to avoid false positives.
"""

import re
from src.common.schemas import GuardrailResult
from src.common.config_loader import load_settings


# ── SQL Injection Patterns ───────────────────────────────────────────
# These detect standalone SQL commands that appear to be injection attempts.
# Each pattern requires the keyword at the START of a statement or after
# a semicolon, so natural language like "I want to update my query" passes.
_SQL_INJECTION_PATTERNS = [
    # SQL syntax patterns (exact SQL commands)
    re.compile(r"(?:^|;\s*)DROP\s+TABLE\b", re.IGNORECASE),
    re.compile(r"(?:^|;\s*)DELETE\s+FROM\b", re.IGNORECASE),
    re.compile(r"(?:^|;\s*)INSERT\s+INTO\b", re.IGNORECASE),
    re.compile(r"(?:^|;\s*)UPDATE\s+\w+\s+SET\b", re.IGNORECASE),
    re.compile(r"(?:^|;\s*)ALTER\s+TABLE\b", re.IGNORECASE),
    re.compile(r"(?:^|;\s*)TRUNCATE\s+TABLE\b", re.IGNORECASE),
    re.compile(r"(?:^|;\s*)GRANT\b", re.IGNORECASE),
    re.compile(r"(?:^|;\s*)REVOKE\b", re.IGNORECASE),
    re.compile(r"UNION\s+SELECT\b", re.IGNORECASE),
    re.compile(r";\s*--", re.IGNORECASE),
    re.compile(r";\s*DROP\b", re.IGNORECASE),
]

# Natural language patterns requesting destructive operations.
# Users may not write SQL but say "drop the table" or "delete all data".
# This is a READ-ONLY analytics system — any modification request should be refused.
_DESTRUCTIVE_INTENT_PATTERNS = [
    re.compile(r"\b(drop|delete|remove|destroy|erase|wipe|clear|truncate|purge)\b.*\b(table|data|database|records|rows|everything|all)\b", re.IGNORECASE),
    re.compile(r"\b(table|data|database|records|rows)\b.*\b(drop|delete|remove|destroy|erase|wipe|clear|truncate|purge)\b", re.IGNORECASE),
    re.compile(r"\b(modify|change|update|alter|edit)\b.*\b(schema|table|column|structure)\b", re.IGNORECASE),
]


# ── Prompt Injection Detection ───────────────────────────────────────

def _load_prompt_injection_patterns() -> list[re.Pattern]:
    """Load prompt injection patterns from settings.yaml and compile them.

    Patterns are defined in config/settings.yaml under:
        guardrails:
          injection_patterns:
            - "ignore\\s+(previous|above)\\s+instructions"
            - "you\\s+are\\s+now"
            ...

    Falls back to a hardcoded set of common prompt injection patterns
    if the config file cannot be loaded (e.g., during testing).

    Returns:
        List of compiled regex patterns for prompt injection detection.
    """
    try:
        settings = load_settings()
        raw_patterns = settings.get("guardrails", {}).get("injection_patterns", [])
        return [re.compile(p, re.IGNORECASE) for p in raw_patterns]
    except Exception:
        # Fallback patterns if config loading fails
        return [
            re.compile(r"ignore\s+(previous|above)\s+instructions", re.IGNORECASE),
            re.compile(r"disregard\s+(all|your)", re.IGNORECASE),
            re.compile(r"you\s+are\s+now", re.IGNORECASE),
            re.compile(r"new\s+instructions", re.IGNORECASE),
            re.compile(r"system\s+prompt", re.IGNORECASE),
            re.compile(r"forget\s+(your|all)", re.IGNORECASE),
        ]


# ── Main Validation Function ─────────────────────────────────────────

def validate_input(user_input: str) -> GuardrailResult:
    """Validate user input before sending to any LLM.

    Runs all checks in order and returns immediately on the first failure.
    On success, the sanitized (trimmed) input is returned in the result.

    Args:
        user_input: Raw user input string from the chat interface.

    Returns:
        GuardrailResult with:
        - passed=True and sanitized_input if all checks pass
        - passed=False with reason and blocked_patterns if any check fails
    """
    # Load max length from settings (with fallback for robustness)
    try:
        settings = load_settings()
        max_length = settings.get("guardrails", {}).get("max_input_length", 1000)
    except Exception:
        max_length = 1000

    # ── Check 1: Strip whitespace ────────────────────────────────────
    stripped = user_input.strip()

    # ── Check 2: Reject empty input ──────────────────────────────────
    if not stripped:
        return GuardrailResult(
            passed=False,
            reason="Input is empty.",
            sanitized_input=None,
        )

    # ── Check 3: Reject too-short input ──────────────────────────────
    if len(stripped) <= 1:
        return GuardrailResult(
            passed=False,
            reason="Input is too short. Please enter a meaningful question.",
            sanitized_input=None,
        )

    # ── Check 4: Reject too-long input ───────────────────────────────
    if len(stripped) > max_length:
        return GuardrailResult(
            passed=False,
            reason=f"Input is too long ({len(stripped)} chars). Maximum is {max_length} characters.",
            sanitized_input=None,
        )

    # ── Check 5: SQL injection detection ─────────────────────────────
    blocked = []
    for pattern in _SQL_INJECTION_PATTERNS:
        if pattern.search(stripped):
            blocked.append(pattern.pattern)
    if blocked:
        return GuardrailResult(
            passed=False,
            reason="Input contains potentially dangerous SQL patterns.",
            sanitized_input=None,
            blocked_patterns=blocked,
        )

    # ── Check 6: Destructive intent detection ─────────────────────────
    # Catches natural language requests to modify/delete data.
    # "drop the table", "delete all data", "remove everything", etc.
    # This is a READ-ONLY system — all modification requests are refused.
    for pattern in _DESTRUCTIVE_INTENT_PATTERNS:
        if pattern.search(stripped):
            return GuardrailResult(
                passed=False,
                reason="This is a read-only analytics system. Data modification (drop, delete, update) is not supported.",
                sanitized_input=None,
                blocked_patterns=[pattern.pattern],
            )

    # ── Check 7: Prompt injection detection ──────────────────────────
    prompt_patterns = _load_prompt_injection_patterns()
    for pattern in prompt_patterns:
        if pattern.search(stripped):
            blocked.append(pattern.pattern)
    if blocked:
        return GuardrailResult(
            passed=False,
            reason="Input contains prompt injection patterns.",
            sanitized_input=None,
            blocked_patterns=blocked,
        )

    # ── All checks passed ────────────────────────────────────────────
    return GuardrailResult(
        passed=True,
        reason=None,
        sanitized_input=stripped,
    )
