"""SQL Pitfall Checker — catch common SQL mistakes BEFORE execution.

This is separate from output guardrails (which check safety/DML).
This module checks for CORRECTNESS issues that LLMs commonly get wrong.
The checks catch subtle bugs that would produce wrong results silently:

1. NOT IN with nullable columns (fails silently — returns no rows)
2. UNION instead of UNION ALL (unnecessary dedup overhead)
3. Missing GROUP BY with aggregate functions (undefined behavior)
4. JOIN without ON clause (produces a cartesian product)
5. Date comparison without date() function (SQLite-specific, actually OK)
6. SELECT * in JOINs (ambiguous column names)
7. HAVING without GROUP BY (semantically meaningless)

Returns warnings (not hard blocks) — the SQL might still be intentional.
Warnings are appended to the SQL explanation so the user can review them.
"""

import re
from dataclasses import dataclass


@dataclass
class PitfallWarning:
    """A potential SQL issue found by the checker.

    Attributes:
        severity: "error" (likely wrong) or "warning" (might be intentional).
        message: Human-readable description of the issue.
        suggestion: How to fix the issue.
    """
    severity: str
    message: str
    suggestion: str


def check_sql_pitfalls(sql: str) -> list[PitfallWarning]:
    """Check SQL for common pitfalls and return warnings.

    Runs a series of regex-based checks against the SQL string.
    Each check is independent — multiple warnings can be returned.

    Args:
        sql: The SQL query string to check.

    Returns:
        List of PitfallWarning objects. Empty list means no issues found.
    """
    warnings = []
    sql_upper = sql.upper()

    # ── Check 1: NOT IN with potentially nullable column ─────────
    # NOT IN returns no rows when the subquery contains any NULL values.
    # This is a well-known SQL gotcha that catches even experienced devs.
    if re.search(r"NOT\s+IN\s*\(", sql_upper):
        warnings.append(PitfallWarning(
            severity="warning",
            message="NOT IN can return unexpected results if the subquery contains NULL values.",
            suggestion="Consider using NOT EXISTS instead: WHERE NOT EXISTS (SELECT 1 FROM ... WHERE ...)",
        ))

    # ── Check 2: UNION without ALL (unnecessary dedup) ───────────
    # UNION removes duplicates (expensive sort+dedup). UNION ALL is
    # almost always what you want unless you specifically need dedup.
    if re.search(r"\bUNION\b(?!\s+ALL)", sql_upper):
        warnings.append(PitfallWarning(
            severity="warning",
            message="UNION removes duplicates (slow). If duplicates aren't a concern, use UNION ALL.",
            suggestion="Change UNION to UNION ALL unless you specifically need deduplication.",
        ))

    # ── Check 3: Aggregate without GROUP BY ──────────────────────
    # Having COUNT/SUM/AVG with multiple columns but no GROUP BY
    # produces undefined behavior in most SQL engines.
    aggregates = re.findall(r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(", sql_upper)
    has_group_by = "GROUP BY" in sql_upper
    if aggregates and not has_group_by:
        # Only warn if there are multiple columns in SELECT (single agg is fine)
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql_upper, re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            if "," in select_clause:
                warnings.append(PitfallWarning(
                    severity="error",
                    message="Multiple columns in SELECT with aggregate functions but no GROUP BY clause.",
                    suggestion="Add GROUP BY for all non-aggregated columns in the SELECT.",
                ))

    # ── Check 4: JOIN without ON clause ──────────────────────────
    # A JOIN without ON produces a cartesian product (every row x every row),
    # which is almost never intentional and can return millions of rows.
    joins = re.findall(r"\bJOIN\s+(\w+)\s*(?:\w+\s*)?(?=JOIN|WHERE|GROUP|ORDER|LIMIT|$)", sql_upper)
    for table in joins:
        pattern = rf"JOIN\s+{table}\s+\w*\s*(?!ON)"
        if re.search(pattern, sql_upper) and "ON" not in sql_upper:
            warnings.append(PitfallWarning(
                severity="error",
                message=f"JOIN with {table} appears to be missing an ON clause.",
                suggestion=f"Add ON clause: JOIN {table} ON <condition>",
            ))
            break  # One warning is enough for this check

    # ── Check 5: SELECT * with JOINs (ambiguous columns) ────────
    # When joining tables that share column names (e.g., both have "id"),
    # SELECT * produces ambiguous results that vary by engine.
    if "SELECT *" in sql_upper and "JOIN" in sql_upper:
        warnings.append(PitfallWarning(
            severity="warning",
            message="SELECT * with JOINs can cause ambiguous column names.",
            suggestion="Specify explicit columns: SELECT t1.col1, t2.col2 instead of SELECT *.",
        ))

    # ── Check 6: Date string comparison ──────────────────────────
    # In SQLite, ISO date strings (YYYY-MM-DD) compare correctly as strings.
    # This check exists as a placeholder but does NOT warn for SQLite.
    date_columns = ["booking_date", "invoice_date", "due_date", "payment_date",
                    "estimated_departure", "actual_departure", "estimated_arrival",
                    "actual_arrival", "event_timestamp", "upload_timestamp"]
    for col in date_columns:
        pattern = rf"{col}\s*[><=!]+\s*'?\d{{4}}"
        if re.search(pattern, sql, re.IGNORECASE):
            # ISO date strings compare correctly in SQLite — no warning needed
            break

    # ── Check 7: HAVING without GROUP BY ─────────────────────────
    # HAVING filters groups, but without GROUP BY there are no groups
    # to filter. This is semantically meaningless.
    if "HAVING" in sql_upper and not has_group_by:
        warnings.append(PitfallWarning(
            severity="error",
            message="HAVING clause without GROUP BY doesn't make sense.",
            suggestion="Add GROUP BY before HAVING, or move the condition to WHERE.",
        ))

    # Note: Check 8 (ORDER BY on column not in SELECT) was removed
    # because it produced too many false positives with valid SQL.

    return warnings


def format_pitfall_warnings(warnings: list[PitfallWarning]) -> str:
    """Format warnings into a human-readable string.

    Used both for:
    - Appending to the SQL explanation shown to users
    - Including in retry prompts so the LLM can fix the issues

    Args:
        warnings: List of PitfallWarning objects from check_sql_pitfalls().

    Returns:
        Formatted string with numbered warnings, or empty string if none.
    """
    if not warnings:
        return ""

    lines = ["SQL Quality Warnings:"]
    for i, w in enumerate(warnings, 1):
        severity_icon = "🔴" if w.severity == "error" else "⚠️"
        lines.append(f"  {severity_icon} {i}. {w.message}")
        lines.append(f"     Fix: {w.suggestion}")

    return "\n".join(lines)
