"""Verifier — validate SQL safety, estimate cost, and execute against the database.

This is the GATEKEEPER between LLM-generated SQL and the actual database.
No SQL reaches the database without passing through this module first.

Three-step verification process:
1. Validate SQL via output guardrails (SELECT only, no DML, auto-LIMIT)
2. Cost estimation via EXPLAIN QUERY PLAN (block full scans on large tables)
3. Execute via AnalyticsRepository (readonly mode with additional checks)

If any step fails, returns a VerificationResult with is_safe=False or
is_valid=False and an error message. The error message is used by the
analytics agent to feed back into a retry prompt.

Cost estimation is critical for production with large tables:
- EXPLAIN QUERY PLAN reveals whether the query uses indexes or does full scans
- Full table scans on million+ row tables are blocked with a helpful message
- The threshold is configurable (default: 100K rows)
- For our POC (10K rows), this rarely triggers but demonstrates production thinking
"""

import re
import logging

from src.common.schemas import SQLGenerationResult, VerificationResult
from src.common.config_loader import load_settings
from src.guardrails.output_guards import validate_sql
from src.repositories.analytics_repo import AnalyticsRepository

logger = logging.getLogger(__name__)


def _get_full_scan_threshold() -> int:
    """Load full_scan_row_threshold from config/settings.yaml.

    Configurable so you can set it low (1K) for POC demos or
    high (1M+) for production with large tables.

    Returns:
        Row count threshold. Full table scans on tables exceeding
        this count are blocked with a helpful error message.
    """
    settings = load_settings()
    return settings.get("analytics", {}).get("full_scan_row_threshold", 1000)


def verify_and_execute(sql_result: SQLGenerationResult, db_path: str | None = None) -> VerificationResult:
    """Validate SQL safety, estimate cost, execute readonly, return results.

    Three-layer defense:
    - Layer 1 (guardrails): Regex-based checks for DML/DDL, multi-statement
    - Layer 2 (cost estimation): EXPLAIN QUERY PLAN to catch full table scans
    - Layer 3 (repository): AnalyticsRepository validates SELECT-only + executes

    Args:
        sql_result: The SQL generation output from the generator module.
        db_path: Optional database path override (for testing with temp databases).

    Returns:
        VerificationResult with:
        - is_safe: Whether the SQL passed guardrail + cost checks
        - is_valid: Whether the SQL executed without database errors
        - result_rows: The query results (list of dicts)
        - row_count: Number of rows returned
        - columns: Column names from the result set
        - error: Error message if any step failed
    """
    sql = sql_result.sql.strip()

    # ── Step 1: Validate SQL via output guardrails ───────────────
    # Checks: SELECT-only, no DML/DDL, no multi-statement, auto-LIMIT
    guard_result = validate_sql(sql)

    if not guard_result.passed:
        return VerificationResult(
            is_safe=False,
            is_valid=False,
            error=guard_result.reason,
        )

    # Use the sanitized SQL (may have LIMIT appended by guardrails)
    safe_sql = guard_result.sanitized_input or sql

    repo = AnalyticsRepository(db_path=db_path)

    # ── Step 2: Cost estimation via EXPLAIN QUERY PLAN ───────────
    # Detect full table scans on large tables BEFORE executing.
    # This prevents queries that would scan billions of rows in production.
    cost_error = _estimate_query_cost(repo, safe_sql)
    if cost_error:
        return VerificationResult(
            is_safe=True,       # SQL is syntactically safe
            is_valid=False,     # But too expensive to run
            error=cost_error,
        )

    # ── Step 3: Execute via repository in readonly mode ──────────
    try:
        rows = repo.execute_readonly(safe_sql)

        # Extract column names from the first row's keys
        columns = list(rows[0].keys()) if rows else []

        return VerificationResult(
            is_safe=True,
            is_valid=True,
            result_rows=rows,
            row_count=len(rows),
            columns=columns,
        )

    except ValueError as e:
        # execute_readonly blocked the query (DML detected at repo level)
        return VerificationResult(
            is_safe=False,
            is_valid=False,
            error=str(e),
        )

    except Exception as e:
        # SQL execution error (syntax error, missing column, etc.)
        # The SQL passed guardrails (is_safe=True) but failed at runtime
        return VerificationResult(
            is_safe=True,
            is_valid=False,
            error=f"SQL execution error: {str(e)}",
        )


# ── Cost Estimation ──────────────────────────────────────────────────

def _estimate_query_cost(repo: AnalyticsRepository, sql: str) -> str | None:
    """Run EXPLAIN QUERY PLAN and check for expensive full table scans.

    SQLite's EXPLAIN QUERY PLAN returns rows like:
        SCAN shipments              ← full table scan (no index, BAD for large tables)
        SEARCH carriers USING INDEX ← index lookup (fast, GOOD)
        SEARCH shipments USING INDEX idx_shipments_status ← index scan (GOOD)

    We flag SCAN operations on tables exceeding FULL_SCAN_ROW_THRESHOLD.

    In production (PostgreSQL/MySQL), this would use:
        EXPLAIN ANALYZE for actual row estimates and cost numbers.

    Args:
        repo: AnalyticsRepository instance for executing EXPLAIN.
        sql: The SQL query to analyze.

    Returns:
        Error message string if the query is too expensive, None if OK.
    """
    try:
        # Load the configurable threshold from settings.yaml
        threshold = _get_full_scan_threshold()

        # Run EXPLAIN QUERY PLAN (this does NOT execute the actual query)
        conn = repo._get_connection()
        cursor = conn.execute(f"EXPLAIN QUERY PLAN {sql}")
        plan_rows = cursor.fetchall()

        expensive_scans = []

        for row in plan_rows:
            # SQLite EXPLAIN QUERY PLAN returns: id, parent, notused, detail
            detail = str(dict(row).get("detail", ""))

            # Look for "SCAN <table>" which means full table scan (no index)
            # "SEARCH <table> USING INDEX" means index is used (fast)
            scan_match = re.match(r"SCAN\s+(\w+)", detail)
            if scan_match:
                table_name = scan_match.group(1)

                # Check if this table is large enough to worry about
                try:
                    row_count = repo.get_row_count(table_name)
                except ValueError:
                    continue  # Unknown table in EXPLAIN (subquery alias, etc.)

                if row_count > threshold:
                    expensive_scans.append((table_name, row_count))

        if expensive_scans:
            # Build a helpful error message suggesting how to narrow the query
            scan_details = ", ".join(
                f"{table} ({count:,} rows)" for table, count in expensive_scans
            )
            return (
                f"Query would perform a full table scan on: {scan_details}. "
                f"This could be very slow on large datasets. "
                f"Please add a WHERE filter to narrow the results — "
                f"for example: date range, specific carrier, status, mode, or customer."
            )

        return None  # Query plan looks OK

    except Exception as e:
        # If EXPLAIN itself fails, log it but don't block execution.
        # The query might still work fine — EXPLAIN failures are non-critical.
        logger.warning(f"EXPLAIN QUERY PLAN failed (non-blocking): {e}")
        return None
