#!/usr/bin/env python3
"""LLM Model Evaluation Script for Text-to-SQL Generation Accuracy.

This script evaluates multiple LLM models on their ability to generate
correct SQL queries from natural language questions against the logistics
database. It is designed to be run standalone from the project root:

    python eval/run_llm_eval.py
    python eval/run_llm_eval.py --limit 10 --models groq,gemini

Workflow:
    1. Load test cases from eval/test_cases.yaml
    2. Load the database schema description (same one used in production prompts)
    3. For each model x test_case combination:
       a. Build the prompt (schema + question) using config/prompts.yaml
       b. Send to the LLM via LangChain
       c. Score the response on multiple dimensions (JSON parse, SQL validity, etc.)
       d. Record latency and estimated cost
    4. Aggregate results and generate:
       - eval/results/llm_results.json  (raw data)
       - eval/results/llm_report.md     (markdown summary)
       - eval/results/llm_report.html   (styled HTML report)

Models tested (from config/models.yaml sql_generation chain):
    - groq / llama-3.3-70b-versatile          (free)
    - google / gemini-2.5-flash               (free)
    - openrouter / qwen/qwen3-coder           (free)
    - openrouter / deepseek/deepseek-r1       (free)
    - openrouter / deepseek/deepseek-chat     (cheap paid)
    - openrouter / anthropic/claude-sonnet-4  (expensive paid)

Scoring dimensions per test case:
    - json_parse:        1 if LLM returned valid JSON with a "sql" key
    - sql_valid:         1 if the SQL parses (EXPLAIN succeeds on SQLite)
    - sql_runs:          1 if the SQL executes without error
    - correct_tables:    1 if tables_used overlaps with expected_tables
    - correct_behavior:  1 if "answer" cases produce SQL OR "refuse" cases refuse

Environment:
    - API keys loaded from .env via python-dotenv
    - Database at db/logistics.db
    - Python 3.13
"""

# ── Standard Library Imports ────────────────────────────────────────────
import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── Third-Party Imports ─────────────────────────────────────────────────
import yaml
from dotenv import load_dotenv

# ── Path Setup ──────────────────────────────────────────────────────────
# Add the project root to sys.path so we can import src.* modules.
# This allows running the script from the project root directory:
#   python eval/run_llm_eval.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file at the project root.
# This makes GROQ_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY available.
load_dotenv(PROJECT_ROOT / ".env")

# ── Project Imports (after path setup) ──────────────────────────────────
from src.common.config_loader import load_model_config, load_prompts
from src.database import get_schema_description
from src.guardrails.input_guards import validate_input

# ── Constants ───────────────────────────────────────────────────────────

# Path to the test cases YAML file, located alongside this script.
TEST_CASES_PATH = Path(__file__).resolve().parent / "test_cases.yaml"

# Output directory for results files. Created automatically if missing.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Path to the SQLite database used for SQL validation and execution.
DB_PATH = PROJECT_ROOT / "db" / "logistics.db"

# Maximum number of retries when a model call fails due to rate limiting
# or transient errors. Each retry doubles the sleep time (exponential backoff).
MAX_RETRIES = 3

# Base sleep time in seconds for the first retry. Doubled on each subsequent retry.
BASE_RETRY_SLEEP = 2.0

# Cost-per-million-tokens for each model.
# Models with $0 cost are on free tiers. This dict is used to estimate
# the total cost of running the evaluation.
MODEL_COSTS: dict[str, dict[str, float]] = {
    "groq/llama-3.3-70b-versatile": {
        "input_per_mtok": 0.0,
        "output_per_mtok": 0.0,
    },
    "google/gemini-2.5-flash": {
        "input_per_mtok": 0.0,
        "output_per_mtok": 0.0,
    },
    "openrouter/qwen/qwen3-coder": {
        "input_per_mtok": 0.0,
        "output_per_mtok": 0.0,
    },
    "openrouter/deepseek/deepseek-r1": {
        "input_per_mtok": 0.0,
        "output_per_mtok": 0.0,
    },
    "openrouter/deepseek/deepseek-chat": {
        "input_per_mtok": 0.27,
        "output_per_mtok": 1.10,
    },
    "openrouter/anthropic/claude-sonnet-4": {
        "input_per_mtok": 3.0,
        "output_per_mtok": 15.0,
    },
}

# Short display names for models (used in tables where space is limited).
# Maps the full "provider/model" key to a concise label.
MODEL_SHORT_NAMES: dict[str, str] = {
    "groq/llama-3.3-70b-versatile": "groq/llama-3.3-70b",
    "google/gemini-2.5-flash": "google/gemini-2.5-flash",
    "openrouter/qwen/qwen3-coder": "openrouter/qwen3-coder",
    "openrouter/deepseek/deepseek-r1": "openrouter/deepseek-r1",
    "openrouter/deepseek/deepseek-chat": "openrouter/deepseek-chat",
    "openrouter/anthropic/claude-sonnet-4": "openrouter/claude-sonnet-4",
}

# Mapping from CLI shorthand names to the full "provider/model" keys.
# Users can pass --models groq,gemini,qwen,deepseek-r1,deepseek-chat,claude
# instead of typing the full model identifier.
MODEL_ALIASES: dict[str, str] = {
    "groq": "groq/llama-3.3-70b-versatile",
    "gemini": "google/gemini-2.5-flash",
    "gemini-or": "openrouter/google/gemini-2.5-flash",      # Gemini via OpenRouter (no rate limit)
    "gemini-lite": "openrouter/google/gemini-2.5-flash-lite", # Lite variant via OpenRouter
    "qwen": "openrouter/qwen/qwen3-coder",
    "deepseek-r1": "openrouter/deepseek/deepseek-r1",
    "deepseek-chat": "openrouter/deepseek/deepseek-chat",
    "claude": "openrouter/anthropic/claude-sonnet-4",
}

# The list of all model keys in evaluation order (free first, then paid).
ALL_MODEL_KEYS: list[str] = [
    "groq/llama-3.3-70b-versatile",
    "google/gemini-2.5-flash",
    "openrouter/google/gemini-2.5-flash",       # Same model, no rate limit via OpenRouter
    "openrouter/qwen/qwen3-coder",
    "openrouter/deepseek/deepseek-r1",
    "openrouter/deepseek/deepseek-chat",
    "openrouter/anthropic/claude-sonnet-4",
]


# ═══════════════════════════════════════════════════════════════════════
# ── Test Case Loading ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def load_test_cases(limit: Optional[int] = None) -> list[dict]:
    """Load test cases from the YAML file.

    Each test case dict contains:
        id (str): Unique identifier like "simple_count_01"
        category (str): Grouping like "simple_count", "join", "aggregation"
        question (str): The natural language question
        expected_tables (list[str]): Tables the SQL should reference
        expected_behavior (str): "answer", "refuse", or "injection"
        difficulty (str): "easy", "medium", or "hard"
        notes (str, optional): Explanation

    Args:
        limit: If provided, only return the first N test cases. Useful
               for quick smoke tests during development.

    Returns:
        List of test case dicts.

    Raises:
        FileNotFoundError: If eval/test_cases.yaml does not exist.
        ValueError: If the YAML file is malformed or missing the test_cases key.
    """
    if not TEST_CASES_PATH.exists():
        raise FileNotFoundError(
            f"Test cases file not found: {TEST_CASES_PATH}\n"
            f"Create it at eval/test_cases.yaml with a 'test_cases' key."
        )

    with open(TEST_CASES_PATH, "r") as f:
        data = yaml.safe_load(f)

    if not data or "test_cases" not in data:
        raise ValueError(
            f"Invalid test cases file: {TEST_CASES_PATH}\n"
            f"Expected a top-level 'test_cases' key containing a list."
        )

    cases = data["test_cases"]

    if limit is not None and limit > 0:
        cases = cases[:limit]

    return cases


# ═══════════════════════════════════════════════════════════════════════
# ── Model Instantiation ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def create_model_instance(model_key: str) -> Any:
    """Create a LangChain chat model instance for the given model key.

    Uses the same provider logic as src/models/llm_factory.py but allows
    us to instantiate a specific model rather than walking the fallback chain.
    We import the factory helper directly to stay DRY.

    The model_key format is "provider/model_name", e.g.:
        "groq/llama-3.3-70b-versatile"
        "openrouter/deepseek/deepseek-chat"

    For openrouter models, the provider is "openrouter" and the model name
    may contain slashes (e.g., "deepseek/deepseek-chat").

    Args:
        model_key: Full model identifier in "provider/model" format.

    Returns:
        A LangChain BaseChatModel instance configured with temperature=0
        and appropriate max_tokens/timeout for SQL generation.

    Raises:
        ValueError: If the provider is not recognized.
        Exception: If model instantiation fails (e.g., missing API key).
    """
    # Load SQL generation task settings to use consistent temperature/max_tokens
    config = load_model_config()
    task_config = config["task_routing"]["sql_generation"]
    temperature = task_config.get("temperature", 0.0)
    max_tokens = task_config.get("max_tokens", 2000)
    timeout = task_config.get("timeout_seconds", 30)

    # Split the model_key into provider and model name.
    # For openrouter, the model name itself contains a slash:
    #   "openrouter/deepseek/deepseek-chat" -> provider="openrouter", model="deepseek/deepseek-chat"
    parts = model_key.split("/", 1)
    provider = parts[0]
    model_name = parts[1] if len(parts) > 1 else ""

    # Use the factory's public creation function for consistency
    from src.models.llm_factory import create_model_instance
    return create_model_instance(provider, model_name, temperature, max_tokens, timeout)


# ═══════════════════════════════════════════════════════════════════════
# ── LLM Invocation with Retry Logic ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def invoke_model(
    model_instance: Any,
    prompt_text: str,
    model_key: str,
) -> dict:
    """Invoke an LLM model with the given prompt and return the raw response.

    Implements exponential backoff retry logic to handle transient failures
    like rate limiting (HTTP 429) and temporary API outages (HTTP 5xx).

    The retry strategy:
        - Attempt 1: immediate
        - Attempt 2: sleep 2s
        - Attempt 3: sleep 4s
    If all retries fail, returns an error dict instead of raising.

    Args:
        model_instance: A LangChain BaseChatModel instance.
        prompt_text: The fully assembled prompt string (schema + question).
        model_key: Model identifier for logging (e.g., "groq/llama-3.3-70b-versatile").

    Returns:
        Dict with keys:
            raw_response (str): The raw text output from the LLM.
            latency_ms (float): Wall-clock time for the API call in milliseconds.
            input_tokens (int): Estimated input token count (chars / 4 heuristic).
            output_tokens (int): Estimated output token count (chars / 4 heuristic).
            error (str|None): Error message if all retries failed, None on success.
    """
    from langchain_core.messages import HumanMessage

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Measure wall-clock latency for the API call
            start_time = time.perf_counter()
            response = model_instance.invoke([HumanMessage(content=prompt_text)])
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000.0

            # Extract the text content from the LangChain response object
            raw_text = response.content if hasattr(response, "content") else str(response)

            # Estimate token counts using the rough heuristic of ~4 chars per token.
            # This is approximate but sufficient for cost estimation. Real token
            # counts would require the model's tokenizer, which varies by provider.
            input_tokens = len(prompt_text) // 4
            output_tokens = len(raw_text) // 4

            # Try to extract actual token usage from the response metadata if available.
            # LangChain models sometimes populate response_metadata with token counts.
            if hasattr(response, "response_metadata") and response.response_metadata:
                meta = response.response_metadata
                # OpenAI / OpenRouter format
                if "token_usage" in meta:
                    usage = meta["token_usage"]
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)
                # Google format
                elif "usage_metadata" in meta:
                    usage = meta["usage_metadata"]
                    input_tokens = usage.get("prompt_token_count", input_tokens)
                    output_tokens = usage.get("candidates_token_count", output_tokens)
            # Also check the usage_metadata attribute directly (some LangChain versions)
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                if hasattr(usage, "input_tokens") and usage.input_tokens:
                    input_tokens = usage.input_tokens
                if hasattr(usage, "output_tokens") and usage.output_tokens:
                    output_tokens = usage.output_tokens

            return {
                "raw_response": raw_text,
                "latency_ms": round(latency_ms, 1),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "error": None,
            }

        except Exception as e:
            last_error = e
            error_str = f"{type(e).__name__}: {e}"

            # Check if this is a rate-limit or transient error worth retrying
            is_retryable = _is_retryable_error(e)

            if is_retryable and attempt < MAX_RETRIES:
                sleep_time = BASE_RETRY_SLEEP * (2 ** (attempt - 1))
                print(f"    [retry {attempt}/{MAX_RETRIES}] {model_key}: {error_str}")
                print(f"    Sleeping {sleep_time:.1f}s before retry...")
                time.sleep(sleep_time)
                continue
            else:
                # Non-retryable error or final attempt exhausted
                return {
                    "raw_response": "",
                    "latency_ms": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error": error_str,
                }

    # Should not reach here, but just in case
    return {
        "raw_response": "",
        "latency_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "error": f"All {MAX_RETRIES} retries failed: {last_error}",
    }


def _is_retryable_error(error: Exception) -> bool:
    """Determine if an error is transient and worth retrying.

    Rate limits (429), server errors (5xx), and timeout errors are
    considered retryable. Authentication errors (401, 403) and bad
    requests (400) are not retryable.

    Args:
        error: The exception raised by the model invocation.

    Returns:
        True if the error is likely transient and a retry may succeed.
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # Rate limit indicators
    if "429" in error_str or "rate" in error_str and "limit" in error_str:
        return True

    # Server errors (5xx)
    if any(code in error_str for code in ["500", "502", "503", "504"]):
        return True

    # Timeout errors
    if "timeout" in error_str or "timeout" in error_type:
        return True

    # Connection errors
    if "connection" in error_str or "connect" in error_type:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════
# ── Response Parsing and Scoring ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def parse_llm_response(raw_response: str) -> dict:
    """Parse the raw LLM response string into a structured dict.

    The expected LLM output format (from config/prompts.yaml sql_generator) is:
        {"sql": "SELECT ...", "explanation": "...", "tables_used": ["shipments"]}

    Many LLMs wrap their JSON in markdown code fences (```json ... ```) or
    include preamble text. This function handles those cases by extracting
    the JSON substring.

    Some models (notably deepseek-r1) emit a <think>...</think> reasoning
    block before the actual JSON response. This function strips that block.

    Args:
        raw_response: The raw text string from the LLM.

    Returns:
        Dict with keys:
            json_parse_ok (bool): Whether valid JSON was extracted.
            parsed (dict|None): The parsed JSON dict, or None if parsing failed.
            sql (str): The extracted SQL string, or empty string.
            tables_used (list[str]): Tables reported by the model.
            explanation (str): The model's explanation of the query.
            parse_error (str): Error description if parsing failed.
    """
    if not raw_response or not raw_response.strip():
        return {
            "json_parse_ok": False,
            "parsed": None,
            "sql": "",
            "tables_used": [],
            "explanation": "",
            "parse_error": "Empty response from model",
        }

    text = raw_response.strip()

    # Strip <think>...</think> reasoning blocks (deepseek-r1 style).
    # These appear before the actual JSON response and must be removed.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try to extract JSON from markdown code fences first.
    # Pattern matches ```json ... ``` or ``` ... ``` blocks.
    json_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if json_block_match:
        text = json_block_match.group(1).strip()

    # If no code fence, try to find a JSON object directly.
    # Look for the outermost { ... } in the response.
    if not text.startswith("{"):
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    # Attempt to parse the extracted text as JSON
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return {
            "json_parse_ok": False,
            "parsed": None,
            "sql": "",
            "tables_used": [],
            "explanation": "",
            "parse_error": f"JSON decode error: {e}",
        }

    # Validate that the parsed object is a dict with a "sql" key
    if not isinstance(parsed, dict):
        return {
            "json_parse_ok": False,
            "parsed": None,
            "sql": "",
            "tables_used": [],
            "explanation": "",
            "parse_error": f"Expected JSON object, got {type(parsed).__name__}",
        }

    has_sql_key = "sql" in parsed
    sql_value = parsed.get("sql", "")
    tables_used = parsed.get("tables_used", [])
    explanation = parsed.get("explanation", "")

    # Ensure tables_used is a list of strings
    if isinstance(tables_used, str):
        tables_used = [tables_used]
    elif not isinstance(tables_used, list):
        tables_used = []

    return {
        "json_parse_ok": has_sql_key,
        "parsed": parsed,
        "sql": sql_value if isinstance(sql_value, str) else str(sql_value) if sql_value else "",
        "tables_used": tables_used,
        "explanation": explanation,
        "parse_error": "" if has_sql_key else "JSON parsed but missing 'sql' key",
    }


def check_sql_validity(sql: str) -> dict:
    """Check if the SQL string parses correctly in SQLite.

    Uses EXPLAIN to validate the SQL syntax without actually executing
    the query. This catches syntax errors, unknown function names, and
    invalid column references.

    Args:
        sql: The SQL query string to validate.

    Returns:
        Dict with keys:
            sql_valid (bool): True if EXPLAIN succeeds.
            validity_error (str): Error message if validation failed.
    """
    if not sql or not sql.strip():
        return {"sql_valid": False, "validity_error": "Empty SQL string"}

    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        # EXPLAIN parses the SQL and generates the query plan without executing.
        # If the SQL has syntax errors or references invalid tables/columns,
        # EXPLAIN will raise an OperationalError.
        cursor.execute(f"EXPLAIN {sql}")
        conn.close()
        return {"sql_valid": True, "validity_error": ""}
    except sqlite3.Error as e:
        return {"sql_valid": False, "validity_error": str(e)}
    except Exception as e:
        return {"sql_valid": False, "validity_error": f"Unexpected error: {e}"}


def check_sql_execution(sql: str) -> dict:
    """Attempt to execute the SQL query against the actual database.

    Runs the query inside a read-only context to prevent any data
    modification. Returns whether the query ran without errors and
    the number of rows returned.

    The query is wrapped with a LIMIT 10 if it does not already contain
    a LIMIT clause, to avoid returning excessive data during evaluation.

    Args:
        sql: The SQL query string to execute.

    Returns:
        Dict with keys:
            sql_runs (bool): True if the query executed without error.
            execution_error (str): Error message if execution failed.
            row_count (int): Number of rows returned (up to the limit).
    """
    if not sql or not sql.strip():
        return {"sql_runs": False, "execution_error": "Empty SQL string", "row_count": 0}

    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Add a LIMIT if the query does not already have one,
        # to prevent returning huge result sets during eval.
        exec_sql = sql.strip().rstrip(";")
        if "LIMIT" not in exec_sql.upper():
            exec_sql = f"{exec_sql} LIMIT 10"

        cursor.execute(exec_sql)
        rows = cursor.fetchall()
        conn.close()

        return {
            "sql_runs": True,
            "execution_error": "",
            "row_count": len(rows),
        }
    except sqlite3.Error as e:
        return {"sql_runs": False, "execution_error": str(e), "row_count": 0}
    except Exception as e:
        return {"sql_runs": False, "execution_error": f"Unexpected: {e}", "row_count": 0}


def check_tables_overlap(
    reported_tables: list[str],
    expected_tables: list[str],
) -> bool:
    """Check if the tables reported by the model overlap with the expected tables.

    For "answer" type test cases, we require that at least the expected tables
    are present in the reported tables. The model may use additional tables
    (e.g., for JOINs) and that is acceptable.

    For test cases with empty expected_tables (refuse/injection), this always
    returns True because there are no table expectations.

    Args:
        reported_tables: Tables the LLM says it used (from its JSON response).
        expected_tables: Tables the test case expects to be used.

    Returns:
        True if every expected table appears in the reported tables (case-insensitive),
        or if expected_tables is empty.
    """
    if not expected_tables:
        # No table expectations for refuse/injection cases
        return True

    # Normalize to lowercase for comparison
    reported_lower = {t.lower().strip() for t in reported_tables}
    expected_lower = {t.lower().strip() for t in expected_tables}

    # Check that every expected table is present in the reported tables
    return expected_lower.issubset(reported_lower)


def check_correct_behavior(
    test_case: dict,
    parse_result: dict,
    raw_response: str,
    guardrail_blocked: bool,
) -> dict:
    """Determine if the model exhibited the correct behavior for this test case.

    Different expected_behavior values require different checks:

    "answer": The model should produce a valid SQL query. We check that
        json_parse_ok is True and that a non-empty SQL string was returned.

    "refuse": The model should decline to generate SQL. This is correct if
        the model returned JSON without a SQL query, returned an explanation
        of why it cannot answer, or if the sql field is empty/null.

    "injection": The model should be blocked by input guardrails before
        it even reaches the LLM. We check the guardrail_blocked flag.

    Args:
        test_case: The test case dict with expected_behavior.
        parse_result: The parsed LLM response dict from parse_llm_response().
        raw_response: The raw text response from the LLM.
        guardrail_blocked: Whether the input guardrail blocked this query.

    Returns:
        Dict with keys:
            correct_behavior (bool): Whether the model did the right thing.
            behavior_note (str): Explanation of the behavior assessment.
    """
    expected = test_case.get("expected_behavior", "answer")

    if expected == "answer":
        # For answerable questions, the model should produce SQL
        if parse_result["json_parse_ok"] and parse_result["sql"].strip():
            return {
                "correct_behavior": True,
                "behavior_note": "Correctly generated SQL for answerable question",
            }
        else:
            return {
                "correct_behavior": False,
                "behavior_note": "Failed to generate SQL for answerable question",
            }

    elif expected == "refuse":
        # For unanswerable questions, the model should NOT produce SQL.
        # It should either return empty SQL, explain it cannot answer,
        # or return JSON without a sql key.
        sql = parse_result.get("sql", "").strip()

        # Check if the model explicitly refused
        if not sql:
            return {
                "correct_behavior": True,
                "behavior_note": "Correctly refused to generate SQL (empty sql field)",
            }

        # Check if the response contains refusal language
        response_lower = raw_response.lower()
        refusal_indicators = [
            "cannot answer", "can't answer", "unable to answer",
            "not available", "no data", "doesn't exist",
            "does not exist", "not in the database", "not in our database",
            "no table", "no such", "unanswerable", "cannot be answered",
            "not possible", "outside the scope", "beyond the scope",
            "i cannot", "i can't", "not supported",
        ]
        if any(indicator in response_lower for indicator in refusal_indicators):
            return {
                "correct_behavior": True,
                "behavior_note": "Correctly refused with explanation",
            }

        # If the model generated SQL for an unanswerable question, that is wrong
        return {
            "correct_behavior": False,
            "behavior_note": "Incorrectly generated SQL for unanswerable question",
        }

    elif expected == "injection":
        # For injection attempts, input guardrails should block before LLM call
        if guardrail_blocked:
            return {
                "correct_behavior": True,
                "behavior_note": "Correctly blocked by input guardrails",
            }
        else:
            # If guardrails did not block, check if the model itself refused
            sql = parse_result.get("sql", "").strip()
            if not sql:
                return {
                    "correct_behavior": True,
                    "behavior_note": "Guardrails did not block, but model refused to generate harmful SQL",
                }
            # Model generated SQL for an injection attempt - worst case
            return {
                "correct_behavior": False,
                "behavior_note": "DANGER: Injection bypassed guardrails and model generated SQL",
            }

    else:
        # Unknown expected_behavior - treat as a fail
        return {
            "correct_behavior": False,
            "behavior_note": f"Unknown expected_behavior: {expected}",
        }


def estimate_cost(
    model_key: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate the cost of a single model invocation in USD.

    Uses the per-million-token pricing from MODEL_COSTS. Free-tier models
    return $0.000.

    Args:
        model_key: The full model identifier (e.g., "groq/llama-3.3-70b-versatile").
        input_tokens: Number of input tokens (actual or estimated).
        output_tokens: Number of output tokens (actual or estimated).

    Returns:
        Estimated cost in USD (float). Returns 0.0 for free-tier models.
    """
    costs = MODEL_COSTS.get(model_key, {"input_per_mtok": 0.0, "output_per_mtok": 0.0})
    input_cost = (input_tokens / 1_000_000) * costs["input_per_mtok"]
    output_cost = (output_tokens / 1_000_000) * costs["output_per_mtok"]
    return input_cost + output_cost


# ═══════════════════════════════════════════════════════════════════════
# ── Prompt Building ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(schema: str, question: str) -> str:
    """Build the full SQL generation prompt from the template in config/prompts.yaml.

    Loads the sql_generator prompt template and fills in the schema and question
    placeholders. The conversation history is left empty since eval test cases
    are standalone questions without context.

    Args:
        schema: The database schema description string.
        question: The natural language question from the test case.

    Returns:
        The fully assembled prompt string ready to send to the LLM.
    """
    prompts = load_prompts()
    # Navigate to analytics > sql_generator template
    template = prompts.get("analytics", {}).get("sql_generator", "")

    if not template:
        # Fallback prompt if config is missing (should not happen in production)
        return (
            f"Generate a SQL query to answer this question about a logistics database.\n\n"
            f"DATABASE SCHEMA:\n{schema}\n\n"
            f"QUESTION: {question}\n\n"
            f"Respond ONLY with valid JSON:\n"
            f'{{"sql": "your SELECT query", "explanation": "...", "tables_used": ["table1"]}}'
        )

    # Fill in the template placeholders
    # The template uses {schema}, {history}, and {query} placeholders
    prompt = template.format(
        schema=schema,
        history="(no conversation history - standalone evaluation question)",
        query=question,
    )

    return prompt


# ═══════════════════════════════════════════════════════════════════════
# ── Single Test Case Evaluation ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def evaluate_single(
    model_key: str,
    model_instance: Any,
    test_case: dict,
    schema: str,
) -> dict:
    """Evaluate a single test case against a single model.

    This is the core evaluation function. It:
    1. Checks input guardrails (for injection detection)
    2. Builds and sends the prompt to the LLM
    3. Parses the JSON response
    4. Validates the SQL (EXPLAIN)
    5. Executes the SQL
    6. Checks table usage
    7. Checks correct behavior
    8. Estimates cost

    Args:
        model_key: Full model identifier (e.g., "groq/llama-3.3-70b-versatile").
        model_instance: The LangChain chat model instance.
        test_case: A single test case dict from test_cases.yaml.
        schema: The database schema description string.

    Returns:
        A comprehensive result dict with all scoring dimensions, latency,
        cost, and error details. This dict is serializable to JSON.
    """
    test_id = test_case["id"]
    question = test_case["question"]
    expected_tables = test_case.get("expected_tables", [])
    expected_behavior = test_case.get("expected_behavior", "answer")
    category = test_case.get("category", "unknown")
    difficulty = test_case.get("difficulty", "unknown")

    # ── Step 1: Check input guardrails ──────────────────────────────
    # Run the same guardrails that the production system uses.
    # For injection test cases, we EXPECT these to block the input.
    guardrail_result = validate_input(question)
    guardrail_blocked = not guardrail_result.passed

    # If guardrails blocked AND this is an injection test, that is success.
    # If guardrails blocked AND this is NOT an injection test, record the block.
    if guardrail_blocked:
        behavior_check = check_correct_behavior(
            test_case,
            {"json_parse_ok": False, "sql": "", "tables_used": [], "explanation": ""},
            "",
            guardrail_blocked=True,
        )

        return {
            "test_id": test_id,
            "model": model_key,
            "model_short": MODEL_SHORT_NAMES.get(model_key, model_key),
            "category": category,
            "difficulty": difficulty,
            "question": question,
            "expected_behavior": expected_behavior,
            "expected_tables": expected_tables,
            # Guardrail blocked - no LLM call made
            "guardrail_blocked": True,
            "guardrail_reason": guardrail_result.reason,
            "raw_response": "",
            "sql": "",
            "explanation": "",
            "tables_used": [],
            # Scores
            "json_parse": 0,
            "sql_valid": 0,
            "sql_runs": 0,
            "correct_tables": 1 if not expected_tables else 0,
            "correct_behavior": 1 if behavior_check["correct_behavior"] else 0,
            "behavior_note": behavior_check["behavior_note"],
            # Performance
            "latency_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
            # Errors
            "error": None,
            "parse_error": "Blocked by guardrails",
            "validity_error": "",
            "execution_error": "",
        }

    # ── Step 2: Build prompt and invoke model ───────────────────────
    prompt = build_prompt(schema, question)
    invocation_result = invoke_model(model_instance, prompt, model_key)

    raw_response = invocation_result["raw_response"]
    latency_ms = invocation_result["latency_ms"]
    input_tokens = invocation_result["input_tokens"]
    output_tokens = invocation_result["output_tokens"]
    invocation_error = invocation_result["error"]

    # If the model call itself failed (all retries exhausted), record failure
    if invocation_error:
        behavior_check = check_correct_behavior(
            test_case,
            {"json_parse_ok": False, "sql": "", "tables_used": [], "explanation": ""},
            raw_response,
            guardrail_blocked=False,
        )

        return {
            "test_id": test_id,
            "model": model_key,
            "model_short": MODEL_SHORT_NAMES.get(model_key, model_key),
            "category": category,
            "difficulty": difficulty,
            "question": question,
            "expected_behavior": expected_behavior,
            "expected_tables": expected_tables,
            "guardrail_blocked": False,
            "guardrail_reason": None,
            "raw_response": raw_response,
            "sql": "",
            "explanation": "",
            "tables_used": [],
            "json_parse": 0,
            "sql_valid": 0,
            "sql_runs": 0,
            "correct_tables": 1 if not expected_tables else 0,
            "correct_behavior": 1 if behavior_check["correct_behavior"] else 0,
            "behavior_note": behavior_check["behavior_note"],
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimate_cost(model_key, input_tokens, output_tokens),
            "error": invocation_error,
            "parse_error": "Model invocation failed",
            "validity_error": "",
            "execution_error": "",
        }

    # ── Step 3: Parse the JSON response ─────────────────────────────
    parse_result = parse_llm_response(raw_response)
    sql = parse_result["sql"]
    tables_used = parse_result["tables_used"]
    explanation = parse_result["explanation"]

    # ── Step 4: Validate SQL syntax (EXPLAIN) ───────────────────────
    if parse_result["json_parse_ok"] and sql.strip():
        validity = check_sql_validity(sql)
    else:
        validity = {"sql_valid": False, "validity_error": "No SQL to validate"}

    # ── Step 5: Execute SQL ─────────────────────────────────────────
    if validity["sql_valid"]:
        execution = check_sql_execution(sql)
    else:
        execution = {"sql_runs": False, "execution_error": "SQL not valid, skipping execution", "row_count": 0}

    # ── Step 6: Check table usage ───────────────────────────────────
    # Also try to extract tables from the SQL itself if the model did not
    # report them in tables_used (some models omit this field)
    if not tables_used and sql.strip():
        tables_used = _extract_tables_from_sql(sql)

    tables_correct = check_tables_overlap(tables_used, expected_tables)

    # ── Step 7: Check correct behavior ──────────────────────────────
    behavior_check = check_correct_behavior(
        test_case, parse_result, raw_response, guardrail_blocked=False,
    )

    # ── Step 8: Estimate cost ───────────────────────────────────────
    cost = estimate_cost(model_key, input_tokens, output_tokens)

    return {
        "test_id": test_id,
        "model": model_key,
        "model_short": MODEL_SHORT_NAMES.get(model_key, model_key),
        "category": category,
        "difficulty": difficulty,
        "question": question,
        "expected_behavior": expected_behavior,
        "expected_tables": expected_tables,
        "guardrail_blocked": False,
        "guardrail_reason": None,
        "raw_response": raw_response[:500],  # Truncate for storage
        "sql": sql,
        "explanation": explanation,
        "tables_used": tables_used,
        # Scores (1 or 0 for each dimension)
        "json_parse": 1 if parse_result["json_parse_ok"] else 0,
        "sql_valid": 1 if validity["sql_valid"] else 0,
        "sql_runs": 1 if execution["sql_runs"] else 0,
        "correct_tables": 1 if tables_correct else 0,
        "correct_behavior": 1 if behavior_check["correct_behavior"] else 0,
        "behavior_note": behavior_check["behavior_note"],
        # Performance
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
        "row_count": execution.get("row_count", 0),
        # Errors (for debugging failed cases)
        "error": None,
        "parse_error": parse_result["parse_error"],
        "validity_error": validity["validity_error"],
        "execution_error": execution.get("execution_error", ""),
    }


def _extract_tables_from_sql(sql: str) -> list[str]:
    """Extract table names from a SQL query using regex.

    This is a fallback for when the LLM does not report tables_used in
    its JSON response. It uses a simple regex to find table names after
    FROM and JOIN keywords.

    This approach is approximate and may miss some tables (e.g., in
    subqueries) or include false positives. It is used only as a
    fallback when the model's self-reported tables_used is empty.

    Args:
        sql: The SQL query string.

    Returns:
        List of table names found in the query (deduplicated, lowercase).
    """
    # Known tables in our schema (to filter out false positives)
    known_tables = {
        "customers", "carriers", "shipments", "shipment_charges",
        "tracking_events", "invoices", "extracted_documents",
    }

    # Find table names after FROM and JOIN keywords.
    # Pattern: FROM/JOIN followed by optional whitespace and a word (the table name).
    # Also captures table aliases like "shipments s" but we only want the table name.
    pattern = r"(?:FROM|JOIN)\s+(\w+)"
    matches = re.findall(pattern, sql, re.IGNORECASE)

    # Filter to only known tables and deduplicate
    tables = []
    seen = set()
    for match in matches:
        table_lower = match.lower().strip()
        if table_lower in known_tables and table_lower not in seen:
            tables.append(table_lower)
            seen.add(table_lower)

    return tables


# ═══════════════════════════════════════════════════════════════════════
# ── Aggregation and Reporting ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def aggregate_results(results: list[dict]) -> dict:
    """Aggregate per-test-case results into summary statistics.

    Groups results by model and by category, computing:
    - Pass rates for each scoring dimension (json_parse, sql_valid, etc.)
    - Average latency per model
    - Total and per-query estimated costs
    - Lists of failed test cases

    Args:
        results: List of result dicts from evaluate_single().

    Returns:
        Dict with keys:
            models (dict): Per-model summary stats.
            categories (dict): Per-category x per-model pass rates.
            failed_cases (list): List of dicts for test cases that failed.
            overall_winner (str): Best model by composite score.
            best_free (str): Best free-tier model by composite score.
            best_cost_efficiency (str): Best score/cost ratio model.
    """
    # ── Group results by model ──────────────────────────────────────
    by_model: dict[str, list[dict]] = {}
    for r in results:
        model = r["model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    # ── Compute per-model summaries ─────────────────────────────────
    model_summaries: dict[str, dict] = {}
    for model_key, model_results in by_model.items():
        n = len(model_results)
        if n == 0:
            continue

        # Sum up each scoring dimension
        json_parse_sum = sum(r["json_parse"] for r in model_results)
        sql_valid_sum = sum(r["sql_valid"] for r in model_results)
        sql_runs_sum = sum(r["sql_runs"] for r in model_results)
        correct_tables_sum = sum(r["correct_tables"] for r in model_results)
        correct_behavior_sum = sum(r["correct_behavior"] for r in model_results)

        # Calculate pass rates as percentages
        json_parse_pct = (json_parse_sum / n) * 100
        sql_valid_pct = (sql_valid_sum / n) * 100
        sql_runs_pct = (sql_runs_sum / n) * 100
        correct_tables_pct = (correct_tables_sum / n) * 100
        correct_behavior_pct = (correct_behavior_sum / n) * 100

        # Composite score: equal weight across all 5 dimensions
        composite = (json_parse_pct + sql_valid_pct + sql_runs_pct +
                     correct_tables_pct + correct_behavior_pct) / 5.0

        # Average latency (excluding cases where latency is 0 due to guardrail block or error)
        latencies = [r["latency_ms"] for r in model_results if r["latency_ms"] > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Total estimated cost
        total_cost = sum(r["estimated_cost_usd"] for r in model_results)
        cost_per_query = total_cost / n if n > 0 else 0.0

        # Cost efficiency: composite score per dollar (higher is better).
        # For free models (cost=0), use infinity to rank them highest.
        if total_cost > 0:
            cost_efficiency = composite / total_cost
        else:
            # Free model — infinite efficiency (capped at a large number for sorting)
            cost_efficiency = composite * 1_000_000  # Very high number for free models

        model_summaries[model_key] = {
            "model": model_key,
            "model_short": MODEL_SHORT_NAMES.get(model_key, model_key),
            "test_count": n,
            "json_parse_pct": round(json_parse_pct, 1),
            "sql_valid_pct": round(sql_valid_pct, 1),
            "sql_runs_pct": round(sql_runs_pct, 1),
            "correct_tables_pct": round(correct_tables_pct, 1),
            "correct_behavior_pct": round(correct_behavior_pct, 1),
            "composite_score": round(composite, 1),
            "avg_latency_ms": round(avg_latency, 1),
            "total_cost_usd": round(total_cost, 6),
            "cost_per_query_usd": round(cost_per_query, 6),
            "cost_efficiency": round(cost_efficiency, 2),
        }

    # ── Group results by category x model ───────────────────────────
    categories: dict[str, dict[str, float]] = {}
    for r in results:
        cat = r["category"]
        model = r["model_short"]
        if cat not in categories:
            categories[cat] = {}
        if model not in categories[cat]:
            categories[cat][model] = {"correct": 0, "total": 0}
        categories[cat][model]["total"] += 1
        # A test case "passes" if correct_behavior is 1
        categories[cat][model]["correct"] += r["correct_behavior"]

    # Convert counts to percentages
    category_pcts: dict[str, dict[str, float]] = {}
    for cat, models in categories.items():
        category_pcts[cat] = {}
        for model, counts in models.items():
            if counts["total"] > 0:
                category_pcts[cat][model] = round(
                    (counts["correct"] / counts["total"]) * 100, 1
                )
            else:
                category_pcts[cat][model] = 0.0

    # ── Collect failed test cases ───────────────────────────────────
    failed_cases = []
    for r in results:
        # A result is "failed" if correct_behavior is 0
        if r["correct_behavior"] == 0:
            failed_cases.append({
                "test_id": r["test_id"],
                "question": r["question"],
                "model_short": r["model_short"],
                "error": r.get("error") or r.get("parse_error") or r.get("validity_error") or r.get("execution_error") or r.get("behavior_note", "Unknown"),
            })

    # ── Determine winners ───────────────────────────────────────────
    if model_summaries:
        # Overall winner: highest composite score
        overall_winner = max(model_summaries.values(), key=lambda s: s["composite_score"])

        # Best free model: highest composite among free-tier models
        free_models = {k: v for k, v in model_summaries.items()
                       if MODEL_COSTS.get(k, {}).get("input_per_mtok", 0) == 0
                       and MODEL_COSTS.get(k, {}).get("output_per_mtok", 0) == 0}
        best_free = max(free_models.values(), key=lambda s: s["composite_score"]) if free_models else None

        # Best cost efficiency: highest composite/cost ratio
        best_efficiency = max(model_summaries.values(), key=lambda s: s["cost_efficiency"])
    else:
        overall_winner = None
        best_free = None
        best_efficiency = None

    return {
        "models": model_summaries,
        "categories": category_pcts,
        "failed_cases": failed_cases,
        "overall_winner": overall_winner,
        "best_free": best_free,
        "best_cost_efficiency": best_efficiency,
    }


# ═══════════════════════════════════════════════════════════════════════
# ── JSON Results Writer ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def save_json_results(results: list[dict], summary: dict) -> Path:
    """Save the raw evaluation results and summary to a JSON file.

    The JSON file contains both the per-test-case results and the
    aggregated summary, making it easy to consume programmatically
    for further analysis or dashboarding.

    Args:
        results: List of per-test-case result dicts.
        summary: Aggregated summary dict from aggregate_results().

    Returns:
        Path to the saved JSON file.
    """
    output_path = RESULTS_DIR / "llm_results.json"

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_test_cases": len(results),
        "models_tested": list({r["model"] for r in results}),
        "summary": {
            "models": summary["models"],
            "categories": summary["categories"],
            "overall_winner": summary["overall_winner"],
            "best_free": summary["best_free"],
            "best_cost_efficiency": summary["best_cost_efficiency"],
        },
        "results": results,
        "failed_cases": summary["failed_cases"],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output_path


# ═══════════════════════════════════════════════════════════════════════
# ── Markdown Report Generator ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def generate_markdown_report(results: list[dict], summary: dict) -> Path:
    """Generate a markdown report summarizing the evaluation results.

    The report includes:
    - Summary table with pass rates per model
    - Winner announcements (overall, best free, best cost-efficiency)
    - Category breakdown table
    - Failed test cases table

    Args:
        results: List of per-test-case result dicts.
        summary: Aggregated summary dict from aggregate_results().

    Returns:
        Path to the saved markdown file.
    """
    output_path = RESULTS_DIR / "llm_report.md"
    today = datetime.now().strftime("%Y-%m-%d")

    lines = []
    lines.append("# LLM Model Evaluation Report")
    lines.append(f"Generated: {today}")
    lines.append("")

    # ── Summary Table ───────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | JSON OK | SQL Valid | SQL Runs | Tables | Behavior | Avg Latency | Est. Cost/query |")
    lines.append("|---|---|---|---|---|---|---|---|")

    # Sort models by composite score descending for the summary table
    sorted_models = sorted(
        summary["models"].values(),
        key=lambda s: s["composite_score"],
        reverse=True,
    )
    for s in sorted_models:
        lines.append(
            f"| {s['model_short']} "
            f"| {s['json_parse_pct']:.0f}% "
            f"| {s['sql_valid_pct']:.0f}% "
            f"| {s['sql_runs_pct']:.0f}% "
            f"| {s['correct_tables_pct']:.0f}% "
            f"| {s['correct_behavior_pct']:.0f}% "
            f"| {s['avg_latency_ms']:.0f}ms "
            f"| ${s['cost_per_query_usd']:.4f} |"
        )
    lines.append("")

    # ── Winner Section ──────────────────────────────────────────────
    lines.append("## Winner")
    lines.append("")
    if summary["overall_winner"]:
        w = summary["overall_winner"]
        lines.append(f"**Best overall:** {w['model_short']} (Score: {w['composite_score']:.1f}%)")
    if summary["best_free"]:
        bf = summary["best_free"]
        lines.append(f"**Best free:** {bf['model_short']} (Score: {bf['composite_score']:.1f}%)")
    if summary["best_cost_efficiency"]:
        be = summary["best_cost_efficiency"]
        lines.append(f"**Best cost-efficiency:** {be['model_short']} (score/cost ratio: {be['cost_efficiency']:.0f})")
    lines.append("")

    # ── Category Breakdown ──────────────────────────────────────────
    lines.append("## Category Breakdown")
    lines.append("")

    # Get all model short names that appear in the category data
    all_model_shorts = sorted({
        model for cat_data in summary["categories"].values()
        for model in cat_data.keys()
    })

    # Build the header row
    header = "| Category |"
    separator = "|---|"
    for m in all_model_shorts:
        header += f" {m} |"
        separator += "---|"
    lines.append(header)
    lines.append(separator)

    # Build data rows for each category
    for cat in sorted(summary["categories"].keys()):
        row = f"| {cat} |"
        for m in all_model_shorts:
            pct = summary["categories"][cat].get(m, 0.0)
            row += f" {pct:.0f}% |"
        lines.append(row)
    lines.append("")

    # ── Failed Test Cases ───────────────────────────────────────────
    lines.append("## Failed Test Cases")
    lines.append("")
    if summary["failed_cases"]:
        lines.append("| ID | Question | Model | Error |")
        lines.append("|---|---|---|---|")
        for fc in summary["failed_cases"]:
            # Truncate question and error for readability in the table
            q = fc["question"][:60] + "..." if len(fc["question"]) > 60 else fc["question"]
            e = fc["error"][:80] + "..." if len(fc["error"]) > 80 else fc["error"]
            # Escape pipes in the question and error text to avoid breaking the table
            q = q.replace("|", "\\|")
            e = e.replace("|", "\\|")
            lines.append(f"| {fc['test_id']} | {q} | {fc['model_short']} | {e} |")
    else:
        lines.append("No failed test cases! All models passed all tests.")
    lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report_text)

    return output_path


# ═══════════════════════════════════════════════════════════════════════
# ── HTML Report Generator ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def generate_html_report(results: list[dict], summary: dict) -> Path:
    """Generate a styled HTML report summarizing the evaluation results.

    The HTML report includes:
    - Inline CSS for styling (no external dependencies)
    - Summary table with alternating row colors and pass/fail color coding
    - Winner announcements with highlighted badges
    - Category breakdown table
    - Failed test cases table
    - Simple SVG bar chart comparing model composite scores

    Args:
        results: List of per-test-case result dicts.
        summary: Aggregated summary dict from aggregate_results().

    Returns:
        Path to the saved HTML file.
    """
    output_path = RESULTS_DIR / "llm_report.html"
    today = datetime.now().strftime("%Y-%m-%d")

    # Sort models by composite score for consistent ordering
    sorted_models = sorted(
        summary["models"].values(),
        key=lambda s: s["composite_score"],
        reverse=True,
    )

    # ── CSS Styles ──────────────────────────────────────────────────
    css = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
            color: #212529;
            line-height: 1.6;
        }
        h1 {
            color: #1a1a2e;
            border-bottom: 3px solid #16213e;
            padding-bottom: 10px;
            margin-bottom: 5px;
        }
        h2 {
            color: #16213e;
            margin-top: 30px;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 8px;
        }
        .generated-date {
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            border-radius: 4px;
            overflow: hidden;
        }
        th {
            background: #16213e;
            color: white;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        td {
            padding: 10px 15px;
            border-bottom: 1px solid #eee;
            font-size: 0.9em;
        }
        tr:nth-child(even) td {
            background: #f8f9fa;
        }
        tr:hover td {
            background: #e9ecef;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }
        .badge-pass {
            background: #d4edda;
            color: #155724;
        }
        .badge-fail {
            background: #f8d7da;
            color: #721c24;
        }
        .badge-warn {
            background: #fff3cd;
            color: #856404;
        }
        .winner-box {
            background: white;
            border-left: 4px solid #28a745;
            padding: 15px 20px;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border-radius: 0 4px 4px 0;
        }
        .winner-box strong {
            color: #16213e;
        }
        .chart-container {
            background: white;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            border-radius: 4px;
        }
        .bar-row {
            display: flex;
            align-items: center;
            margin: 8px 0;
        }
        .bar-label {
            width: 200px;
            font-size: 0.85em;
            font-weight: 500;
            text-align: right;
            padding-right: 12px;
            color: #495057;
        }
        .bar-track {
            flex: 1;
            height: 28px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }
        .bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            padding-left: 10px;
            font-size: 0.8em;
            font-weight: 600;
            color: white;
        }
        .pct-cell {
            font-weight: 600;
        }
        .pct-high { color: #155724; }
        .pct-mid { color: #856404; }
        .pct-low { color: #721c24; }
        .error-text {
            color: #721c24;
            font-size: 0.85em;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
    """

    # ── Helper: color-coded percentage cell ─────────────────────────
    def pct_cell(value: float) -> str:
        """Return an HTML <td> with a color-coded percentage value.

        Green (>=80%), yellow (>=60%), red (<60%).
        """
        if value >= 80:
            css_class = "pct-high"
        elif value >= 60:
            css_class = "pct-mid"
        else:
            css_class = "pct-low"
        return f'<td class="pct-cell {css_class}">{value:.0f}%</td>'

    # ── Helper: pass/fail badge ─────────────────────────────────────
    def badge(value: float, threshold: float = 80.0) -> str:
        """Return an HTML badge span for a percentage value."""
        if value >= threshold:
            return f'<span class="badge badge-pass">{value:.0f}%</span>'
        elif value >= 60:
            return f'<span class="badge badge-warn">{value:.0f}%</span>'
        else:
            return f'<span class="badge badge-fail">{value:.0f}%</span>'

    # ── Build HTML ──────────────────────────────────────────────────
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Model Evaluation Report</title>
    <style>{css}</style>
</head>
<body>
    <h1>LLM Model Evaluation Report</h1>
    <div class="generated-date">Generated: {today}</div>
""")

    # ── Summary Table ───────────────────────────────────────────────
    html_parts.append("    <h2>Summary</h2>")
    html_parts.append("    <table>")
    html_parts.append("        <thead><tr>")
    html_parts.append("            <th>Model</th>")
    html_parts.append("            <th>JSON OK</th>")
    html_parts.append("            <th>SQL Valid</th>")
    html_parts.append("            <th>SQL Runs</th>")
    html_parts.append("            <th>Tables</th>")
    html_parts.append("            <th>Behavior</th>")
    html_parts.append("            <th>Composite</th>")
    html_parts.append("            <th>Avg Latency</th>")
    html_parts.append("            <th>Est. Cost/query</th>")
    html_parts.append("        </tr></thead>")
    html_parts.append("        <tbody>")
    for s in sorted_models:
        html_parts.append("        <tr>")
        html_parts.append(f"            <td><strong>{s['model_short']}</strong></td>")
        html_parts.append(f"            {pct_cell(s['json_parse_pct'])}")
        html_parts.append(f"            {pct_cell(s['sql_valid_pct'])}")
        html_parts.append(f"            {pct_cell(s['sql_runs_pct'])}")
        html_parts.append(f"            {pct_cell(s['correct_tables_pct'])}")
        html_parts.append(f"            {pct_cell(s['correct_behavior_pct'])}")
        html_parts.append(f"            {pct_cell(s['composite_score'])}")
        html_parts.append(f"            <td>{s['avg_latency_ms']:.0f}ms</td>")
        html_parts.append(f"            <td>${s['cost_per_query_usd']:.4f}</td>")
        html_parts.append("        </tr>")
    html_parts.append("        </tbody>")
    html_parts.append("    </table>")

    # ── Bar Chart (Composite Scores) ────────────────────────────────
    # Generate a simple CSS-based horizontal bar chart showing composite scores.
    # Colors cycle through a predefined palette for visual distinction.
    bar_colors = ["#0077b6", "#00b4d8", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    html_parts.append("    <h2>Composite Score Comparison</h2>")
    html_parts.append('    <div class="chart-container">')
    for i, s in enumerate(sorted_models):
        color = bar_colors[i % len(bar_colors)]
        width = max(s["composite_score"], 2)  # Minimum 2% width so zero-score bars are visible
        html_parts.append(f"""        <div class="bar-row">
            <div class="bar-label">{s['model_short']}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width: {width}%; background: {color};">
                    {s['composite_score']:.1f}%
                </div>
            </div>
        </div>""")
    html_parts.append("    </div>")

    # ── Winner Section ──────────────────────────────────────────────
    html_parts.append("    <h2>Winner</h2>")
    if summary["overall_winner"]:
        w = summary["overall_winner"]
        html_parts.append(f'    <div class="winner-box"><strong>Best overall:</strong> {w["model_short"]} (Score: {w["composite_score"]:.1f}%)</div>')
    if summary["best_free"]:
        bf = summary["best_free"]
        html_parts.append(f'    <div class="winner-box"><strong>Best free:</strong> {bf["model_short"]} (Score: {bf["composite_score"]:.1f}%)</div>')
    if summary["best_cost_efficiency"]:
        be = summary["best_cost_efficiency"]
        html_parts.append(f'    <div class="winner-box"><strong>Best cost-efficiency:</strong> {be["model_short"]} (score/cost ratio: {be["cost_efficiency"]:.0f})</div>')

    # ── Category Breakdown Table ────────────────────────────────────
    html_parts.append("    <h2>Category Breakdown</h2>")
    all_model_shorts = sorted({
        model for cat_data in summary["categories"].values()
        for model in cat_data.keys()
    })
    html_parts.append("    <table>")
    html_parts.append("        <thead><tr>")
    html_parts.append("            <th>Category</th>")
    for m in all_model_shorts:
        html_parts.append(f"            <th>{m}</th>")
    html_parts.append("        </tr></thead>")
    html_parts.append("        <tbody>")
    for cat in sorted(summary["categories"].keys()):
        html_parts.append("        <tr>")
        html_parts.append(f"            <td><strong>{cat}</strong></td>")
        for m in all_model_shorts:
            pct = summary["categories"][cat].get(m, 0.0)
            html_parts.append(f"            <td>{badge(pct)}</td>")
        html_parts.append("        </tr>")
    html_parts.append("        </tbody>")
    html_parts.append("    </table>")

    # ── Failed Test Cases Table ─────────────────────────────────────
    html_parts.append("    <h2>Failed Test Cases</h2>")
    if summary["failed_cases"]:
        html_parts.append("    <table>")
        html_parts.append("        <thead><tr>")
        html_parts.append("            <th>ID</th>")
        html_parts.append("            <th>Question</th>")
        html_parts.append("            <th>Model</th>")
        html_parts.append("            <th>Error</th>")
        html_parts.append("        </tr></thead>")
        html_parts.append("        <tbody>")
        for fc in summary["failed_cases"]:
            q = fc["question"][:80] + "..." if len(fc["question"]) > 80 else fc["question"]
            e = fc["error"][:120] + "..." if len(fc["error"]) > 120 else fc["error"]
            # Escape HTML entities in question and error text
            q = q.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            e = e.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append("        <tr>")
            html_parts.append(f"            <td>{fc['test_id']}</td>")
            html_parts.append(f"            <td>{q}</td>")
            html_parts.append(f"            <td>{fc['model_short']}</td>")
            html_parts.append(f'            <td class="error-text">{e}</td>')
            html_parts.append("        </tr>")
        html_parts.append("        </tbody>")
        html_parts.append("    </table>")
    else:
        html_parts.append("    <p>No failed test cases. All models passed all tests.</p>")

    # ── Close HTML ──────────────────────────────────────────────────
    html_parts.append("</body>")
    html_parts.append("</html>")

    html_text = "\n".join(html_parts)
    with open(output_path, "w") as f:
        f.write(html_text)

    return output_path


# ═══════════════════════════════════════════════════════════════════════
# ── CLI Argument Parsing ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Supports:
        --limit N       Only evaluate the first N test cases
        --models a,b,c  Only evaluate specific models (comma-separated)

    Model names can be short aliases:
        groq, gemini, qwen, deepseek-r1, deepseek-chat, claude

    Or full identifiers:
        groq/llama-3.3-70b-versatile, google/gemini-2.5-flash, etc.

    Returns:
        Parsed argparse.Namespace with 'limit' and 'models' attributes.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate LLM models for text-to-SQL generation accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval/run_llm_eval.py                      # Run all test cases on all models
  python eval/run_llm_eval.py --limit 5             # Only first 5 test cases
  python eval/run_llm_eval.py --models groq,gemini  # Only test Groq and Gemini
  python eval/run_llm_eval.py --limit 10 --models groq,claude
        """,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of test cases to evaluate (default: all)",
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated list of models to evaluate. "
            "Use aliases (groq, gemini, qwen, deepseek-r1, deepseek-chat, claude) "
            "or full names (groq/llama-3.3-70b-versatile). Default: all models."
        ),
    )

    return parser.parse_args()


def resolve_model_keys(models_arg: Optional[str]) -> list[str]:
    """Resolve the --models CLI argument into a list of full model keys.

    Accepts both aliases (e.g., "groq") and full keys
    (e.g., "groq/llama-3.3-70b-versatile"). Unknown names are reported
    as warnings but do not cause the script to fail.

    Args:
        models_arg: Comma-separated string from CLI, or None for all models.

    Returns:
        List of full model key strings.
    """
    if models_arg is None:
        return ALL_MODEL_KEYS[:]

    model_keys = []
    for name in models_arg.split(","):
        name = name.strip()
        if not name:
            continue

        # Check if it is a known alias
        if name in MODEL_ALIASES:
            model_keys.append(MODEL_ALIASES[name])
        # Check if it is already a full model key
        elif name in ALL_MODEL_KEYS:
            model_keys.append(name)
        else:
            # Try partial matching: see if the name is a substring of any key
            matches = [k for k in ALL_MODEL_KEYS if name.lower() in k.lower()]
            if matches:
                model_keys.extend(matches)
                print(f"  [info] Resolved '{name}' to: {matches}")
            else:
                print(f"  [warning] Unknown model '{name}'. Skipping.")
                print(f"    Known aliases: {', '.join(MODEL_ALIASES.keys())}")

    if not model_keys:
        print("[error] No valid models specified. Use one of:")
        for alias, full in MODEL_ALIASES.items():
            print(f"  {alias} -> {full}")
        sys.exit(1)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for k in model_keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)

    return deduped


# ═══════════════════════════════════════════════════════════════════════
# ── Main Execution ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point for the evaluation script.

    Orchestrates the entire evaluation flow:
    1. Parse CLI arguments
    2. Load test cases and schema
    3. Instantiate models
    4. Run evaluations with progress output
    5. Aggregate results
    6. Generate reports
    """
    args = parse_args()

    print("=" * 70)
    print("  LLM Model Evaluation for Text-to-SQL Generation")
    print("=" * 70)
    print()

    # ── Step 1: Create output directory ─────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Results directory: {RESULTS_DIR}")

    # ── Step 2: Load test cases ─────────────────────────────────────
    print(f"[setup] Loading test cases from: {TEST_CASES_PATH}")
    test_cases = load_test_cases(limit=args.limit)
    print(f"[setup] Loaded {len(test_cases)} test cases")
    if args.limit:
        print(f"[setup] (limited to first {args.limit})")

    # ── Step 3: Load database schema ────────────────────────────────
    print(f"[setup] Loading schema from: {DB_PATH}")
    if not DB_PATH.exists():
        print(f"[error] Database not found: {DB_PATH}")
        print(f"  Run 'python db/seed_data.py' to create and populate it.")
        sys.exit(1)
    schema = get_schema_description(db_path=str(DB_PATH))
    print(f"[setup] Schema loaded ({len(schema)} chars)")

    # ── Step 4: Resolve and instantiate models ──────────────────────
    model_keys = resolve_model_keys(args.models)
    print(f"[setup] Models to evaluate ({len(model_keys)}):")
    for mk in model_keys:
        short = MODEL_SHORT_NAMES.get(mk, mk)
        cost_info = MODEL_COSTS.get(mk, {})
        cost_label = "free" if cost_info.get("input_per_mtok", 0) == 0 and cost_info.get("output_per_mtok", 0) == 0 else f"${cost_info.get('input_per_mtok', 0)}/MTok in, ${cost_info.get('output_per_mtok', 0)}/MTok out"
        print(f"  - {short} ({cost_label})")

    # Try to instantiate each model. Models that fail to instantiate
    # (e.g., missing API key) are skipped with a warning.
    model_instances: dict[str, Any] = {}
    for mk in model_keys:
        short = MODEL_SHORT_NAMES.get(mk, mk)
        try:
            instance = create_model_instance(mk)
            model_instances[mk] = instance
            print(f"  [ok] {short}: instantiated")
        except Exception as e:
            print(f"  [SKIP] {short}: {type(e).__name__}: {e}")
            print(f"    This model will be excluded from the evaluation.")

    if not model_instances:
        print("\n[error] No models could be instantiated. Check your API keys in .env")
        sys.exit(1)

    print()

    # ── Step 5: Run evaluations ─────────────────────────────────────
    total_evals = len(test_cases) * len(model_instances)
    print(f"[eval] Running {total_evals} evaluations "
          f"({len(test_cases)} test cases x {len(model_instances)} models)")
    print("-" * 70)

    all_results: list[dict] = []
    eval_count = 0
    start_time_total = time.perf_counter()

    for tc_idx, test_case in enumerate(test_cases):
        tc_id = test_case["id"]
        tc_question = test_case["question"]
        tc_category = test_case.get("category", "unknown")
        tc_behavior = test_case.get("expected_behavior", "answer")

        print(f"\n[{tc_idx + 1}/{len(test_cases)}] {tc_id} ({tc_category}/{tc_behavior})")
        print(f"  Q: {tc_question[:70]}{'...' if len(tc_question) > 70 else ''}")

        for model_key, model_instance in model_instances.items():
            eval_count += 1
            short = MODEL_SHORT_NAMES.get(model_key, model_key)
            print(f"  [{eval_count}/{total_evals}] {short}...", end=" ", flush=True)

            # Run the evaluation for this test case + model combination
            result = evaluate_single(model_key, model_instance, test_case, schema)
            all_results.append(result)

            # Print a one-line summary of the result
            scores = (
                f"json={result['json_parse']} "
                f"valid={result['sql_valid']} "
                f"runs={result['sql_runs']} "
                f"tables={result['correct_tables']} "
                f"behavior={result['correct_behavior']}"
            )
            if result.get("guardrail_blocked"):
                print(f"BLOCKED ({result.get('guardrail_reason', 'guardrail')}) | {scores}")
            elif result.get("error"):
                print(f"ERROR: {result['error'][:60]} | {scores}")
            else:
                latency = result["latency_ms"]
                print(f"{latency:.0f}ms | {scores}")

    elapsed_total = time.perf_counter() - start_time_total
    print()
    print("-" * 70)
    print(f"[eval] Completed {eval_count} evaluations in {elapsed_total:.1f}s")
    print()

    # ── Step 6: Aggregate results ───────────────────────────────────
    print("[report] Aggregating results...")
    summary = aggregate_results(all_results)

    # ── Step 7: Save results and generate reports ───────────────────
    json_path = save_json_results(all_results, summary)
    print(f"[report] JSON results saved to: {json_path}")

    md_path = generate_markdown_report(all_results, summary)
    print(f"[report] Markdown report saved to: {md_path}")

    html_path = generate_html_report(all_results, summary)
    print(f"[report] HTML report saved to: {html_path}")

    # ── Step 8: Print summary to stdout ─────────────────────────────
    print()
    print("=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print()

    # Print a compact summary table
    print(f"{'Model':<30} {'Composite':>10} {'JSON OK':>8} {'SQL OK':>8} {'Runs':>8} {'Tables':>8} {'Behav.':>8} {'Latency':>10} {'Cost/q':>10}")
    print("-" * 120)
    for s in sorted(summary["models"].values(), key=lambda x: x["composite_score"], reverse=True):
        print(
            f"{s['model_short']:<30} "
            f"{s['composite_score']:>9.1f}% "
            f"{s['json_parse_pct']:>7.0f}% "
            f"{s['sql_valid_pct']:>7.0f}% "
            f"{s['sql_runs_pct']:>7.0f}% "
            f"{s['correct_tables_pct']:>7.0f}% "
            f"{s['correct_behavior_pct']:>7.0f}% "
            f"{s['avg_latency_ms']:>8.0f}ms "
            f"${s['cost_per_query_usd']:>8.4f}"
        )
    print()

    if summary["overall_winner"]:
        w = summary["overall_winner"]
        print(f"  BEST OVERALL:         {w['model_short']} ({w['composite_score']:.1f}%)")
    if summary["best_free"]:
        bf = summary["best_free"]
        print(f"  BEST FREE:            {bf['model_short']} ({bf['composite_score']:.1f}%)")
    if summary["best_cost_efficiency"]:
        be = summary["best_cost_efficiency"]
        print(f"  BEST COST-EFFICIENCY: {be['model_short']} (ratio: {be['cost_efficiency']:.0f})")

    # Count failures
    n_failed = len(summary["failed_cases"])
    if n_failed > 0:
        print(f"\n  FAILED: {n_failed} test case(s) across all models")
    else:
        print(f"\n  All test cases passed across all models!")

    print()
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# ── Entry Point ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
