#!/usr/bin/env python3
"""Embedding Model Evaluation for Schema Retrieval Accuracy.

This script evaluates different embedding models to determine which one best
retrieves the correct database tables from ChromaDB given a natural language
question. It is the first step in optimizing the vector-search pipeline that
powers the word-to-SQL system.

How it works:
  1. Loads test cases from eval/test_cases.yaml.  Each test case has a natural
     language question, expected_behavior ("answer", "refuse", or "block"), and
     for "answer" cases a list of expected_tables that the retrieval should
     surface.
  2. For every embedding model candidate, builds a fresh ChromaDB collection
     from the semantic-layer data (TABLE_DESCRIPTIONS + RELATIONSHIP_DESCRIPTIONS).
  3. Queries each "answer" test case against the collection (top-5 retrieval).
  4. Scores results using Recall@5, Precision@5, MRR, and retrieval latency.
  5. Writes structured JSON results to eval/results/embedding_results.json.
  6. Generates a Markdown report at eval/results/embedding_report.md and an
     HTML report at eval/results/embedding_report.html.

Embedding models evaluated (all sentence-transformers, run locally):
  - all-MiniLM-L6-v2          (384 dim) -- ChromaDB default
  - all-mpnet-base-v2         (768 dim) -- best general-purpose open-source
  - paraphrase-MiniLM-L6-v2   (384 dim) -- optimised for paraphrase detection
  - multi-qa-MiniLM-L6-cos-v1 (384 dim) -- optimised for QA / retrieval

Usage:
    python eval/run_embedding_eval.py                    # run all models, all test cases
    python eval/run_embedding_eval.py --limit 50         # cap at 50 test cases
    python eval/run_embedding_eval.py --models miniLM    # run only models matching "miniLM"
    python eval/run_embedding_eval.py --models all       # explicit: run all models

Requirements:
    pip install sentence-transformers chromadb pyyaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import src.knowledge.*
# even when running this script directly from the eval/ directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.knowledge.semantic_layer import (  # noqa: E402
    RELATIONSHIP_DESCRIPTIONS,
    TABLE_DESCRIPTIONS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Directory that holds this script and test_cases.yaml
EVAL_DIR = Path(__file__).resolve().parent

# Where results are written
RESULTS_DIR = EVAL_DIR / "results"

# Path to the test-case definitions
TEST_CASES_PATH = EVAL_DIR / "test_cases.yaml"

# Default number of results returned by each ChromaDB similarity query
TOP_K = 5

# ---------------------------------------------------------------------------
# Embedding model registry
# ---------------------------------------------------------------------------
# Each entry describes a candidate model for evaluation.
# "name" is the sentence-transformers model name (passed directly to
# SentenceTransformerEmbeddingFunction).
# "dimensions" is informational -- used in the report, not at runtime.

EMBEDDING_MODELS: list[dict[str, Any]] = [
    {
        "name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "description": "ChromaDB default; lightweight, fast, reasonable quality",
    },
    {
        "name": "all-mpnet-base-v2",
        "dimensions": 768,
        "description": "Best general-purpose open-source model; larger but more accurate",
    },
    {
        "name": "paraphrase-MiniLM-L6-v2",
        "dimensions": 384,
        "description": "Optimised for paraphrase detection; good at semantic similarity",
    },
    {
        "name": "multi-qa-MiniLM-L6-cos-v1",
        "dimensions": 384,
        "description": "Optimised for QA and retrieval; trained on question-answer pairs",
    },
]


# ===================================================================
# Helper: install sentence-transformers if missing
# ===================================================================

def _ensure_sentence_transformers() -> None:
    """Attempt to import sentence-transformers; install it if absent.

    This is a convenience for first-time runs so the evaluator does not
    fail with an opaque ImportError.  In CI you would pin the dependency
    in requirements-eval.txt instead.
    """
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("[setup] sentence-transformers not found -- installing ...")
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "sentence-transformers"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[setup] sentence-transformers installed successfully.")


# ===================================================================
# Test-case loading
# ===================================================================

def load_test_cases(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load test cases from the YAML file and filter to 'answer' cases only.

    Test cases where expected_behavior is "refuse" or "block" are skipped
    because they have no expected_tables to evaluate retrieval against.

    Args:
        path:  Absolute path to eval/test_cases.yaml.
        limit: If set, return at most this many test cases.

    Returns:
        A list of dicts, each containing at minimum:
          - id (str)
          - question (str)
          - expected_tables (list[str])
          - category (str, optional)
    """
    if not path.exists():
        print(f"[error] Test cases file not found: {path}")
        print("        Create eval/test_cases.yaml with your evaluation questions.")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # The YAML structure is expected to be either:
    #   test_cases:
    #     - id: ...
    #       question: ...
    # or a bare list at top level.
    if isinstance(raw, dict) and "test_cases" in raw:
        cases = raw["test_cases"]
    elif isinstance(raw, list):
        cases = raw
    else:
        print("[error] Unexpected YAML structure in test_cases.yaml.")
        sys.exit(1)

    # Keep only cases where expected_behavior == "answer" (or where the
    # field is absent, which we treat as "answer" by default).
    filtered: list[dict[str, Any]] = []
    for case in cases:
        behaviour = case.get("expected_behavior", "answer")
        if behaviour not in ("answer",):
            continue
        # Validate required fields
        if "question" not in case:
            print(f"[warn] Skipping test case without 'question': {case.get('id', '?')}")
            continue
        if "expected_tables" not in case or not case["expected_tables"]:
            print(f"[warn] Skipping test case without 'expected_tables': {case.get('id', '?')}")
            continue
        filtered.append(case)

    # Apply limit
    if limit is not None and limit > 0:
        filtered = filtered[:limit]

    print(f"[info] Loaded {len(filtered)} 'answer' test cases from {path.name}")
    return filtered


# ===================================================================
# Semantic-layer document preparation
# ===================================================================

def prepare_documents() -> list[dict[str, Any]]:
    """Merge TABLE_DESCRIPTIONS and RELATIONSHIP_DESCRIPTIONS into a single
    list of documents ready for ChromaDB ingestion.

    Each document has keys: id, text, metadata.

    Returns:
        Combined list of all semantic-layer documents.
    """
    all_docs = TABLE_DESCRIPTIONS + RELATIONSHIP_DESCRIPTIONS
    print(f"[info] Prepared {len(all_docs)} semantic-layer documents for embedding "
          f"({len(TABLE_DESCRIPTIONS)} tables + {len(RELATIONSHIP_DESCRIPTIONS)} relationships)")
    return all_docs


# ===================================================================
# Collection building
# ===================================================================

def build_collection(
    model_name: str,
    documents: list[dict[str, Any]],
) -> "chromadb.Collection":  # type: ignore[name-defined]
    """Create a fresh in-memory ChromaDB collection using the specified
    sentence-transformers embedding model and populate it with the
    semantic-layer documents.

    Args:
        model_name: A sentence-transformers model identifier
                    (e.g. "all-MiniLM-L6-v2").
        documents:  List of dicts with id / text / metadata keys.

    Returns:
        A populated chromadb.Collection ready for querying.
    """
    import chromadb
    from chromadb.utils import embedding_functions

    # Create a sentence-transformers embedding function for this model.
    # ChromaDB will use it to embed documents on add() and queries on query().
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
    )

    # EphemeralClient stores everything in memory -- no disk artefacts.
    client = chromadb.EphemeralClient()

    # Use a sanitised collection name (ChromaDB names must match [a-zA-Z0-9_-]).
    safe_name = model_name.replace("/", "_").replace(".", "_")
    collection = client.create_collection(
        name=safe_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Add all semantic-layer documents.
    collection.add(
        ids=[doc["id"] for doc in documents],
        documents=[doc["text"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents],
    )

    return collection


# ===================================================================
# Table-name extraction from ChromaDB results
# ===================================================================

def extract_table_names(results: dict) -> list[str]:
    """Extract an ordered list of table names from ChromaDB query results.

    Table descriptions carry a "table" key in metadata.  Relationship
    descriptions carry "from_table" and "to_table".  We collect all of
    them in the order they appear (preserving rank) and deduplicate.

    Args:
        results: Raw dict returned by collection.query().

    Returns:
        Ordered list of unique table names (first occurrence order).
    """
    seen: set[str] = set()
    ordered: list[str] = []

    if not results["metadatas"] or not results["metadatas"][0]:
        return ordered

    for metadata in results["metadatas"][0]:
        # Table-description documents have a "table" key.
        if "table" in metadata:
            tbl = metadata["table"]
            if tbl not in seen:
                seen.add(tbl)
                ordered.append(tbl)
        # Relationship-description documents have from_table / to_table.
        if "from_table" in metadata:
            tbl = metadata["from_table"]
            if tbl not in seen:
                seen.add(tbl)
                ordered.append(tbl)
        if "to_table" in metadata:
            tbl = metadata["to_table"]
            if tbl not in seen:
                seen.add(tbl)
                ordered.append(tbl)

    return ordered


# ===================================================================
# Per-question evaluation
# ===================================================================

def evaluate_question(
    collection: "chromadb.Collection",  # type: ignore[name-defined]
    question: str,
    expected_tables: list[str],
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Run a single retrieval query and score the results.

    Metrics computed:
      - recall:  1.0 if ALL expected_tables are in the retrieved set, else 0.0.
                 (This is Recall@K in a set-retrieval sense.)
      - precision: fraction of retrieved tables that are in expected_tables.
      - reciprocal_rank: 1 / (rank of the first relevant result).  If no
                         relevant table is found, reciprocal_rank = 0.
      - retrieval_time_ms: wall-clock time for the ChromaDB query.

    Args:
        collection:      A populated ChromaDB collection.
        question:        Natural-language query string.
        expected_tables: Ground-truth table names that should be retrieved.
        top_k:           Number of results to retrieve.

    Returns:
        Dict with keys: retrieved_tables, recall, precision, reciprocal_rank,
        retrieval_time_ms, hit (bool -- all expected tables found).
    """
    # -- Run the similarity search and time it -------------------------
    start = time.perf_counter()
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # -- Extract the table names from the results ----------------------
    retrieved_tables = extract_table_names(results)

    # -- Recall: did we retrieve ALL expected tables? ------------------
    expected_set = set(expected_tables)
    retrieved_set = set(retrieved_tables)
    all_found = expected_set.issubset(retrieved_set)
    recall = 1.0 if all_found else 0.0

    # -- Precision: what fraction of retrieved tables are relevant? ----
    if retrieved_tables:
        n_relevant_retrieved = len(expected_set & retrieved_set)
        precision = n_relevant_retrieved / len(retrieved_tables)
    else:
        precision = 0.0

    # -- MRR: reciprocal rank of the FIRST relevant table --------------
    reciprocal_rank = 0.0
    for rank_idx, tbl in enumerate(retrieved_tables, start=1):
        if tbl in expected_set:
            reciprocal_rank = 1.0 / rank_idx
            break

    return {
        "retrieved_tables": retrieved_tables,
        "recall": recall,
        "precision": precision,
        "reciprocal_rank": reciprocal_rank,
        "retrieval_time_ms": round(elapsed_ms, 3),
        "hit": all_found,
    }


# ===================================================================
# Full evaluation loop for one model
# ===================================================================

def evaluate_model(
    model_name: str,
    documents: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a single embedding model against all test cases.

    Builds a fresh ChromaDB collection, runs every test question, and
    aggregates the metrics.

    Args:
        model_name: Sentence-transformers model identifier.
        documents:  Semantic-layer documents to embed.
        test_cases: Filtered list of 'answer' test cases.

    Returns:
        Dict with keys:
          model, dimensions, description,
          recall_at_5, precision_at_5, mrr, avg_retrieval_time_ms,
          total_test_cases, hits, misses,
          per_question (list of per-question detail dicts),
          category_recall (dict mapping category -> recall).
    """
    # Find model metadata (dimensions, description) from the registry.
    model_info = next(
        (m for m in EMBEDDING_MODELS if m["name"] == model_name),
        {"name": model_name, "dimensions": "?", "description": ""},
    )

    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {model_name} ({model_info['dimensions']} dim)")
    print(f"{'=' * 60}")

    # -- Build the collection (embeds all documents) -------------------
    print(f"  [1/2] Building ChromaDB collection ...")
    build_start = time.perf_counter()
    collection = build_collection(model_name, documents)
    build_time = (time.perf_counter() - build_start) * 1000.0
    print(f"        Done in {build_time:.0f} ms")

    # -- Evaluate every test case --------------------------------------
    print(f"  [2/2] Running {len(test_cases)} test queries ...")
    per_question: list[dict[str, Any]] = []
    total_recall = 0.0
    total_precision = 0.0
    total_rr = 0.0
    total_time = 0.0
    hits = 0

    # Track per-category recall for the category breakdown.
    category_hits: dict[str, int] = {}
    category_totals: dict[str, int] = {}

    for idx, tc in enumerate(test_cases):
        question = tc["question"]
        expected_tables = tc["expected_tables"]
        category = tc.get("category", "uncategorized")

        result = evaluate_question(collection, question, expected_tables)

        # Accumulate totals
        total_recall += result["recall"]
        total_precision += result["precision"]
        total_rr += result["reciprocal_rank"]
        total_time += result["retrieval_time_ms"]
        if result["hit"]:
            hits += 1

        # Category tracking
        category_totals[category] = category_totals.get(category, 0) + 1
        if result["hit"]:
            category_hits[category] = category_hits.get(category, 0) + 1

        # Build per-question record
        per_question.append({
            "id": tc.get("id", f"q_{idx}"),
            "question": question,
            "category": category,
            "expected_tables": expected_tables,
            "retrieved_tables": result["retrieved_tables"],
            "recall": result["recall"],
            "precision": result["precision"],
            "reciprocal_rank": result["reciprocal_rank"],
            "retrieval_time_ms": result["retrieval_time_ms"],
            "hit": result["hit"],
        })

        # Progress indicator (every 10 questions)
        if (idx + 1) % 10 == 0 or idx + 1 == len(test_cases):
            print(f"        ... {idx + 1}/{len(test_cases)} done")

    # -- Aggregate metrics ---------------------------------------------
    n = len(test_cases)
    recall_at_5 = total_recall / n if n else 0.0
    precision_at_5 = total_precision / n if n else 0.0
    mrr = total_rr / n if n else 0.0
    avg_time = total_time / n if n else 0.0

    # Per-category recall (fraction of hits in each category)
    category_recall: dict[str, float] = {}
    for cat in sorted(category_totals.keys()):
        cat_total = category_totals[cat]
        cat_hits = category_hits.get(cat, 0)
        category_recall[cat] = cat_hits / cat_total if cat_total else 0.0

    print(f"\n  Results: Recall@5={recall_at_5:.1%}  Precision@5={precision_at_5:.1%}  "
          f"MRR={mrr:.3f}  AvgTime={avg_time:.1f}ms")

    return {
        "model": model_name,
        "dimensions": model_info["dimensions"],
        "description": model_info["description"],
        "recall_at_5": round(recall_at_5, 4),
        "precision_at_5": round(precision_at_5, 4),
        "mrr": round(mrr, 4),
        "avg_retrieval_time_ms": round(avg_time, 3),
        "total_test_cases": n,
        "hits": hits,
        "misses": n - hits,
        "per_question": per_question,
        "category_recall": category_recall,
        "collection_build_time_ms": round(build_time, 1),
    }


# ===================================================================
# Result persistence (JSON)
# ===================================================================

def save_results_json(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write the full evaluation results to a JSON file.

    The JSON file contains every per-model and per-question metric so it
    can be loaded by downstream analysis scripts or dashboards.

    Args:
        results:     List of per-model result dicts.
        output_path: Absolute path to the output JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_k": TOP_K,
        "models_evaluated": len(results),
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\n[output] JSON results saved to {output_path}")


# ===================================================================
# Markdown report generation
# ===================================================================

def _pct(value: float) -> str:
    """Format a 0-1 float as a percentage string like '85.0%'."""
    return f"{value * 100:.1f}%"


def generate_markdown_report(
    results: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a human-readable Markdown evaluation report.

    The report contains:
      - A summary table comparing all models.
      - A "Winner" section highlighting the best model by each metric.
      - A category-breakdown table showing per-category recall.
      - A failure-analysis section listing questions that no model got right.

    Args:
        results:     List of per-model result dicts from evaluate_model().
        test_cases:  The original list of filtered test cases.
        output_path: Where to write the .md file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    # -- Header --------------------------------------------------------
    lines.append("# Embedding Model Evaluation Report")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append(f"Top-K: {TOP_K} | Test cases: {len(test_cases)}")
    lines.append("")

    # -- Summary table -------------------------------------------------
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Dimensions | Recall@5 | Precision@5 | MRR | Avg Time |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['model']} | {r['dimensions']} | "
            f"{_pct(r['recall_at_5'])} | {_pct(r['precision_at_5'])} | "
            f"{r['mrr']:.3f} | {r['avg_retrieval_time_ms']:.1f}ms |"
        )
    lines.append("")

    # -- Winner section ------------------------------------------------
    lines.append("## Winner")
    lines.append("")

    # Best retrieval accuracy (highest recall)
    best_recall = max(results, key=lambda r: r["recall_at_5"])
    lines.append(f"**Best retrieval accuracy:** {best_recall['model']} "
                 f"(Recall@5 = {_pct(best_recall['recall_at_5'])})")

    # Best speed (lowest avg time)
    best_speed = min(results, key=lambda r: r["avg_retrieval_time_ms"])
    lines.append(f"**Best speed:** {best_speed['model']} "
                 f"(Avg = {best_speed['avg_retrieval_time_ms']:.1f}ms)")

    # Best balance: highest (recall * 0.7 + mrr * 0.2 + speed_score * 0.1)
    # Speed score: normalise to 0-1 where fastest=1.
    max_time = max(r["avg_retrieval_time_ms"] for r in results) or 1.0
    def _balance_score(r: dict) -> float:
        speed_norm = 1.0 - (r["avg_retrieval_time_ms"] / max_time) if max_time > 0 else 0.5
        return r["recall_at_5"] * 0.7 + r["mrr"] * 0.2 + speed_norm * 0.1

    best_balance = max(results, key=_balance_score)
    lines.append(f"**Best balance:** {best_balance['model']} "
                 f"(weighted score = {_balance_score(best_balance):.3f})")
    lines.append("")

    # -- Category breakdown --------------------------------------------
    # Gather all categories across all models.
    all_categories: set[str] = set()
    for r in results:
        all_categories.update(r["category_recall"].keys())
    sorted_categories = sorted(all_categories)

    if sorted_categories:
        lines.append("## Category Breakdown")
        lines.append("")
        # Build header row: Category | model1 | model2 | ...
        short_names = [r["model"].split("/")[-1] for r in results]
        header = "| Category | " + " | ".join(short_names) + " |"
        sep = "|---| " + " | ".join(["---"] * len(results)) + " |"
        lines.append(header)
        lines.append(sep)

        for cat in sorted_categories:
            row = f"| {cat} "
            for r in results:
                val = r["category_recall"].get(cat, 0.0)
                row += f"| {_pct(val)} "
            row += "|"
            lines.append(row)
        lines.append("")

    # -- Failure analysis ----------------------------------------------
    # A "universal failure" is a question that NO model retrieved correctly.
    lines.append("## Failure Analysis")
    lines.append("")
    lines.append("Questions where **no** model retrieved the correct tables:")
    lines.append("")

    # Build a lookup: question_id -> {model: per_question_result}
    question_model_results: dict[str, dict[str, dict]] = {}
    for r in results:
        for pq in r["per_question"]:
            qid = pq["id"]
            if qid not in question_model_results:
                question_model_results[qid] = {}
            question_model_results[qid][r["model"]] = pq

    # Find universal failures (no model hit)
    universal_failures: list[dict[str, Any]] = []
    for tc in test_cases:
        qid = tc.get("id", "?")
        model_results_map = question_model_results.get(qid, {})
        if not model_results_map:
            continue
        any_hit = any(mr.get("hit", False) for mr in model_results_map.values())
        if not any_hit:
            # Find the best retrieval (most expected tables found)
            best_retrieval: list[str] = []
            best_overlap = 0
            expected_set = set(tc["expected_tables"])
            for mr in model_results_map.values():
                overlap = len(expected_set & set(mr.get("retrieved_tables", [])))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_retrieval = mr.get("retrieved_tables", [])
            universal_failures.append({
                "id": qid,
                "question": tc["question"],
                "expected_tables": tc["expected_tables"],
                "best_retrieval": best_retrieval,
            })

    if universal_failures:
        lines.append("| ID | Question | Expected Tables | Best Retrieval |")
        lines.append("|---|---|---|---|")
        for f in universal_failures:
            lines.append(
                f"| {f['id']} "
                f"| {f['question'][:80]}{'...' if len(f['question']) > 80 else ''} "
                f"| {', '.join(f['expected_tables'])} "
                f"| {', '.join(f['best_retrieval'])} |"
            )
    else:
        lines.append("*None -- at least one model retrieved the correct tables for every question.*")
    lines.append("")

    # -- Write ---------------------------------------------------------
    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"[output] Markdown report saved to {output_path}")


# ===================================================================
# HTML report generation
# ===================================================================

def _recall_badge_html(value: float) -> str:
    """Return an HTML badge <span> with green/yellow/red colouring
    based on the recall value.

    Thresholds:
      >= 0.90  -> green
      >= 0.70  -> yellow
      <  0.70  -> red

    Args:
        value: Recall fraction in [0, 1].

    Returns:
        HTML string for a styled badge.
    """
    pct_text = f"{value * 100:.1f}%"
    if value >= 0.90:
        color = "#28a745"  # green
        bg = "#d4edda"
    elif value >= 0.70:
        color = "#856404"  # dark-yellow
        bg = "#fff3cd"
    else:
        color = "#721c24"  # red
        bg = "#f8d7da"
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'background:{bg};color:{color};font-weight:600;">{pct_text}</span>'
    )


def generate_html_report(
    results: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a styled HTML evaluation report.

    Contains the same data as the Markdown report but with CSS styling,
    colour-coded recall badges, and a more visual layout.

    Args:
        results:     List of per-model result dicts.
        test_cases:  Filtered test cases.
        output_path: Where to write the .html file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Collect all categories ----------------------------------------
    all_categories: set[str] = set()
    for r in results:
        all_categories.update(r["category_recall"].keys())
    sorted_categories = sorted(all_categories)

    # -- Failure analysis (same logic as markdown) ---------------------
    question_model_results: dict[str, dict[str, dict]] = {}
    for r in results:
        for pq in r["per_question"]:
            qid = pq["id"]
            if qid not in question_model_results:
                question_model_results[qid] = {}
            question_model_results[qid][r["model"]] = pq

    universal_failures: list[dict[str, Any]] = []
    for tc in test_cases:
        qid = tc.get("id", "?")
        model_results_map = question_model_results.get(qid, {})
        if not model_results_map:
            continue
        any_hit = any(mr.get("hit", False) for mr in model_results_map.values())
        if not any_hit:
            best_retrieval: list[str] = []
            best_overlap = 0
            expected_set = set(tc["expected_tables"])
            for mr in model_results_map.values():
                overlap = len(expected_set & set(mr.get("retrieved_tables", [])))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_retrieval = mr.get("retrieved_tables", [])
            universal_failures.append({
                "id": qid,
                "question": tc["question"],
                "expected_tables": tc["expected_tables"],
                "best_retrieval": best_retrieval,
            })

    # -- Winners -------------------------------------------------------
    best_recall = max(results, key=lambda r: r["recall_at_5"])
    best_speed = min(results, key=lambda r: r["avg_retrieval_time_ms"])
    max_time = max(r["avg_retrieval_time_ms"] for r in results) or 1.0

    def _balance_score(r: dict) -> float:
        speed_norm = 1.0 - (r["avg_retrieval_time_ms"] / max_time) if max_time > 0 else 0.5
        return r["recall_at_5"] * 0.7 + r["mrr"] * 0.2 + speed_norm * 0.1

    best_balance = max(results, key=_balance_score)

    # -- Build HTML ----------------------------------------------------
    html_parts: list[str] = []

    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Embedding Model Evaluation Report</title>
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        max-width: 1100px;
        margin: 0 auto;
        padding: 20px 40px;
        color: #333;
        background: #fafafa;
    }
    h1 { color: #1a1a2e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; }
    h2 { color: #16213e; margin-top: 30px; }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 12px 0 24px 0;
        background: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    th {
        background: #16213e;
        color: #fff;
        padding: 10px 14px;
        text-align: left;
        font-size: 0.9em;
    }
    td {
        padding: 8px 14px;
        border-bottom: 1px solid #e8e8e8;
        font-size: 0.9em;
    }
    tr:hover td { background: #f0f4ff; }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .badge-green  { background: #d4edda; color: #28a745; }
    .badge-yellow { background: #fff3cd; color: #856404; }
    .badge-red    { background: #f8d7da; color: #721c24; }
    .winner-card {
        display: inline-block;
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 12px 20px;
        margin: 6px 8px 6px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .winner-card strong { color: #0f3460; }
    .meta { color: #888; font-size: 0.85em; }
    code { background: #eef; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
    .truncate { max-width: 350px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
</head>
<body>
""")

    # Title
    gen_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    html_parts.append(f"<h1>Embedding Model Evaluation Report</h1>")
    html_parts.append(f'<p class="meta">Generated: {gen_time} | Top-K: {TOP_K} | '
                      f'Test cases: {len(test_cases)}</p>')

    # -- Summary table -------------------------------------------------
    html_parts.append("<h2>Summary</h2>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Model</th><th>Dimensions</th><th>Recall@5</th>"
                      "<th>Precision@5</th><th>MRR</th><th>Avg Time</th><th>Hits/Total</th></tr>")
    for r in results:
        recall_badge = _recall_badge_html(r["recall_at_5"])
        precision_badge = _recall_badge_html(r["precision_at_5"])
        html_parts.append(
            f"<tr>"
            f"<td><code>{r['model']}</code></td>"
            f"<td>{r['dimensions']}</td>"
            f"<td>{recall_badge}</td>"
            f"<td>{precision_badge}</td>"
            f"<td>{r['mrr']:.3f}</td>"
            f"<td>{r['avg_retrieval_time_ms']:.1f}ms</td>"
            f"<td>{r['hits']}/{r['total_test_cases']}</td>"
            f"</tr>"
        )
    html_parts.append("</table>")

    # -- Winners -------------------------------------------------------
    html_parts.append("<h2>Winner</h2>")
    html_parts.append('<div>')
    html_parts.append(
        f'<div class="winner-card"><strong>Best Retrieval Accuracy:</strong><br>'
        f'<code>{best_recall["model"]}</code> '
        f'(Recall@5 = {_pct(best_recall["recall_at_5"])})</div>'
    )
    html_parts.append(
        f'<div class="winner-card"><strong>Best Speed:</strong><br>'
        f'<code>{best_speed["model"]}</code> '
        f'(Avg = {best_speed["avg_retrieval_time_ms"]:.1f}ms)</div>'
    )
    html_parts.append(
        f'<div class="winner-card"><strong>Best Balance:</strong><br>'
        f'<code>{best_balance["model"]}</code> '
        f'(score = {_balance_score(best_balance):.3f})</div>'
    )
    html_parts.append('</div>')

    # -- Category breakdown --------------------------------------------
    if sorted_categories:
        html_parts.append("<h2>Category Breakdown (Recall@5)</h2>")
        html_parts.append("<table>")
        header = "<tr><th>Category</th>"
        for r in results:
            header += f"<th>{r['model'].split('/')[-1]}</th>"
        header += "</tr>"
        html_parts.append(header)

        for cat in sorted_categories:
            row = f"<tr><td><code>{cat}</code></td>"
            for r in results:
                val = r["category_recall"].get(cat, 0.0)
                badge = _recall_badge_html(val)
                row += f"<td>{badge}</td>"
            row += "</tr>"
            html_parts.append(row)
        html_parts.append("</table>")

    # -- Failure analysis ----------------------------------------------
    html_parts.append("<h2>Failure Analysis</h2>")
    html_parts.append("<p>Questions where <strong>no</strong> model retrieved the correct tables:</p>")

    if universal_failures:
        html_parts.append("<table>")
        html_parts.append("<tr><th>ID</th><th>Question</th><th>Expected Tables</th>"
                          "<th>Best Retrieval</th></tr>")
        for f in universal_failures:
            q_display = f["question"]
            if len(q_display) > 80:
                q_display = q_display[:80] + "..."
            html_parts.append(
                f"<tr>"
                f"<td><code>{f['id']}</code></td>"
                f'<td class="truncate">{q_display}</td>'
                f"<td>{', '.join(f['expected_tables'])}</td>"
                f"<td>{', '.join(f['best_retrieval'])}</td>"
                f"</tr>"
            )
        html_parts.append("</table>")
    else:
        html_parts.append("<p><em>None -- at least one model retrieved the correct tables "
                          "for every question.</em></p>")

    # -- Footer --------------------------------------------------------
    html_parts.append('<hr><p class="meta">Report generated by '
                      '<code>eval/run_embedding_eval.py</code></p>')
    html_parts.append("</body>\n</html>")

    html_text = "\n".join(html_parts)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html_text)
    print(f"[output] HTML report saved to {output_path}")


# ===================================================================
# Model filtering
# ===================================================================

def filter_models(keyword: str) -> list[dict[str, Any]]:
    """Return the subset of EMBEDDING_MODELS whose name matches the keyword.

    The special keyword "all" returns all models.  Otherwise, a
    case-insensitive substring match is used.

    Args:
        keyword: Filter string from --models CLI argument.

    Returns:
        List of model-info dicts to evaluate.
    """
    if keyword.lower() == "all":
        return EMBEDDING_MODELS

    matched = [
        m for m in EMBEDDING_MODELS
        if keyword.lower() in m["name"].lower()
    ]

    if not matched:
        print(f"[error] No embedding models matched '{keyword}'.")
        print(f"        Available: {', '.join(m['name'] for m in EMBEDDING_MODELS)}")
        sys.exit(1)

    return matched


# ===================================================================
# CLI argument parsing
# ===================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Supported flags:
      --limit N       Evaluate only the first N test cases.
      --models FILTER Filter which models to evaluate (default: 'all').
                      Use a substring like 'miniLM' or 'mpnet', or 'all'.
      --test-cases P  Path to test_cases.yaml (default: eval/test_cases.yaml).
      --output-dir P  Directory for results (default: eval/results/).

    Returns:
        Parsed argparse.Namespace with attributes:
          limit, models, test_cases, output_dir.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate embedding models for schema retrieval accuracy. "
            "Tests how well each model retrieves the correct database tables "
            "from ChromaDB given a natural language question."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval/run_embedding_eval.py                    # all models, all cases\n"
            "  python eval/run_embedding_eval.py --limit 50         # cap at 50 test cases\n"
            "  python eval/run_embedding_eval.py --models mpnet     # only all-mpnet-base-v2\n"
            "  python eval/run_embedding_eval.py --models all       # explicit all\n"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of test cases to evaluate (default: all).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Which models to evaluate.  'all' runs every model; otherwise "
            "filters by case-insensitive substring match on model name.  "
            "Examples: 'all', 'miniLM', 'mpnet', 'paraphrase', 'multi-qa'."
        ),
    )
    parser.add_argument(
        "--test-cases",
        type=str,
        default=str(TEST_CASES_PATH),
        help=f"Path to test_cases.yaml (default: {TEST_CASES_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help=f"Directory for output files (default: {RESULTS_DIR}).",
    )
    return parser.parse_args()


# ===================================================================
# Main entry point
# ===================================================================

def main() -> None:
    """Orchestrate the full evaluation pipeline.

    Steps:
      1. Parse CLI arguments.
      2. Ensure sentence-transformers is installed.
      3. Load and filter test cases.
      4. Prepare semantic-layer documents.
      5. For each selected model, build a collection and evaluate.
      6. Save JSON results.
      7. Generate Markdown and HTML reports.
    """
    args = parse_args()

    print("=" * 60)
    print("  Embedding Model Evaluation for Schema Retrieval")
    print("=" * 60)
    print()

    # -- Step 1: Ensure dependencies -----------------------------------
    _ensure_sentence_transformers()

    # -- Step 2: Load test cases ---------------------------------------
    test_cases_path = Path(args.test_cases)
    test_cases = load_test_cases(test_cases_path, limit=args.limit)
    if not test_cases:
        print("[error] No test cases to evaluate. Exiting.")
        sys.exit(1)

    # -- Step 3: Determine which models to evaluate --------------------
    models_to_eval = filter_models(args.models)
    print(f"[info] Models to evaluate: {', '.join(m['name'] for m in models_to_eval)}")

    # -- Step 4: Prepare semantic-layer documents ----------------------
    documents = prepare_documents()

    # -- Step 5: Run evaluation for each model -------------------------
    all_results: list[dict[str, Any]] = []
    overall_start = time.perf_counter()

    for model_info in models_to_eval:
        model_name = model_info["name"]
        try:
            result = evaluate_model(model_name, documents, test_cases)
            all_results.append(result)
        except Exception as exc:
            print(f"\n[error] Failed to evaluate {model_name}: {exc}")
            import traceback
            traceback.print_exc()
            # Record the failure so it appears in results
            all_results.append({
                "model": model_name,
                "dimensions": model_info["dimensions"],
                "description": model_info["description"],
                "recall_at_5": 0.0,
                "precision_at_5": 0.0,
                "mrr": 0.0,
                "avg_retrieval_time_ms": 0.0,
                "total_test_cases": len(test_cases),
                "hits": 0,
                "misses": len(test_cases),
                "per_question": [],
                "category_recall": {},
                "collection_build_time_ms": 0.0,
                "error": str(exc),
            })

    total_elapsed = time.perf_counter() - overall_start

    # -- Step 6: Save outputs ------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "embedding_results.json"
    md_path = output_dir / "embedding_report.md"
    html_path = output_dir / "embedding_report.html"

    save_results_json(all_results, json_path)
    generate_markdown_report(all_results, test_cases, md_path)
    generate_html_report(all_results, test_cases, html_path)

    # -- Final summary -------------------------------------------------
    print()
    print("=" * 60)
    print("  Evaluation Complete")
    print("=" * 60)
    print(f"  Models evaluated: {len(all_results)}")
    print(f"  Test cases:       {len(test_cases)}")
    print(f"  Total time:       {total_elapsed:.1f}s")
    print()
    print("  Output files:")
    print(f"    {json_path}")
    print(f"    {md_path}")
    print(f"    {html_path}")
    print()

    # Quick leaderboard
    successful = [r for r in all_results if "error" not in r]
    if successful:
        best = max(successful, key=lambda r: r["recall_at_5"])
        print(f"  Best model: {best['model']} "
              f"(Recall@5={_pct(best['recall_at_5'])}, "
              f"MRR={best['mrr']:.3f})")


if __name__ == "__main__":
    main()
