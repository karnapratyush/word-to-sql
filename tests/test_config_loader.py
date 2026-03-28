"""Tests for src/common/config_loader.py

Tests that config files load correctly, return expected structure,
and handle missing files gracefully.
"""

import pytest

from src.common.config_loader import load_prompts, load_model_config, load_settings


# ── load_prompts ────────────────────────────────────────────────

class TestLoadPrompts:
    def test_returns_dict(self):
        result = load_prompts()
        assert isinstance(result, dict)

    def test_has_analytics_section(self):
        result = load_prompts()
        assert "analytics" in result

    def test_has_vision_section(self):
        result = load_prompts()
        assert "vision" in result

    def test_analytics_has_planner_prompt(self):
        result = load_prompts()
        assert "planner" in result["analytics"]
        assert "{schema}" in result["analytics"]["planner"]
        assert "{query}" in result["analytics"]["planner"]

    def test_analytics_has_sql_generator_prompt(self):
        result = load_prompts()
        assert "sql_generator" in result["analytics"]
        assert "SELECT" in result["analytics"]["sql_generator"]

    def test_analytics_has_sql_retry_prompt(self):
        result = load_prompts()
        assert "sql_retry" in result["analytics"]
        assert "{error}" in result["analytics"]["sql_retry"]

    def test_vision_has_extractor_prompt(self):
        result = load_prompts()
        assert "extractor" in result["vision"]
        assert "{document_type}" in result["vision"]["extractor"]

    def test_has_document_types(self):
        result = load_prompts()
        assert "document_types" in result
        assert "invoice" in result["document_types"]
        assert "bill_of_lading" in result["document_types"]

    def test_invoice_expected_fields(self):
        result = load_prompts()
        fields = result["document_types"]["invoice"]["expected_fields"]
        assert "invoice_number" in fields
        assert "total_amount" in fields


# ── load_model_config ───────────────────────────────────────────

class TestLoadModelConfig:
    def test_returns_dict(self):
        result = load_model_config()
        assert isinstance(result, dict)

    def test_has_providers(self):
        result = load_model_config()
        assert "providers" in result
        assert "groq" in result["providers"]
        assert "google" in result["providers"]
        assert "openrouter" in result["providers"]

    def test_has_task_routing(self):
        result = load_model_config()
        assert "task_routing" in result

    def test_sql_generation_has_chain(self):
        result = load_model_config()
        chain = result["task_routing"]["sql_generation"]["chain"]
        assert isinstance(chain, list)
        assert len(chain) >= 2  # at least 2 fallback options

    def test_chain_entries_have_provider_and_model(self):
        result = load_model_config()
        for task, config in result["task_routing"].items():
            if "chain" in config:
                for entry in config["chain"]:
                    assert "provider" in entry, f"Missing provider in {task} chain"
                    assert "model" in entry, f"Missing model in {task} chain"

    def test_vision_chain_exists(self):
        result = load_model_config()
        assert "vision" in result["task_routing"]
        assert "chain" in result["task_routing"]["vision"]


# ── load_settings ───────────────────────────────────────────────

class TestLoadSettings:
    def test_returns_dict(self):
        result = load_settings()
        assert isinstance(result, dict)

    def test_has_guardrails(self):
        result = load_settings()
        assert "guardrails" in result
        assert "max_input_length" in result["guardrails"]

    def test_has_extraction_settings(self):
        result = load_settings()
        assert "extraction" in result
        assert "confidence_threshold" in result["extraction"]
        assert 0 < result["extraction"]["confidence_threshold"] < 1

    def test_has_database_path(self):
        result = load_settings()
        assert "database" in result
        assert "path" in result["database"]

    def test_sql_blocked_keywords_populated(self):
        result = load_settings()
        blocked = result["guardrails"]["sql_blocked_keywords"]
        assert "DROP" in blocked
        assert "DELETE" in blocked

    def test_injection_patterns_populated(self):
        result = load_settings()
        patterns = result["guardrails"]["injection_patterns"]
        assert len(patterns) >= 3
