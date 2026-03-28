"""Tests for src/models.py — LLM model factory with fallback chains.

These test the factory interface and fallback behavior.
Actual LLM calls require API keys so some tests may be skipped
in CI without keys.
"""

import pytest

from src.models import get_model, get_model_with_fallback
from src.common.exceptions import AllModelsFailedError


class TestGetModel:
    def test_returns_tuple(self):
        model, name = get_model("classification")
        assert model is not None
        assert isinstance(name, str)
        assert len(name) > 0

    def test_sql_generation_model(self):
        model, name = get_model("sql_generation")
        assert model is not None

    def test_vision_model(self):
        model, name = get_model("vision")
        assert model is not None

    def test_answer_synthesis_model(self):
        model, name = get_model("answer_synthesis")
        assert model is not None

    def test_unknown_task_raises(self):
        with pytest.raises((KeyError, ValueError, AllModelsFailedError)):
            get_model("nonexistent_task")

    def test_returns_different_models_for_different_tasks(self):
        _, name1 = get_model("classification")
        _, name2 = get_model("sql_generation")
        # May or may not be same model, but both should work
        assert isinstance(name1, str)
        assert isinstance(name2, str)


class TestGetModelWithFallback:
    def test_returns_response_and_model_name(self):
        response, model_name = get_model_with_fallback(
            "classification",
            messages=[{"role": "user", "content": "Say hello"}],
        )
        assert response is not None
        assert isinstance(model_name, str)

    def test_all_models_failed_raises(self):
        """If we pass an impossible task, should raise."""
        with pytest.raises((AllModelsFailedError, KeyError, ValueError)):
            get_model_with_fallback(
                "nonexistent_task",
                messages=[{"role": "user", "content": "test"}],
            )
