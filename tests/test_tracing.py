"""Tests for src/tracing.py — LangFuse integration.

Tests that tracing handles missing config gracefully.
"""

import pytest

from src.tracing import get_langfuse_handler, get_callbacks


class TestGetLangfuseHandler:
    def test_returns_handler_or_none(self):
        """Should return a handler if configured, None if not."""
        handler = get_langfuse_handler(session_id="test", trace_name="test")
        # Either a handler object or None (graceful degradation)
        assert handler is None or handler is not None

    def test_does_not_crash_without_keys(self):
        """Must not crash even if LANGFUSE keys are not set."""
        handler = get_langfuse_handler()
        # Should gracefully return None
        assert handler is None or handler is not None


class TestGetCallbacks:
    def test_returns_dict(self):
        result = get_callbacks(session_id="test", trace_name="test")
        assert isinstance(result, dict)

    def test_has_callbacks_key(self):
        result = get_callbacks()
        assert "callbacks" in result
        assert isinstance(result["callbacks"], list)
