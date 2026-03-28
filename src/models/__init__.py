"""LLM model factory — re-exports for backward compatibility.

This package provides a unified interface for creating and invoking LLM
models across multiple providers (Groq, Google, OpenRouter). The actual
implementation lives in llm_factory.py; this __init__.py re-exports the
two main functions so callers can use the shorter import path:

    from src.models import get_model, get_model_with_fallback
"""

from src.models.llm_factory import get_model, get_model_with_fallback

__all__ = ["get_model", "get_model_with_fallback"]
