"""LangFuse tracing integration.

Provides callback handlers for LangChain that send traces to LangFuse,
an open-source LLM observability platform. Traces capture model inputs,
outputs, latency, and token usage for debugging and monitoring.

Degrades gracefully: if LangFuse keys are not set, returns empty callbacks.
The app works identically with or without tracing configured — no code
changes needed to enable or disable it.

Environment variables required for tracing:
    LANGFUSE_PUBLIC_KEY: Your LangFuse project's public key
    LANGFUSE_SECRET_KEY: Your LangFuse project's secret key
    LANGFUSE_HOST: (Optional) Custom LangFuse host URL

Usage:
    from src.tracing import get_callbacks
    config = get_callbacks(session_id="abc", trace_name="sql_generation")
    response = chain.invoke(prompt, config=config)
"""

import os
from typing import Any, Optional


def get_langfuse_handler(session_id: str = "", trace_name: str = "") -> Optional[Any]:
    """Return a LangFuse CallbackHandler if credentials are configured.

    Checks for LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars.
    If either is missing, returns None (silent degradation — no tracing).

    Args:
        session_id: Groups related traces under a single session in LangFuse.
        trace_name: Tag applied to the trace for filtering in the dashboard.

    Returns:
        A LangFuse CallbackHandler instance, or None if not configured.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    # Both keys are required; skip tracing if either is missing
    if not public_key or not secret_key:
        return None

    try:
        # Lazy import so the app doesn't fail if langfuse is not installed.
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # LangFuse v4: Initialize the core client first with credentials.
        # This configures the global Langfuse instance that CallbackHandler uses.
        # Ref: https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4
        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        # Then create the callback handler — it picks up the initialized client.
        # In v4, CallbackHandler() takes no constructor args for credentials.
        handler = CallbackHandler()
        return handler
    except Exception:
        # LangFuse not installed or config error — degrade silently
        return None


def get_callbacks(session_id: str = "", trace_name: str = "") -> dict:
    """Return a config dict with callbacks list for LangChain .invoke().

    Always returns a valid config dict. If LangFuse is not configured,
    the callbacks list is empty, making this a no-op wrapper.

    This is the primary entry point — all LLM calls in the codebase
    pass the result of this function as the config= argument.

    Args:
        session_id: Session ID for grouping traces in LangFuse.
        trace_name: Descriptive name for the trace (e.g., "planner", "sql_generator").

    Returns:
        Dict with a "callbacks" key containing a list of handler(s).

    Usage:
        config = get_callbacks(session_id="abc", trace_name="planner")
        result = chain.invoke(input, config=config)
    """
    handler = get_langfuse_handler(session_id=session_id, trace_name=trace_name)
    # Wrap in a list if present; empty list means LangChain skips callbacks
    callbacks = [handler] if handler is not None else []
    return {"callbacks": callbacks}
