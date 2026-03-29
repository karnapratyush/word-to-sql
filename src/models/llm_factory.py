"""LLM model factory with fallback chain support.

Creates LangChain chat model instances based on config/models.yaml.
Each task (sql_generation, classification, vision, etc.) has a fallback
chain of models — free/fast models first, paid/slow as last resort.

If the primary model fails (API down, rate limited, timeout), the factory
automatically tries the next model in the chain until one succeeds.

Providers supported:
- groq: Llama models via Groq's fast inference API
- google: Gemini models via Google AI (Generative AI SDK)
- openrouter: Any model via OpenRouter's unified API (OpenAI-compatible)

Configuration in config/models.yaml:
    task_routing:
      sql_generation:
        chain:
          - {provider: "groq", model: "llama-3.3-70b-versatile"}
          - {provider: "google", model: "gemini-2.0-flash"}
        temperature: 0.0
        max_tokens: 2000
        timeout_seconds: 30
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from src.common.config_loader import load_model_config
from src.common.exceptions import AllModelsFailedError
from src.tracing import get_callbacks

# Load .env file so API keys (GROQ_API_KEY, GOOGLE_API_KEY, etc.)
# are available in os.environ before any model creation
load_dotenv()


# ── Model Instance Creation ──────────────────────────────────────────

def _create_groq(model, temperature, max_tokens, timeout):
    """Create a Groq LLM instance (Llama models via Groq's fast inference API)."""
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout, api_key=os.environ.get("GROQ_API_KEY", ""))


def _create_google(model, temperature, max_tokens, timeout):
    """Create a Google Generative AI instance (Gemini models)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_output_tokens=max_tokens, timeout=timeout, google_api_key=os.environ.get("GOOGLE_API_KEY", ""))


def _create_openrouter(model, temperature, max_tokens, timeout):
    """Create an OpenRouter instance (OpenAI-compatible API with custom base URL)."""
    from langchain_openai import ChatOpenAI
    config = load_model_config()
    base_url = config.get("providers", {}).get("openrouter", {}).get("base_url", "https://openrouter.ai/api/v1")
    return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout, openai_api_key=os.environ.get("OPENROUTER_API_KEY", ""), openai_api_base=base_url)


# Registry dict mapping provider names to their factory functions.
# Replaces the if/elif chain for cleaner dispatch and easier extension.
_PROVIDER_REGISTRY = {
    "groq": _create_groq,
    "google": _create_google,
    "openrouter": _create_openrouter,
}


def create_model_instance(provider: str, model: str, temperature: float, max_tokens: int, timeout: int) -> Any:
    """Create a LangChain chat model instance for the given provider.

    Uses lazy imports for provider SDKs so that only the needed SDK
    is loaded. This avoids import errors if a user hasn't installed
    all provider packages (e.g., only using Groq, not Google).

    Args:
        provider: One of "groq", "google", "openrouter".
        model: Model name/ID for the provider (e.g., "llama-3.3-70b-versatile").
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum output tokens for the response.
        timeout: Request timeout in seconds.

    Returns:
        A LangChain BaseChatModel instance ready for .invoke().

    Raises:
        ValueError: If the provider string is not recognized.
    """
    factory = _PROVIDER_REGISTRY.get(provider)
    if factory is None:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(_PROVIDER_REGISTRY.keys())}")
    return factory(model, temperature, max_tokens, timeout)


# Backward-compatible alias (was private, now public)
_create_model_instance = create_model_instance


# ── Public Factory Functions ─────────────────────────────────────────

def get_model(task: str) -> tuple[Any, str]:
    """Return (model_instance, model_name) for the given task.

    Walks the fallback chain from config/models.yaml and returns the
    first model that can be successfully instantiated. Note: this only
    tests instantiation, not actual invocation — the model's API may
    still fail when called.

    Args:
        task: Task name from config (e.g., "sql_generation", "classification").

    Returns:
        Tuple of (model_instance, "provider/model_name").

    Raises:
        KeyError: If the task is not found in config/models.yaml.
        AllModelsFailedError: If no model in the chain could be created.
    """
    config = load_model_config()
    task_config = config["task_routing"][task]  # KeyError if task unknown

    # Extract task-level settings that apply to all models in the chain
    chain = task_config.get("chain", [])
    temperature = task_config.get("temperature", 0.0)
    max_tokens = task_config.get("max_tokens", 1000)
    timeout = task_config.get("timeout_seconds", 30)

    errors = []
    for entry in chain:
        provider = entry["provider"]
        model = entry["model"]
        model_name = f"{provider}/{model}"
        try:
            instance = create_model_instance(provider, model, temperature, max_tokens, timeout)
            return instance, model_name
        except Exception as e:
            errors.append(f"{model_name}: {e}")
            continue  # Try next model in the fallback chain

    raise AllModelsFailedError(task, errors)


def get_model_with_fallback(task: str, messages: list, **kwargs) -> tuple[Any, str]:
    """Try invoking models in the fallback chain until one succeeds.

    Unlike get_model() which only instantiates, this function actually
    CALLS each model with the provided messages and returns the first
    successful response. This handles runtime failures like rate limits,
    API outages, and timeout errors.

    Args:
        task: Task name from config (e.g., "sql_generation").
        messages: List of message dicts with "role" and "content" keys.
            Roles: "system" (for system prompts) or "user" (for queries).
        **kwargs: Additional keyword arguments:
            session_id (str): For LangFuse tracing session grouping.
            trace_name (str): For LangFuse trace labeling.
            Other kwargs are passed to model.invoke().

    Returns:
        Tuple of (response_content_string, "provider/model_name").

    Raises:
        AllModelsFailedError: If every model in the chain failed invocation.
    """
    config = load_model_config()
    task_config = config["task_routing"][task]

    chain = task_config.get("chain", [])
    temperature = task_config.get("temperature", 0.0)
    max_tokens = task_config.get("max_tokens", 1000)
    timeout = task_config.get("timeout_seconds", 30)

    # Convert message dicts to LangChain message objects
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    # Extract tracing kwargs before passing remaining to model.invoke()
    session_id = kwargs.pop("session_id", "")
    trace_name = kwargs.pop("trace_name", task)
    cb_config = get_callbacks(session_id=session_id, trace_name=trace_name)

    errors = []
    for entry in chain:
        provider = entry["provider"]
        model = entry["model"]
        model_name = f"{provider}/{model}"
        try:
            instance = create_model_instance(provider, model, temperature, max_tokens, timeout)
            # Actually invoke the model — this is where API calls happen
            response = instance.invoke(lc_messages, config=cb_config)
            # Extract string content from the LangChain response object
            content = response.content if hasattr(response, "content") else str(response)
            return content, model_name
        except Exception as e:
            # Log the specific error type for debugging fallback behavior
            errors.append(f"{model_name}: {type(e).__name__}: {e}")
            continue  # Try next model in the fallback chain

    raise AllModelsFailedError(task, errors)
