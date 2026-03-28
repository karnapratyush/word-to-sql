"""Load YAML configuration files.

All config files live in the config/ directory:
- prompts.yaml: LLM prompt templates used by planner, SQL generator, and answer synthesizer
- models.yaml: Model routing and provider config (fallback chains, temperature, etc.)
- settings.yaml: Application settings (guardrails thresholds, retry counts, etc.)

Configs are cached after first load via @lru_cache to avoid re-reading files
on every call. This means changes to YAML files require a process restart.
"""

# ── Imports ──────────────────────────────────────────────────────────
import os
from functools import lru_cache

import yaml

# ── Path Resolution ──────────────────────────────────────────────────
# Navigate from src/common/ up two levels to the project root
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# All YAML config files are stored in the top-level config/ directory
_CONFIG_DIR = os.path.join(_BASE_DIR, "config")


def _load_yaml(filename: str) -> dict:
    """Load a YAML file from the config directory and return its contents as a dict.

    Args:
        filename: Name of the YAML file (e.g., "prompts.yaml").

    Returns:
        Parsed YAML content as a dict. Returns empty dict if file is empty.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
    """
    path = os.path.join(_CONFIG_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        # yaml.safe_load returns None for empty files, so default to {}
        return yaml.safe_load(f) or {}


# ── Public Cached Loaders ────────────────────────────────────────────
# Each loader is cached with maxsize=1 because there is only one config
# per type. The cache persists for the lifetime of the process.

@lru_cache(maxsize=1)
def load_prompts() -> dict:
    """Load all prompt templates from config/prompts.yaml.

    Returns:
        Dict keyed by domain (e.g., "analytics") containing prompt
        template strings with {placeholder} variables.
    """
    return _load_yaml("prompts.yaml")


@lru_cache(maxsize=1)
def load_model_config() -> dict:
    """Load model routing and provider config from config/models.yaml.

    Returns:
        Dict with "providers" (API base URLs) and "task_routing"
        (fallback chains per task like sql_generation, classification).
    """
    return _load_yaml("models.yaml")


@lru_cache(maxsize=1)
def load_settings() -> dict:
    """Load application settings from config/settings.yaml.

    Returns:
        Dict with sections like "guardrails" (max_input_length,
        injection_patterns), "analytics" (max_retries), etc.
    """
    return _load_yaml("settings.yaml")
