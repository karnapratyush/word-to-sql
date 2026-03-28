"""Custom exception hierarchy for the GoCoMet AI application.

All domain-specific exceptions inherit from GoCoMetAIError so that
API-level error handlers can catch them with a single except clause.
Each exception type maps to a specific HTTP status code in the FastAPI
exception handlers (see src/api/main.py):

    GoCoMetAIError          → 500 Internal Server Error
    ├── AllModelsFailedError → 503 Service Unavailable
    ├── SQLGenerationError   → 500 (SQL pipeline failure)
    ├── SQLExecutionError    → 500 (query ran but errored)
    ├── ExtractionError      → 500 (vision pipeline failure)
    ├── GuardrailError       → 400 Bad Request (input/output validation)
    └── UnsupportedFileError → 400 Bad Request (wrong file type)
"""


class GoCoMetAIError(Exception):
    """Base exception for all GoCoMet AI errors.

    Catching this single class in a handler covers every domain error.
    """
    pass


class AllModelsFailedError(GoCoMetAIError):
    """Every model in the fallback chain failed for a given task.

    Stores the task name and per-model error messages so the caller
    can log which providers were attempted and why each failed.

    Attributes:
        task: The task key from models.yaml (e.g., "sql_generation").
        errors: List of "provider/model: error" strings, one per attempt.
    """

    def __init__(self, task: str, errors: list[str]):
        self.task = task
        self.errors = errors
        # Build a human-readable message listing the task and all errors
        super().__init__(f"All models failed for task '{task}': {errors}")


class SQLGenerationError(GoCoMetAIError):
    """SQL generation failed after all retries.

    Raised when the LLM could not produce valid SQL even after
    receiving error feedback in retry prompts.
    """
    pass


class SQLExecutionError(GoCoMetAIError):
    """SQL executed but produced a database error.

    Raised when the generated SQL passed guardrail validation but
    failed during actual execution (e.g., syntax error, missing table).
    """
    pass


class ExtractionError(GoCoMetAIError):
    """Vision extraction failed.

    Raised when the vision LLM could not extract structured fields
    from an uploaded PDF or image document.
    """
    pass


class GuardrailError(GoCoMetAIError):
    """Input or output validation failed.

    Raised when either input guardrails (SQL injection, prompt injection,
    length limits) or output guardrails (DML detection, grounding check)
    reject the content.
    """
    pass


class UnsupportedFileError(GoCoMetAIError):
    """Uploaded file type is not supported.

    Raised when a user uploads a file format that the vision pipeline
    cannot process (only PDF, PNG, JPG, JPEG, TIFF are supported).
    """
    pass
