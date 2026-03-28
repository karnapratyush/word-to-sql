"""FastAPI application factory.

Creates and configures the FastAPI app with:
- CORS middleware (allows Streamlit frontend on port 8501)
- Exception handlers for custom domain exceptions
- Router registration for health and analytics endpoints
- Lifespan management (startup config validation, shutdown cleanup)

Usage:
    from src.api.main import create_app
    app = create_app()                    # Production
    app = create_app(db_path="/tmp/test.db")  # Testing
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.common.exceptions import GoCoMetAIError, GuardrailError, AllModelsFailedError, UnsupportedFileError


# ── Lifespan Manager ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for the FastAPI application.

    Startup: Validates that configuration files can be loaded.
    Shutdown: Closes any open SQLite connections to prevent file locks.
    """
    # Startup: verify configs load without error (fail fast on bad config)
    from src.common.config_loader import load_settings
    load_settings()  # Will raise FileNotFoundError if config missing

    # Rebuild the extracted_document_fields VIEW so the LLM can query
    # extracted JSON fields as regular columns from application start.
    from src.vision.storage import rebuild_view_on_startup
    rebuild_view_on_startup(db_path=getattr(app.state, "db_path", None))

    yield
    # Shutdown: close all thread-local SQLite connections
    from src.repositories.engines.sqlite import _local
    if hasattr(_local, "connections"):
        for conn in _local.connections.values():
            try:
                conn.close()
            except Exception:
                pass  # Best-effort cleanup


# ── Application Factory ──────────────────────────────────────────────

def create_app(db_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Uses the factory pattern so tests can create isolated app instances
    with custom database paths.

    Args:
        db_path: Optional database path override. When set, stored in
            app.state.db_path and used by dependency injection to create
            repositories pointing to the specified database.

    Returns:
        Fully configured FastAPI app instance ready for uvicorn.run().
    """
    app = FastAPI(
        title="GoComet AI Logistics API",
        description="Agentic analytics and document extraction for logistics data",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store db_path in app state for dependency injection access
    app.state.db_path = db_path

    # ── CORS Middleware ───────────────────────────────────────────
    # Allow the Streamlit frontend (default port 8501) and local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",    # Streamlit default
            "http://127.0.0.1:8501",    # Streamlit alternate
            "http://localhost:3000",     # React dev server (future)
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ───────────────────────────────────────
    # Map domain exceptions to appropriate HTTP status codes.
    # More specific handlers are registered first (FastAPI matches most specific).

    @app.exception_handler(GuardrailError)
    async def guardrail_error_handler(request: Request, exc: GuardrailError):
        """400 Bad Request — input or output validation failed."""
        return JSONResponse(
            status_code=400,
            content={"error": "guardrail_violation", "detail": str(exc)},
        )

    @app.exception_handler(UnsupportedFileError)
    async def unsupported_file_handler(request: Request, exc: UnsupportedFileError):
        """400 Bad Request — uploaded file type is not supported."""
        return JSONResponse(
            status_code=400,
            content={"error": "unsupported_file", "detail": str(exc)},
        )

    @app.exception_handler(AllModelsFailedError)
    async def models_failed_handler(request: Request, exc: AllModelsFailedError):
        """503 Service Unavailable — all LLM providers are down."""
        return JSONResponse(
            status_code=503,
            content={"error": "all_models_failed", "detail": str(exc), "task": exc.task},
        )

    @app.exception_handler(GoCoMetAIError)
    async def gocomet_error_handler(request: Request, exc: GoCoMetAIError):
        """500 Internal Server Error — catch-all for domain errors."""
        return JSONResponse(
            status_code=500,
            content={"error": "internal_ai_error", "detail": str(exc)},
        )

    # ── Register Routers ─────────────────────────────────────────
    # Import routers inside the function to avoid circular imports
    from src.api.routers import health, analytics, documents
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
    app.include_router(documents.router, prefix="/api/documents", tags=["documents"])

    return app
