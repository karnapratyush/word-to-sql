"""FastAPI dependency injection providers.

These functions are used with FastAPI's Depends() system to inject
service instances into route handlers. This decouples route logic
from service construction and enables testing with mock services.

Each provider reads app-level configuration from app.state (set in
create_app) and passes it to the service constructor. This allows
the entire app to use a custom database path, which is especially
useful for testing with temporary databases.

Usage in routes:
    @router.post("/query")
    def query(service: AnalyticsService = Depends(get_analytics_service)):
        return service.query(request)

    @router.post("/extract")
    def extract(service: VisionService = Depends(get_vision_service)):
        return service.extract_document(...)
"""

from fastapi import Request

from src.services.analytics_service import AnalyticsService
from src.services.vision_service import VisionService


def get_analytics_service(request: Request) -> AnalyticsService:
    """Provide an AnalyticsService instance for route handler injection.

    Reads the app-level db_path from app.state (set in create_app) and
    passes it to the service. This allows the entire app to use a custom
    database path (useful for testing with temporary databases).

    Args:
        request: FastAPI Request object (provides access to app.state).

    Returns:
        An AnalyticsService instance configured with the app's db_path.
    """
    db_path = getattr(request.app.state, "db_path", None)
    return AnalyticsService(db_path=db_path)


def get_vision_service(request: Request) -> VisionService:
    """Provide a VisionService instance for route handler injection.

    Reads the app-level db_path from app.state (set in create_app) and
    passes it to the service. The VisionService handles document
    extraction, approval/rejection, and retrieval.

    Args:
        request: FastAPI Request object (provides access to app.state).

    Returns:
        A VisionService instance configured with the app's db_path.
    """
    db_path = getattr(request.app.state, "db_path", None)
    return VisionService(db_path=db_path)
