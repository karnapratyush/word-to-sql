"""Analytics Service — orchestrates the analytics pipeline.

This is the service layer between the FastAPI controller and the domain logic.
It wraps the analytics agent with dependency injection support and adds
cross-cutting concerns like schema caching.

In the MVCR (Model-View-Controller-Repository) pattern:
- Controller (FastAPI route) handles HTTP concerns (request/response)
- Service (this file) handles business orchestration and caching
- Domain modules (planner, sql_generator, etc.) handle specific logic
- Repository handles data access and SQL execution
"""

from src.common.schemas import AnalyticsRequest, AnalyticsResponse
from src.analytics.agent import run_analytics_query
from src.database import get_schema_description


class AnalyticsService:
    """Service for analytics query processing.

    Wraps the analytics agent pipeline with:
    - Database path injection (from FastAPI app.state)
    - Schema description caching (avoid re-querying on every request)
    - Health check functionality for the /api/health endpoint

    Attributes:
        _db_path: Optional database path override (None uses default).
        _schema_cache: Cached schema description (populated on first access).
    """

    def __init__(self, db_path: str | None = None):
        """Initialize the service with an optional database path.

        Args:
            db_path: Optional database path. When None, uses the default
                path (db/logistics.db) configured in the repository layer.
        """
        self._db_path = db_path
        self._schema_cache: str | None = None

    def query(self, request: AnalyticsRequest) -> AnalyticsResponse:
        """Process an analytics query through the full pipeline.

        Delegates to the analytics agent which orchestrates all 7 steps
        (guardrails -> planner -> SQL gen -> verify -> synthesize -> visualize).

        Args:
            request: AnalyticsRequest with user_query, history, session_id.

        Returns:
            AnalyticsResponse with answer, SQL, table, chart, and metadata.
        """
        return run_analytics_query(request)

    def get_schema(self) -> str:
        """Return the database schema description, with caching.

        The schema is cached after the first call because it does not
        change during the lifetime of the application (tables are static).

        Returns:
            Multi-line string describing all tables, columns, and relationships.
        """
        if self._schema_cache is None:
            self._schema_cache = get_schema_description(db_path=self._db_path)
        return self._schema_cache

    def health_check(self) -> dict:
        """Check database connectivity and return table information.

        Attempts to query table names and row counts. If any database
        operation fails, returns an "unhealthy" status with the error.

        Returns:
            Dict with keys:
            - status: "healthy" or "unhealthy"
            - tables: List of table names (only when healthy)
            - row_counts: Dict of table -> count (only when healthy)
            - error: Error message (only when unhealthy)
        """
        from src.database import get_table_names, get_row_count
        try:
            tables = get_table_names(db_path=self._db_path)
            row_counts = {}
            for table in tables:
                row_counts[table] = get_row_count(table, db_path=self._db_path)
            return {
                "status": "healthy",
                "tables": tables,
                "row_counts": row_counts,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
