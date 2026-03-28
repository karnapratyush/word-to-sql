"""Health and schema endpoints.

These endpoints are used for:
- Monitoring: Check if the API and database are operational
- Debugging: View the database schema as the LLM sees it

Endpoints:
    GET /api/health — Database connectivity check + table row counts
    GET /api/schema — Human-readable schema description
"""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_analytics_service
from src.api.schemas.analytics import HealthResponse, SchemaResponse
from src.services.analytics_service import AnalyticsService

# Create a router instance for health-related endpoints
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(service: AnalyticsService = Depends(get_analytics_service)):
    """Check database connectivity and return table information.

    Returns status ("healthy" or "unhealthy"), list of table names,
    and row counts for each table. Used by the Streamlit home page
    to display system status.

    Args:
        service: Injected AnalyticsService instance.

    Returns:
        HealthResponse with status, tables, and row_counts.
    """
    result = service.health_check()
    return HealthResponse(**result)


@router.get("/schema", response_model=SchemaResponse)
def get_schema(service: AnalyticsService = Depends(get_analytics_service)):
    """Return the database schema description.

    Returns the same human-readable schema that is injected into LLM
    prompts. Useful for debugging to see what context the LLM receives.

    Args:
        service: Injected AnalyticsService instance.

    Returns:
        SchemaResponse with full schema description and table name list.
    """
    schema = service.get_schema()
    # Parse table names from the schema text (lines starting with "TABLE: ")
    tables = []
    for line in schema.split("\n"):
        if line.startswith("TABLE: "):
            # Extract table name from "TABLE: shipments (10,000 rows)" format
            table_name = line.split("TABLE: ")[1].split(" (")[0]
            tables.append(table_name)
    return SchemaResponse(
        schema_description=schema,
        tables=tables,
    )
