"""Visualizer — suggest chart type and create Plotly figures.

This module handles the visualization step of the analytics pipeline.
It is entirely rule-based (no LLM needed) to keep it fast and cheap.

Two functions:
1. suggest_chart(): Rule-based chart type suggestion based on query
   keywords, column types, and row count. Returns None when no
   visualization makes sense.
2. create_chart(): Creates a Plotly figure dict from result data,
   suitable for rendering in Streamlit or returning via the API.

Supported chart types: bar, line, pie.
Returns None for single-row results (just a number, no chart needed).
"""

from typing import Optional
import plotly.graph_objects as go


# ── Keyword Sets for Chart Type Detection ────────────────────────────

# Keywords in the user's query that suggest a time-series (line chart)
_TIME_KEYWORDS = {"over time", "trend", "monthly", "daily", "weekly", "yearly",
                  "by month", "by date", "by year", "by week", "per month", "per day"}

# Keywords that suggest a distribution view (pie chart)
_PIE_KEYWORDS = {"distribution", "breakdown", "share", "percentage", "proportion", "by mode", "by type"}

# Column name patterns that indicate date/time data (also suggests line chart)
_DATE_COLUMNS = {"date", "month", "year", "week", "day", "period", "quarter",
                 "booking_date", "invoice_date", "event_timestamp", "created_at"}


# ── Chart Type Suggestion ────────────────────────────────────────────

def suggest_chart(query: str, columns: list[str], row_count: int) -> Optional[str]:
    """Rule-based chart type suggestion based on query and result shape.

    Decision logic (in priority order):
    1. No rows or 1 row → None (no chart for scalar results)
    2. Single column → None (no x vs y relationship)
    3. Time keywords or date columns → "line"
    4. Distribution keywords with few categories → "pie"
    5. Few categories (<=5) with 2 columns → "pie"
    6. Default → "bar" (most versatile chart type)

    Args:
        query: The user's original question (for keyword matching).
        columns: Column names in the query result.
        row_count: Number of rows in the result.

    Returns:
        "bar", "line", "pie", or None if no chart is appropriate.
    """
    # No data to chart
    if row_count <= 0:
        return None

    # Single row = just a number, no chart needed
    if row_count == 1:
        return None

    # Single column = no x vs y relationship to visualize
    if len(columns) < 2:
        return None

    query_lower = query.lower()
    columns_lower = [c.lower() for c in columns]

    # Check for time-series indicators (keywords or date columns)
    has_time_keyword = any(kw in query_lower for kw in _TIME_KEYWORDS)
    has_date_column = any(
        any(dc in col for dc in _DATE_COLUMNS)
        for col in columns_lower
    )

    if has_time_keyword or has_date_column:
        return "line"

    # Check for pie chart indicators (small number of categories)
    has_pie_keyword = any(kw in query_lower for kw in _PIE_KEYWORDS)
    if has_pie_keyword and row_count <= 8:
        return "pie"

    # Few categories with exactly 2 columns → pie is a good fit
    if row_count <= 5 and len(columns) == 2:
        return "pie"

    # Default: bar chart (works well for most comparison queries)
    return "bar"


# ── Chart Creation ───────────────────────────────────────────────────

def create_chart(
    chart_type: str,
    data: list[dict],
    columns: list[str],
    query: str,
) -> Optional[dict]:
    """Create a Plotly figure from query result data.

    Uses the first column as labels/x-axis and the second column as
    values/y-axis. The query text is used as the chart title.

    Args:
        chart_type: One of "bar", "line", "pie".
        data: List of row dicts from query results.
        columns: Column names (used for axis labels).
        query: Original user query (used for chart title).

    Returns:
        Serialized Plotly figure dict (JSON-compatible) suitable for
        rendering with plotly.graph_objects.Figure(data), or None on failure.
    """
    if not data or not columns:
        return None

    if chart_type not in ("bar", "line", "pie"):
        return None

    try:
        # Use first column as labels (x-axis) and second as values (y-axis)
        x_col = columns[0]
        y_col = columns[1] if len(columns) > 1 else columns[0]

        x_values = [row.get(x_col, "") for row in data]
        y_values = [row.get(y_col, 0) for row in data]

        # Ensure y-values are numeric for charting
        y_values = [_to_number(v) for v in y_values]

        # Truncate long queries for the chart title
        title = query[:80] + "..." if len(query) > 80 else query

        fig = go.Figure()

        if chart_type == "bar":
            fig.add_trace(go.Bar(
                x=[str(v) for v in x_values],  # Convert to strings for categorical axis
                y=y_values,
                name=y_col,
            ))
            fig.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col,
            )

        elif chart_type == "line":
            fig.add_trace(go.Scatter(
                x=[str(v) for v in x_values],
                y=y_values,
                mode="lines+markers",  # Show both line and data points
                name=y_col,
            ))
            fig.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col,
            )

        elif chart_type == "pie":
            fig.add_trace(go.Pie(
                labels=[str(v) for v in x_values],
                values=y_values,
                name=y_col,
            ))
            fig.update_layout(title=title)

        # Serialize to JSON-compatible dict for API transport
        return fig.to_plotly_json()

    except Exception:
        # Charting failures are non-critical — return None and skip visualization
        return None


# ── Helpers ──────────────────────────────────────────────────────────

def _to_number(value) -> float:
    """Safely convert a value to float for charting.

    Handles int, float, and string representations. Returns 0.0 for
    any value that cannot be converted (e.g., None, non-numeric strings).

    Args:
        value: The value to convert.

    Returns:
        Float representation, or 0.0 if conversion fails.
    """
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0
