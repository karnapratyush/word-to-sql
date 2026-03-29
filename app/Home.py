"""  AI Logistics Assistant — Home Page.

This is the Streamlit landing page that provides:
- App title and description
- Database health check status with table row counts
- Architecture overview diagram
- Navigation links to capabilities (Analytics, Document Upload, Query)
- Sample analytics questions for quick start
"""

import sys
import os

# Add project root to Python path so 'app' and 'src' packages are importable.
# Streamlit runs from app/ directory, but our imports expect the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="  AI Logistics Assistant",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢   AI Logistics Assistant")
st.markdown("**Agentic Analytics + Vision Document Extraction** for logistics data")

st.divider()

# ── Health Check ─────────────────────────────────────────────────────
# Display API and database connectivity status on page load
st.subheader("System Status")

try:
    from app.api_client import get_api_client
    client = get_api_client()
    health = client.health_check()

    if health["status"] == "healthy":
        st.success("✅ API is healthy")
        # Display row counts for each table as metric cards
        cols = st.columns(len(health.get("row_counts", {})))
        for i, (table, count) in enumerate(health.get("row_counts", {}).items()):
            cols[i % len(cols)].metric(table, f"{count:,}")
    else:
        st.error(f"❌ API unhealthy: {health.get('error', 'unknown')}")
except Exception as e:
    # API is not running — show instructions to start it
    st.warning(f"⚠️ Cannot connect to API at localhost:8000. Start it with: `python run_api.py`")
    st.code(str(e), language="text")

st.divider()

# ── Architecture ─────────────────────────────────────────────────────
# Show the pipeline architecture as an ASCII diagram
st.subheader("Architecture")

st.markdown("""
```
User Question (Streamlit)
    │
    ▼ HTTP POST
FastAPI Controller (/api/analytics/query)
    │
    ▼
Analytics Service
    │
    ├── Input Guardrails (injection detection, length)
    ├── Planner (LLM classifies intent)
    ├── SQL Generator (LLM → SQL with retry)
    ├── Verifier (validate + execute against SQLite)
    ├── Answer Synthesizer (LLM → natural language)
    └── Visualizer (Plotly chart suggestion)
    │
    ▼
AnalyticsResponse (answer + SQL + table + chart)
```
""")

st.divider()

# ── Navigation ───────────────────────────────────────────────────────
# Link to the three main capabilities
st.subheader("Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Analytics Chat")
    st.markdown("Ask questions about shipments, costs, delays, AND extracted documents — all in one place.")
    st.page_link("pages/1_Analytics.py", label="Open Analytics Chat")

with col2:
    st.markdown("### Document Upload")
    st.markdown("Upload invoices or bills of lading. AI extracts fields, you review and approve.")
    st.page_link("pages/2_Document_Upload.py", label="Upload Documents")

st.divider()

st.subheader("Try These Questions")

samples = [
    "How many shipments are delayed?",
    "Average freight cost by carrier",
    "Top 5 carriers by delay rate",
    "Show all extracted documents",
    "Which shipments have linked documents?",
]

for q in samples:
    st.code(q, language="text")
