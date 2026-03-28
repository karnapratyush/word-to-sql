"""GoComet AI Logistics Assistant — Home Page.

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
    page_title="GoComet AI Logistics Assistant",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 GoComet AI Logistics Assistant")
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

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Analytics")
    st.markdown("Ask questions about shipments, carriers, costs, and delays in natural language.")
    st.page_link("pages/1_Analytics.py", label="Open Analytics Chat", icon="💬")

with col2:
    st.markdown("### 📄 Document Upload")
    st.markdown("Upload invoices, bills of lading, or packing lists for AI extraction.")
    st.page_link("pages/2_Document_Upload.py", label="Upload Documents", icon="📤")

with col3:
    st.markdown("### 🔗 Query Documents")
    st.markdown("Query extracted document data alongside shipment analytics.")
    st.page_link("pages/3_Query_Documents.py", label="Query Documents", icon="🔍")

st.divider()

# ── Sample Questions ─────────────────────────────────────────────────
# Provide example queries to help users get started
st.subheader("Sample Analytics Questions")

samples = [
    "How many shipments are currently delayed?",
    "What is the average freight cost by carrier for ocean shipments?",
    "Show me the top 5 routes by shipment volume",
    "Which carriers have the highest delay rate?",
    "Monthly shipment count trend for 2024",
    "Compare ocean vs air shipping costs",
    "What are the most common delay reasons?",
    "Show total invoice amount by payment status",
]

for q in samples:
    st.code(q, language="text")
