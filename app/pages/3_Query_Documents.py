"""Query Documents Page -- End-to-End Linkage (Capability C).

This Streamlit page demonstrates the linkage between the analytics
pipeline and the vision extraction pipeline. Users can ask natural
language questions that span BOTH shipment data AND extracted document
data, because the analytics agent already has access to the
extracted_documents table in the same SQLite database.

This uses the same analytics API endpoint (POST /api/analytics/query)
as the Analytics page, but provides sample questions specifically about
extracted documents and their relationship to shipments.

Key insight: No special backend changes are needed for linkage. The
analytics agent's SQL generator already sees the extracted_documents
table in the schema description, so it can generate JOINs between
shipments and extracted_documents when the user's question requires it.

Architecture:
    Streamlit (this page)
        -> APIClient.query_analytics()
            -> FastAPI POST /api/analytics/query
                -> Analytics Agent (same pipeline as Analytics page)
                    -> SQL Generator sees extracted_documents + shipment tables
                    -> Can generate JOINs across both data sources
"""

import sys
import os

# Add project root to Python path so 'app' and 'src' packages are importable.
# Streamlit runs from app/ directory, but our imports expect the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import plotly.graph_objects as go
import json

# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Query Documents | GoComet AI",
    page_icon="🔗",
    layout="wide",
)

st.title("🔗 Query Extracted Documents")
st.markdown(
    "Ask questions about extracted documents alongside shipment analytics. "
    "This demonstrates **Capability C: End-to-End Linkage** -- the analytics "
    "agent can query both shipment data and extracted document data because "
    "they share the same database."
)

st.divider()

# ── Session State Initialization ─────────────────────────────────────
# Streamlit reruns the entire script on each interaction, so we use
# session_state to persist conversation history across reruns.

if "doc_messages" not in st.session_state:
    # Separate message history from the Analytics page
    st.session_state.doc_messages = []

if "doc_session_id" not in st.session_state:
    import uuid
    st.session_state.doc_session_id = str(uuid.uuid4())

# ── Sidebar ──────────────────────────────────────────────────────────
# Document-specific sample questions and conversation controls
with st.sidebar:
    st.subheader("Sample Document Queries")
    st.markdown(
        "These questions query the `extracted_documents` table "
        "and can be joined with shipment data."
    )

    # Sample questions that reference extracted document data
    doc_samples = [
        "Show all extracted documents",
        "What is the total amount from extracted invoices?",
        "Which shipments have linked documents?",
        "Show documents with confidence below 0.8",
        "How many documents have been approved vs pending?",
        "List all bill of lading documents",
        "What is the average extraction confidence by document type?",
        "Show extracted invoice fields for the most recent document",
        "Which documents were corrected by reviewers?",
        "Count documents by type and review status",
    ]

    # Each sample question is a button that sets a pending query
    for sample in doc_samples:
        if st.button(sample, key=f"doc_sample_{sample[:25]}", use_container_width=True):
            st.session_state.doc_pending_query = sample

    st.divider()

    st.subheader("Cross-Data Queries")
    st.markdown("These join shipment data with extracted documents.")

    cross_samples = [
        "Show shipments that have linked extracted documents",
        "Compare freight costs for shipments with and without documents",
        "Which carriers have the most linked documents?",
    ]

    for sample in cross_samples:
        if st.button(sample, key=f"cross_sample_{sample[:25]}", use_container_width=True):
            st.session_state.doc_pending_query = sample

    st.divider()

    # Clear conversation button resets messages and generates a new session
    if st.button("🗑️ Clear Conversation", key="doc_clear", use_container_width=True):
        st.session_state.doc_messages = []
        st.session_state.doc_session_id = __import__("uuid").uuid4().hex
        st.rerun()

    st.divider()
    # Display truncated session ID for debugging/tracking
    st.caption(f"Session: `{st.session_state.doc_session_id[:8]}...`")

# ── Chat History Display ─────────────────────────────────────────────
# Render all previous messages from session state
for msg in st.session_state.doc_messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

        # Show additional data for assistant messages (SQL, table, chart)
        if role == "assistant" and "extras" in msg:
            extras = msg["extras"]

            # SQL query used (collapsible for transparency)
            if extras.get("sql_query"):
                with st.expander("🔍 SQL Query Used", expanded=False):
                    st.code(extras["sql_query"], language="sql")

            # Result data table (collapsible)
            if extras.get("result_table"):
                with st.expander(
                    f"📋 Result Table ({len(extras['result_table'])} rows)",
                    expanded=False,
                ):
                    st.dataframe(extras["result_table"], use_container_width=True)

            # Plotly chart visualization
            if extras.get("chart_data"):
                try:
                    fig = go.Figure(extras["chart_data"])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass  # Skip chart if Plotly data is malformed

            # Model attribution
            if extras.get("model_used"):
                st.caption(f"Model: `{extras['model_used']}`")

            # Error display
            if extras.get("error"):
                st.error(f"⚠️ {extras['error']}")

# ── Chat Input ───────────────────────────────────────────────────────
# Handle both manual input and sidebar sample question clicks

# Check if a sample question was clicked (stored in session_state)
pending = st.session_state.pop("doc_pending_query", None)
user_input = st.chat_input("Ask a question about extracted documents or shipments...")

# Use pending sample question if available, otherwise use typed input
query = pending or user_input

if query:
    # Display the user's message immediately
    st.session_state.doc_messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Build conversation history for context (exclude the current message).
    # This enables follow-up questions like "Show me details for the first one"
    # after a listing query.
    history = []
    for msg in st.session_state.doc_messages[:-1]:
        history.append({
            "role": msg["role"],
            "content": msg["content"],
        })

    # Call the FastAPI backend via APIClient.
    # We use the SAME analytics endpoint because the analytics agent
    # already has access to the extracted_documents table in the schema.
    with st.chat_message("assistant"):
        with st.spinner("Querying documents and shipments..."):
            try:
                from app.api_client import get_api_client
                client = get_api_client()
                result = client.query_analytics(
                    user_query=query,
                    conversation_history=history,
                    session_id=st.session_state.doc_session_id,
                )

                # Display the natural language answer
                answer = result.get("answer", "No answer received.")
                st.markdown(answer)

                # Collect all extra data for display and history storage
                extras = {
                    "sql_query": result.get("sql_query"),
                    "result_table": result.get("result_table", []),
                    "chart_data": result.get("chart_data"),
                    "chart_type": result.get("chart_type"),
                    "model_used": result.get("model_used", ""),
                    "error": result.get("error"),
                }

                # Show SQL query (collapsible)
                if extras["sql_query"]:
                    with st.expander("🔍 SQL Query Used", expanded=False):
                        st.code(extras["sql_query"], language="sql")

                # Show result table (collapsible)
                if extras["result_table"]:
                    with st.expander(
                        f"📋 Result Table ({len(extras['result_table'])} rows)",
                        expanded=False,
                    ):
                        st.dataframe(extras["result_table"], use_container_width=True)

                # Show chart visualization
                if extras["chart_data"]:
                    try:
                        fig = go.Figure(extras["chart_data"])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass  # Skip chart if Plotly data is malformed

                # Show model attribution
                if extras["model_used"]:
                    st.caption(f"Model: `{extras['model_used']}`")

                # Show error if present
                if extras["error"]:
                    st.error(f"⚠️ {extras['error']}")

                # Save assistant response to session state for history
                st.session_state.doc_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "extras": extras,
                })

            except Exception as e:
                # API connection error -- display helpful message
                error_msg = f"Failed to connect to API: {str(e)}"
                st.error(error_msg)
                st.info("Make sure the API is running: `python run_api.py`")
                st.session_state.doc_messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "extras": {"error": str(e)},
                })

# ── Bottom Info Section ──────────────────────────────────────────────
st.divider()

with st.expander("How does linkage work?", expanded=False):
    st.markdown("""
**End-to-End Linkage** means the analytics engine can query both:
- **Shipment data** (`shipments`, `carriers`, `customers`, `invoices`, etc.)
- **Extracted documents** (`extracted_documents` table)

...in a single SQL query. This is possible because:

1. The vision pipeline stores extracted document data in the same SQLite database
2. The `extracted_documents` table has a `linked_shipment_id` foreign key to `shipments`
3. The analytics agent's SQL generator sees ALL tables in the schema description
4. When you ask a question that spans both data sources, the LLM generates JOINs

**Example**: "Which shipments have linked documents?" generates:
```sql
SELECT s.shipment_id, s.status, ed.document_type, ed.overall_confidence
FROM shipments s
JOIN extracted_documents ed ON s.shipment_id = ed.linked_shipment_id
```

This demonstrates that the analytics and vision capabilities are not siloed --
they share a unified data layer.
""")
