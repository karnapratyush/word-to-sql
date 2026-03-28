"""Analytics Chat Page — NL-to-SQL conversational interface.

This Streamlit page provides a chat-style interface for querying the
logistics database using natural language. It communicates with the
FastAPI backend via the APIClient.

Features:
- Chat-style input with full conversation history
- Shows: text answer, SQL used, result data table, Plotly visualization
- Follow-up support (conversation context sent with each query)
- Error display with helpful suggestions
- Sample questions in sidebar for quick access
- Session management with clear conversation button
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
import json

# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics | GoComet AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Analytics Chat")
st.markdown("Ask questions about your logistics data in natural language.")

# ── Session State Initialization ─────────────────────────────────────
# Streamlit reruns the entire script on each interaction, so we use
# session_state to persist conversation history across reruns.

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# ── Sidebar ──────────────────────────────────────────────────────────
# Sample questions and conversation controls
with st.sidebar:
    st.subheader("Sample Questions")

    samples = [
        "How many shipments are currently delayed?",
        "Average freight cost by carrier for ocean shipments",
        "Top 5 routes by shipment volume",
        "Which carriers have the highest delay rate?",
        "Monthly shipment count trend",
        "Compare ocean vs air shipping costs",
        "Most common delay reasons",
        "Total invoice amount by payment status",
        "Shipments from India to USA",
        "Top 10 customers by total shipping cost",
    ]

    # Each sample question is a button that sets a pending query
    for sample in samples:
        if st.button(sample, key=f"sample_{sample[:20]}", use_container_width=True):
            st.session_state.pending_query = sample

    st.divider()

    # Clear conversation button resets messages and generates a new session
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = __import__("uuid").uuid4().hex
        st.rerun()

    st.divider()
    # Display truncated session ID for debugging/tracking
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

# ── Chat History Display ─────────────────────────────────────────────
# Render all previous messages from session state
for msg in st.session_state.messages:
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
                with st.expander(f"📋 Result Table ({len(extras['result_table'])} rows)", expanded=False):
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
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask a question about your logistics data...")

# Use pending sample question if available, otherwise use typed input
query = pending or user_input

if query:
    # Display the user's message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Build conversation history for context (exclude the current message)
    history = []
    for msg in st.session_state.messages[:-1]:
        history.append({
            "role": msg["role"],
            "content": msg["content"],
        })

    # Call the FastAPI backend via APIClient
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):
            try:
                from app.api_client import get_api_client
                client = get_api_client()
                result = client.query_analytics(
                    user_query=query,
                    conversation_history=history,
                    session_id=st.session_state.session_id,
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
                    with st.expander(f"📋 Result Table ({len(extras['result_table'])} rows)", expanded=False):
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
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "extras": extras,
                })

            except Exception as e:
                # API connection error — display helpful message
                error_msg = f"Failed to connect to API: {str(e)}"
                st.error(error_msg)
                st.info("Make sure the API is running: `python run_api.py`")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "extras": {"error": str(e)},
                })
