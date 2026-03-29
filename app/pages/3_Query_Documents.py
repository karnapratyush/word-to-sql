"""Query Documents — redirects to Analytics page.

Extracted document data is queryable through the same Analytics chat
because both share the same SQLite database. No separate page needed.
This is Capability C (linkage) — the analytics agent sees both
shipment tables and extracted_documents in the same schema.
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

st.set_page_config(page_title="Query Documents |   AI", page_icon="🔗", layout="wide")

st.title("🔗 Query Extracted Documents")

st.info(
    "Extracted document data is queried through the **Analytics** page. "
    "Both shipment data and extracted documents share the same database, "
    "so you can ask questions about either — or both — in one place."
)

st.page_link("pages/1_Analytics.py", label="Go to Analytics", icon="📊")

st.markdown(
    "**Try these on the Analytics page:**\n"
    "- Show all extracted documents\n"
    "- Give me details about BL number BL-2024-KR-US-0841\n"
    "- Which shipments have linked documents?\n"
    "- Compare extracted invoice amount with actual shipment charges"
)
