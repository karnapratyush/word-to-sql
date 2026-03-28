"""Document Upload Page -- Vision Document Extraction with Human-in-the-Loop Review.

This Streamlit page provides a full document upload and review interface
for the GoComet AI vision extraction pipeline. Users can:

1. Upload a PDF, PNG, or JPG document
2. Optionally select a document type hint (or let the AI auto-detect)
3. Click "Extract" to run the vision pipeline
4. Review extracted fields with confidence score indicators
5. Edit low-confidence fields that need review
6. Approve, correct, or reject the extraction
7. View previously extracted documents stored in the database

The page communicates with the FastAPI backend via the APIClient. It
never imports src/ modules directly -- all business logic runs on the
backend through the /api/documents/ endpoints.

Architecture:
    Streamlit (this page)
        -> APIClient (app/api_client.py)
            -> FastAPI POST /api/documents/extract
                -> VisionService -> Vision Agent -> Extractor + Validator
            -> FastAPI POST /api/documents/approve
                -> VisionService -> Vision Agent -> Storage + Cleanup
            -> FastAPI DELETE /api/documents/reject/{file_name}
                -> VisionService -> Vision Agent -> Cleanup
"""

import sys
import os

# Add project root to Python path so 'app' and 'src' packages are importable.
# Streamlit runs from app/ directory, but our imports expect the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd

# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Document Upload | GoComet AI",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Upload & Extraction")
st.markdown("Upload invoices, bills of lading, or packing lists for AI-powered field extraction.")

st.divider()


# ── Helper Functions ─────────────────────────────────────────────────

def get_confidence_color(confidence: float) -> str:
    """Return a color string based on the confidence score.

    Green for high confidence (>= 0.8), yellow/orange for medium
    (0.5 to 0.8), red for low (< 0.5). Used by the UI to color-code
    confidence indicators.

    Args:
        confidence: A float between 0.0 and 1.0.

    Returns:
        A CSS color string (e.g., "green", "orange", "red").
    """
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.5:
        return "orange"
    else:
        return "red"


def get_confidence_icon(confidence: float) -> str:
    """Return an emoji indicator based on the confidence score.

    Args:
        confidence: A float between 0.0 and 1.0.

    Returns:
        An emoji string indicating confidence level.
    """
    if confidence >= 0.8:
        return "✅"
    elif confidence >= 0.5:
        return "⚠️"
    else:
        return "🚩"


def render_confidence_badge(confidence: float) -> str:
    """Render a colored confidence badge as HTML/markdown.

    Args:
        confidence: A float between 0.0 and 1.0.

    Returns:
        A markdown string with a colored confidence indicator.
    """
    icon = get_confidence_icon(confidence)
    percentage = f"{confidence * 100:.0f}%"
    return f"{icon} {percentage}"


# ── Session State Initialization ─────────────────────────────────────
# Streamlit reruns the entire script on each interaction, so we use
# session_state to persist extraction results across reruns.

if "extraction_result" not in st.session_state:
    # Holds the extraction response dict from the API after upload
    st.session_state.extraction_result = None

if "extraction_saved" not in st.session_state:
    # Flag to show the success message after saving
    st.session_state.extraction_saved = False

if "extraction_rejected" not in st.session_state:
    # Flag to show the rejection message after rejecting
    st.session_state.extraction_rejected = False


# ── Section 1: File Upload ───────────────────────────────────────────
st.subheader("Upload Document")

# File uploader accepts PDF, PNG, and JPG files
uploaded_file = st.file_uploader(
    "Choose a document file",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Supported formats: PDF, PNG, JPG. Maximum size: 10 MB.",
)

# Optional document type dropdown -- user can let the AI auto-detect
# or provide a hint to improve extraction accuracy.
doc_type_options = [
    "auto-detect",
    "invoice",
    "bill_of_lading",
    "packing_list",
    "customs_declaration",
]
selected_doc_type = st.selectbox(
    "Document Type (optional)",
    options=doc_type_options,
    index=0,
    help="Select a document type to improve extraction accuracy, or leave as 'auto-detect' for AI classification.",
)

# Convert the selection to a hint value (None for auto-detect)
doc_type_hint = None if selected_doc_type == "auto-detect" else selected_doc_type

# Extract button -- only enabled when a file is uploaded
extract_button = st.button(
    "🔍 Extract Fields",
    disabled=(uploaded_file is None),
    type="primary",
    use_container_width=True,
)

# ── Handle Extraction ────────────────────────────────────────────────
# When the user clicks Extract, upload the file to the API and get results.

if extract_button and uploaded_file is not None:
    # Reset any previous state
    st.session_state.extraction_result = None
    st.session_state.extraction_saved = False
    st.session_state.extraction_rejected = False

    with st.spinner("Extracting fields from document... This may take 15-30 seconds."):
        try:
            from app.api_client import get_api_client
            client = get_api_client()

            # Read the uploaded file bytes
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name

            # Call the extraction API endpoint
            result = client.upload_document(
                file_bytes=file_bytes,
                file_name=file_name,
                doc_type_hint=doc_type_hint,
            )

            # Store the result in session state for review
            st.session_state.extraction_result = result
            # Store the original file name for approve/reject calls
            st.session_state.extraction_file_name = file_name

            st.rerun()

        except Exception as e:
            st.error(f"Extraction failed: {str(e)}")
            st.info("Make sure the API is running: `python run_api.py`")


# ── Show Success / Rejection Messages ────────────────────────────────
# These are shown after the user approves or rejects an extraction.

if st.session_state.extraction_saved:
    st.success("Document saved successfully! You can query it on the Query Documents page.")
    st.page_link("pages/3_Query_Documents.py", label="Go to Query Documents", icon="🔍")
    # Reset the flag so it does not persist across further interactions
    st.session_state.extraction_saved = False
    st.session_state.extraction_result = None

if st.session_state.extraction_rejected:
    st.info("Extraction rejected. The temporary file has been deleted.")
    # Reset the flag
    st.session_state.extraction_rejected = False
    st.session_state.extraction_result = None

st.divider()

# ── Section 2: Extraction Review ─────────────────────────────────────
# Show extraction results when available for review.

if st.session_state.extraction_result is not None:
    result = st.session_state.extraction_result

    st.subheader("Extraction Results — Review")

    # ── Overall Confidence Badge ─────────────────────────────────────
    overall_conf = result.get("overall_confidence", 0.0)
    conf_color = get_confidence_color(overall_conf)
    conf_icon = get_confidence_icon(overall_conf)

    col_meta1, col_meta2, col_meta3 = st.columns(3)

    with col_meta1:
        st.metric(
            "Overall Confidence",
            f"{overall_conf * 100:.0f}%",
        )
        # Color indicator
        if overall_conf >= 0.8:
            st.success(f"{conf_icon} High confidence extraction")
        elif overall_conf >= 0.5:
            st.warning(f"{conf_icon} Medium confidence -- review recommended")
        else:
            st.error(f"{conf_icon} Low confidence -- careful review required")

    with col_meta2:
        st.metric("Document Type", result.get("document_type", "unknown").replace("_", " ").title())

    with col_meta3:
        st.metric("Model Used", result.get("model_used", "N/A"))

    # Show notes/warnings if present
    notes = result.get("notes", "")
    if notes:
        st.warning(f"Notes: {notes}")

    st.divider()

    # ── Field-by-Field Review Table ──────────────────────────────────
    st.subheader("Extracted Fields")

    fields = result.get("fields", {})
    confidence_scores = result.get("confidence_scores", {})
    needs_review_flags = result.get("needs_review", {})

    if not fields:
        st.warning("No fields were extracted from this document.")
    else:
        # Build a review table with editable fields for low-confidence items.
        # We use a form so all edits are submitted together.

        # First, show a summary table
        summary_data = []
        for field_name, value in fields.items():
            conf = confidence_scores.get(field_name, 0.0)
            needs_review = needs_review_flags.get(field_name, False)
            summary_data.append({
                "Field": field_name.replace("_", " ").title(),
                "Value": str(value) if value is not None else "(empty)",
                "Confidence": f"{conf * 100:.0f}%",
                "Status": get_confidence_icon(conf),
                "Needs Review": "Yes" if needs_review else "No",
            })

        # Display as a dataframe for overview
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # ── Editable Fields — ALL fields are editable ────────────────
        # Every extracted field gets a text input so the user can
        # correct any value, not just low-confidence ones.
        # Low-confidence fields are highlighted with a warning badge.

        st.subheader("Edit Fields")
        st.markdown(
            "Review and edit any field below. "
            "Fields with low confidence are flagged with warnings."
        )

        # Track which fields the user has edited
        edited_fields = {}
        has_edits = False

        # Sort fields: low-confidence (needs_review) first, then the rest
        sorted_fields = sorted(
            fields.items(),
            key=lambda x: (not needs_review_flags.get(x[0], False), x[0]),
        )

        for field_name, original_value in sorted_fields:
            conf = confidence_scores.get(field_name, 0.0)
            needs_review = needs_review_flags.get(field_name, False)
            badge = render_confidence_badge(conf)

            # Format the value for display in text input
            # Lists and dicts are serialized to string for editing
            if isinstance(original_value, (list, dict)):
                display_value = str(original_value)
            elif original_value is not None:
                display_value = str(original_value)
            else:
                display_value = ""

            display_name = field_name.replace("_", " ").title()

            # Add warning label for low-confidence fields
            label = f"{badge} {display_name}"
            if needs_review:
                label += " (needs review)"

            new_value = st.text_input(
                label,
                value=display_value,
                key=f"edit_{field_name}",
                help=f"Confidence: {conf * 100:.0f}%. Edit this value if incorrect.",
            )

            # Track the edited value
            edited_fields[field_name] = new_value

            # Check if the user actually changed the value
            if str(new_value) != display_value:
                has_edits = True

        st.divider()

        # ── Action Buttons ───────────────────────────────────────────
        # Three buttons: Approve, Save as Corrected, Reject
        col_approve, col_correct, col_reject = st.columns(3)

        with col_approve:
            approve_button = st.button(
                "✅ Approve & Save",
                type="primary",
                use_container_width=True,
                help="Save the extraction as-is (no corrections needed).",
            )

        with col_correct:
            correct_button = st.button(
                "📝 Save as Corrected",
                use_container_width=True,
                disabled=(not has_edits),
                help="Save with your corrections applied. Only enabled when you edit a field.",
            )

        with col_reject:
            reject_button = st.button(
                "❌ Reject",
                use_container_width=True,
                help="Reject the extraction and delete the temporary file.",
            )

        # ── Handle Approve ───────────────────────────────────────────
        if approve_button:
            with st.spinner("Saving extraction to database..."):
                try:
                    from app.api_client import get_api_client
                    client = get_api_client()

                    # Build the approve request data
                    approve_data = {
                        "temp_file_name": st.session_state.get("extraction_file_name", "unknown"),
                        "document_type": result.get("document_type", "unknown"),
                        "extracted_fields": fields,
                        "confidence_scores": confidence_scores,
                        "overall_confidence": overall_conf,
                        "extraction_model": result.get("model_used", ""),
                        "review_status": "approved",
                        "notes": notes,
                    }

                    # Call the approve API endpoint
                    record = client.approve_document(approve_data)

                    # Set the saved flag and clear the extraction result
                    st.session_state.extraction_saved = True
                    st.session_state.extraction_result = None
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to save document: {str(e)}")

        # ── Handle Save as Corrected ─────────────────────────────────
        if correct_button and has_edits:
            with st.spinner("Saving corrected extraction to database..."):
                try:
                    from app.api_client import get_api_client
                    client = get_api_client()

                    # Merge the edited fields into the original fields
                    corrected_fields = dict(fields)  # Start with original
                    for field_name, new_value in edited_fields.items():
                        corrected_fields[field_name] = new_value

                    # Build the approve request data with "corrected" status
                    approve_data = {
                        "temp_file_name": st.session_state.get("extraction_file_name", "unknown"),
                        "document_type": result.get("document_type", "unknown"),
                        "extracted_fields": corrected_fields,
                        "confidence_scores": confidence_scores,
                        "overall_confidence": overall_conf,
                        "extraction_model": result.get("model_used", ""),
                        "review_status": "corrected",
                        "notes": f"User corrected fields: {', '.join(edited_fields.keys())}. {notes}",
                    }

                    # Call the approve API endpoint
                    record = client.approve_document(approve_data)

                    # Set the saved flag and clear the extraction result
                    st.session_state.extraction_saved = True
                    st.session_state.extraction_result = None
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to save corrected document: {str(e)}")

        # ── Handle Reject ────────────────────────────────────────────
        if reject_button:
            with st.spinner("Rejecting extraction and cleaning up..."):
                try:
                    from app.api_client import get_api_client
                    client = get_api_client()

                    # Call the reject API endpoint to delete the temp file
                    file_name = st.session_state.get("extraction_file_name", "unknown")
                    client.reject_upload(file_name)

                    # Set the rejected flag and clear the extraction result
                    st.session_state.extraction_rejected = True
                    st.session_state.extraction_result = None
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to reject document: {str(e)}")


# ── Section 3: Previously Extracted Documents ────────────────────────
st.divider()
st.subheader("Previously Extracted Documents")

try:
    from app.api_client import get_api_client
    client = get_api_client()

    # Fetch all documents from the database
    documents = client.list_documents()

    if not documents:
        st.info("No documents have been extracted yet. Upload a document above to get started.")
    else:
        # Build a summary table for display
        doc_table_data = []
        for doc in documents:
            doc_table_data.append({
                "Document ID": doc.get("document_id", "")[:8] + "...",
                "Type": doc.get("document_type", "unknown").replace("_", " ").title(),
                "File Name": doc.get("file_name", ""),
                "Confidence": f"{doc.get('overall_confidence', 0) * 100:.0f}%",
                "Status": doc.get("review_status", "pending").title(),
                "Model": doc.get("extraction_model", "N/A") or "N/A",
                "Fields": len(doc.get("extracted_fields", {})),
            })

        doc_df = pd.DataFrame(doc_table_data)
        st.dataframe(
            doc_df,
            use_container_width=True,
            hide_index=True,
        )

        # Expandable detail view for each document
        with st.expander("View Document Details", expanded=False):
            # Let the user select a document to view details
            doc_ids = [doc.get("document_id", "") for doc in documents]
            doc_labels = [
                f"{doc.get('document_type', 'unknown')} - {doc.get('file_name', '')} ({doc.get('document_id', '')[:8]}...)"
                for doc in documents
            ]

            if doc_labels:
                selected_label = st.selectbox("Select a document", doc_labels)
                selected_idx = doc_labels.index(selected_label)
                selected_doc = documents[selected_idx]

                # Show the full extracted fields
                st.markdown(f"**Document ID:** `{selected_doc.get('document_id', '')}`")
                st.markdown(f"**Type:** {selected_doc.get('document_type', 'unknown').replace('_', ' ').title()}")
                st.markdown(f"**File:** {selected_doc.get('file_name', '')}")
                st.markdown(f"**Overall Confidence:** {selected_doc.get('overall_confidence', 0) * 100:.0f}%")
                st.markdown(f"**Review Status:** {selected_doc.get('review_status', 'pending').title()}")
                st.markdown(f"**Model:** {selected_doc.get('extraction_model', 'N/A')}")

                # Show extracted fields as a table
                ext_fields = selected_doc.get("extracted_fields", {})
                conf_scores = selected_doc.get("confidence_scores", {})

                if ext_fields:
                    detail_data = []
                    for name, value in ext_fields.items():
                        conf = conf_scores.get(name, 0.0)
                        detail_data.append({
                            "Field": name.replace("_", " ").title(),
                            "Value": str(value) if value is not None else "(empty)",
                            "Confidence": f"{conf * 100:.0f}%" if isinstance(conf, (int, float)) else "N/A",
                        })

                    detail_df = pd.DataFrame(detail_data)
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.warning(f"Could not load documents: {str(e)}")
    st.info("Make sure the API is running: `python run_api.py`")
