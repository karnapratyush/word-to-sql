"""Verification Page -- Document Verification Against Customer Rules.

This Streamlit page provides a full document verification interface with
four states:

1. Incoming: Upload a document with a customer selector, view recent
   verifications in a summary table.

2. Result: After verification, show a field-by-field comparison table
   with color-coded status badges (green=match, red=mismatch,
   yellow=uncertain, grey=no_rule).

3. Detail: Expandable discrepancy detail per field, showing extracted
   vs. expected values and the specific rule that was violated.

4. Draft: Editable text area with the generated email draft, plus
   action buttons to approve, send for amendment, or request re-review.

The page communicates with the FastAPI backend via the APIClient. It
never imports src/ modules directly -- all business logic runs on the
backend through the /api/verification/ endpoints.

Architecture:
    Streamlit (this page)
        -> APIClient (app/api_client.py)
            -> FastAPI POST /api/verification/process
                -> VerificationService -> Verification Agent
            -> FastAPI GET /api/verification
            -> FastAPI GET /api/verification/{id}
            -> FastAPI PUT /api/verification/{id}/review
            -> FastAPI GET /api/verification/stats
            -> FastAPI GET /api/verification/customers
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
    page_title="Verification | AI",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Document Verification")
st.markdown(
    "Verify shipping documents against customer-specific rules. "
    "Upload a document, select the customer, and the system will "
    "automatically compare extracted fields against the rules."
)

st.divider()


# ── Helper Functions ─────────────────────────────────────────────────

def get_status_badge(status: str) -> str:
    """Return a colored emoji badge for a verification status.

    Args:
        status: One of match, mismatch, uncertain, no_rule,
            approved, amendment_required, extraction_failed, no_rules.

    Returns:
        An emoji string indicating the status.
    """
    badges = {
        "match": "🟢",
        "approved": "🟢",
        "mismatch": "🔴",
        "amendment_required": "🔴",
        "uncertain": "🟡",
        "no_rule": "⚪",
        "extraction_failed": "🔴",
        "no_rules": "🔴",
    }
    return badges.get(status, "⚪")


def get_status_label(status: str) -> str:
    """Return a human-readable label for a verification status.

    Args:
        status: The raw status string.

    Returns:
        A formatted status label.
    """
    labels = {
        "match": "Match",
        "mismatch": "Mismatch",
        "uncertain": "Uncertain",
        "no_rule": "No Rule",
        "approved": "Approved",
        "amendment_required": "Amendment Required",
        "extraction_failed": "Extraction Failed",
        "no_rules": "No Customer Rules",
    }
    return labels.get(status, status.replace("_", " ").title())


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


# ── Session State Initialization ─────────────────────────────────────
# Streamlit reruns the entire script on each interaction, so we use
# session_state to persist verification results across reruns.

if "verification_result" not in st.session_state:
    # Holds the verification response dict from the API after processing
    st.session_state.verification_result = None

if "verification_reviewed" not in st.session_state:
    # Flag to show the success message after review submission
    st.session_state.verification_reviewed = False

if "selected_verification_id" not in st.session_state:
    # Tracks which verification the user selected for detail view
    st.session_state.selected_verification_id = None


# ── STATE 1: Incoming — Upload and Customer Selection ────────────────
st.subheader("Upload Document for Verification")

# Load available customers from the API for the dropdown
customers = []
try:
    from app.api_client import get_api_client
    client = get_api_client()
    customers = client.list_verification_customers()
except Exception as e:
    st.warning(f"Could not load customer list: {e}")
    st.info("Make sure the API is running: `python run_api.py`")

# Customer selector dropdown
if customers:
    customer_options = {
        f"{c['customer_name']} ({c['customer_id']}) - {c['rule_count']} rules": c["customer_id"]
        for c in customers
    }
    selected_customer_label = st.selectbox(
        "Select Customer",
        options=list(customer_options.keys()),
        help="Choose the customer whose rules will be used for verification.",
    )
    selected_customer_id = customer_options[selected_customer_label]
else:
    st.warning("No customers with verification rules found. Add YAML files to config/customer_rules/.")
    selected_customer_id = st.text_input(
        "Customer ID",
        value="sample_customer",
        help="Enter the customer ID manually (must match a YAML filename).",
    )

# Shipment reference (required -- used for grouping multiple docs in a batch)
shipment_ref = st.text_input(
    "Shipment Reference",
    value="",
    help="Required. Shipment reference number used to group and track documents for the same shipment.",
)

# File uploader — accepts multiple files for the same shipment
uploaded_files = st.file_uploader(
    "Choose documents to verify",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Upload one or more documents for the same shipment. Each will be verified independently against the customer's rules.",
)

# ── Session state for two-phase flow ────────────────────────────────
if "extraction_results" not in st.session_state:
    st.session_state.extraction_results = []  # Phase 1 results
if "verification_results_batch" not in st.session_state:
    st.session_state.verification_results_batch = []  # Phase 2 results
if "phase" not in st.session_state:
    st.session_state.phase = "upload"  # upload → review → verify → results

# ── PHASE 1: Extract ────────────────────────────────────────────────
extract_button = st.button(
    "Step 1: Extract Fields",
    disabled=(len(uploaded_files) == 0),
    type="primary",
    use_container_width=True,
)

if extract_button and not shipment_ref.strip():
    st.warning("Shipment Reference is required.")
    extract_button = False

if extract_button and len(uploaded_files) > 0:
    st.session_state.extraction_results = []
    st.session_state.verification_results_batch = []
    st.session_state.verification_result = None
    st.session_state.phase = "review"

    from app.api_client import get_api_client
    client = get_api_client()

    extractions = []
    progress = st.progress(0, text="Extracting fields...")

    for i, uploaded_file in enumerate(uploaded_files):
        progress.progress(i / len(uploaded_files),
                         text=f"Extracting {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
        try:
            result = client.upload_document(
                file_bytes=uploaded_file.getvalue(),
                file_name=uploaded_file.name,
            )
            result["_file_name"] = uploaded_file.name
            extractions.append(result)
        except Exception as e:
            extractions.append({
                "_file_name": uploaded_file.name,
                "error": str(e),
                "fields": {},
                "confidence_scores": {},
                "document_type": "unknown",
            })

    progress.progress(1.0, text=f"Extracted {len(extractions)} document(s)")
    st.session_state.extraction_results = extractions
    st.rerun()

# ── PHASE 2: Review Extracted Fields ────────────────────────────────
if st.session_state.extraction_results and st.session_state.phase == "review":
    st.divider()
    st.subheader("Step 2: Review Extracted Fields")
    st.markdown("Review the fields extracted from each document. Edit any incorrect values before verification.")

    edited_documents = []

    for idx, extraction in enumerate(st.session_state.extraction_results):
        file_name = extraction.get("_file_name", f"Document {idx+1}")
        doc_type = extraction.get("document_type", "unknown")
        fields = extraction.get("fields", {})
        scores = extraction.get("confidence_scores", {})
        needs_review = extraction.get("needs_review", {})
        error = extraction.get("error")

        with st.expander(f"{file_name} ({doc_type.replace('_', ' ').title()})", expanded=True):
            if error:
                st.error(f"Extraction failed: {error}")
                continue

            st.caption(f"Overall confidence: {extraction.get('overall_confidence', 0) * 100:.0f}% | Model: {extraction.get('model_used', 'N/A')}")

            edited_fields = {}
            for field_name, value in fields.items():
                conf = scores.get(field_name, 0)
                review_flag = needs_review.get(field_name, False)
                icon = get_confidence_icon(conf)
                label = f"{icon} {field_name.replace('_', ' ').title()} ({conf*100:.0f}%)"
                if review_flag:
                    label += " — needs review"

                display_val = str(value) if value is not None else ""
                if isinstance(value, (list, dict)):
                    display_val = str(value)

                edited_val = st.text_input(
                    label,
                    value=display_val,
                    key=f"edit_{idx}_{field_name}",
                )
                edited_fields[field_name] = edited_val

            edited_documents.append({
                "file_name": file_name,
                "document_type": doc_type,
                "extracted_fields": edited_fields,
                "confidence_scores": scores,
            })

    st.divider()

    # Confirm and verify button
    verify_button = st.button(
        "Step 3: Confirm & Verify Against Customer Rules",
        type="primary",
        use_container_width=True,
    )

    if verify_button:
        st.session_state.phase = "verify"
        from app.api_client import get_api_client
        client = get_api_client()

        with st.spinner("Comparing against customer rules..."):
            try:
                result = client.compare_confirmed_fields(
                    customer_id=selected_customer_id,
                    shipment_ref=shipment_ref,
                    documents=edited_documents,
                )
                st.session_state.verification_result = result
                st.session_state.verification_results_batch = [result]
                st.session_state.phase = "results"
                st.rerun()
            except Exception as e:
                st.error(f"Verification failed: {e}")

# ── Show Review Success Message ──────────────────────────────────────
if st.session_state.verification_reviewed:
    st.success("Review submitted successfully!")
    st.session_state.verification_reviewed = False

st.divider()


# ── STATE 2 & 3: Result + Detail View ───────────────────────────────
if st.session_state.verification_result is not None:
    result = st.session_state.verification_result

    # Check for pipeline errors
    error = result.get("error")
    if error:
        st.error(f"Verification Error: {error}")
        overall_status = result.get("overall_status", "unknown")
        st.markdown(f"**Status:** {get_status_badge(overall_status)} {get_status_label(overall_status)}")

        # Show a button to clear and try again
        if st.button("Clear and Try Again"):
            st.session_state.verification_result = None
            st.rerun()
    else:
        # ── Overall Status Banner ────────────────────────────────────
        overall_status = result.get("overall_status", "unknown")
        status_badge = get_status_badge(overall_status)
        status_label = get_status_label(overall_status)

        # Color the banner based on status
        if overall_status == "approved":
            st.success(f"{status_badge} Verification Result: **{status_label}**")
        elif overall_status == "amendment_required":
            st.error(f"{status_badge} Verification Result: **{status_label}**")
        else:
            st.warning(f"{status_badge} Verification Result: **{status_label}**")

        # ── Metadata Row ─────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Customer", result.get("customer_name", "N/A"))

        with col2:
            st.metric("Document Type", (result.get("document_type") or "N/A").replace("_", " ").title())

        with col3:
            st.metric("Shipment Ref", result.get("shipment_ref") or "N/A")

        with col4:
            st.metric("Verification ID", result.get("verification_id", "")[:8] + "...")

        # ── Document Type Verification Warning ──────────────────────
        # If the classified document type does not match the filename
        # hint, show a warning. This is informational, not blocking.
        doc_type_check = result.get("doc_type_check")
        if doc_type_check and doc_type_check.get("warning"):
            st.warning(f"Document Type Warning: {doc_type_check['warning']}")

        st.divider()

        # ── STATE 2: Field-by-Field Comparison Table ─────────────────
        st.subheader("Field Comparison Results")

        comparisons = result.get("comparisons", [])

        if comparisons:
            # Build the summary table data
            table_data = []
            for comp in comparisons:
                field_display = comp.get("field_name", "").replace("_", " ").title()
                status = comp.get("status", "no_rule")
                confidence = comp.get("confidence", 0.0)

                table_data.append({
                    "Status": f"{get_status_badge(status)} {get_status_label(status)}",
                    "Field": field_display,
                    "Extracted": str(comp.get("extracted_value") or "(empty)"),
                    "Expected": str(comp.get("expected_value") or "(no rule)"),
                    "Confidence": f"{get_confidence_icon(confidence)} {confidence * 100:.0f}%",
                    "Rule Type": (comp.get("rule_type") or "N/A").replace("_", " ").title(),
                })

            # Display as a dataframe
            df = pd.DataFrame(table_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            # ── Comparison summary metrics ───────────────────────────
            match_count = sum(1 for c in comparisons if c.get("status") == "match")
            mismatch_count = sum(1 for c in comparisons if c.get("status") == "mismatch")
            uncertain_count = sum(1 for c in comparisons if c.get("status") == "uncertain")
            no_rule_count = sum(1 for c in comparisons if c.get("status") == "no_rule")

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Matches", f"🟢 {match_count}")
            with mcol2:
                st.metric("Mismatches", f"🔴 {mismatch_count}")
            with mcol3:
                st.metric("Uncertain", f"🟡 {uncertain_count}")
            with mcol4:
                st.metric("No Rule", f"⚪ {no_rule_count}")

            st.divider()

            # ── STATE 3: Expandable Discrepancy Detail ───────────────
            # Show detail only for non-matching fields
            issues = [c for c in comparisons if c.get("status") in ("mismatch", "uncertain")]

            if issues:
                st.subheader("Discrepancy Details")

                for comp in issues:
                    field_display = comp.get("field_name", "").replace("_", " ").title()
                    status = comp.get("status", "")
                    badge = get_status_badge(status)

                    with st.expander(f"{badge} {field_display} - {get_status_label(status)}", expanded=True):
                        detail_col1, detail_col2 = st.columns(2)

                        with detail_col1:
                            st.markdown("**Extracted Value:**")
                            st.code(str(comp.get("extracted_value") or "(empty)"))

                        with detail_col2:
                            st.markdown("**Expected Value:**")
                            st.code(str(comp.get("expected_value") or "(no rule)"))

                        st.markdown(f"**Confidence:** {get_confidence_icon(comp.get('confidence', 0))} {comp.get('confidence', 0) * 100:.0f}%")
                        st.markdown(f"**Rule Type:** {(comp.get('rule_type') or 'N/A').replace('_', ' ').title()}")

                        violation = comp.get("rule_violated")
                        if violation:
                            st.markdown(f"**Rule Violated:** {violation}")

                st.divider()

        else:
            st.info("No field comparisons available for this verification.")

        # ── STATE 4: Draft Email ─────────────────────────────────────
        st.subheader("Draft Reply Email")

        draft = result.get("draft_reply", "")

        if draft:
            # Editable text area for the draft email
            edited_draft = st.text_area(
                "Edit the draft email before sending",
                value=draft,
                height=300,
                key="draft_email_editor",
            )

            # Notes from the reviewer
            reviewer_name = st.text_input(
                "Your Name (for review record)",
                value="",
                key="reviewer_name",
            )

            review_notes = st.text_input(
                "Review Notes (optional)",
                value="",
                key="review_notes",
            )

            # Action buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)

            with btn_col1:
                approve_btn = st.button(
                    "✅ Approve Verification",
                    type="primary",
                    use_container_width=True,
                    disabled=(not reviewer_name),
                    help="Confirm the automated verification result. Enter your name first.",
                )

            with btn_col2:
                override_btn = st.button(
                    "🔄 Override Status",
                    use_container_width=True,
                    disabled=(not reviewer_name),
                    help="Override the automated status (e.g., approve despite mismatches).",
                )

            with btn_col3:
                clear_btn = st.button(
                    "🗑️ Clear Result",
                    use_container_width=True,
                    help="Clear the current verification result.",
                )

            # Handle approve
            if approve_btn and reviewer_name:
                with st.spinner("Submitting review..."):
                    try:
                        from app.api_client import get_api_client
                        client = get_api_client()

                        client.review_verification(
                            verification_id=result["verification_id"],
                            reviewed_by=reviewer_name,
                            notes=review_notes if review_notes else None,
                            draft_reply=edited_draft if edited_draft != draft else None,
                        )

                        st.session_state.verification_reviewed = True
                        st.session_state.verification_result = None
                        st.rerun()

                    except Exception as e:
                        st.error(f"Failed to submit review: {e}")

            # Handle override
            if override_btn and reviewer_name:
                # Show a dropdown to select the new status
                new_status = st.selectbox(
                    "Select new status",
                    ["approved", "amendment_required", "uncertain"],
                    key="override_status",
                )
                confirm_override = st.button(
                    "Confirm Override",
                    key="confirm_override",
                )
                if confirm_override:
                    with st.spinner("Submitting override..."):
                        try:
                            from app.api_client import get_api_client
                            client = get_api_client()

                            client.review_verification(
                                verification_id=result["verification_id"],
                                reviewed_by=reviewer_name,
                                notes=f"Status overridden. {review_notes}".strip(),
                                overall_status=new_status,
                                draft_reply=edited_draft if edited_draft != draft else None,
                            )

                            st.session_state.verification_reviewed = True
                            st.session_state.verification_result = None
                            st.rerun()

                        except Exception as e:
                            st.error(f"Failed to submit override: {e}")

            # Handle clear
            if clear_btn:
                st.session_state.verification_result = None
                st.rerun()

        else:
            st.info("No draft email was generated for this verification.")

        # Show any notes/warnings
        notes = result.get("notes", "")
        if notes:
            st.warning(f"Notes: {notes}")


# ── Recent Verifications Table ───────────────────────────────────────
st.divider()
st.subheader("Recent Verifications")

try:
    from app.api_client import get_api_client
    client = get_api_client()

    # Fetch all verifications from the API
    verifications = client.list_verifications()

    if not verifications:
        st.info("No verifications have been processed yet. Upload a document above to get started.")
    else:
        # Build summary table
        v_table_data = []
        for v in verifications:
            v_id = v.get("verification_id", "")
            status = v.get("overall_status", "unknown")

            v_table_data.append({
                "ID": v_id[:8] + "..." if v_id else "",
                "Status": f"{get_status_badge(status)} {get_status_label(status)}",
                "Customer": v.get("customer_id", ""),
                "Shipment": v.get("shipment_ref") or "N/A",
                "Received": v.get("received_at", "")[:19] if v.get("received_at") else "N/A",
                "Reviewed By": v.get("reviewed_by") or "Pending",
            })

        v_df = pd.DataFrame(v_table_data)
        st.dataframe(v_df, use_container_width=True, hide_index=True)

        # Detail view for a selected verification
        with st.expander("View Verification Details", expanded=False):
            v_ids = [v.get("verification_id", "") for v in verifications]
            v_labels = [
                f"{v.get('overall_status', '?')} - {v.get('customer_id', '?')} ({v.get('verification_id', '')[:8]}...)"
                for v in verifications
            ]

            if v_labels:
                selected_label = st.selectbox("Select a verification", v_labels, key="detail_select")
                selected_idx = v_labels.index(selected_label)
                selected_v_id = v_ids[selected_idx]

                if st.button("Load Details", key="load_details"):
                    try:
                        detail = client.get_verification(selected_v_id)

                        st.markdown(f"**Verification ID:** `{detail.get('verification_id', '')}`")
                        st.markdown(f"**Status:** {get_status_badge(detail.get('overall_status', ''))} {get_status_label(detail.get('overall_status', ''))}")
                        st.markdown(f"**Customer:** {detail.get('customer_id', '')}")
                        st.markdown(f"**Shipment:** {detail.get('shipment_ref') or 'N/A'}")
                        st.markdown(f"**Reviewed By:** {detail.get('reviewed_by') or 'Not yet reviewed'}")

                        # Show field comparisons
                        fields = detail.get("fields", [])
                        if fields:
                            field_data = []
                            for f in fields:
                                f_status = f.get("status", "no_rule")
                                f_conf = f.get("confidence", 0.0)
                                field_data.append({
                                    "Status": f"{get_status_badge(f_status)} {get_status_label(f_status)}",
                                    "Field": f.get("field_name", "").replace("_", " ").title(),
                                    "Extracted": str(f.get("extracted_value") or "(empty)"),
                                    "Expected": str(f.get("expected_value") or "(no rule)"),
                                    "Confidence": f"{f_conf * 100:.0f}%" if f_conf else "N/A",
                                })

                            field_df = pd.DataFrame(field_data)
                            st.dataframe(field_df, use_container_width=True, hide_index=True)

                        # Show draft reply
                        draft_text = detail.get("draft_reply", "")
                        if draft_text:
                            st.markdown("**Draft Reply:**")
                            st.text_area("Draft", value=draft_text, height=200, disabled=True, key="detail_draft")

                    except Exception as e:
                        st.error(f"Failed to load details: {e}")

except Exception as e:
    st.warning(f"Could not load verifications: {str(e)}")
    st.info("Make sure the API is running: `python run_api.py`")


# ── Statistics Section ───────────────────────────────────────────────
st.divider()
st.subheader("Verification Statistics")

try:
    from app.api_client import get_api_client
    client = get_api_client()

    stats = client.get_verification_stats()

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Total Verifications", stats.get("total", 0))

    with stat_col2:
        st.metric("Reviewed", stats.get("reviewed", 0))

    with stat_col3:
        st.metric("Pending Review", stats.get("pending_review", 0))

    with stat_col4:
        by_status = stats.get("by_status", {})
        approved = by_status.get("approved", 0)
        amendment = by_status.get("amendment_required", 0)
        total = stats.get("total", 0)
        rate = f"{approved / total * 100:.0f}%" if total > 0 else "N/A"
        st.metric("Approval Rate", rate)

    # Status breakdown
    by_status = stats.get("by_status", {})
    if by_status:
        status_data = []
        for s, count in by_status.items():
            status_data.append({
                "Status": f"{get_status_badge(s)} {get_status_label(s)}",
                "Count": count,
            })
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.info(f"Statistics unavailable: {e}")
