"""Draft email generator for verification results.

Generates amendment request or approval emails based on verification
outcomes. Uses the LLM (via get_model_with_fallback) to produce
professional, context-aware emails. Falls back to a simple template
if the LLM is unavailable (all models fail).

The prompts are loaded from config/prompts.yaml under the 'verification'
section. This keeps prompt engineering separate from code, allowing
non-developers to tune the email tone and format.

Draft types:
    amendment — sent when verification finds mismatches. Lists the
        discrepant fields and asks the shipper to amend the document.
    approval  — sent when all fields match. Confirms the document
        has been verified and accepted.

Public functions:
    generate_draft(comparisons, overall_status, customer_name,
                   shipment_ref) -> str
"""

import logging
from typing import Optional

from src.common.config_loader import load_prompts
from src.common.exceptions import AllModelsFailedError
from src.models.llm_factory import get_model_with_fallback
from src.verification.comparator import FieldComparison

# ── Logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────────────

def generate_draft(
    comparisons: list[FieldComparison],
    overall_status: str,
    customer_name: str = "Customer",
    shipment_ref: Optional[str] = None,
    document_summaries: Optional[list[dict]] = None,
    cross_document_inconsistencies: Optional[list[dict]] = None,
) -> str:
    """Generate a draft email based on the verification result.

    For amendment_required status, generates an email listing all
    discrepancies and requesting corrections. For approved status,
    generates a confirmation email. For uncertain status, generates
    an email noting that manual review is needed.

    Supports multi-document batches: when document_summaries is provided,
    the draft lists all documents processed and groups discrepancies by
    document. Cross-document inconsistencies (if any) are mentioned as
    an additional concern.

    Uses the LLM to generate professional emails. If the LLM fails
    (all models down), falls back to a simple template-based email.

    Args:
        comparisons: List of FieldComparison results from the comparator.
            For single-document flow, these are the comparisons for that doc.
            For multi-document flow, these are all comparisons combined.
        overall_status: One of "amendment_required", "uncertain", "approved".
        customer_name: The customer's display name for the email greeting.
        shipment_ref: Optional shipment reference number for the email subject.
        document_summaries: Optional list of dicts for multi-document batches.
            Each dict has keys: "file_name", "document_type", "status",
            "comparisons" (list of FieldComparison for that document).
        cross_document_inconsistencies: Optional list of inconsistency dicts
            from check_cross_document_consistency(). Only included if there
            are actual inconsistencies.

    Returns:
        A draft email string ready for human review and editing.
    """
    logger.info(
        "Generating draft email: status=%s, customer=%s, shipment=%s, docs=%d",
        overall_status, customer_name, shipment_ref,
        len(document_summaries) if document_summaries else 1,
    )

    # Build the discrepancy summary for the prompt.
    # For multi-document batches, group discrepancies by document.
    if document_summaries and len(document_summaries) > 1:
        discrepancy_summary = _build_multi_doc_discrepancy_summary(
            document_summaries, cross_document_inconsistencies,
        )
    else:
        discrepancy_summary = _build_discrepancy_summary(comparisons)

    # Try LLM-based generation first
    try:
        draft = _generate_with_llm(
            comparisons=comparisons,
            overall_status=overall_status,
            customer_name=customer_name,
            shipment_ref=shipment_ref,
            discrepancy_summary=discrepancy_summary,
        )
        logger.info("Draft email generated via LLM (%d chars)", len(draft))
        return draft

    except (AllModelsFailedError, Exception) as e:
        # LLM generation failed — fall back to template
        logger.warning(
            "LLM draft generation failed, falling back to template: %s", e,
        )
        draft = _generate_from_template(
            comparisons=comparisons,
            overall_status=overall_status,
            customer_name=customer_name,
            shipment_ref=shipment_ref,
            discrepancy_summary=discrepancy_summary,
        )
        logger.info("Draft email generated via fallback template (%d chars)", len(draft))
        return draft


# ── LLM-Based Generation ───────────────────────────────────────────────

def _generate_with_llm(
    comparisons: list[FieldComparison],
    overall_status: str,
    customer_name: str,
    shipment_ref: Optional[str],
    discrepancy_summary: str,
) -> str:
    """Generate a draft email using the LLM with prompts from config/prompts.yaml.

    Loads the appropriate prompt template (draft_amendment or draft_approval)
    from the 'verification' section of prompts.yaml, fills in the variables,
    and invokes the LLM via get_model_with_fallback.

    Args:
        comparisons: Field comparison results.
        overall_status: Verification status.
        customer_name: Customer display name.
        shipment_ref: Optional shipment reference.
        discrepancy_summary: Pre-formatted discrepancy text.

    Returns:
        LLM-generated email string.

    Raises:
        AllModelsFailedError: If all LLM models in the chain fail.
    """
    prompts = load_prompts()
    verification_prompts = prompts.get("verification", {})

    # Select the appropriate prompt template based on overall status.
    # For multi-document batches (detected by the "DOCUMENTS PROCESSED" header
    # in the discrepancy summary), prefer the batch-specific template.
    is_batch = "DOCUMENTS PROCESSED:" in discrepancy_summary
    if overall_status == "amendment_required" or overall_status == "uncertain":
        if is_batch:
            # Use batch template if available, fall back to single-doc template
            prompt_template = verification_prompts.get(
                "draft_amendment_batch",
                verification_prompts.get("draft_amendment", ""),
            )
        else:
            prompt_template = verification_prompts.get("draft_amendment", "")
    elif overall_status == "approved":
        prompt_template = verification_prompts.get("draft_approval", "")
    else:
        prompt_template = verification_prompts.get("draft_amendment", "")

    if not prompt_template:
        # No prompt configured — raise to trigger fallback
        raise ValueError("No verification prompt template configured in prompts.yaml")

    # Build the match summary for the prompt context
    match_count = sum(1 for c in comparisons if c.status == "match")
    mismatch_count = sum(1 for c in comparisons if c.status == "mismatch")
    uncertain_count = sum(1 for c in comparisons if c.status == "uncertain")
    total_fields = len(comparisons)

    # Fill in the prompt template variables
    prompt = prompt_template.format(
        customer_name=customer_name,
        shipment_ref=shipment_ref or "N/A",
        overall_status=overall_status,
        discrepancy_summary=discrepancy_summary,
        match_count=match_count,
        mismatch_count=mismatch_count,
        uncertain_count=uncertain_count,
        total_fields=total_fields,
    )

    # Invoke the LLM using the verification task from models.yaml
    # Falls back through the model chain if the primary model fails
    messages = [
        {"role": "system", "content": "You are a professional logistics email writer. Write clear, concise business emails."},
        {"role": "user", "content": prompt},
    ]

    response_text, model_name = get_model_with_fallback(
        task="verification",
        messages=messages,
        trace_name="draft_email_generation",
    )

    logger.debug("Draft generated by model: %s", model_name)
    return response_text.strip()


# ── Template-Based Fallback ─────────────────────────────────────────────

def _generate_from_template(
    comparisons: list[FieldComparison],
    overall_status: str,
    customer_name: str,
    shipment_ref: Optional[str],
    discrepancy_summary: str,
) -> str:
    """Generate a draft email using a simple template (LLM fallback).

    This is the fallback when all LLM models are unavailable. The email
    is functional but less polished than an LLM-generated one.

    Args:
        comparisons: Field comparison results.
        overall_status: Verification status.
        customer_name: Customer display name.
        shipment_ref: Optional shipment reference.
        discrepancy_summary: Pre-formatted discrepancy text.

    Returns:
        Template-based email string.
    """
    ref_line = f" for shipment {shipment_ref}" if shipment_ref else ""

    if overall_status == "amendment_required":
        return (
            f"Dear {customer_name},\n\n"
            f"We have completed the automated verification of the shipping documents"
            f"{ref_line}. Our system has identified the following discrepancies that "
            f"require your attention:\n\n"
            f"{discrepancy_summary}\n\n"
            f"Please review the above items and submit amended documents at your "
            f"earliest convenience. If you believe any of these flags are incorrect, "
            f"please contact us with supporting documentation.\n\n"
            f"Thank you for your prompt attention to this matter.\n\n"
            f"Best regards,\n"
            f"Document Verification Team"
        )

    elif overall_status == "approved":
        match_count = sum(1 for c in comparisons if c.status == "match")
        return (
            f"Dear {customer_name},\n\n"
            f"We are pleased to confirm that the shipping documents{ref_line} have "
            f"passed our automated verification checks. All {match_count} verified "
            f"fields match our records.\n\n"
            f"No further action is required from your side. The documents have been "
            f"accepted and filed in our system.\n\n"
            f"Best regards,\n"
            f"Document Verification Team"
        )

    else:
        # uncertain status
        uncertain_fields = [c for c in comparisons if c.status == "uncertain"]
        uncertain_names = ", ".join(c.field_name.replace("_", " ").title()
                                    for c in uncertain_fields)
        return (
            f"Dear {customer_name},\n\n"
            f"We have completed an initial review of the shipping documents"
            f"{ref_line}. However, our system was unable to verify the following "
            f"fields with sufficient confidence:\n\n"
            f"Fields requiring manual review: {uncertain_names}\n\n"
            f"{discrepancy_summary}\n\n"
            f"A member of our team will review these items manually and follow up "
            f"with you shortly. No action is required from you at this time.\n\n"
            f"Best regards,\n"
            f"Document Verification Team"
        )


# ── Internal Helpers ────────────────────────────────────────────────────

def _build_discrepancy_summary(comparisons: list[FieldComparison]) -> str:
    """Build a formatted text summary of field discrepancies.

    Lists all fields that are not "match" or "no_rule" status, with
    their extracted vs. expected values. Used in both LLM prompts and
    fallback templates.

    Args:
        comparisons: List of FieldComparison results.

    Returns:
        A formatted multi-line string listing discrepancies.
        Returns "No discrepancies found." if all fields match.
    """
    # Collect mismatches and uncertain fields
    issues = [c for c in comparisons if c.status in ("mismatch", "uncertain")]

    if not issues:
        return "No discrepancies found."

    lines = []
    for i, comp in enumerate(issues, start=1):
        field_display = comp.field_name.replace("_", " ").title()
        status_label = "MISMATCH" if comp.status == "mismatch" else "UNCERTAIN"

        line = (
            f"{i}. {field_display} [{status_label}]\n"
            f"   Extracted: {comp.extracted_value or '(empty)'}\n"
            f"   Expected:  {comp.expected_value or '(no rule)'}\n"
            f"   Confidence: {comp.confidence:.0%}"
        )

        if comp.rule_violated:
            line += f"\n   Issue: {comp.rule_violated}"

        lines.append(line)

    return "\n\n".join(lines)


def _build_multi_doc_discrepancy_summary(
    document_summaries: list[dict],
    cross_document_inconsistencies: Optional[list[dict]] = None,
) -> str:
    """Build a formatted discrepancy summary for a multi-document batch.

    Groups discrepancies by document and appends cross-document
    inconsistencies (if any) as a separate section.

    Args:
        document_summaries: List of dicts with keys:
            "file_name", "document_type", "status", "comparisons".
        cross_document_inconsistencies: Optional list of inconsistency dicts
            (only those where consistent=False).

    Returns:
        A formatted multi-line string listing all discrepancies grouped
        by document, plus any cross-document inconsistencies.
    """
    sections = []

    # Section 1: Documents processed
    doc_list_lines = ["DOCUMENTS PROCESSED:"]
    for i, doc in enumerate(document_summaries, start=1):
        doc_type = (doc.get("document_type") or "unknown").replace("_", " ").title()
        status_label = doc.get("status", "unknown").replace("_", " ").title()
        doc_list_lines.append(
            f"  {i}. {doc.get('file_name', 'unknown')} ({doc_type}) — {status_label}"
        )
    sections.append("\n".join(doc_list_lines))

    # Section 2: Per-document discrepancies
    for doc in document_summaries:
        comparisons = doc.get("comparisons", [])
        issues = [c for c in comparisons if c.status in ("mismatch", "uncertain")]
        if not issues:
            continue

        file_name = doc.get("file_name", "unknown")
        doc_type = (doc.get("document_type") or "unknown").replace("_", " ").title()
        doc_section_lines = [f"DISCREPANCIES IN: {file_name} ({doc_type})"]

        for j, comp in enumerate(issues, start=1):
            field_display = comp.field_name.replace("_", " ").title()
            status_label = "MISMATCH" if comp.status == "mismatch" else "UNCERTAIN"
            doc_section_lines.append(
                f"  {j}. {field_display} [{status_label}]\n"
                f"     Extracted: {comp.extracted_value or '(empty)'}\n"
                f"     Expected:  {comp.expected_value or '(no rule)'}\n"
                f"     Confidence: {comp.confidence:.0%}"
            )
            if comp.rule_violated:
                doc_section_lines[-1] += f"\n     Issue: {comp.rule_violated}"

        sections.append("\n".join(doc_section_lines))

    # Section 3: Cross-document inconsistencies (if any)
    if cross_document_inconsistencies:
        actual_inconsistencies = [
            i for i in cross_document_inconsistencies if not i.get("consistent", True)
        ]
        if actual_inconsistencies:
            inc_lines = [
                "CROSS-DOCUMENT INCONSISTENCIES:",
                "(The same field has different values in different documents)",
            ]
            for inc in actual_inconsistencies:
                field_display = inc["field_name"].replace("_", " ").title()
                inc_lines.append(f"  - {field_display}:")
                for doc_type, value in inc.get("values", {}).items():
                    inc_lines.append(f"      {doc_type}: {value}")
            sections.append("\n".join(inc_lines))

    if len(sections) <= 1:
        # Only the document list, no discrepancies
        return sections[0] + "\n\nNo discrepancies found across any documents."

    return "\n\n".join(sections)
