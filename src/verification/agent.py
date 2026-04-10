"""Verification Agent — full document verification pipeline orchestrator.

This module orchestrates the complete document verification flow:

    Uploaded Document (PDF/Image)
        -> Step 1: Extract (reuse Part 1 vision pipeline)
        -> Step 2: Load customer rules (from YAML config)
        -> Step 3: Compare fields against rules (comparator)
        -> Step 4: Determine overall status (approved/amendment/uncertain)
        -> Step 5: Generate draft email (LLM or template fallback)
        -> Step 6: Store results (verification_results + verification_fields)
        -> Return VerificationOutcome for UI display

The agent handles 5 failure scenarios from the assignment:

1. Extraction Failure — the vision pipeline cannot extract fields from
   the document (corrupt file, unsupported format, LLM down).
   -> Caught: returns error outcome with "extraction_failed" status.

2. Unknown Customer — no YAML rule file exists for the customer_id.
   -> Caught: returns error outcome with "no_rules" status.

3. Low Confidence Fields — extraction confidence below threshold for
   some fields.
   -> Handled: comparator marks those fields as "uncertain", overall
      status becomes "uncertain" (unless there are also mismatches).

4. LLM Draft Generation Failure — all LLM models are unavailable for
   email generation.
   -> Handled: draft_generator falls back to a template-based email.
      The verification result is still saved successfully.

5. Database Storage Failure — cannot write to the database (disk full,
   permissions, locked).
   -> Caught: logs the error, returns the verification result with a
      warning note. The user can still see results in the UI.

Public functions:
    run_verification(file_bytes, file_name, customer_id,
                     shipment_ref, db_path) -> dict
    run_verification_from_document(document_id, customer_id,
                                   shipment_ref, db_path) -> dict
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from src.common.config_loader import load_settings
from src.common.exceptions import ExtractionError, UnsupportedFileError
from src.common.schemas import ExtractionRequest
from src.verification.comparator import (
    FieldComparison,
    check_cross_document_consistency,
    compare_fields,
    determine_overall_status,
)
from src.verification.draft_generator import generate_draft
from src.verification.rules_loader import load_customer_rules

# ── Logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────────────

def run_verification(
    file_bytes: bytes,
    file_name: str,
    customer_id: str,
    shipment_ref: Optional[str] = None,
    db_path: Optional[str] = None,
    doc_type_hint: Optional[str] = None,
) -> dict:
    """Run the full verification pipeline on a newly uploaded document.

    This is the main entry point for verifying a new document. It:
    1. Extracts fields using the Part 1 vision pipeline
    2. Loads customer-specific rules from YAML config
    3. Compares extracted fields against customer rules
    4. Determines the overall verification status
    5. Generates a draft amendment or approval email
    6. Stores the verification results in the database

    Args:
        file_bytes: Raw bytes of the uploaded document (PDF, PNG, JPG).
        file_name: Original filename (used for logging and extraction).
        customer_id: Customer identifier matching a YAML config filename
            (e.g., "sample_customer" loads sample_customer.yaml).
        shipment_ref: Optional shipment reference number for cross-referencing.
        db_path: Optional database path override (for testing).
        doc_type_hint: Optional document type hint for the vision pipeline.

    Returns:
        Dict with verification outcome containing:
            - verification_id: Unique ID for this verification
            - document_id: UUID of the stored extraction
            - customer_id: The customer identifier
            - customer_name: Human-readable customer name
            - shipment_ref: Shipment reference (or None)
            - overall_status: "approved", "amendment_required", "uncertain",
              or "extraction_failed", "no_rules"
            - comparisons: List of field comparison dicts
            - draft_reply: Generated email draft
            - notes: Any warnings or error messages
            - error: Error message if the pipeline failed
    """
    logger.info(
        "Starting verification: file=%s, customer=%s, shipment=%s",
        file_name, customer_id, shipment_ref,
    )

    verification_id = str(uuid.uuid4())

    # ── Step 1: Load customer rules ─────────────────────────────────
    # Load BEFORE extraction so we fail fast if customer is unknown
    rules_data = load_customer_rules(customer_id)

    if rules_data is None:
        # Failure scenario 2: Unknown customer — no rules file found
        logger.warning("No rules found for customer '%s'", customer_id)
        return _error_outcome(
            verification_id=verification_id,
            customer_id=customer_id,
            shipment_ref=shipment_ref,
            overall_status="no_rules",
            error=f"No verification rules found for customer '{customer_id}'. "
                  f"Please ensure a YAML file exists at config/customer_rules/{customer_id}.yaml",
        )

    customer_name = rules_data.get("customer_name", customer_id)
    customer_rules = rules_data.get("rules", {})

    # ── Step 2: Extract fields from the document ────────────────────
    # Reuse the Part 1 vision pipeline for extraction
    try:
        from src.vision.agent import process_document

        request = ExtractionRequest(
            file_bytes=file_bytes,
            file_name=file_name,
            document_type_hint=doc_type_hint,
        )

        extraction_result = process_document(request)

        # Flatten extraction fields and confidence scores from FieldExtraction objects
        extracted_fields = {}
        confidence_scores = {}
        for field_name, field_obj in extraction_result.fields.items():
            extracted_fields[field_name] = field_obj.value
            confidence_scores[field_name] = field_obj.confidence

        document_id = str(uuid.uuid4())  # Assign a document ID
        extraction_model = extraction_result.model_used
        document_type = extraction_result.document_type

        logger.info(
            "Extraction complete: %d fields, overall confidence %.2f",
            len(extracted_fields), extraction_result.overall_confidence,
        )

        # Verify the classified document type against filename hints.
        # This is a soft check -- warnings are added to notes, not blocking.
        doc_type_check = _verify_document_type(file_name, document_type)

    except (ExtractionError, UnsupportedFileError) as e:
        # Failure scenario 1: Extraction failed
        logger.error("Extraction failed for '%s': %s", file_name, e)
        return _error_outcome(
            verification_id=verification_id,
            customer_id=customer_id,
            customer_name=customer_name,
            shipment_ref=shipment_ref,
            overall_status="extraction_failed",
            error=f"Document extraction failed: {e}",
        )
    except Exception as e:
        # Unexpected extraction error
        logger.error("Unexpected extraction error: %s", e, exc_info=True)
        return _error_outcome(
            verification_id=verification_id,
            customer_id=customer_id,
            customer_name=customer_name,
            shipment_ref=shipment_ref,
            overall_status="extraction_failed",
            error=f"Unexpected extraction error: {e}",
        )

    # ── Step 3: Compare fields against customer rules ───────────────
    # Load confidence threshold from settings (not hardcoded)
    settings = load_settings()
    confidence_threshold = settings.get("verification", {}).get(
        "confidence_threshold", 0.7
    )

    comparisons = compare_fields(
        extracted_fields=extracted_fields,
        confidence_scores=confidence_scores,
        customer_rules=customer_rules,
        confidence_threshold=confidence_threshold,
    )

    # ── Step 4: Determine overall status ────────────────────────────
    overall_status = determine_overall_status(comparisons)

    logger.info(
        "Verification status: %s (%d comparisons)",
        overall_status, len(comparisons),
    )

    # ── Step 5: Generate draft email ────────────────────────────────
    # Failure scenario 4 is handled inside generate_draft (LLM fallback)
    draft_reply = generate_draft(
        comparisons=comparisons,
        overall_status=overall_status,
        customer_name=customer_name,
        shipment_ref=shipment_ref,
    )

    # ── Step 6: Store results in the database ───────────────────────
    # Collect any warnings (e.g., document type mismatch from filename check)
    notes = ""
    if doc_type_check.get("warning"):
        notes = doc_type_check["warning"]

    try:
        from src.repositories.verification_repo import VerificationRepository

        repo = VerificationRepository(db_path=db_path)

        # Store the overall verification result
        repo.insert_verification_result(
            verification_id=verification_id,
            document_id=document_id,
            shipment_ref=shipment_ref,
            customer_id=customer_id,
            overall_status=overall_status,
            draft_reply=draft_reply,
        )

        # Store individual field comparisons
        for comp in comparisons:
            repo.insert_verification_field(
                verification_id=verification_id,
                field_name=comp.field_name,
                extracted_value=comp.extracted_value,
                expected_value=comp.expected_value,
                status=comp.status,
                confidence=comp.confidence,
                rule_type=comp.rule_type,
                rule_violated=comp.rule_violated,
            )

        logger.info("Verification results stored: %s", verification_id)

    except Exception as e:
        # Failure scenario 5: Database storage failed
        # The verification result is still returned to the user
        logger.error(
            "Failed to store verification results: %s", e, exc_info=True,
        )
        notes = f"Warning: verification results could not be saved to database: {e}"

    # ── Build and return the verification outcome ───────────────────
    return {
        "verification_id": verification_id,
        "document_id": document_id,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "shipment_ref": shipment_ref,
        "overall_status": overall_status,
        "document_type": document_type,
        "extraction_model": extraction_model,
        "extracted_fields": extracted_fields,
        "confidence_scores": confidence_scores,
        "comparisons": [_comparison_to_dict(c) for c in comparisons],
        "draft_reply": draft_reply,
        "notes": notes,
        "doc_type_check": doc_type_check,
        "error": None,
    }


def run_verification_from_document(
    document_id: str,
    customer_id: str,
    shipment_ref: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Run verification on an already-extracted document from the database.

    Instead of extracting from a new file, this function loads the
    previously extracted fields from the extracted_documents table and
    verifies them against customer rules. This is useful for re-verifying
    a document with different customer rules or after rule changes.

    Args:
        document_id: UUID of the previously extracted document.
        customer_id: Customer identifier matching a YAML config filename.
        shipment_ref: Optional shipment reference number.
        db_path: Optional database path override.

    Returns:
        Dict with the same structure as run_verification().
    """
    logger.info(
        "Re-verification from document: doc=%s, customer=%s",
        document_id, customer_id,
    )

    verification_id = str(uuid.uuid4())

    # ── Load customer rules ─────────────────────────────────────────
    rules_data = load_customer_rules(customer_id)
    if rules_data is None:
        return _error_outcome(
            verification_id=verification_id,
            customer_id=customer_id,
            shipment_ref=shipment_ref,
            overall_status="no_rules",
            error=f"No verification rules found for customer '{customer_id}'",
        )

    customer_name = rules_data.get("customer_name", customer_id)
    customer_rules = rules_data.get("rules", {})

    # ── Load the existing extraction from the database ──────────────
    try:
        from src.repositories.document_repo import DocumentRepository

        doc_repo = DocumentRepository(db_path=db_path)
        doc_record = doc_repo.get_document_by_id(document_id)

        if doc_record is None:
            return _error_outcome(
                verification_id=verification_id,
                customer_id=customer_id,
                customer_name=customer_name,
                shipment_ref=shipment_ref,
                overall_status="extraction_failed",
                error=f"Document not found: {document_id}",
            )

        extracted_fields = doc_record.get("extracted_fields", {})
        confidence_scores = doc_record.get("confidence_scores", {})
        document_type = doc_record.get("document_type", "unknown")
        extraction_model = doc_record.get("extraction_model", "")

    except Exception as e:
        logger.error("Failed to load document %s: %s", document_id, e, exc_info=True)
        return _error_outcome(
            verification_id=verification_id,
            customer_id=customer_id,
            customer_name=customer_name,
            shipment_ref=shipment_ref,
            overall_status="extraction_failed",
            error=f"Failed to load document: {e}",
        )

    # ── Compare, determine status, generate draft, and store ────────
    settings = load_settings()
    confidence_threshold = settings.get("verification", {}).get(
        "confidence_threshold", 0.7
    )

    comparisons = compare_fields(
        extracted_fields=extracted_fields,
        confidence_scores=confidence_scores,
        customer_rules=customer_rules,
        confidence_threshold=confidence_threshold,
    )

    overall_status = determine_overall_status(comparisons)

    draft_reply = generate_draft(
        comparisons=comparisons,
        overall_status=overall_status,
        customer_name=customer_name,
        shipment_ref=shipment_ref,
    )

    # Store results
    notes = ""
    try:
        from src.repositories.verification_repo import VerificationRepository

        repo = VerificationRepository(db_path=db_path)
        repo.insert_verification_result(
            verification_id=verification_id,
            document_id=document_id,
            shipment_ref=shipment_ref,
            customer_id=customer_id,
            overall_status=overall_status,
            draft_reply=draft_reply,
        )
        for comp in comparisons:
            repo.insert_verification_field(
                verification_id=verification_id,
                field_name=comp.field_name,
                extracted_value=comp.extracted_value,
                expected_value=comp.expected_value,
                status=comp.status,
                confidence=comp.confidence,
                rule_type=comp.rule_type,
                rule_violated=comp.rule_violated,
            )
        logger.info("Re-verification results stored: %s", verification_id)

    except Exception as e:
        logger.error("Failed to store re-verification results: %s", e, exc_info=True)
        notes = f"Warning: results could not be saved to database: {e}"

    return {
        "verification_id": verification_id,
        "document_id": document_id,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "shipment_ref": shipment_ref,
        "overall_status": overall_status,
        "document_type": document_type,
        "extraction_model": extraction_model,
        "extracted_fields": extracted_fields,
        "confidence_scores": confidence_scores,
        "comparisons": [_comparison_to_dict(c) for c in comparisons],
        "draft_reply": draft_reply,
        "notes": notes,
        "doc_type_check": None,  # Not available for re-verification (no filename)
        "error": None,
    }


# ── Internal Helpers ────────────────────────────────────────────────────

def _comparison_to_dict(comp: FieldComparison) -> dict:
    """Convert a FieldComparison dataclass to a serializable dict.

    Args:
        comp: A FieldComparison instance from the comparator.

    Returns:
        Dict with all FieldComparison attributes, including normalized value.
    """
    return {
        "field_name": comp.field_name,
        "extracted_value": comp.extracted_value,
        "extracted_normalized": comp.extracted_normalized,
        "expected_value": comp.expected_value,
        "status": comp.status,
        "confidence": comp.confidence,
        "rule_type": comp.rule_type,
        "rule_violated": comp.rule_violated,
    }


def _verify_document_type(file_name: str, classified_type: str) -> dict:
    """Check if the classified document type matches what was expected from the filename.

    Simple heuristic: filenames often contain hints about the document type
    (e.g., "ACME_BOL_2025.pdf" is likely a bill of lading). If the vision
    pipeline classifies it differently, we add a warning but do NOT block
    the verification -- the LLM classification may be more accurate.

    Args:
        file_name: Original uploaded filename (e.g., "invoice_toyota_jan.pdf").
        classified_type: The document type classified by the vision pipeline
            (e.g., "bill_of_lading", "invoice", "packing_list").

    Returns:
        Dict with keys:
            expected: The type guessed from the filename (or None if no hint).
            classified: The type from the vision pipeline.
            match: Whether they agree (True/False/None if no hint).
            warning: A warning string if they disagree, else None.
    """
    name_lower = file_name.lower()
    classified_lower = (classified_type or "").lower()

    # Mapping of filename keywords to expected document types
    keyword_map = {
        "bol": "bill_of_lading",
        "b/l": "bill_of_lading",
        "bl_": "bill_of_lading",
        "bill_of_lading": "bill_of_lading",
        "inv": "invoice",
        "invoice": "invoice",
        "pack": "packing_list",
        "packing": "packing_list",
        "customs": "customs_declaration",
        "declaration": "customs_declaration",
    }

    expected_type = None
    for keyword, doc_type in keyword_map.items():
        if keyword in name_lower:
            expected_type = doc_type
            break

    if expected_type is None:
        # No hint in the filename -- cannot verify, no warning
        return {
            "expected": None,
            "classified": classified_type,
            "match": None,
            "warning": None,
        }

    is_match = expected_type == classified_lower

    warning = None
    if not is_match:
        warning = (
            f"Filename suggests '{expected_type}' but the document was classified "
            f"as '{classified_type}'. The classification may be correct -- please "
            f"verify the document type is accurate."
        )
        logger.warning(
            "Document type mismatch: filename '%s' suggests '%s', classified as '%s'",
            file_name, expected_type, classified_type,
        )

    return {
        "expected": expected_type,
        "classified": classified_type,
        "match": is_match,
        "warning": warning,
    }


def _error_outcome(
    verification_id: str,
    customer_id: str,
    shipment_ref: Optional[str],
    overall_status: str,
    error: str,
    customer_name: str = "",
) -> dict:
    """Build an error outcome dict when the pipeline fails early.

    Used for failure scenarios (unknown customer, extraction failure)
    where we cannot complete the full verification but still need to
    return a structured response to the caller.

    Args:
        verification_id: The UUID assigned to this verification attempt.
        customer_id: The customer identifier.
        shipment_ref: Optional shipment reference.
        overall_status: Error status (e.g., "no_rules", "extraction_failed").
        error: Human-readable error description.
        customer_name: Optional customer display name.

    Returns:
        Dict with the standard verification outcome structure, populated
        with empty/default values for fields that could not be computed.
    """
    return {
        "verification_id": verification_id,
        "document_id": None,
        "customer_id": customer_id,
        "customer_name": customer_name or customer_id,
        "shipment_ref": shipment_ref,
        "overall_status": overall_status,
        "document_type": None,
        "extraction_model": None,
        "extracted_fields": {},
        "confidence_scores": {},
        "comparisons": [],
        "draft_reply": "",
        "notes": "",
        "doc_type_check": None,
        "error": error,
    }
