"""Vision Agent -- full document extraction pipeline orchestrator.

This module orchestrates the complete document extraction flow:

    Uploaded PDF/Image
        -> Extractor (vision LLM extracts structured fields)
        -> Validator (confidence checks, consistency, completeness)
        -> Return for user review (NOT auto-stored)
        -> User approves/corrects -> Storage (separate step)
        -> Temp file cleanup (after approval or rejection)

The key design decision: storage happens AFTER user approval, not
automatically. This prevents low-confidence extractions from polluting
the database. The UI flow is:

1. User uploads a document
2. process_document() extracts and validates
3. UI shows the result with confidence indicators
4. User reviews, edits, and either approves or rejects
5. On approval: store_approved_document() persists and cleans up
6. On rejection: delete_uploaded_file() just cleans up

Public functions:
- process_document(request) -> ExtractionResult
- store_approved_document(result, file_name, review_status) -> DocumentRecord
- delete_uploaded_file(file_name) -> None
"""

import logging
import os
from typing import Optional

from src.common.config_loader import load_settings
from src.common.exceptions import ExtractionError, UnsupportedFileError
from src.common.schemas import (
    DocumentRecord,
    ExtractionRequest,
    ExtractionResult,
)
from src.vision.extractor import (
    extract_from_document,
    delete_temp_file,
    get_temp_file_path,
    UPLOAD_DIR,
)
from src.vision.storage import store_extraction, update_review_status
from src.vision.validator import validate_extraction

# ── Logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────────────

def process_document(request: ExtractionRequest) -> ExtractionResult:
    """Full vision pipeline: extract -> validate -> return for review.

    This is the main entry point for the vision pipeline. It takes an
    ExtractionRequest containing the uploaded file and returns a validated
    ExtractionResult ready for human review in the UI.

    Storage does NOT happen here. The result is returned to the UI, where
    the user can review, edit, and then approve or reject. This separation
    ensures humans review extractions before they become queryable data
    in the analytics system.

    Args:
        request: ExtractionRequest with file_bytes, file_name, and
            optional document_type_hint.

    Returns:
        ExtractionResult with extracted fields, confidence scores, and
        needs_review flags set by the validator. Ready for UI display.

    Raises:
        UnsupportedFileError: If the file type or size is invalid.
        ExtractionError: If extraction fails completely after all retries.
    """
    logger.info(
        "Processing document: %s (%d bytes, hint=%s)",
        request.file_name,
        len(request.file_bytes),
        request.document_type_hint,
    )

    # Step 1: Extract structured fields from the document
    # This validates the file, saves it temporarily, converts PDF to images,
    # classifies the document type, and extracts fields via vision LLM.
    extraction_result = extract_from_document(
        file_bytes=request.file_bytes,
        file_name=request.file_name,
        document_type_hint=request.document_type_hint,
    )

    logger.info(
        "Extraction produced %d fields with overall confidence %.2f",
        len(extraction_result.fields),
        extraction_result.overall_confidence,
    )

    # Step 2: Validate the extraction result
    # Flags low-confidence fields, checks consistency, verifies completeness,
    # and recalculates the overall confidence score.
    settings = load_settings()
    validated_result = validate_extraction(extraction_result, settings)

    logger.info(
        "Validation complete: %d fields need review, overall confidence %.2f",
        sum(1 for f in validated_result.fields.values() if f.needs_review),
        validated_result.overall_confidence,
    )

    # Step 3: Return for user review (no storage yet)
    return validated_result


def store_approved_document(
    result: ExtractionResult,
    file_name: str,
    review_status: str = "approved",
    db_path: Optional[str] = None,
) -> DocumentRecord:
    """Store an approved (or corrected) extraction to the database.

    Called by the UI after the user reviews and approves the extraction.
    This function:
    1. Stores the ExtractionResult to the extracted_documents table
    2. Updates the review status (approved or corrected)
    3. Deletes the temporary uploaded file from db/uploads/

    If the user edited any fields before approving, the corrected values
    should already be in the ExtractionResult.fields.

    Args:
        result: The ExtractionResult (possibly with user edits).
        file_name: Original uploaded file name.
        review_status: Review status to set ("approved" or "corrected").
            Defaults to "approved".
        db_path: Optional database path override (for testing).

    Returns:
        DocumentRecord representing the stored and reviewed document.

    Raises:
        ExtractionError: If storage fails.
    """
    logger.info(
        "Storing approved document: %s (type=%s, status=%s)",
        file_name, result.document_type, review_status,
    )

    try:
        # Step 1: Store the extraction result in the database
        record = store_extraction(result, file_name, db_path=db_path)
        logger.info("Document stored with ID: %s", record.document_id)

        # Step 2: Update the review status from "pending" to "approved"/"corrected"
        if review_status != "pending":
            update_review_status(
                record.document_id, review_status, db_path=db_path
            )
            logger.info("Review status updated to: %s", review_status)

        # Step 3: Delete the temporary uploaded file
        temp_path = get_temp_file_path(file_name)
        if temp_path:
            delete_temp_file(temp_path)
            logger.info("Cleaned up temp file: %s", temp_path)

        return record

    except Exception as e:
        logger.error("Failed to store approved document: %s", e, exc_info=True)
        raise ExtractionError(f"Failed to store document: {e}")


def delete_uploaded_file(file_name: str) -> None:
    """Delete the temporary uploaded file when the user rejects the extraction.

    Called by the UI when the user rejects the extraction. Simply deletes
    the temporary file from db/uploads/ without storing anything to the
    database. This keeps the uploads directory clean.

    If the file is not found (already deleted, or was never saved), this
    function silently succeeds -- it is idempotent.

    Args:
        file_name: The original filename of the uploaded document.
    """
    logger.info("Deleting uploaded file (user rejected): %s", file_name)

    temp_path = get_temp_file_path(file_name)
    if temp_path:
        delete_temp_file(temp_path)
        logger.info("Deleted rejected upload: %s", temp_path)
    else:
        logger.debug(
            "Temp file for '%s' not found in %s (may already be deleted)",
            file_name, UPLOAD_DIR,
        )
