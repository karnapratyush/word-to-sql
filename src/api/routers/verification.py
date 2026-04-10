"""Verification API endpoints.

This router handles the document verification workflow:

1. Process:   POST /api/verification/process
   - Accepts a multipart file upload + customer_id
   - Runs the full verification pipeline (extract -> compare -> draft)
   - Returns the verification outcome with field comparisons

2. List:      GET /api/verification
   - Returns all verification results (summary view)
   - Supports optional status and customer query filters

3. Detail:    GET /api/verification/{id}
   - Returns a single verification with full field comparison details

4. Review:    PUT /api/verification/{id}/review
   - Human reviews and optionally overrides the automated decision

5. Stats:     GET /api/verification/stats
   - Returns aggregate statistics for the dashboard

6. Customers: GET /api/verification/customers
   - Lists all customers with configured verification rules

Error handling:
- Unknown customer -> error in response body (not HTTP 404, since the
  pipeline handles it gracefully and returns an outcome)
- Extraction failure -> error in response body
- Verification not found -> 404 Not Found
- Review of non-existent verification -> 404 Not Found
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from src.api.dependencies import get_verification_service
from src.api.schemas.verification import (
    CustomerResponse,
    FieldComparisonResponse,
    VerificationDetailResponse,
    VerificationListResponse,
    VerificationProcessResponse,
    VerificationReviewRequest,
    VerificationStatsResponse,
)
from src.services.verification_service import VerificationService

# ── Logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Router ──────────────────────────────────────────────────────────────
# Registered in src/api/main.py with prefix="/api/verification"
router = APIRouter()


# ── POST /process — Upload and verify a document ────────────────────────

@router.post("/process", response_model=VerificationProcessResponse)
async def process_document(
    file: UploadFile = File(..., description="PDF, PNG, or JPG document to verify"),
    customer_id: str = Form(..., description="Customer ID matching a YAML rules file"),
    shipment_ref: Optional[str] = Form(default=None, description="Optional shipment reference"),
    doc_type_hint: Optional[str] = Form(default=None, description="Optional document type hint"),
    service: VerificationService = Depends(get_verification_service),
):
    """Upload a document and run the full verification pipeline.

    Accepts a multipart file upload along with the customer_id and
    optional metadata. Runs the complete pipeline:
    1. Extracts fields using the vision LLM
    2. Loads customer-specific rules from YAML config
    3. Compares each extracted field against its rule
    4. Determines overall status (approved/amendment/uncertain)
    5. Generates a draft email (with LLM fallback to template)
    6. Stores results in the database

    The response always has HTTP 200 status. Pipeline failures (unknown
    customer, extraction error) are reported in the response body's
    'error' and 'overall_status' fields, not as HTTP errors.

    Args:
        file: Uploaded document file (multipart form data).
        customer_id: Customer identifier matching a YAML config filename.
        shipment_ref: Optional shipment reference for cross-referencing.
        doc_type_hint: Optional document type hint (invoice, bill_of_lading, etc.).
        service: Injected VerificationService instance.

    Returns:
        VerificationProcessResponse with the full verification outcome.
    """
    logger.info(
        "POST /verification/process: file=%s, customer=%s, shipment=%s",
        file.filename, customer_id, shipment_ref,
    )

    try:
        # Read the uploaded file bytes
        file_bytes = await file.read()
        file_name = file.filename or "uploaded_document"

        # Run the full verification pipeline
        result = service.process_document(
            file_bytes=file_bytes,
            file_name=file_name,
            customer_id=customer_id,
            shipment_ref=shipment_ref,
            doc_type_hint=doc_type_hint,
        )

        # Convert field comparison dicts to response models
        comparisons = [
            FieldComparisonResponse(**comp)
            for comp in result.get("comparisons", [])
        ]

        return VerificationProcessResponse(
            verification_id=result["verification_id"],
            document_id=result.get("document_id"),
            customer_id=result["customer_id"],
            customer_name=result.get("customer_name", ""),
            shipment_ref=result.get("shipment_ref"),
            overall_status=result["overall_status"],
            document_type=result.get("document_type"),
            extraction_model=result.get("extraction_model"),
            extracted_fields=result.get("extracted_fields", {}),
            confidence_scores=result.get("confidence_scores", {}),
            comparisons=comparisons,
            draft_reply=result.get("draft_reply", ""),
            notes=result.get("notes", ""),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(
            "Unexpected error in process_document: %s", e, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {e}",
        )


# ── GET / — List all verifications ──────────────────────────────────────

@router.get("", response_model=list[VerificationListResponse])
def list_verifications(
    status: Optional[str] = Query(
        default=None,
        description="Filter by overall_status (approved, amendment_required, uncertain)",
    ),
    customer: Optional[str] = Query(
        default=None,
        description="Filter by customer_id",
    ),
    service: VerificationService = Depends(get_verification_service),
):
    """List all verification results with optional filtering.

    Returns a summary view of all verifications, ordered by newest first.
    Use the status or customer query parameters to filter results.

    Args:
        status: Optional overall_status filter.
        customer: Optional customer_id filter.
        service: Injected VerificationService instance.

    Returns:
        List of VerificationListResponse objects.
    """
    logger.info(
        "GET /verification: status=%s, customer=%s", status, customer,
    )

    try:
        results = service.list_verifications(
            status_filter=status,
            customer_filter=customer,
        )
        return [VerificationListResponse(**r) for r in results]

    except Exception as e:
        logger.error("Failed to list verifications: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list verifications: {e}",
        )


# ── GET /stats — Verification statistics ────────────────────────────────

@router.get("/stats", response_model=VerificationStatsResponse)
def get_stats(
    service: VerificationService = Depends(get_verification_service),
):
    """Get aggregate verification statistics for the dashboard.

    Returns total count, counts by status, and reviewed vs. pending
    review counts.

    Args:
        service: Injected VerificationService instance.

    Returns:
        VerificationStatsResponse with aggregate counts.
    """
    logger.info("GET /verification/stats")

    try:
        stats = service.get_stats()
        return VerificationStatsResponse(**stats)

    except Exception as e:
        logger.error("Failed to get stats: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get verification stats: {e}",
        )


# ── GET /customers — List customers with rules ─────────────────────────

@router.get("/customers", response_model=list[CustomerResponse])
def list_customers(
    service: VerificationService = Depends(get_verification_service),
):
    """List all customers that have verification rules configured.

    Scans config/customer_rules/ for YAML files and returns basic info
    for each customer. Used by the UI to populate the customer selector.

    Args:
        service: Injected VerificationService instance.

    Returns:
        List of CustomerResponse objects.
    """
    logger.info("GET /verification/customers")

    try:
        customers = service.list_customers()
        return [CustomerResponse(**c) for c in customers]

    except Exception as e:
        logger.error("Failed to list customers: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list customers: {e}",
        )


# ── GET /{verification_id} — Get verification detail ───────────────────

@router.get("/{verification_id}", response_model=VerificationDetailResponse)
def get_verification(
    verification_id: str,
    service: VerificationService = Depends(get_verification_service),
):
    """Retrieve a single verification with full field comparison details.

    Returns the verification result enriched with all field-level
    comparison details (extracted vs. expected, match status, etc.).

    Args:
        verification_id: UUID of the verification to retrieve.
        service: Injected VerificationService instance.

    Returns:
        VerificationDetailResponse with field comparison details.

    Raises:
        HTTPException 404: If no verification with the given ID exists.
    """
    logger.info("GET /verification/%s", verification_id)

    try:
        result = service.get_verification(verification_id)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Verification not found: {verification_id}",
            )

        # Convert field dicts to response models
        fields = [
            FieldComparisonResponse(**f) for f in result.get("fields", [])
        ]

        return VerificationDetailResponse(
            verification_id=result["verification_id"],
            document_id=result.get("document_id", ""),
            customer_id=result.get("customer_id", ""),
            shipment_ref=result.get("shipment_ref"),
            overall_status=result.get("overall_status", ""),
            received_at=result.get("received_at"),
            reviewed_by=result.get("reviewed_by"),
            reviewed_at=result.get("reviewed_at"),
            draft_reply=result.get("draft_reply"),
            notes=result.get("notes"),
            fields=fields,
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions without wrapping

    except Exception as e:
        logger.error(
            "Failed to get verification %s: %s",
            verification_id, e, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get verification: {e}",
        )


# ── PUT /{verification_id}/review — Human review ───────────────────────

@router.put("/{verification_id}/review", response_model=VerificationDetailResponse)
def review_verification(
    verification_id: str,
    request: VerificationReviewRequest,
    service: VerificationService = Depends(get_verification_service),
):
    """Review a verification result and optionally override the decision.

    Called when a human reviews the automated verification. Records who
    reviewed it and when. Optionally allows overriding the automated
    status and updating the draft email.

    Args:
        verification_id: UUID of the verification to review.
        request: VerificationReviewRequest with reviewer info and
            optional overrides.
        service: Injected VerificationService instance.

    Returns:
        VerificationDetailResponse with the updated verification.

    Raises:
        HTTPException 404: If the verification is not found.
    """
    logger.info(
        "PUT /verification/%s/review: by=%s",
        verification_id, request.reviewed_by,
    )

    try:
        result = service.update_review(
            verification_id=verification_id,
            reviewed_by=request.reviewed_by,
            notes=request.notes,
            overall_status=request.overall_status,
            draft_reply=request.draft_reply,
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Verification not found: {verification_id}",
            )

        # Convert field dicts to response models
        fields = [
            FieldComparisonResponse(**f) for f in result.get("fields", [])
        ]

        return VerificationDetailResponse(
            verification_id=result["verification_id"],
            document_id=result.get("document_id", ""),
            customer_id=result.get("customer_id", ""),
            shipment_ref=result.get("shipment_ref"),
            overall_status=result.get("overall_status", ""),
            received_at=result.get("received_at"),
            reviewed_by=result.get("reviewed_by"),
            reviewed_at=result.get("reviewed_at"),
            draft_reply=result.get("draft_reply"),
            notes=result.get("notes"),
            fields=fields,
        )

    except HTTPException:
        raise

    except ValueError as e:
        # Raised by the repository when verification_id is not found
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.error(
            "Failed to review verification %s: %s",
            verification_id, e, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to review verification: {e}",
        )
