"""API request/response schemas for verification endpoints.

These Pydantic models define the HTTP API contract for document
verification. They translate between the verification pipeline's
internal dicts and the JSON structures sent/received over HTTP.

Schemas:
    VerificationProcessRequest — POST /api/verification/process
    VerificationProcessResponse — returned after verification
    VerificationListResponse — GET /api/verification (list view)
    VerificationDetailResponse — GET /api/verification/{id} (detail view)
    VerificationReviewRequest — PUT /api/verification/{id}/review
    VerificationStatsResponse — GET /api/verification/stats
    CustomerListResponse — GET /api/verification/customers
    FieldComparisonResponse — nested model for field comparisons
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Field Comparison (nested in responses) ──────────────────────────────

class FieldComparisonResponse(BaseModel):
    """A single field comparison result in the verification output.

    Attributes:
        field_name: Name of the compared field (e.g., "consignee_name").
        extracted_value: Raw value extracted from the document (as seen by CG).
        extracted_normalized: Cleaned value used for comparison (currency/units stripped).
        expected_value: Expected value from the customer rule.
        status: Comparison result (match, mismatch, uncertain, no_rule).
        confidence: Extraction confidence score (0.0-1.0).
        rule_type: Match type used (exact, prefix, one_of, etc.).
        rule_violated: Description of the violated rule (or None).
    """
    field_name: str
    extracted_value: Optional[str] = None
    extracted_normalized: Optional[str] = None
    expected_value: Optional[str] = None
    status: str  # match | mismatch | uncertain | no_rule
    confidence: float = 0.0
    rule_type: Optional[str] = None
    rule_violated: Optional[str] = None


# ── Process Request / Response ──────────────────────────────────────────

class VerificationProcessRequest(BaseModel):
    """POST /api/verification/process — request body.

    Sent when the user uploads a document for verification. The file
    itself is uploaded as multipart form data; this model captures the
    additional metadata sent alongside the file.

    Attributes:
        customer_id: Customer identifier matching a YAML rules file.
        shipment_ref: Shipment reference number. Strongly recommended --
            required by the UI, used for grouping batch documents.
            The API still accepts None for backward compatibility with
            programmatic callers, but the UI enforces it.
        doc_type_hint: Optional document type hint for the vision pipeline.
    """
    customer_id: str
    shipment_ref: Optional[str] = Field(
        default=None,
        description="Shipment reference number. Required by the UI for batch grouping.",
    )
    doc_type_hint: Optional[str] = None


class VerificationProcessResponse(BaseModel):
    """POST /api/verification/process — response body.

    Returned after the full verification pipeline completes. Contains
    the verification outcome, field comparisons, and draft email.

    Attributes:
        verification_id: Unique ID for this verification.
        document_id: UUID of the extracted document (or None on failure).
        customer_id: Customer identifier used for verification.
        customer_name: Human-readable customer name.
        shipment_ref: Shipment reference (or None).
        overall_status: Verification outcome (approved, amendment_required,
            uncertain, extraction_failed, no_rules).
        document_type: Detected document type (or None on failure).
        extraction_model: LLM model used for extraction (or None).
        extracted_fields: Dict of field_name -> extracted value.
        confidence_scores: Dict of field_name -> confidence float.
        comparisons: List of field comparison details.
        draft_reply: Generated draft email text.
        notes: Any warnings or informational notes.
        error: Error message if the pipeline failed (or None).
    """
    verification_id: str
    document_id: Optional[str] = None
    customer_id: str
    customer_name: str = ""
    shipment_ref: Optional[str] = None
    overall_status: str
    document_type: Optional[str] = None
    extraction_model: Optional[str] = None
    extracted_fields: dict = {}
    confidence_scores: dict[str, float] = {}
    comparisons: list[FieldComparisonResponse] = []
    draft_reply: str = ""
    notes: str = ""
    doc_type_check: Optional[dict] = None
    error: Optional[str] = None


# ── List Response (summary view) ────────────────────────────────────────

class VerificationListResponse(BaseModel):
    """GET /api/verification — response model for list view.

    A lightweight summary of a verification result, without the full
    field comparison details. Used for the table/list view.

    Attributes:
        verification_id: Unique ID for this verification.
        document_id: UUID of the verified document.
        customer_id: Customer identifier.
        shipment_ref: Shipment reference (or None).
        document_type: Document type (e.g., invoice, bill_of_lading).
        overall_status: Verification outcome.
        received_at: ISO timestamp of when the verification was processed.
        reviewed_by: Name of the reviewer (or None if not yet reviewed).
        reviewed_at: ISO timestamp of review (or None).
        notes: Any notes.
        draft_reply: The draft email text.
    """
    verification_id: str
    document_id: str = ""
    customer_id: str = ""
    shipment_ref: Optional[str] = None
    document_type: Optional[str] = None
    overall_status: str = ""
    received_at: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    notes: Optional[str] = None
    draft_reply: Optional[str] = None


# ── Detail Response (full view with fields) ─────────────────────────────

class VerificationDetailResponse(BaseModel):
    """GET /api/verification/{id} — response model for detail view.

    The full verification result including all field comparison details.
    Used for the single-verification detail page.

    Attributes:
        verification_id: Unique ID for this verification.
        document_id: UUID of the verified document.
        customer_id: Customer identifier.
        shipment_ref: Shipment reference (or None).
        overall_status: Verification outcome.
        received_at: ISO timestamp of processing.
        reviewed_by: Name of the reviewer (or None).
        reviewed_at: ISO timestamp of review (or None).
        draft_reply: Generated draft email text.
        notes: Any notes.
        fields: List of field comparison details.
    """
    verification_id: str
    document_id: str = ""
    customer_id: str = ""
    shipment_ref: Optional[str] = None
    overall_status: str = ""
    received_at: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    draft_reply: Optional[str] = None
    notes: Optional[str] = None
    fields: list[FieldComparisonResponse] = []


# ── Review Request ──────────────────────────────────────────────────────

class VerificationReviewRequest(BaseModel):
    """PUT /api/verification/{id}/review — request body.

    Sent when a human reviews a verification result and either confirms
    or overrides the automated decision.

    Attributes:
        reviewed_by: Name or email of the person reviewing.
        notes: Optional notes from the reviewer.
        overall_status: Optional status override. If provided, replaces
            the automated status (e.g., changing "amendment_required"
            to "approved" after manual review).
        draft_reply: Optional updated draft email text.
    """
    reviewed_by: str
    notes: Optional[str] = None
    overall_status: Optional[str] = None
    draft_reply: Optional[str] = None


# ── Stats Response ──────────────────────────────────────────────────────

class VerificationStatsResponse(BaseModel):
    """GET /api/verification/stats — response model.

    Aggregate statistics for the verification dashboard.

    Attributes:
        total: Total number of verifications.
        by_status: Dict mapping status to count.
        reviewed: Number reviewed by a human.
        pending_review: Number awaiting human review.
    """
    total: int = 0
    by_status: dict[str, int] = {}
    reviewed: int = 0
    pending_review: int = 0


# ── Customer List Response ──────────────────────────────────────────────

class CustomerResponse(BaseModel):
    """A single customer with verification rules.

    Attributes:
        customer_id: The customer identifier (YAML filename stem).
        customer_name: Human-readable customer name.
        file_name: The YAML filename.
        rule_count: Number of rules defined for this customer.
    """
    customer_id: str
    customer_name: str = ""
    file_name: str = ""
    rule_count: int = 0
