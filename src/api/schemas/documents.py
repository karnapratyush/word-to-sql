"""API request/response schemas for document extraction endpoints.

These Pydantic models define the HTTP API contract for document
upload, extraction, review, and retrieval. They translate between
the vision pipeline's domain schemas (ExtractionResult, DocumentRecord)
and the JSON structures sent/received over HTTP.

Schemas:
    DocumentExtractResponse — returned after extraction (before approval)
    DocumentApproveRequest — sent when user approves/corrects extraction
    DocumentRecordResponse — returned when reading stored documents
    DocumentRejectResponse — returned after rejecting an upload
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Extraction Response ─────────────────────────────────────────────

class DocumentExtractResponse(BaseModel):
    """POST /api/documents/extract -- response body.

    Returned after the vision pipeline extracts fields from an uploaded
    document. Contains all extracted data plus confidence metadata so the
    UI can render a review interface with field-level indicators.

    Attributes:
        fields: Dict mapping field names to their extracted values.
            Values can be strings, numbers, or None.
        confidence_scores: Dict mapping field names to confidence floats
            (0.0 to 1.0). Used by the UI to color-code each field.
        needs_review: Dict mapping field names to boolean flags. True
            when the field's confidence is below the review threshold.
        document_type: Detected or hinted document type (e.g., "invoice").
        overall_confidence: Average confidence across all extracted fields.
        model_used: Which vision LLM model performed the extraction.
        temp_file_name: Name of the temporary file stored in db/uploads/.
            Needed for the approve/reject endpoints to clean up the file.
        notes: Any validation warnings or extraction notes.
    """
    fields: dict = {}
    confidence_scores: dict[str, float] = {}
    needs_review: dict[str, bool] = {}
    document_type: str = "unknown"
    overall_confidence: float = 0.0
    model_used: str = ""
    temp_file_name: str = ""
    notes: str = ""


# ── Approve / Save Request ──────────────────────────────────────────

class DocumentApproveRequest(BaseModel):
    """POST /api/documents/approve -- request body.

    Sent by the UI when the user approves (or corrects and approves) an
    extraction. Contains the full extraction data so it can be persisted
    to the extracted_documents table.

    If the user edited any fields, extracted_fields should contain the
    corrected values (not the original LLM output). The review_status
    should be set to "corrected" in that case.

    Attributes:
        temp_file_name: Original uploaded file name (for temp file cleanup).
        document_type: Document type (invoice, bill_of_lading, etc.).
        extracted_fields: Dict of field_name -> value (possibly user-corrected).
        confidence_scores: Dict of field_name -> confidence float.
        overall_confidence: Aggregate confidence score.
        extraction_model: Name of the vision model that performed extraction.
        review_status: "approved" if unchanged, "corrected" if user edited fields.
        linked_shipment_id: Optional shipment ID to link this document to.
        notes: Optional notes about the extraction or review.
    """
    temp_file_name: str
    document_type: str
    extracted_fields: dict
    confidence_scores: dict[str, float] = {}
    overall_confidence: float = 0.0
    extraction_model: str = ""
    review_status: str = "approved"
    linked_shipment_id: Optional[str] = None
    notes: Optional[str] = None


# ── Document Record Response ────────────────────────────────────────

class DocumentRecordResponse(BaseModel):
    """Response model for stored document records.

    Returned by GET /api/documents and GET /api/documents/{id}.
    Represents a fully persisted extraction record from the
    extracted_documents table.

    Attributes:
        document_id: Unique UUID assigned at storage time.
        document_type: Type of document (invoice, bill_of_lading, etc.).
        file_name: Original uploaded file name.
        extraction_model: LLM model that performed the extraction.
        overall_confidence: Aggregate confidence score (0.0 to 1.0).
        review_status: Current review state (pending, approved, corrected, rejected).
        extracted_fields: Dict of field_name -> extracted value.
        confidence_scores: Dict of field_name -> confidence float.
        linked_shipment_id: Optional FK to shipments.shipment_id.
        upload_timestamp: ISO timestamp of when the document was uploaded.
        reviewed_at: ISO timestamp of when the document was reviewed (or None).
        notes: Any notes about the extraction quality or review.
    """
    document_id: str
    document_type: str
    file_name: str
    extraction_model: Optional[str] = None
    overall_confidence: float = 0.0
    review_status: str = "pending"
    extracted_fields: dict = {}
    confidence_scores: dict[str, float] = {}
    linked_shipment_id: Optional[str] = None
    upload_timestamp: Optional[str] = None
    reviewed_at: Optional[str] = None
    notes: Optional[str] = None


# ── Reject Response ─────────────────────────────────────────────────

class DocumentRejectResponse(BaseModel):
    """DELETE /api/documents/reject/{file_name} -- response body.

    Simple confirmation that the temporary file was deleted.

    Attributes:
        status: Always "rejected".
        message: Human-readable confirmation message.
    """
    status: str = "rejected"
    message: str = "File deleted successfully."
