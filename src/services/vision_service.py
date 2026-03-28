"""Vision Service -- service layer wrapping the vision agent.

This is the service layer between the FastAPI controller and the vision
domain logic. It follows the same pattern as AnalyticsService: the
controller handles HTTP concerns, the service handles business
orchestration, and the domain modules (vision agent, extractor,
validator, storage) handle the specific logic.

The service translates between API-level data (dicts, file bytes) and
domain-level schemas (ExtractionRequest, ExtractionResult, DocumentRecord).

In the MVCR (Model-View-Controller-Repository) pattern:
- Controller (FastAPI route in routers/documents.py) handles HTTP
- Service (this file) orchestrates domain calls
- Domain (src/vision/agent.py) runs the extraction pipeline
- Repository (src/repositories/document_repo.py) handles persistence

Public methods:
    extract_document(file_bytes, file_name, doc_type_hint) -> dict
    approve_document(request_data) -> dict
    reject_document(file_name) -> dict
    list_documents() -> list[dict]
    get_document(document_id) -> dict | None
"""

import logging
from typing import Optional

from src.common.schemas import (
    ExtractionRequest,
    ExtractionResult,
    FieldExtraction,
)
from src.vision.agent import (
    process_document,
    store_approved_document,
    delete_uploaded_file,
)
from src.repositories.document_repo import DocumentRepository

# ── Logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


class VisionService:
    """Service for document extraction, approval, and retrieval.

    Wraps the vision agent pipeline with:
    - Database path injection (from FastAPI app.state)
    - Translation between API dicts and domain schemas
    - Error handling with meaningful log messages

    Attributes:
        _db_path: Optional database path override (None uses default).
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the service with an optional database path.

        Args:
            db_path: Optional path to a SQLite database file.
                When None, uses the default path (db/logistics.db)
                configured in the repository layer.
        """
        self._db_path = db_path

    # ── Extract ─────────────────────────────────────────────────────────

    def extract_document(
        self,
        file_bytes: bytes,
        file_name: str,
        doc_type_hint: Optional[str] = None,
    ) -> dict:
        """Run the full vision extraction pipeline on an uploaded document.

        Delegates to the vision agent's process_document() which handles:
        1. File validation (type, size)
        2. Temporary file storage in db/uploads/
        3. PDF-to-image conversion (if PDF)
        4. Document type classification (if no hint)
        5. Field extraction via vision LLM
        6. Validation (confidence flags, consistency, completeness)

        The result is NOT stored to the database yet -- it is returned
        for human review in the UI.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            file_name: Original filename (used for type detection and storage).
            doc_type_hint: Optional hint about document type (e.g., "invoice")
                to skip the classification step.

        Returns:
            Dict with keys: fields, confidence_scores, needs_review,
            document_type, overall_confidence, model_used, temp_file_name, notes.

        Raises:
            UnsupportedFileError: If the file type or size is invalid.
            ExtractionError: If extraction fails completely.
        """
        logger.info(
            "VisionService.extract_document: file=%s, size=%d, hint=%s",
            file_name, len(file_bytes), doc_type_hint,
        )

        # Build the domain request schema
        request = ExtractionRequest(
            file_bytes=file_bytes,
            file_name=file_name,
            document_type_hint=doc_type_hint,
        )

        # Run the full vision pipeline (extract + validate)
        result: ExtractionResult = process_document(request)

        # Translate domain ExtractionResult to API-friendly dict.
        # The domain uses FieldExtraction objects; the API needs flat dicts.
        fields = {}
        confidence_scores = {}
        needs_review = {}

        for field_name, field_obj in result.fields.items():
            fields[field_name] = field_obj.value
            confidence_scores[field_name] = field_obj.confidence
            needs_review[field_name] = field_obj.needs_review

        return {
            "fields": fields,
            "confidence_scores": confidence_scores,
            "needs_review": needs_review,
            "document_type": result.document_type,
            "overall_confidence": result.overall_confidence,
            "model_used": result.model_used,
            "temp_file_name": file_name,
            "notes": result.notes or "",
        }

    # ── Approve / Save ──────────────────────────────────────────────────

    def approve_document(self, request_data: dict) -> dict:
        """Store an approved (or corrected) extraction to the database.

        Called after the user reviews the extraction in the UI and clicks
        "Approve" or "Save as Corrected". Builds an ExtractionResult from
        the request data and delegates to the vision agent's
        store_approved_document() which handles:
        1. Inserting the record into extracted_documents table
        2. Updating review status (approved or corrected)
        3. Deleting the temporary upload file

        Args:
            request_data: Dict from DocumentApproveRequest with keys:
                temp_file_name, document_type, extracted_fields,
                confidence_scores, overall_confidence, extraction_model,
                review_status, linked_shipment_id, notes.

        Returns:
            Dict representing the stored DocumentRecord with keys:
            document_id, document_type, file_name, extraction_model,
            overall_confidence, review_status, extracted_fields,
            confidence_scores, linked_shipment_id.

        Raises:
            ExtractionError: If storage fails.
        """
        logger.info(
            "VisionService.approve_document: file=%s, status=%s",
            request_data.get("temp_file_name"),
            request_data.get("review_status"),
        )

        # Rebuild FieldExtraction objects from the flat dicts.
        # This is the reverse of what extract_document() does.
        extracted_fields = request_data.get("extracted_fields", {})
        confidence_scores = request_data.get("confidence_scores", {})

        field_extractions = {}
        for field_name, value in extracted_fields.items():
            confidence = confidence_scores.get(field_name, 0.5)
            field_extractions[field_name] = FieldExtraction(
                value=value,
                confidence=confidence,
                needs_review=False,  # Already reviewed by user
            )

        # Build the domain ExtractionResult
        result = ExtractionResult(
            document_type=request_data.get("document_type", "unknown"),
            fields=field_extractions,
            overall_confidence=request_data.get("overall_confidence", 0.0),
            model_used=request_data.get("extraction_model", ""),
            notes=request_data.get("notes", ""),
        )

        # Store to database and clean up temp file
        record = store_approved_document(
            result=result,
            file_name=request_data.get("temp_file_name", "unknown"),
            review_status=request_data.get("review_status", "approved"),
            db_path=self._db_path,
        )

        # Convert the DocumentRecord to a serializable dict
        return {
            "document_id": record.document_id,
            "document_type": record.document_type,
            "file_name": record.file_name,
            "extraction_model": record.extraction_model,
            "overall_confidence": record.overall_confidence,
            "review_status": record.review_status.value
                if hasattr(record.review_status, "value")
                else record.review_status,
            "extracted_fields": record.extracted_fields,
            "confidence_scores": record.confidence_scores,
            "linked_shipment_id": record.linked_shipment_id,
        }

    # ── Reject ──────────────────────────────────────────────────────────

    def reject_document(self, file_name: str) -> dict:
        """Delete the temporary upload file when user rejects the extraction.

        Called when the user clicks "Reject" in the review UI. Simply
        deletes the temporary file from db/uploads/ without storing
        anything to the database.

        Args:
            file_name: The original filename of the uploaded document.

        Returns:
            Dict with status="rejected" and a confirmation message.
        """
        logger.info("VisionService.reject_document: file=%s", file_name)

        delete_uploaded_file(file_name)

        return {
            "status": "rejected",
            "message": f"File '{file_name}' deleted successfully.",
        }

    # ── List / Get ──────────────────────────────────────────────────────

    def list_documents(self) -> list[dict]:
        """Retrieve all extracted documents from the database.

        Returns documents ordered by upload time (newest first),
        with JSON fields parsed back to Python dicts.

        Returns:
            List of document dicts with parsed extracted_fields
            and confidence_scores.
        """
        logger.info("VisionService.list_documents")
        repo = DocumentRepository(db_path=self._db_path)
        return repo.get_all_documents()

    def get_document(self, document_id: str) -> Optional[dict]:
        """Retrieve a single document by its document_id.

        Args:
            document_id: The UUID of the document to retrieve.

        Returns:
            Dict with parsed JSON fields, or None if not found.
        """
        logger.info("VisionService.get_document: id=%s", document_id)
        repo = DocumentRepository(db_path=self._db_path)
        return repo.get_document_by_id(document_id)
