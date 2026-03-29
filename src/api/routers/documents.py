"""Document extraction API endpoints.

This router handles the document upload, extraction, review, and retrieval
workflow. The flow follows a human-in-the-loop pattern:

1. Upload + Extract: POST /api/documents/extract
   - Accepts a multipart file upload (PDF, PNG, JPG)
   - Runs the vision pipeline (extract + validate)
   - Returns extraction results with confidence scores for review

2. Approve + Store: POST /api/documents/approve
   - User reviewed the extraction and approves (or corrects + approves)
   - Stores the extraction to the extracted_documents table
   - Deletes the temporary upload file
   - Returns the stored DocumentRecord

3. Reject: DELETE /api/documents/reject/{file_name}
   - User rejects the extraction entirely
   - Deletes the temporary upload file without storing anything

4. List: GET /api/documents
   - Returns all extracted documents from the database

5. Get: GET /api/documents/{document_id}
   - Returns a single document by its UUID

Error handling:
- UnsupportedFileError -> 400 Bad Request
- ExtractionError -> 500 Internal Server Error
- Document not found -> 404 Not Found
"""

import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from src.api.dependencies import get_vision_service
from src.api.schemas.documents import (
    DocumentApproveRequest,
    DocumentExtractResponse,
    DocumentRecordResponse,
    DocumentRejectResponse,
)
from src.common.exceptions import ExtractionError, UnsupportedFileError
from src.services.vision_service import VisionService

# ── Logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Router ──────────────────────────────────────────────────────────────
# This router is registered in src/api/main.py with prefix="/api/documents"
router = APIRouter()


# ── POST /extract — Upload and extract fields from a document ───────────

@router.post("/extract", response_model=DocumentExtractResponse)
async def extract_document(
    file: UploadFile = File(..., description="PDF, PNG, or JPG document to extract fields from"),
    doc_type_hint: Optional[str] = Query(
        default=None,
        description="Optional document type hint: invoice, bill_of_lading, packing_list, customs_declaration",
    ),
    service: VisionService = Depends(get_vision_service),
):
    """Upload a document file and extract structured fields using a vision LLM.

    Accepts a multipart file upload (PDF, PNG, JPG, JPEG) and runs the full
    vision extraction pipeline:
    1. Validates file type and size
    2. Saves temporarily to db/uploads/
    3. Converts PDF to images (if PDF)
    4. Classifies document type (if no hint provided)
    5. Extracts fields via vision LLM with retry on parse failure
    6. Validates extractions (confidence flags, consistency, completeness)

    The extraction result is returned for human review -- nothing is stored
    to the database at this point. The user must call /approve or /reject.

    Args:
        file: Uploaded file (multipart form data).
        doc_type_hint: Optional hint about document type to skip classification.
            One of: invoice, bill_of_lading, packing_list, customs_declaration.
        service: Injected VisionService instance.

    Returns:
        DocumentExtractResponse with fields, confidence scores, and metadata.

    Raises:
        HTTPException 400: If the file type or size is not supported.
        HTTPException 500: If extraction fails completely after retries.
    """
    logger.info(
        "POST /extract: filename=%s, content_type=%s, hint=%s",
        file.filename, file.content_type, doc_type_hint,
    )

    try:
        # Read the uploaded file bytes
        file_bytes = await file.read()

        # Determine the filename (use the uploaded name, or fall back to a default)
        file_name = file.filename or "uploaded_document"

        # Call the vision service to run the extraction pipeline
        result = service.extract_document(
            file_bytes=file_bytes,
            file_name=file_name,
            doc_type_hint=doc_type_hint,
        )

        # Return the extraction result for user review
        return DocumentExtractResponse(**result)

    except UnsupportedFileError as e:
        # File type or size validation failed -- return 400 Bad Request
        logger.warning("File validation failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    except ExtractionError as e:
        # Extraction pipeline failed -- return 500 Internal Server Error
        logger.error("Extraction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        # Unexpected error -- wrap and return 500
        logger.error("Unexpected error in extract_document: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")


# ── POST /approve — Approve extraction and store to DB ──────────────────

@router.post("/approve", response_model=DocumentRecordResponse)
def approve_document(
    request: DocumentApproveRequest,
    service: VisionService = Depends(get_vision_service),
):
    """Approve an extraction result and persist it to the database.

    Called after the user reviews the extraction in the UI and decides to
    approve it (possibly after correcting some fields). This endpoint:
    1. Stores the extraction to the extracted_documents table
    2. Updates the review status (approved or corrected)
    3. Deletes the temporary upload file from db/uploads/

    If the user edited any fields before approving, the corrected values
    should be in extracted_fields and review_status should be "corrected".

    Args:
        request: DocumentApproveRequest with extraction data and review status.
        service: Injected VisionService instance.

    Returns:
        DocumentRecordResponse representing the stored document.

    Raises:
        HTTPException 500: If storage fails.
    """
    logger.info(
        "POST /approve: file=%s, type=%s, status=%s",
        request.temp_file_name, request.document_type, request.review_status,
    )

    try:
        # Convert the Pydantic model to a dict for the service layer
        request_data = request.model_dump()

        # Store the extraction and clean up the temp file
        record = service.approve_document(request_data)

        # Return the stored record
        return DocumentRecordResponse(**record)

    except ExtractionError as e:
        logger.error("Failed to approve document: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error("Unexpected error in approve_document: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to store document: {e}")


# ── DELETE /reject/{file_name} — Reject and delete temp file ────────────

@router.delete("/reject/{file_name}", response_model=DocumentRejectResponse)
def reject_document(
    file_name: str,
    service: VisionService = Depends(get_vision_service),
):
    """Reject an extraction and delete the temporary upload file.

    Called when the user decides to reject the extraction entirely. Simply
    deletes the temporary file from db/uploads/ without storing anything
    to the database. This endpoint is idempotent -- calling it multiple
    times for the same file_name will not cause errors.

    Args:
        file_name: The original filename of the uploaded document.
        service: Injected VisionService instance.

    Returns:
        DocumentRejectResponse confirming the file was deleted.
    """
    logger.info("DELETE /reject/%s", file_name)

    # Sanitize file_name to prevent path traversal
    safe_name = re.sub(r'[/\\]', '', file_name)
    safe_name = safe_name.replace('..', '')
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid file name")

    result = service.reject_document(safe_name)

    return DocumentRejectResponse(
        status=result["status"],
        message=result["message"],
    )


# ── GET / — List all extracted documents ────────────────────────────────

@router.get("", response_model=list[DocumentRecordResponse])
def list_documents(
    service: VisionService = Depends(get_vision_service),
):
    """List all extracted documents from the database.

    Returns all documents in the extracted_documents table, ordered by
    upload timestamp (newest first). JSON fields (extracted_fields,
    confidence_scores) are parsed back to Python dicts.

    Args:
        service: Injected VisionService instance.

    Returns:
        List of DocumentRecordResponse objects.
    """
    logger.info("GET /documents (list all)")

    try:
        documents = service.list_documents()

        # Convert each raw dict to the response model.
        # Database rows may have extra fields (id, raw_text, etc.) that
        # are not in the response model -- Pydantic will ignore them.
        return [DocumentRecordResponse(**doc) for doc in documents]

    except Exception as e:
        logger.error("Failed to list documents: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")


# ── GET /{document_id} — Get single document by ID ─────────────────────

@router.get("/{document_id}", response_model=DocumentRecordResponse)
def get_document(
    document_id: str,
    service: VisionService = Depends(get_vision_service),
):
    """Retrieve a single extracted document by its UUID.

    Args:
        document_id: The UUID of the document to retrieve.
        service: Injected VisionService instance.

    Returns:
        DocumentRecordResponse with the document's fields and metadata.

    Raises:
        HTTPException 404: If no document with the given ID exists.
    """
    logger.info("GET /documents/%s", document_id)

    try:
        document = service.get_document(document_id)

        if document is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}",
            )

        return DocumentRecordResponse(**document)

    except HTTPException:
        # Re-raise HTTP exceptions (like 404) without wrapping
        raise

    except Exception as e:
        logger.error("Failed to get document %s: %s", document_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get document: {e}")
