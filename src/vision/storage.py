"""Vision document storage facade.

This module provides module-level functions that delegate to DocumentRepository.

    from src.vision.storage import store_extraction, get_extracted_documents

Key responsibilities:
- Generate UUIDs for new documents
- Flatten FieldExtraction objects into simple dicts for JSON storage
- Auto-link documents to shipments via shipment_ref / po_number matching
- Maintain an in-memory set of known JSON field names (_known_fields)
- Provide get_known_fields() for the schema description builder

The _known_fields set:
  On startup, we read all stored documents and collect every unique JSON
  field name. When new documents are uploaded, the set grows. This set is
  used by analytics_repo.get_schema_description() to tell the LLM what
  fields exist inside extracted_documents.extracted_fields JSON, so the
  LLM can write json_extract() queries against them.
"""

import json
import logging
import threading
import uuid
from typing import Optional

from src.common.schemas import ExtractionResult, DocumentRecord, ReviewStatus
from src.repositories.document_repo import DocumentRepository

logger = logging.getLogger(__name__)


# ── Known Fields Set ─────────────────────────────────────────────────
#
# In-memory set of all JSON field names seen across extracted documents.
# Used by the schema description to tell the LLM what columns exist
# inside the extracted_fields JSON blob.
#
# Flow:
#   Startup:  load all field names from DB → _known_fields
#   Upload:   add new field names to _known_fields (if any)
#   Query:    analytics_repo reads _known_fields via get_known_fields()
#             and includes them in the schema description sent to the LLM

_known_fields: set[str] = set()
# Thread lock for atomic updates to _known_fields
_fields_lock = threading.Lock()


def _load_known_fields(repo: DocumentRepository) -> set[str]:
    """Read all stored documents and collect every unique JSON field name.

    Called once at startup to populate _known_fields.

    Args:
        repo: DocumentRepository instance for database access.

    Returns:
        Set of all unique field names across all stored documents.
    """
    fields: set[str] = set()
    try:
        rows = repo._execute("SELECT extracted_fields FROM extracted_documents")
        for row in rows:
            fields_str = row.get("extracted_fields", "{}")
            if isinstance(fields_str, str):
                parsed = json.loads(fields_str)
            else:
                parsed = fields_str
            fields.update(parsed.keys())
    except Exception as e:
        logger.warning("Failed to load known fields: %s", e)
    return fields


def init_known_fields(db_path: str | None = None) -> None:
    """Initialize the known fields set at application startup.

    Called from FastAPI lifespan to populate _known_fields before any
    queries arrive. After this, get_known_fields() returns the full set.

    Args:
        db_path: Optional database path override.
    """
    global _known_fields
    repo = DocumentRepository(db_path=db_path)
    _known_fields = _load_known_fields(repo)
    logger.info("Startup: loaded %d known extracted fields: %s",
                len(_known_fields), sorted(_known_fields))


def get_known_fields() -> set[str]:
    """Return the current set of known JSON field names.

    Used by analytics_repo.get_schema_description() to include
    extracted document field names in the schema sent to the LLM.

    Returns:
        Frozen copy of the known fields set.
    """
    return _known_fields.copy()


def _update_known_fields(new_fields: dict) -> None:
    """Add any new field names from an uploaded document to the set.

    Called after each document insert. If the document has field names
    we haven't seen before, they're added to the set. Uses a lock
    and atomic replacement for thread safety.

    Args:
        new_fields: Field name → value dict from the newly stored document.
    """
    global _known_fields
    incoming = set(new_fields.keys())
    new_names = incoming - _known_fields
    if new_names:
        with _fields_lock:
            _known_fields = _known_fields | new_names  # atomic replacement
        logger.info("New fields discovered: %s", new_names)


# ── Auto-Linking ─────────────────────────────────────────────────────

def _try_auto_link(extracted_fields: dict, repo: DocumentRepository) -> str | None:
    """Try to automatically link an extracted document to a shipment.

    Searches multiple tables in priority order for a matching reference:
    1. shipments.shipment_id (direct match on shipment reference)
    2. shipments.po_number (match on purchase order number)
    3. invoices.invoice_number → invoices.shipment_id (match via invoice)

    Stops at the first match found. Returns the shipment_id to store
    in extracted_documents.linked_shipment_id for JOIN queries.

    Args:
        extracted_fields: Flat dict of field_name → value from extraction.
        repo: DocumentRepository instance for database lookups.

    Returns:
        A shipment_id string if a match is found, None otherwise.
    """
    # Priority 1: Direct shipment reference fields → shipments.shipment_id
    for field_name in ["shipment_ref", "shipment_reference", "shipment_id"]:
        ref_value = _get_clean_value(extracted_fields, field_name)
        if not ref_value:
            continue
        try:
            rows = repo._execute(
                "SELECT shipment_id FROM shipments WHERE shipment_id = ? LIMIT 1",
                (ref_value,),
            )
            if rows:
                logger.info("Auto-linked via %s → shipments.shipment_id: %s",
                           field_name, rows[0]["shipment_id"])
                return rows[0]["shipment_id"]
        except Exception as e:
            # Log at debug level — auto-link failures are non-critical
            logger.debug("Auto-link lookup failed for %s: %s", field_name, e)

    # Priority 2: PO number → shipments.po_number
    po_value = _get_clean_value(extracted_fields, "po_number")
    if po_value:
        try:
            rows = repo._execute(
                "SELECT shipment_id FROM shipments WHERE po_number = ? LIMIT 1",
                (po_value,),
            )
            if rows:
                logger.info("Auto-linked via po_number → shipments.po_number: %s",
                           rows[0]["shipment_id"])
                return rows[0]["shipment_id"]
        except Exception as e:
            # Log at debug level — auto-link failures are non-critical
            logger.debug("Auto-link lookup failed for po_number: %s", e)

    # Priority 3: Invoice number → invoices.invoice_number → invoices.shipment_id
    inv_value = _get_clean_value(extracted_fields, "invoice_number")
    if inv_value:
        try:
            rows = repo._execute(
                "SELECT s.shipment_id FROM invoices i "
                "JOIN shipments s ON i.shipment_id = s.id "
                "WHERE i.invoice_number = ? LIMIT 1",
                (inv_value,),
            )
            if rows:
                logger.info("Auto-linked via invoice_number → invoices → shipments: %s",
                           rows[0]["shipment_id"])
                return rows[0]["shipment_id"]
        except Exception as e:
            # Log at debug level — auto-link failures are non-critical
            logger.debug("Auto-link lookup failed for invoice_number: %s", e)

    return None


def _get_clean_value(fields: dict, key: str) -> str | None:
    """Get a trimmed string value from a dict, or None if empty/missing.

    Args:
        fields: Flat dict of field_name → value.
        key: The key to look up.

    Returns:
        Trimmed string value, or None if missing/empty/non-string.
    """
    value = fields.get(key)
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    return value if value else None


# ── Document Storage ─────────────────────────────────────────────────

def store_extraction(
    result: ExtractionResult,
    file_name: str,
    db_path: Optional[str] = None,
) -> DocumentRecord:
    """Store a validated extraction result into the extracted_documents table.

    Also:
    - Auto-links to a shipment if a matching reference is found
    - Updates the in-memory _known_fields set with any new field names

    Args:
        result: The extraction result from the vision pipeline.
        file_name: Original uploaded file name.
        db_path: Optional database path override (for testing).

    Returns:
        DocumentRecord with the generated document_id and all stored fields.
    """
    repo = DocumentRepository(db_path=db_path)
    document_id = str(uuid.uuid4())

    extracted_fields = {
        name: field.value for name, field in result.fields.items()
    }
    confidence_scores = {
        name: field.confidence for name, field in result.fields.items()
    }

    linked_shipment_id = _try_auto_link(extracted_fields, repo)

    repo.insert_document(
        document_id=document_id,
        document_type=result.document_type,
        file_name=file_name,
        extraction_model=result.model_used,
        overall_confidence=result.overall_confidence,
        extracted_fields=extracted_fields,
        confidence_scores=confidence_scores,
        raw_text=result.raw_text,
        notes=result.notes or None,
        linked_shipment_id=linked_shipment_id,
    )

    # Update the in-memory set with any new field names
    _update_known_fields(extracted_fields)

    return DocumentRecord(
        linked_shipment_id=linked_shipment_id,
        document_id=document_id,
        document_type=result.document_type,
        file_name=file_name,
        extraction_model=result.model_used,
        overall_confidence=result.overall_confidence,
        review_status=ReviewStatus.PENDING,
        extracted_fields=extracted_fields,
        confidence_scores=confidence_scores,
    )


# ── Review Status Updates ────────────────────────────────────────────

def update_review_status(
    document_id: str,
    status: str,
    corrected_fields: Optional[dict] = None,
    db_path: Optional[str] = None,
) -> None:
    """Update the review status of an extracted document.

    Args:
        document_id: UUID of the document to update.
        status: New status string (pending, approved, rejected, corrected).
        corrected_fields: If provided, replaces the stored extracted_fields.
        db_path: Optional database path override.
    """
    repo = DocumentRepository(db_path=db_path)
    repo.update_review_status(document_id, status, corrected_fields)


# ── Document Retrieval ───────────────────────────────────────────────

def get_extracted_documents(db_path: Optional[str] = None) -> list[dict]:
    """Retrieve all extracted documents from the database.

    Args:
        db_path: Optional database path override.

    Returns:
        List of document dicts with parsed extracted_fields and confidence_scores.
    """
    repo = DocumentRepository(db_path=db_path)
    return repo.get_all_documents()
