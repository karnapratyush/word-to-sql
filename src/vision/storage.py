"""Vision document storage facade.

This module provides backward-compatible module-level functions that
delegate to DocumentRepository. Existing code can continue to use:

    from src.vision.storage import store_extraction, get_extracted_documents

These functions handle the translation between domain schemas
(ExtractionResult, DocumentRecord) and the repository's flat
parameter interface. Specifically, they:
- Generate a UUID for new documents
- Flatten FieldExtraction objects into simple dicts for storage
- Create DocumentRecord objects from the stored data
- Rebuild the extracted_document_fields SQL VIEW after each insert

The VIEW trick: extracted_documents stores fields as JSON. Instead of
making the LLM write json_extract() calls, we create a SQL VIEW that
flattens all JSON fields into regular columns. The LLM sees it as a
normal table and writes simple SELECT queries against it.
"""

import json
import logging
import uuid
from typing import Optional

from src.common.schemas import ExtractionResult, DocumentRecord, ReviewStatus
from src.repositories.document_repo import DocumentRepository

logger = logging.getLogger(__name__)


# ── Auto-Linking ─────────────────────────────────────────────────────

def _try_auto_link(extracted_fields: dict, repo: DocumentRepository) -> str | None:
    """Try to automatically link an extracted document to a shipment.

    Checks extracted fields for shipment references (shipment_ref, po_number,
    shipment_id) and looks them up in the shipments table. Returns the
    matching shipment_id if found, None otherwise.

    This enables Capability C (linkage) — extracted documents become
    queryable alongside shipment data via JOINs.

    Args:
        extracted_fields: Flat dict of field_name → value from extraction.
        repo: DocumentRepository instance for database lookups.

    Returns:
        A shipment_id string if a match is found, None otherwise.
    """
    # Fields that might contain a shipment reference
    ref_fields = ["shipment_ref", "shipment_reference", "shipment_id", "po_number"]

    for field_name in ref_fields:
        ref_value = extracted_fields.get(field_name)
        if not ref_value or not isinstance(ref_value, str):
            continue

        ref_value = ref_value.strip()
        if not ref_value:
            continue

        # Try matching against shipments.shipment_id
        try:
            rows = repo._execute(
                "SELECT shipment_id FROM shipments WHERE shipment_id = ? LIMIT 1",
                (ref_value,),
            )
            if rows:
                return rows[0]["shipment_id"]
        except Exception:
            pass

        # Try matching against shipments.po_number
        if field_name == "po_number":
            try:
                rows = repo._execute(
                    "SELECT shipment_id FROM shipments WHERE po_number = ? LIMIT 1",
                    (ref_value,),
                )
                if rows:
                    return rows[0]["shipment_id"]
            except Exception:
                pass

    return None


# ── Known Fields Set + Dynamic VIEW ──────────────────────────────────
#
# Instead of rebuilding the VIEW on every insert, we maintain an in-memory
# set of known field names. The VIEW is only rebuilt when NEW fields are
# discovered (i.e., a document introduces a field name we haven't seen).
#
# Flow:
#   Startup: load all field names from DB → _known_fields set → build VIEW
#   Upload:  check if new fields exist → if yes, update set + rebuild VIEW
#                                       → if no, skip (fast path)

_known_fields: set[str] = set()


def _load_known_fields(repo: DocumentRepository) -> set[str]:
    """Read all stored documents and collect every unique JSON field name.

    This is called once at startup to populate the _known_fields set.

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


def _build_view(repo: DocumentRepository, fields: set[str]) -> None:
    """Create or replace the extracted_document_fields SQL VIEW.

    The VIEW flattens extracted_documents.extracted_fields JSON into
    regular columns. The LLM queries it like a normal table:
        SELECT bl_number, vessel_name FROM extracted_document_fields

    Internally, each column is a json_extract() call, but the LLM
    never needs to know that.

    Args:
        repo: DocumentRepository instance for database access.
        fields: Set of JSON field names to expose as columns.
    """
    if not fields:
        return

    # Base columns from extracted_documents (non-JSON, always present)
    base_columns = [
        "document_id",
        "document_type",
        "file_name",
        "overall_confidence",
        "review_status",
        "linked_shipment_id",
        "upload_timestamp",
    ]

    # JSON field columns — one json_extract() call per field name
    json_columns = []
    for field_name in sorted(fields):
        # Sanitize: only alphanumeric and underscore allowed in column names
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in field_name)
        json_columns.append(
            f"json_extract(extracted_fields, '$.{field_name}') as {safe_name}"
        )

    all_columns = base_columns + json_columns
    columns_sql = ",\n        ".join(all_columns)

    view_sql = f"""
    CREATE VIEW IF NOT EXISTS extracted_document_fields AS
    SELECT
        {columns_sql}
    FROM extracted_documents
    """

    try:
        conn = repo._get_connection()
        conn.execute("DROP VIEW IF EXISTS extracted_document_fields")
        conn.execute(view_sql)
        conn.commit()
        logger.info(
            "Built extracted_document_fields VIEW with %d JSON columns: %s",
            len(json_columns), sorted(fields),
        )
    except Exception as e:
        logger.warning("Failed to build extracted_document_fields VIEW: %s", e)


def _maybe_rebuild_view(repo: DocumentRepository, new_fields: dict) -> None:
    """Rebuild the VIEW only if the new document introduces unknown fields.

    Compares the incoming field names against _known_fields. If there are
    new ones, updates the set and rebuilds the VIEW. Otherwise, skips
    (fast path — no DB operations).

    Args:
        repo: DocumentRepository instance.
        new_fields: Field name → value dict from the newly stored document.
    """
    global _known_fields
    incoming = set(new_fields.keys())
    new_names = incoming - _known_fields

    if new_names:
        logger.info("New fields discovered: %s — rebuilding VIEW", new_names)
        _known_fields.update(new_names)
        _build_view(repo, _known_fields)
    else:
        logger.debug("No new fields — VIEW rebuild skipped")


def rebuild_view_on_startup(db_path: str | None = None) -> None:
    """Initialize the known fields set and build the VIEW at startup.

    Called from FastAPI lifespan to ensure the VIEW exists and the
    _known_fields set is populated before any queries arrive.

    Args:
        db_path: Optional database path override.
    """
    global _known_fields
    repo = DocumentRepository(db_path=db_path)
    _known_fields = _load_known_fields(repo)

    if _known_fields:
        _build_view(repo, _known_fields)
        logger.info("Startup: loaded %d known fields, VIEW built", len(_known_fields))
    else:
        logger.info("Startup: no extracted documents yet, VIEW not created")


# ── Document Storage ─────────────────────────────────────────────────

def store_extraction(
    result: ExtractionResult,
    file_name: str,
    db_path: Optional[str] = None,
) -> DocumentRecord:
    """Store a validated extraction result into the extracted_documents table.

    Generates a unique document_id (UUID), flattens the FieldExtraction
    objects into simple value/confidence dicts, and inserts the record.

    Args:
        result: The extraction result from the vision pipeline.
        file_name: Original uploaded file name.
        db_path: Optional database path override (for testing).

    Returns:
        DocumentRecord with the generated document_id and all stored fields.
    """
    repo = DocumentRepository(db_path=db_path)
    document_id = str(uuid.uuid4())

    # Flatten FieldExtraction objects into simple dicts for JSON storage.
    # FieldExtraction has .value and .confidence attributes; we extract
    # each into a separate dict for the repository.
    extracted_fields = {
        name: field.value for name, field in result.fields.items()
    }
    confidence_scores = {
        name: field.confidence for name, field in result.fields.items()
    }

    # Auto-link: check if extracted fields contain a shipment reference
    # that matches a real shipment in the database. This enables the
    # linkage queries (Capability C) between extracted docs and shipments.
    linked_shipment_id = _try_auto_link(extracted_fields, repo)

    # Insert into the database via repository
    repo.insert_document(
        document_id=document_id,
        document_type=result.document_type,
        file_name=file_name,
        extraction_model=result.model_used,
        overall_confidence=result.overall_confidence,
        extracted_fields=extracted_fields,
        confidence_scores=confidence_scores,
        raw_text=result.raw_text,
        notes=result.notes or None,  # Convert empty string to None
        linked_shipment_id=linked_shipment_id,
    )

    # Check if this document introduces new fields. If yes, rebuild the VIEW.
    # If no (same fields as before), skip — fast path, no DB operations.
    _maybe_rebuild_view(repo, extracted_fields)

    # Return a DocumentRecord with the linked shipment if found
    return DocumentRecord(
        linked_shipment_id=linked_shipment_id,
        document_id=document_id,
        document_type=result.document_type,
        file_name=file_name,
        extraction_model=result.model_used,
        overall_confidence=result.overall_confidence,
        review_status=ReviewStatus.PENDING,  # All new documents start as pending
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
        corrected_fields: If provided, replaces the stored extracted_fields
            with human-corrected values.
        db_path: Optional database path override.
    """
    repo = DocumentRepository(db_path=db_path)
    repo.update_review_status(document_id, status, corrected_fields)


# ── Document Retrieval ───────────────────────────────────────────────

def get_extracted_documents(db_path: Optional[str] = None) -> list[dict]:
    """Retrieve all extracted documents from the database.

    Returns documents ordered by upload time (newest first),
    with JSON fields parsed back to Python dicts.

    Args:
        db_path: Optional database path override.

    Returns:
        List of document dicts with parsed extracted_fields and confidence_scores.
    """
    repo = DocumentRepository(db_path=db_path)
    return repo.get_all_documents()
