"""Repository for extracted document CRUD operations.

This is the write-capable repository — unlike AnalyticsRepository which
is strictly read-only, DocumentRepository supports INSERT and UPDATE
operations for managing documents extracted by the vision pipeline.

Table: extracted_documents
    Stores structured field data (as JSON), confidence scores,
    review status, and optional links to shipment records.
"""

import json
from datetime import datetime
from typing import Optional

from src.repositories.base import BaseRepository


class DocumentRepository(BaseRepository):
    """CRUD operations for the extracted_documents table.

    Provides methods to insert new extraction records, retrieve them,
    update review status (approve/reject/correct), and link documents
    to shipments for cross-referencing in analytics queries.

    JSON fields (extracted_fields, confidence_scores) are serialized
    to JSON strings on write and deserialized back to dicts on read.
    """

    def insert_document(
        self,
        document_id: str,
        document_type: str,
        file_name: str,
        extraction_model: str,
        overall_confidence: float,
        extracted_fields: dict,
        confidence_scores: dict,
        raw_text: Optional[str] = None,
        notes: Optional[str] = None,
        linked_shipment_id: Optional[str] = None,
    ) -> None:
        """Insert a new extracted document record.

        The document starts with review_status='pending' and awaits
        human review before being marked as approved or corrected.

        Args:
            document_id: Unique UUID for this document.
            document_type: Type (invoice, bill_of_lading, packing_list, etc.).
            file_name: Original uploaded file name.
            extraction_model: Name of the LLM model used for extraction.
            overall_confidence: Overall confidence score (0.0-1.0).
            extracted_fields: Dict of field_name -> extracted value.
            confidence_scores: Dict of field_name -> confidence (0.0-1.0).
            raw_text: Optional raw OCR text from the document.
            notes: Optional notes about the extraction quality or issues.
            linked_shipment_id: Optional FK to shipments.shipment_id.
        """
        self._execute_write(
            """INSERT INTO extracted_documents
            (document_id, document_type, file_name, extraction_model,
             overall_confidence, extracted_fields, confidence_scores,
             raw_text, notes, linked_shipment_id, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
            (
                document_id,
                document_type,
                file_name,
                extraction_model,
                overall_confidence,
                json.dumps(extracted_fields),   # Serialize dict to JSON string
                json.dumps(confidence_scores),  # Serialize dict to JSON string
                raw_text,
                notes,
                linked_shipment_id,
            ),
        )

    def get_all_documents(self) -> list[dict]:
        """Retrieve all extracted documents, ordered by upload time (newest first).

        Returns:
            List of dicts with extracted_fields and confidence_scores
            parsed from JSON strings back to Python dicts.
        """
        rows = self._execute(
            "SELECT * FROM extracted_documents ORDER BY upload_timestamp DESC"
        )
        # Parse JSON string columns back to dicts for each row
        return [self._parse_json_fields(row) for row in rows]

    def get_document_by_id(self, document_id: str) -> Optional[dict]:
        """Retrieve a single document by its document_id.

        Args:
            document_id: The UUID of the document.

        Returns:
            Dict with parsed JSON fields, or None if not found.
        """
        rows = self._execute(
            "SELECT * FROM extracted_documents WHERE document_id = ?",
            (document_id,),
        )
        if not rows:
            return None
        return self._parse_json_fields(rows[0])

    def update_review_status(
        self,
        document_id: str,
        status: str,
        corrected_fields: Optional[dict] = None,
    ) -> None:
        """Update the review status of an extracted document.

        When corrected_fields is provided, the stored extracted_fields
        are replaced with the corrected values (status should be 'corrected').

        Args:
            document_id: The UUID of the document to update.
            status: New status (pending, approved, rejected, corrected).
            corrected_fields: If provided, replaces extracted_fields
                with human-corrected values.

        Raises:
            ValueError: If document_id is not found in the database.
        """
        # Verify document exists before updating
        existing = self.get_document_by_id(document_id)
        if existing is None:
            raise ValueError(f"Document not found: {document_id}")

        if corrected_fields is not None:
            # Update both status and fields when corrections are provided
            self._execute_write(
                """UPDATE extracted_documents
                SET review_status = ?, extracted_fields = ?, reviewed_at = ?
                WHERE document_id = ?""",
                (
                    status,
                    json.dumps(corrected_fields),  # Serialize corrected dict to JSON
                    datetime.now().isoformat(),     # Record when the review happened
                    document_id,
                ),
            )
        else:
            # Update only the status (approve or reject without field changes)
            self._execute_write(
                """UPDATE extracted_documents
                SET review_status = ?, reviewed_at = ?
                WHERE document_id = ?""",
                (
                    status,
                    datetime.now().isoformat(),
                    document_id,
                ),
            )

    def link_shipment(self, document_id: str, shipment_id: str) -> None:
        """Link an extracted document to a shipment record.

        This creates a foreign-key relationship between the document and
        a shipment, allowing cross-referencing in analytics queries.

        Args:
            document_id: The UUID of the document.
            shipment_id: The shipment_id to link to (e.g., 'SHP-2024-000001').

        Raises:
            ValueError: If document_id is not found in the database.
        """
        existing = self.get_document_by_id(document_id)
        if existing is None:
            raise ValueError(f"Document not found: {document_id}")

        self._execute_write(
            "UPDATE extracted_documents SET linked_shipment_id = ? WHERE document_id = ?",
            (shipment_id, document_id),
        )

    # ── Internal Helpers ─────────────────────────────────────────────

    @staticmethod
    def _parse_json_fields(row: dict) -> dict:
        """Parse JSON string fields back to Python dicts.

        The extracted_fields and confidence_scores columns are stored as
        JSON strings in the database. This method deserializes them so
        callers receive native dicts instead of raw JSON strings.

        Args:
            row: A single row dict from the database.

        Returns:
            A copy of the row with JSON fields parsed to dicts.
        """
        result = dict(row)
        for field in ("extracted_fields", "confidence_scores"):
            if field in result and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    pass  # Leave as-is if not valid JSON (shouldn't happen)
        return result
