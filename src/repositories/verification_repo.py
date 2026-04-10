"""Repository for verification CRUD operations.

Follows the same pattern as DocumentRepository (document_repo.py):
extends BaseRepository for engine-agnostic database access, provides
typed methods for INSERT, SELECT, and UPDATE operations.

Tables managed:
    verification_results — overall verification outcome per document
    verification_fields  — field-by-field comparison details

This repository handles two related tables because they form a single
aggregate in DDD terms: a verification result is meaningless without
its field details. The repository ensures they are always written and
read together consistently.
"""

from datetime import datetime
from typing import Optional

from src.repositories.base import BaseRepository


class VerificationRepository(BaseRepository):
    """CRUD operations for verification_results and verification_fields tables.

    Provides methods to:
    - Insert verification results and their field comparisons
    - Retrieve verifications by ID, status, or customer
    - Update review status (when a human reviews the verification)
    - Get aggregate statistics for the dashboard

    All methods use parameterized queries to prevent SQL injection.
    """

    # ── Insert Operations ───────────────────────────────────────────────

    def insert_verification_result(
        self,
        verification_id: str,
        document_id: str,
        customer_id: str,
        overall_status: str,
        shipment_ref: Optional[str] = None,
        draft_reply: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Insert a new verification result record.

        The verification starts with reviewed_by=NULL and reviewed_at=NULL,
        indicating it has not yet been reviewed by a human.

        Args:
            verification_id: Unique UUID for this verification.
            document_id: UUID of the document that was verified.
            customer_id: Customer identifier (matches YAML config filename).
            overall_status: Verification outcome (approved, amendment_required,
                uncertain, extraction_failed, no_rules).
            shipment_ref: Optional shipment reference number.
            draft_reply: Optional generated draft email text.
            notes: Optional notes about the verification.
        """
        self._execute_write(
            """INSERT INTO verification_results
            (verification_id, document_id, shipment_ref, customer_id,
             overall_status, draft_reply, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                verification_id,
                document_id,
                shipment_ref,
                customer_id,
                overall_status,
                draft_reply,
                notes,
            ),
        )

    def insert_verification_field(
        self,
        verification_id: str,
        field_name: str,
        status: str,
        extracted_value: Optional[str] = None,
        expected_value: Optional[str] = None,
        confidence: Optional[float] = None,
        rule_type: Optional[str] = None,
        rule_violated: Optional[str] = None,
    ) -> None:
        """Insert a single field comparison record.

        Each field comparison is stored as a separate row in the
        verification_fields table, linked to the parent verification
        result via verification_id.

        Args:
            verification_id: FK to verification_results.verification_id.
            field_name: Name of the field that was compared.
            status: Comparison result (match, mismatch, uncertain, no_rule).
            extracted_value: The value extracted from the document.
            expected_value: The expected value from the customer rule.
            confidence: Extraction confidence score (0.0-1.0).
            rule_type: Match type used (exact, prefix, one_of, etc.).
            rule_violated: Description of which rule was violated (or None).
        """
        self._execute_write(
            """INSERT INTO verification_fields
            (verification_id, field_name, extracted_value, expected_value,
             status, confidence, rule_type, rule_violated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                verification_id,
                field_name,
                extracted_value,
                expected_value,
                status,
                confidence,
                rule_type,
                rule_violated,
            ),
        )

    # ── Read Operations ─────────────────────────────────────────────────

    def get_all_verifications(self) -> list[dict]:
        """Retrieve all verification results, ordered by newest first.

        Returns:
            List of dicts with verification result columns.
            Does NOT include field details (call get_fields_for_verification
            separately for the detail view).
        """
        return self._execute(
            "SELECT * FROM verification_results ORDER BY received_at DESC"
        )

    def get_verification_by_id(self, verification_id: str) -> Optional[dict]:
        """Retrieve a single verification result by its verification_id.

        Args:
            verification_id: The UUID of the verification.

        Returns:
            Dict with verification result columns, or None if not found.
        """
        rows = self._execute(
            "SELECT * FROM verification_results WHERE verification_id = ?",
            (verification_id,),
        )
        return rows[0] if rows else None

    def get_fields_for_verification(self, verification_id: str) -> list[dict]:
        """Retrieve all field comparisons for a specific verification.

        Args:
            verification_id: The UUID of the parent verification result.

        Returns:
            List of dicts with field comparison details, ordered by
            field name for consistent display.
        """
        return self._execute(
            """SELECT * FROM verification_fields
            WHERE verification_id = ?
            ORDER BY field_name""",
            (verification_id,),
        )

    def get_verifications_by_status(self, status: str) -> list[dict]:
        """Retrieve all verifications with a specific overall status.

        Useful for filtering the dashboard view (e.g., show only
        amendment_required verifications).

        Args:
            status: The overall_status to filter by.

        Returns:
            List of dicts ordered by newest first.
        """
        return self._execute(
            """SELECT * FROM verification_results
            WHERE overall_status = ?
            ORDER BY received_at DESC""",
            (status,),
        )

    def get_verifications_by_customer(self, customer_id: str) -> list[dict]:
        """Retrieve all verifications for a specific customer.

        Args:
            customer_id: The customer identifier to filter by.

        Returns:
            List of dicts ordered by newest first.
        """
        return self._execute(
            """SELECT * FROM verification_results
            WHERE customer_id = ?
            ORDER BY received_at DESC""",
            (customer_id,),
        )

    # ── Update Operations ───────────────────────────────────────────────

    def update_review_status(
        self,
        verification_id: str,
        reviewed_by: str,
        notes: Optional[str] = None,
        overall_status: Optional[str] = None,
        draft_reply: Optional[str] = None,
    ) -> None:
        """Update the review status of a verification result.

        Called when a human reviews the verification and either confirms
        the automated result or overrides it. Records who reviewed it
        and when.

        Args:
            verification_id: UUID of the verification to update.
            reviewed_by: Name/email of the person who reviewed.
            notes: Optional updated notes from the reviewer.
            overall_status: Optional status override (if reviewer changes
                the automated decision).
            draft_reply: Optional updated draft email text.

        Raises:
            ValueError: If the verification_id is not found.
        """
        # Verify the record exists before updating
        existing = self.get_verification_by_id(verification_id)
        if existing is None:
            raise ValueError(f"Verification not found: {verification_id}")

        # Build the SET clause dynamically based on which fields are provided
        set_parts = ["reviewed_by = ?", "reviewed_at = ?"]
        params = [reviewed_by, datetime.now().isoformat()]

        if notes is not None:
            set_parts.append("notes = ?")
            params.append(notes)

        if overall_status is not None:
            set_parts.append("overall_status = ?")
            params.append(overall_status)

        if draft_reply is not None:
            set_parts.append("draft_reply = ?")
            params.append(draft_reply)

        params.append(verification_id)

        sql = f"UPDATE verification_results SET {', '.join(set_parts)} WHERE verification_id = ?"

        self._execute_write(sql, tuple(params))

    # ── Statistics ──────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get aggregate statistics for the verification dashboard.

        Returns counts grouped by overall_status, plus total count and
        the number of reviewed vs. unreviewed verifications.

        Returns:
            Dict with keys:
                total: Total number of verifications.
                by_status: Dict mapping status to count.
                reviewed: Number of verifications reviewed by a human.
                pending_review: Number awaiting human review.
        """
        # Total count
        total_rows = self._execute("SELECT COUNT(*) as cnt FROM verification_results")
        total = total_rows[0]["cnt"] if total_rows else 0

        # Counts by status
        status_rows = self._execute(
            """SELECT overall_status, COUNT(*) as cnt
            FROM verification_results
            GROUP BY overall_status"""
        )
        by_status = {row["overall_status"]: row["cnt"] for row in status_rows}

        # Reviewed vs. pending
        reviewed_rows = self._execute(
            "SELECT COUNT(*) as cnt FROM verification_results WHERE reviewed_by IS NOT NULL"
        )
        reviewed = reviewed_rows[0]["cnt"] if reviewed_rows else 0

        return {
            "total": total,
            "by_status": by_status,
            "reviewed": reviewed,
            "pending_review": total - reviewed,
        }
