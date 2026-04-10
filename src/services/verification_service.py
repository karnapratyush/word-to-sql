"""Verification Service — service layer wrapping the verification agent.

Follows the same pattern as VisionService (vision_service.py):
the controller (FastAPI route) handles HTTP concerns, the service
handles business orchestration, and the domain modules (verification
agent, comparator, rules_loader, draft_generator) handle specifics.

In the MVCR (Model-View-Controller-Repository) pattern:
- Controller (FastAPI route in routers/verification.py) handles HTTP
- Service (this file) orchestrates domain calls
- Domain (src/verification/agent.py) runs the verification pipeline
- Repository (src/repositories/verification_repo.py) handles persistence

Public methods:
    process_document(file_bytes, file_name, customer_id, ...) -> dict
    list_verifications(status_filter, customer_filter) -> list[dict]
    get_verification(verification_id) -> dict | None
    update_review(verification_id, reviewed_by, ...) -> dict
    get_stats() -> dict
    list_customers() -> list[dict]
"""

import logging
from typing import Optional

from src.repositories.verification_repo import VerificationRepository
from src.verification.agent import run_verification, run_verification_from_document
from src.verification.rules_loader import list_available_customers

# ── Logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


class VerificationService:
    """Service for document verification, review, and retrieval.

    Wraps the verification agent pipeline with:
    - Database path injection (from FastAPI app.state)
    - Translation between API-level data and domain calls
    - Error handling with meaningful log messages
    - Repository instantiation for read/update operations

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

    # ── Process (Verify) ───────────────────────────────────────────────

    def process_document(
        self,
        file_bytes: bytes,
        file_name: str,
        customer_id: str,
        shipment_ref: Optional[str] = None,
        doc_type_hint: Optional[str] = None,
    ) -> dict:
        """Run the full verification pipeline on an uploaded document.

        Delegates to the verification agent which orchestrates:
        1. Field extraction (reusing Part 1 vision pipeline)
        2. Customer rule loading
        3. Field-by-field comparison
        4. Overall status determination
        5. Draft email generation (with LLM fallback)
        6. Database storage

        Args:
            file_bytes: Raw bytes of the uploaded file.
            file_name: Original filename.
            customer_id: Customer identifier matching YAML config.
            shipment_ref: Optional shipment reference number.
            doc_type_hint: Optional document type hint.

        Returns:
            Dict with verification outcome (see agent.run_verification).
        """
        logger.info(
            "VerificationService.process_document: file=%s, customer=%s",
            file_name, customer_id,
        )

        return run_verification(
            file_bytes=file_bytes,
            file_name=file_name,
            customer_id=customer_id,
            shipment_ref=shipment_ref,
            db_path=self._db_path,
            doc_type_hint=doc_type_hint,
        )

    def process_existing_document(
        self,
        document_id: str,
        customer_id: str,
        shipment_ref: Optional[str] = None,
    ) -> dict:
        """Re-verify a previously extracted document against customer rules.

        Loads the extraction from the database (no re-extraction needed)
        and runs the comparison + draft generation + storage steps.

        Args:
            document_id: UUID of the previously extracted document.
            customer_id: Customer identifier matching YAML config.
            shipment_ref: Optional shipment reference number.

        Returns:
            Dict with verification outcome.
        """
        logger.info(
            "VerificationService.process_existing_document: doc=%s, customer=%s",
            document_id, customer_id,
        )

        return run_verification_from_document(
            document_id=document_id,
            customer_id=customer_id,
            shipment_ref=shipment_ref,
            db_path=self._db_path,
        )

    # ── List ────────────────────────────────────────────────────────────

    def list_verifications(
        self,
        status_filter: Optional[str] = None,
        customer_filter: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve verification results with optional filtering.

        Args:
            status_filter: Optional overall_status to filter by
                (e.g., "amendment_required", "approved", "uncertain").
            customer_filter: Optional customer_id to filter by.

        Returns:
            List of verification result dicts, ordered by newest first.
        """
        logger.info(
            "VerificationService.list_verifications: status=%s, customer=%s",
            status_filter, customer_filter,
        )

        repo = VerificationRepository(db_path=self._db_path)

        if status_filter:
            return repo.get_verifications_by_status(status_filter)
        elif customer_filter:
            return repo.get_verifications_by_customer(customer_filter)
        else:
            return repo.get_all_verifications()

    # ── Get Detail ──────────────────────────────────────────────────────

    def get_verification(self, verification_id: str) -> Optional[dict]:
        """Retrieve a single verification with its field comparisons.

        Returns the verification result enriched with the field-level
        comparison details. This is the full detail view.

        Args:
            verification_id: UUID of the verification to retrieve.

        Returns:
            Dict with verification result + 'fields' key containing
            the list of field comparison dicts. Returns None if not found.
        """
        logger.info(
            "VerificationService.get_verification: id=%s", verification_id,
        )

        repo = VerificationRepository(db_path=self._db_path)

        # Get the verification result
        result = repo.get_verification_by_id(verification_id)
        if result is None:
            return None

        # Enrich with field comparisons
        fields = repo.get_fields_for_verification(verification_id)
        result["fields"] = fields

        return result

    # ── Update Review ───────────────────────────────────────────────────

    def update_review(
        self,
        verification_id: str,
        reviewed_by: str,
        notes: Optional[str] = None,
        overall_status: Optional[str] = None,
        draft_reply: Optional[str] = None,
    ) -> dict:
        """Update the review status of a verification.

        Called when a human reviews the automated verification and either
        confirms or overrides the result. Records who reviewed it and when.

        Args:
            verification_id: UUID of the verification to review.
            reviewed_by: Name/email of the reviewer.
            notes: Optional notes from the reviewer.
            overall_status: Optional status override.
            draft_reply: Optional updated draft email.

        Returns:
            Dict with the updated verification result.

        Raises:
            ValueError: If the verification_id is not found.
        """
        logger.info(
            "VerificationService.update_review: id=%s, by=%s",
            verification_id, reviewed_by,
        )

        repo = VerificationRepository(db_path=self._db_path)

        repo.update_review_status(
            verification_id=verification_id,
            reviewed_by=reviewed_by,
            notes=notes,
            overall_status=overall_status,
            draft_reply=draft_reply,
        )

        # Return the updated verification (with field details)
        return self.get_verification(verification_id)

    # ── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get aggregate verification statistics for the dashboard.

        Returns:
            Dict with total, by_status, reviewed, pending_review counts.
        """
        logger.info("VerificationService.get_stats")

        repo = VerificationRepository(db_path=self._db_path)
        return repo.get_stats()

    # ── Customers ───────────────────────────────────────────────────────

    def list_customers(self) -> list[dict]:
        """List all customers with configured verification rules.

        Scans config/customer_rules/ for YAML files and returns basic
        info for each customer. Used by the UI for the customer selector.

        Returns:
            List of dicts with customer_id, customer_name, rule_count.
        """
        logger.info("VerificationService.list_customers")
        return list_available_customers()
