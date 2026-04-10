"""API client for Streamlit to communicate with the FastAPI backend.

All Streamlit pages use this client instead of importing src/ modules
directly. This maintains the View <-> Controller separation in the
MVCR architecture — the Streamlit UI (View) only talks to the FastAPI
backend (Controller) via HTTP.

The client uses httpx for synchronous HTTP requests. A generous default
timeout of 120 seconds accounts for LLM inference latency.

Usage:
    from app.api_client import get_api_client
    client = get_api_client()
    result = client.query_analytics("How many shipments are delayed?")
"""

import os
import httpx
from typing import Optional

# ── Configuration ────────────────────────────────────────────────────
# Base URL for the FastAPI backend, configurable via environment variable
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api")
# Generous timeout because LLM calls can take 30-60+ seconds
DEFAULT_TIMEOUT = 120.0


class APIClient:
    """HTTP client for the   AI Logistics API.

    Wraps httpx to provide typed methods for each API endpoint.
    Each method creates a fresh httpx.Client context manager to
    ensure connections are properly closed after each request.

    Attributes:
        base_url: Base URL for the API (e.g., "http://localhost:8000/api").
        timeout: Request timeout in seconds.
    """

    def __init__(self, base_url: str = API_BASE, timeout: float = DEFAULT_TIMEOUT):
        """Initialize the API client.

        Args:
            base_url: Base URL for the FastAPI backend.
            timeout: Request timeout in seconds (default 120s for LLM calls).
        """
        self.base_url = base_url
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        """Create a new httpx Client configured with base URL and timeout.

        Returns:
            An httpx.Client instance (use as context manager).
        """
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    # ── Health & Schema Endpoints ────────────────────────────────────

    def health_check(self) -> dict:
        """GET /api/health — Check API and database connectivity.

        Returns:
            Dict with status, tables, and row_counts.
        """
        with self._client() as client:
            resp = client.get("/health")
            resp.raise_for_status()
            return resp.json()

    def get_schema(self) -> dict:
        """GET /api/schema — Retrieve the database schema description.

        Returns:
            Dict with schema_description and tables list.
        """
        with self._client() as client:
            resp = client.get("/schema")
            resp.raise_for_status()
            return resp.json()

    # ── Analytics Endpoint ───────────────────────────────────────────

    def query_analytics(
        self,
        user_query: str,
        conversation_history: Optional[list[dict]] = None,
        session_id: str = "",
    ) -> dict:
        """POST /api/analytics/query — Run the full NL-to-SQL pipeline.

        Args:
            user_query: Natural language question from the user.
            conversation_history: Previous turns for follow-up context.
            session_id: Session identifier for tracing.

        Returns:
            Dict with answer, sql_query, result_table, chart_data, etc.
        """
        with self._client() as client:
            resp = client.post("/analytics/query", json={
                "user_query": user_query,
                "conversation_history": conversation_history or [],
                "session_id": session_id,
            })
            resp.raise_for_status()
            return resp.json()

    # ── Document Endpoints ───────────────────────────────────────────

    def upload_document(
        self,
        file_bytes: bytes,
        file_name: str,
        doc_type_hint: Optional[str] = None,
    ) -> dict:
        """POST /api/documents/extract — Upload and extract a document.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            file_name: Original filename.
            doc_type_hint: Optional document type hint (e.g., "invoice").

        Returns:
            Dict with extraction results (fields, confidence, etc.).
        """
        with self._client() as client:
            files = {"file": (file_name, file_bytes)}
            params = {}
            if doc_type_hint:
                params["doc_type_hint"] = doc_type_hint
            resp = client.post("/documents/extract", files=files, params=params)
            resp.raise_for_status()
            return resp.json()

    def list_documents(self) -> list[dict]:
        """GET /api/documents — List all extracted documents.

        Returns:
            List of document dicts ordered by upload time (newest first).
        """
        with self._client() as client:
            resp = client.get("/documents")
            resp.raise_for_status()
            return resp.json()

    def get_document(self, document_id: str) -> dict:
        """GET /api/documents/{document_id} — Retrieve a single document.

        Args:
            document_id: UUID of the document to retrieve.

        Returns:
            Dict with document fields, confidence scores, and review status.
        """
        with self._client() as client:
            resp = client.get(f"/documents/{document_id}")
            resp.raise_for_status()
            return resp.json()

    def approve_document(self, request_data: dict) -> dict:
        """POST /api/documents/approve — Approve and store extraction.

        Sends the full extraction data (possibly with user corrections)
        to the backend for persistence. The backend stores the record
        in the extracted_documents table and deletes the temp upload file.

        Args:
            request_data: Dict with keys matching DocumentApproveRequest:
                temp_file_name, document_type, extracted_fields,
                confidence_scores, overall_confidence, extraction_model,
                review_status, linked_shipment_id, notes.

        Returns:
            Dict representing the stored DocumentRecord.
        """
        with self._client() as client:
            resp = client.post("/documents/approve", json=request_data)
            resp.raise_for_status()
            return resp.json()

    def reject_upload(self, file_name: str) -> dict:
        """DELETE /api/documents/reject/{file_name} — Reject and delete temp file.

        Deletes the temporary upload file without storing anything to
        the database. Called when the user rejects the extraction.

        Args:
            file_name: The original filename of the uploaded document.

        Returns:
            Dict with status="rejected" and confirmation message.
        """
        with self._client() as client:
            resp = client.delete(f"/documents/reject/{file_name}")
            resp.raise_for_status()
            return resp.json()

    # ── Verification Endpoints ──────────────────────────────────────

    def verify_document(
        self,
        file_bytes: bytes,
        file_name: str,
        customer_id: str,
        shipment_ref: Optional[str] = None,
        doc_type_hint: Optional[str] = None,
    ) -> dict:
        """POST /api/verification/process — Upload and verify a document.

        Uploads a document and runs the full verification pipeline:
        extract fields, compare against customer rules, generate draft email.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            file_name: Original filename.
            customer_id: Customer ID matching a YAML rules file.
            shipment_ref: Optional shipment reference number.
            doc_type_hint: Optional document type hint.

        Returns:
            Dict with verification outcome (comparisons, draft, status).
        """
        with self._client() as client:
            files = {"file": (file_name, file_bytes)}
            data = {"customer_id": customer_id}
            if shipment_ref:
                data["shipment_ref"] = shipment_ref
            if doc_type_hint:
                data["doc_type_hint"] = doc_type_hint
            resp = client.post("/verification/process", files=files, data=data)
            resp.raise_for_status()
            return resp.json()

    def compare_confirmed_fields(
        self,
        customer_id: str,
        shipment_ref: str,
        documents: list[dict],
    ) -> dict:
        """POST /api/verification/compare — Compare confirmed fields against rules.

        Phase 2 of the two-phase flow: CG already reviewed extracted fields,
        now compare the confirmed values against customer rules.

        Args:
            customer_id: Customer ID for rule lookup.
            shipment_ref: Shipment reference.
            documents: List of dicts with file_name, document_type,
                extracted_fields, confidence_scores.

        Returns:
            Verification result dict with comparisons, status, draft.
        """
        with self._client() as client:
            resp = client.post("/verification/compare", json={
                "customer_id": customer_id,
                "shipment_ref": shipment_ref,
                "documents": documents,
            })
            resp.raise_for_status()
            return resp.json()

    def list_verifications(
        self,
        status: Optional[str] = None,
        customer: Optional[str] = None,
        shipment: Optional[str] = None,
    ) -> list[dict]:
        """GET /api/verification — List all verification results.

        Args:
            status: Optional overall_status filter.
            customer: Optional customer_id filter.
            shipment: Optional shipment_ref filter (for cross-session tracking).

        Returns:
            List of verification result dicts.
        """
        with self._client() as client:
            params = {}
            if status:
                params["status"] = status
            if customer:
                params["customer"] = customer
            if shipment:
                params["shipment"] = shipment
            resp = client.get("/verification", params=params)
            resp.raise_for_status()
            return resp.json()

    def get_verification(self, verification_id: str) -> dict:
        """GET /api/verification/{id} — Get verification detail.

        Args:
            verification_id: UUID of the verification to retrieve.

        Returns:
            Dict with verification result and field comparison details.
        """
        with self._client() as client:
            resp = client.get(f"/verification/{verification_id}")
            resp.raise_for_status()
            return resp.json()

    def review_verification(
        self,
        verification_id: str,
        reviewed_by: str,
        notes: Optional[str] = None,
        overall_status: Optional[str] = None,
        draft_reply: Optional[str] = None,
    ) -> dict:
        """PUT /api/verification/{id}/review — Submit human review.

        Args:
            verification_id: UUID of the verification to review.
            reviewed_by: Name/email of the reviewer.
            notes: Optional review notes.
            overall_status: Optional status override.
            draft_reply: Optional updated draft email.

        Returns:
            Dict with updated verification result.
        """
        with self._client() as client:
            body = {"reviewed_by": reviewed_by}
            if notes is not None:
                body["notes"] = notes
            if overall_status is not None:
                body["overall_status"] = overall_status
            if draft_reply is not None:
                body["draft_reply"] = draft_reply
            resp = client.put(f"/verification/{verification_id}/review", json=body)
            resp.raise_for_status()
            return resp.json()

    def get_verification_stats(self) -> dict:
        """GET /api/verification/stats — Get verification statistics.

        Returns:
            Dict with total, by_status, reviewed, pending_review counts.
        """
        with self._client() as client:
            resp = client.get("/verification/stats")
            resp.raise_for_status()
            return resp.json()

    def list_verification_customers(self) -> list[dict]:
        """GET /api/verification/customers — List customers with rules.

        Returns:
            List of dicts with customer_id, customer_name, rule_count.
        """
        with self._client() as client:
            resp = client.get("/verification/customers")
            resp.raise_for_status()
            return resp.json()


# ── Singleton ────────────────────────────────────────────────────────
# Shared client instance for Streamlit (cached across page reruns).
# Streamlit reruns the entire page script on each interaction, so
# using a singleton avoids creating new client instances constantly.

_client_instance: Optional[APIClient] = None


def get_api_client() -> APIClient:
    """Return the shared APIClient singleton (created on first call).

    Returns:
        The global APIClient instance.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = APIClient()
    return _client_instance
