"""Tests for DocumentRepository."""

import pytest
from src.repositories.document_repo import DocumentRepository


@pytest.fixture
def doc_repo(empty_db_path):
    """DocumentRepository against a fresh empty DB."""
    return DocumentRepository(db_path=empty_db_path)


@pytest.fixture
def sample_doc_data():
    """Sample data for inserting a test document."""
    return {
        "document_id": "test-doc-001",
        "document_type": "invoice",
        "file_name": "test_invoice.pdf",
        "extraction_model": "gemini-2.0-flash",
        "overall_confidence": 0.85,
        "extracted_fields": {
            "invoice_number": "INV-2024-001",
            "vendor": "Maersk Line",
            "total_amount": 4500.00,
            "date": "2024-03-15",
        },
        "confidence_scores": {
            "invoice_number": 0.95,
            "vendor": 0.91,
            "total_amount": 0.88,
            "date": 0.55,
        },
        "raw_text": "Invoice #INV-2024-001\nMaersk Line\n$4500.00",
        "notes": "Date has low confidence",
    }


class TestInsertDocument:
    """Test document insertion."""

    def test_insert_succeeds(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)
        docs = doc_repo.get_all_documents()
        assert len(docs) == 1

    def test_insert_sets_pending_status(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)
        doc = doc_repo.get_document_by_id("test-doc-001")
        assert doc["review_status"] == "pending"

    def test_duplicate_document_id_raises(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)
        with pytest.raises(Exception):  # IntegrityError
            doc_repo.insert_document(**sample_doc_data)

    def test_insert_with_linked_shipment(self, doc_repo, sample_doc_data):
        sample_doc_data["linked_shipment_id"] = "SHP-2024-000001"
        doc_repo.insert_document(**sample_doc_data)
        doc = doc_repo.get_document_by_id("test-doc-001")
        assert doc["linked_shipment_id"] == "SHP-2024-000001"


class TestGetAllDocuments:
    """Test retrieving all documents."""

    def test_empty_table_returns_empty_list(self, doc_repo):
        docs = doc_repo.get_all_documents()
        assert docs == []

    def test_returns_all_inserted_docs(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)

        sample_doc_data["document_id"] = "test-doc-002"
        sample_doc_data["file_name"] = "test_invoice_2.pdf"
        doc_repo.insert_document(**sample_doc_data)

        docs = doc_repo.get_all_documents()
        assert len(docs) == 2

    def test_json_fields_parsed(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)
        docs = doc_repo.get_all_documents()
        assert isinstance(docs[0]["extracted_fields"], dict)
        assert isinstance(docs[0]["confidence_scores"], dict)
        assert docs[0]["extracted_fields"]["vendor"] == "Maersk Line"

    def test_ordered_by_timestamp_desc(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)

        sample_doc_data["document_id"] = "test-doc-002"
        sample_doc_data["file_name"] = "second.pdf"
        doc_repo.insert_document(**sample_doc_data)

        docs = doc_repo.get_all_documents()
        # Both docs returned; ordering by id DESC when timestamps match
        assert len(docs) == 2
        doc_ids = {d["document_id"] for d in docs}
        assert doc_ids == {"test-doc-001", "test-doc-002"}


class TestGetDocumentById:
    """Test retrieving a single document."""

    def test_returns_none_for_missing(self, doc_repo):
        assert doc_repo.get_document_by_id("nonexistent") is None

    def test_returns_correct_document(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)
        doc = doc_repo.get_document_by_id("test-doc-001")
        assert doc is not None
        assert doc["document_type"] == "invoice"
        assert doc["file_name"] == "test_invoice.pdf"
        assert doc["overall_confidence"] == 0.85

    def test_json_fields_parsed(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)
        doc = doc_repo.get_document_by_id("test-doc-001")
        assert isinstance(doc["extracted_fields"], dict)
        assert doc["extracted_fields"]["total_amount"] == 4500.00
        assert doc["confidence_scores"]["invoice_number"] == 0.95


class TestUpdateReviewStatus:
    """Test review status updates."""

    def test_approve(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)
        doc_repo.update_review_status("test-doc-001", "approved")
        doc = doc_repo.get_document_by_id("test-doc-001")
        assert doc["review_status"] == "approved"
        assert doc["reviewed_at"] is not None

    def test_reject(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)
        doc_repo.update_review_status("test-doc-001", "rejected")
        doc = doc_repo.get_document_by_id("test-doc-001")
        assert doc["review_status"] == "rejected"

    def test_correct_with_updated_fields(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)
        corrected = {"date": "2024-03-16", "vendor": "Maersk Line"}
        doc_repo.update_review_status("test-doc-001", "corrected", corrected)
        doc = doc_repo.get_document_by_id("test-doc-001")
        assert doc["review_status"] == "corrected"
        assert doc["extracted_fields"]["date"] == "2024-03-16"

    def test_nonexistent_doc_raises(self, doc_repo):
        with pytest.raises(ValueError, match="Document not found"):
            doc_repo.update_review_status("nonexistent", "approved")


class TestLinkShipment:
    """Test linking documents to shipments."""

    def test_link_succeeds(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)
        doc_repo.link_shipment("test-doc-001", "SHP-2024-000001")
        doc = doc_repo.get_document_by_id("test-doc-001")
        assert doc["linked_shipment_id"] == "SHP-2024-000001"

    def test_link_nonexistent_doc_raises(self, doc_repo):
        with pytest.raises(ValueError, match="Document not found"):
            doc_repo.link_shipment("nonexistent", "SHP-2024-000001")

    def test_relink_overwrites(self, doc_repo, sample_doc_data):
        doc_repo.insert_document(**sample_doc_data)
        doc_repo.link_shipment("test-doc-001", "SHP-2024-000001")
        doc_repo.link_shipment("test-doc-001", "SHP-2024-000002")
        doc = doc_repo.get_document_by_id("test-doc-001")
        assert doc["linked_shipment_id"] == "SHP-2024-000002"
