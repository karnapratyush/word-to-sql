"""Tests for the vision pipeline modules.

Covers: extractor, validator, storage, and the full agent.
"""

import json
import os
import uuid

import pytest

from src.vision.extractor import extract_from_document
from src.vision.validator import validate_extraction
from src.vision.storage import store_extraction, update_review_status, get_extracted_documents
from src.vision.agent import process_document
from src.common.schemas import (
    DocumentRecord,
    ExtractionRequest,
    ExtractionResult,
    FieldExtraction,
    ReviewStatus,
)
from src.common.exceptions import UnsupportedFileError


# ═══════════════════════════════════════════════════════════════
# EXTRACTOR
# ═══════════════════════════════════════════════════════════════

class TestExtractFromDocument:
    def test_returns_extraction_result(self):
        # Use a minimal valid PDF (1x1 white pixel PNG as fallback)
        fake_png = _make_minimal_png()
        result = extract_from_document(fake_png, "test_invoice.png", document_type_hint="invoice")
        assert isinstance(result, ExtractionResult)

    def test_has_fields(self):
        fake_png = _make_minimal_png()
        result = extract_from_document(fake_png, "invoice.png", document_type_hint="invoice")
        assert isinstance(result.fields, dict)

    def test_has_confidence_scores(self):
        fake_png = _make_minimal_png()
        result = extract_from_document(fake_png, "invoice.png", document_type_hint="invoice")
        for field_name, field in result.fields.items():
            assert 0.0 <= field.confidence <= 1.0

    def test_unsupported_file_type(self):
        with pytest.raises((UnsupportedFileError, Exception)):
            extract_from_document(b"not a file", "test.xyz")

    def test_includes_model_used(self):
        fake_png = _make_minimal_png()
        result = extract_from_document(fake_png, "invoice.png", document_type_hint="invoice")
        assert result.model_used != ""


# ═══════════════════════════════════════════════════════════════
# VALIDATOR
# ═══════════════════════════════════════════════════════════════

class TestValidateExtraction:
    def test_flags_low_confidence_fields(self):
        result = ExtractionResult(
            document_type="invoice",
            fields={
                "invoice_number": FieldExtraction(value="INV-001", confidence=0.95),
                "date": FieldExtraction(value="2024-03-15", confidence=0.45),
            },
            overall_confidence=0.7,
        )
        settings = {"extraction": {"confidence_threshold": 0.7, "low_confidence_threshold": 0.4}}
        validated = validate_extraction(result, settings)
        assert validated.fields["date"].needs_review is True
        assert validated.fields["invoice_number"].needs_review is False

    def test_high_confidence_passes(self):
        result = ExtractionResult(
            document_type="invoice",
            fields={
                "invoice_number": FieldExtraction(value="INV-001", confidence=0.95),
                "total": FieldExtraction(value=1500.0, confidence=0.92),
            },
            overall_confidence=0.93,
        )
        settings = {"extraction": {"confidence_threshold": 0.7, "low_confidence_threshold": 0.4}}
        validated = validate_extraction(result, settings)
        assert all(not f.needs_review for f in validated.fields.values())

    def test_recalculates_overall_confidence(self):
        result = ExtractionResult(
            document_type="invoice",
            fields={
                "a": FieldExtraction(value="x", confidence=0.8),
                "b": FieldExtraction(value="y", confidence=0.4),
            },
            overall_confidence=0.99,  # wrong, should be recalculated
        )
        settings = {"extraction": {"confidence_threshold": 0.7, "low_confidence_threshold": 0.4}}
        validated = validate_extraction(result, settings)
        assert validated.overall_confidence < 0.99  # should be average of 0.8 and 0.4 = 0.6

    def test_empty_fields(self):
        result = ExtractionResult(
            document_type="unknown",
            fields={},
            overall_confidence=0.0,
        )
        settings = {"extraction": {"confidence_threshold": 0.7, "low_confidence_threshold": 0.4}}
        validated = validate_extraction(result, settings)
        assert validated.overall_confidence == 0.0


# ═══════════════════════════════════════════════════════════════
# STORAGE
# ═══════════════════════════════════════════════════════════════

class TestStoreExtraction:
    def test_stores_and_returns_record(self, empty_db_path):
        result = ExtractionResult(
            document_type="invoice",
            fields={
                "invoice_number": FieldExtraction(value="INV-TEST-001", confidence=0.95),
            },
            overall_confidence=0.95,
            model_used="test-model",
        )
        record = store_extraction(result, "test_invoice.pdf", db_path=empty_db_path)
        assert isinstance(record, DocumentRecord)
        assert record.document_type == "invoice"
        assert record.review_status == ReviewStatus.PENDING

    def test_generates_unique_document_id(self, empty_db_path):
        result = ExtractionResult(
            document_type="invoice",
            fields={"a": FieldExtraction(value="x", confidence=0.9)},
            overall_confidence=0.9,
            model_used="test",
        )
        r1 = store_extraction(result, "a.pdf", db_path=empty_db_path)
        r2 = store_extraction(result, "b.pdf", db_path=empty_db_path)
        assert r1.document_id != r2.document_id

    def test_stored_record_queryable(self, empty_db_path):
        result = ExtractionResult(
            document_type="bill_of_lading",
            fields={"bl_number": FieldExtraction(value="BL-001", confidence=0.88)},
            overall_confidence=0.88,
            model_used="test",
        )
        store_extraction(result, "bol.pdf", db_path=empty_db_path)
        docs = get_extracted_documents(db_path=empty_db_path)
        assert len(docs) == 1
        assert docs[0]["document_type"] == "bill_of_lading"


class TestUpdateReviewStatus:
    def test_approve(self, empty_db_path):
        result = ExtractionResult(
            document_type="invoice",
            fields={"x": FieldExtraction(value="y", confidence=0.9)},
            overall_confidence=0.9,
            model_used="test",
        )
        record = store_extraction(result, "test.pdf", db_path=empty_db_path)
        update_review_status(record.document_id, "approved", db_path=empty_db_path)
        docs = get_extracted_documents(db_path=empty_db_path)
        assert docs[0]["review_status"] == "approved"

    def test_correct_with_updated_fields(self, empty_db_path):
        result = ExtractionResult(
            document_type="invoice",
            fields={"amount": FieldExtraction(value=100.0, confidence=0.5)},
            overall_confidence=0.5,
            model_used="test",
        )
        record = store_extraction(result, "test.pdf", db_path=empty_db_path)
        update_review_status(
            record.document_id,
            "corrected",
            corrected_fields={"amount": 150.0},
            db_path=empty_db_path,
        )
        docs = get_extracted_documents(db_path=empty_db_path)
        assert docs[0]["review_status"] == "corrected"


class TestGetExtractedDocuments:
    def test_empty_table(self, empty_db_path):
        docs = get_extracted_documents(db_path=empty_db_path)
        assert docs == []

    def test_returns_list_of_dicts(self, empty_db_path):
        result = ExtractionResult(
            document_type="invoice",
            fields={"x": FieldExtraction(value="y", confidence=0.9)},
            overall_confidence=0.9,
            model_used="test",
        )
        store_extraction(result, "test.pdf", db_path=empty_db_path)
        docs = get_extracted_documents(db_path=empty_db_path)
        assert isinstance(docs, list)
        assert isinstance(docs[0], dict)


# ═══════════════════════════════════════════════════════════════
# FULL VISION AGENT
# ═══════════════════════════════════════════════════════════════

class TestProcessDocument:
    def test_returns_extraction_result(self):
        fake_png = _make_minimal_png()
        req = ExtractionRequest(
            file_bytes=fake_png,
            file_name="invoice.png",
            document_type_hint="invoice",
        )
        result = process_document(req)
        assert isinstance(result, ExtractionResult)

    def test_fields_have_review_flags(self):
        fake_png = _make_minimal_png()
        req = ExtractionRequest(
            file_bytes=fake_png,
            file_name="invoice.png",
            document_type_hint="invoice",
        )
        result = process_document(req)
        for field in result.fields.values():
            assert isinstance(field.needs_review, bool)

    def test_unsupported_format_raises(self):
        req = ExtractionRequest(
            file_bytes=b"random bytes",
            file_name="mystery.bmp",
        )
        with pytest.raises((UnsupportedFileError, Exception)):
            process_document(req)


# ── Helper ──────────────────────────────────────────────────────

def _make_minimal_png() -> bytes:
    """Create a minimal valid 1x1 white PNG file in memory."""
    import struct
    import zlib

    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = b"\x00\xff\xff\xff"
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend
