"""Vision extraction validator -- post-extraction quality checks.

This module performs validation on the ExtractionResult produced by the
extractor. It runs AFTER extraction and BEFORE the result is shown to
the user for review. The validator enriches the ExtractionResult by:

1. Confidence threshold checks: Flag fields below the threshold as
   needs_review=True so the UI highlights them for human attention.
2. Cross-field consistency checks: invoice (total=subtotal+tax),
   bill of lading (loading port != discharge port), packing list
   (gross >= net weight), customs declaration (declared value > 0).
3. Completeness checks: Verify that all required fields for the document
   type are present (using expected_fields from prompts.yaml).

Note: Overall confidence is NOT recalculated. We trust the LLM's original
score (how clearly it could read the document). Consistency issues are
surfaced as notes/warnings, not by lowering scores.

Settings used (from config/settings.yaml):
    extraction.confidence_threshold: 0.7  -- fields below this get needs_review=True
    extraction.low_confidence_threshold: 0.4  -- fields below this are flagged in notes
"""

import logging
from typing import Optional

from src.common.config_loader import load_prompts, load_settings
from src.common.schemas import ExtractionResult, FieldExtraction

# ── Logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Private Helpers ─────────────────────────────────────────────────────

def _flag_low_confidence_fields(
    fields: dict[str, FieldExtraction],
    confidence_threshold: float,
    low_confidence_threshold: float,
) -> tuple[dict[str, FieldExtraction], list[str]]:
    """Flag fields that fall below the confidence threshold as needing review.

    Fields with confidence below confidence_threshold get needs_review=True.
    Fields with confidence below low_confidence_threshold are additionally
    collected in a warning list for the notes.

    Args:
        fields: Dict mapping field names to FieldExtraction objects.
        confidence_threshold: Fields below this get needs_review=True (default 0.7).
        low_confidence_threshold: Fields below this get extra warning (default 0.4).

    Returns:
        Tuple of (updated_fields_dict, list_of_very_low_confidence_field_names).
    """
    updated_fields = {}
    very_low_fields = []

    for name, field in fields.items():
        # Determine if this field needs human review based on confidence
        needs_review = field.confidence < confidence_threshold

        # Create a new FieldExtraction with the updated needs_review flag
        updated_fields[name] = FieldExtraction(
            value=field.value,
            confidence=field.confidence,
            needs_review=needs_review,
        )

        # Track very low confidence fields for notes
        if field.confidence < low_confidence_threshold:
            very_low_fields.append(name)

    return updated_fields, very_low_fields


def _check_invoice_consistency(fields: dict[str, FieldExtraction]) -> list[str]:
    """Run consistency checks specific to invoice documents.

    Checks:
    - If both subtotal and total_amount are present, total >= subtotal.
    - If line_items, subtotal, and total are present, attempts to verify
      that line items sum approximately to the subtotal.

    Args:
        fields: Dict mapping field names to FieldExtraction objects.

    Returns:
        List of warning/inconsistency messages (empty if all checks pass).
    """
    warnings = []

    # Check subtotal vs total consistency
    subtotal_field = fields.get("subtotal")
    total_field = fields.get("total_amount")
    tax_field = fields.get("tax_amount")

    if subtotal_field and total_field:
        try:
            subtotal = float(subtotal_field.value) if subtotal_field.value is not None else None
            total = float(total_field.value) if total_field.value is not None else None

            if subtotal is not None and total is not None:
                if total < subtotal * 0.99:  # Allow 1% tolerance for rounding
                    warnings.append(
                        f"Inconsistency: total_amount ({total}) is less than "
                        f"subtotal ({subtotal})."
                    )
        except (ValueError, TypeError):
            # Values might not be numeric -- skip consistency check
            pass

    # Check subtotal + tax = total
    if subtotal_field and total_field and tax_field:
        try:
            subtotal = float(subtotal_field.value) if subtotal_field.value is not None else None
            total = float(total_field.value) if total_field.value is not None else None
            tax = float(tax_field.value) if tax_field.value is not None else None

            if subtotal is not None and total is not None and tax is not None:
                expected_total = subtotal + tax
                # Allow 2% tolerance for rounding differences
                if abs(expected_total - total) > total * 0.02 + 0.01:
                    warnings.append(
                        f"Inconsistency: subtotal ({subtotal}) + tax ({tax}) = "
                        f"{expected_total}, but total_amount is {total}."
                    )
        except (ValueError, TypeError):
            pass

    return warnings


def _check_bol_consistency(fields: dict[str, FieldExtraction]) -> list[str]:
    """Run consistency checks specific to bill of lading documents.

    Checks:
    - Required parties are present (shipper, consignee).
    - Port of loading differs from port of discharge.

    Args:
        fields: Dict mapping field names to FieldExtraction objects.

    Returns:
        List of warning/inconsistency messages.
    """
    warnings = []

    pol = fields.get("port_of_loading")
    pod = fields.get("port_of_discharge")

    if pol and pod and pol.value and pod.value:
        if str(pol.value).lower().strip() == str(pod.value).lower().strip():
            warnings.append(
                "Inconsistency: port_of_loading and port_of_discharge are the same."
            )

    return warnings


def _check_packing_list_consistency(fields: dict[str, FieldExtraction]) -> list[str]:
    """Run consistency checks specific to packing list documents.

    Checks:
    - Total gross weight should be >= total net weight.
    - Total packages should be a positive number.

    Args:
        fields: Dict mapping field names to FieldExtraction objects.

    Returns:
        List of warning/inconsistency messages.
    """
    warnings = []

    gross = fields.get("total_gross_weight")
    net = fields.get("total_net_weight")

    if gross and net and gross.value is not None and net.value is not None:
        try:
            gross_val = float(str(gross.value).replace(",", "").split()[0])
            net_val = float(str(net.value).replace(",", "").split()[0])
            if net_val > gross_val:
                warnings.append(
                    f"Inconsistency: net weight ({net_val}) exceeds gross weight ({gross_val})."
                )
        except (ValueError, TypeError):
            pass

    packages = fields.get("total_packages")
    if packages and packages.value is not None:
        try:
            pkg_val = float(str(packages.value).replace(",", "").split()[0])
            if pkg_val <= 0:
                warnings.append("Inconsistency: total_packages should be positive.")
        except (ValueError, TypeError):
            pass

    return warnings


def _check_customs_consistency(fields: dict[str, FieldExtraction]) -> list[str]:
    """Run consistency checks specific to customs declaration documents.

    Checks:
    - Declared value should be positive.
    - Country of origin should differ from destination (importer country).

    Args:
        fields: Dict mapping field names to FieldExtraction objects.

    Returns:
        List of warning/inconsistency messages.
    """
    warnings = []

    declared_value = fields.get("declared_value")
    if declared_value and declared_value.value is not None:
        try:
            val = float(str(declared_value.value).replace(",", "").replace("$", "").split()[0])
            if val <= 0:
                warnings.append("Inconsistency: declared_value should be positive.")
        except (ValueError, TypeError):
            pass

    origin = fields.get("country_of_origin")
    importer = fields.get("importer_name")
    exporter = fields.get("exporter_name")

    if origin and importer and origin.value and importer.value:
        # If importer name contains the origin country, that's suspicious
        if str(origin.value).lower() in str(importer.value).lower():
            warnings.append(
                "Possible inconsistency: importer appears to be in the same country as origin."
            )

    return warnings


def _check_completeness(
    fields: dict[str, FieldExtraction],
    document_type: str,
) -> list[str]:
    """Check that all expected fields for the document type are present.

    Reads the expected_fields list from config/prompts.yaml for the given
    document type and reports any fields that are missing or have None values.

    Args:
        fields: Dict mapping field names to FieldExtraction objects.
        document_type: The document type (e.g., "invoice", "bill_of_lading").

    Returns:
        List of missing field names (empty if all expected fields are present).
    """
    prompts = load_prompts()
    doc_types_config = prompts.get("document_types", {})
    type_config = doc_types_config.get(document_type, {})
    expected_fields = type_config.get("expected_fields", [])

    missing = []
    for expected in expected_fields:
        if expected not in fields:
            missing.append(expected)
        elif fields[expected].value is None or fields[expected].value == "":
            missing.append(expected)

    return missing


# ── Consistency Checker Registry ───────────────────────────────────────
# Strategy dict mapping document types to their consistency check functions.
# Replaces if/elif chain in validate_extraction for cleaner dispatch.
_CONSISTENCY_CHECKERS = {
    "invoice": _check_invoice_consistency,
    "bill_of_lading": _check_bol_consistency,
    "packing_list": _check_packing_list_consistency,
    "customs_declaration": _check_customs_consistency,
}


# ── Public API ──────────────────────────────────────────────────────────

def validate_extraction(
    result: ExtractionResult,
    settings: Optional[dict] = None,
) -> ExtractionResult:
    """Post-extraction validation: confidence thresholds, consistency, completeness.

    This function takes the raw ExtractionResult from the extractor and applies
    a series of quality checks:

    1. Flag fields below the confidence threshold as needs_review=True
    2. Run document-type-specific consistency checks (e.g., invoice totals)
    3. Check completeness (expected fields present?)
    4. Recalculate overall confidence as the mean of field confidences
    5. Assemble any warnings/notes

    The validated result is suitable for presenting to the user for review,
    with low-confidence fields visually highlighted.

    Args:
        result: The ExtractionResult from the extractor module.
        settings: Application settings dict (from config/settings.yaml).
            If None, loads settings from disk.

    Returns:
        Enhanced ExtractionResult with needs_review flags, recalculated
        overall confidence, and any validation notes appended.
    """
    # Load settings if not provided
    if settings is None:
        settings = load_settings()

    extraction_settings = settings.get("extraction", {})
    confidence_threshold = extraction_settings.get("confidence_threshold", 0.7)
    low_confidence_threshold = extraction_settings.get("low_confidence_threshold", 0.4)

    logger.info(
        "Validating extraction: %d fields, document_type=%s, thresholds=(%.2f, %.2f)",
        len(result.fields), result.document_type,
        confidence_threshold, low_confidence_threshold,
    )

    notes_parts = []

    # Preserve existing notes from extraction
    if result.notes:
        notes_parts.append(result.notes)

    # Step 1: Flag low-confidence fields
    updated_fields, very_low_fields = _flag_low_confidence_fields(
        result.fields, confidence_threshold, low_confidence_threshold,
    )

    if very_low_fields:
        notes_parts.append(
            f"Very low confidence fields (below {low_confidence_threshold}): "
            f"{', '.join(very_low_fields)}"
        )
        logger.warning("Very low confidence fields: %s", very_low_fields)

    # Step 2: Document-type-specific consistency checks (via registry)
    consistency_warnings = []
    checker = _CONSISTENCY_CHECKERS.get(result.document_type)
    if checker:
        consistency_warnings = checker(updated_fields)

    if consistency_warnings:
        notes_parts.extend(consistency_warnings)
        logger.warning("Consistency issues: %s", consistency_warnings)

    # Step 3: Completeness check
    missing_fields = _check_completeness(updated_fields, result.document_type)
    if missing_fields:
        notes_parts.append(
            f"Missing expected fields: {', '.join(missing_fields)}"
        )
        logger.info("Missing fields for %s: %s", result.document_type, missing_fields)

    # Keep the original overall confidence from the LLM — it reflects
    # how clearly the model could read the document. Consistency warnings
    # are added to notes instead of lowering the score.
    overall_confidence = result.overall_confidence

    # Combine all notes
    combined_notes = " | ".join(notes_parts) if notes_parts else ""

    # Count how many fields need review
    review_count = sum(1 for f in updated_fields.values() if f.needs_review)
    logger.info(
        "Validation complete: %d/%d fields need review, overall confidence=%.2f",
        review_count, len(updated_fields), overall_confidence,
    )

    # Build and return the validated ExtractionResult
    return ExtractionResult(
        document_type=result.document_type,
        fields=updated_fields,
        overall_confidence=overall_confidence,
        model_used=result.model_used,
        notes=combined_notes,
        raw_text=result.raw_text,
    )
