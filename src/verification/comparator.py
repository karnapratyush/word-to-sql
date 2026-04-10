"""Field-by-field document verification comparator.

THE CORE MODULE for Part 2 verification. This module compares extracted
document fields against customer-specific rules and produces a list of
FieldComparison results indicating whether each field matches, mismatches,
is uncertain, or has no applicable rule.

This module is intentionally SEPARATE from the extraction logic (assignment
requirement). It receives already-extracted fields and compares them against
rules — it never calls the vision pipeline or touches the LLM.

Match types supported:
    exact       — extracted value must exactly equal the expected value
                  (case-insensitive string comparison)
    prefix      — extracted value must start with the expected prefix
    one_of      — extracted value must be one of the allowed values
    tolerance   — numeric extracted value must be within N% of reference
    contains_any — extracted value must contain at least one of the keywords

Confidence handling:
    If a field's extraction confidence is below the configured threshold,
    the comparison is automatically marked as "uncertain" regardless of
    whether the values match. This prevents auto-approving fields where
    the extraction itself is unreliable.

Statuses:
    match       — field value matches the customer rule
    mismatch    — field value violates the customer rule
    uncertain   — extraction confidence too low to trust the comparison
    no_rule     — no customer rule exists for this field (informational)

Overall status determination:
    any mismatch         -> "amendment_required"
    no mismatch but uncertain -> "uncertain"
    all match (or no_rule)   -> "approved"

This module is independently testable — it requires no database, no LLM,
no network. Feed it dicts and get back dataclass instances.

Public functions:
    compare_fields(extracted_fields, confidence_scores, customer_rules,
                   confidence_threshold) -> list[FieldComparison]
    determine_overall_status(comparisons) -> str
"""

import logging
from dataclasses import dataclass
from typing import Optional

# ── Logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Data Classes ────────────────────────────────────────────────────────

@dataclass
class FieldComparison:
    """Result of comparing a single extracted field against a customer rule.

    Attributes:
        field_name: The name of the field being compared (e.g., "consignee_name").
        extracted_value: The value extracted from the document by the vision LLM.
        expected_value: The expected value from the customer rule (for display).
        status: Comparison result — one of: match, mismatch, uncertain, no_rule.
        confidence: The extraction confidence score for this field (0.0-1.0).
        rule_type: The match type used (exact, prefix, one_of, tolerance,
            contains_any). None if no rule exists for this field.
        rule_violated: Human-readable description of which rule was violated.
            None if the field matches or has no rule.
    """
    field_name: str
    extracted_value: str | None
    expected_value: str | None
    status: str  # match | mismatch | uncertain | no_rule
    confidence: float
    rule_type: str | None
    rule_violated: str | None


# ── Match Functions ─────────────────────────────────────────────────────
# Each match function takes the extracted value and the rule dict, and
# returns a tuple of (is_match: bool, expected_display: str, violation_msg: str | None).

def _match_exact(extracted: str, rule: dict) -> tuple[bool, str, Optional[str]]:
    """Check if extracted value exactly matches the expected value (case-insensitive).

    Args:
        extracted: The extracted value as a string.
        rule: Rule dict with key "expected" containing the expected string.

    Returns:
        Tuple of (is_match, expected_display_value, violation_message_or_None).
    """
    expected = str(rule.get("expected", ""))
    is_match = extracted.strip().lower() == expected.strip().lower()
    violation = None if is_match else f"Expected exact match '{expected}', got '{extracted}'"
    return is_match, expected, violation


def _match_prefix(extracted: str, rule: dict) -> tuple[bool, str, Optional[str]]:
    """Check if extracted value starts with the expected prefix.

    Args:
        extracted: The extracted value as a string.
        rule: Rule dict with key "expected_prefix" containing the prefix string.

    Returns:
        Tuple of (is_match, expected_display_value, violation_message_or_None).
    """
    prefix = str(rule.get("expected_prefix", ""))
    is_match = extracted.strip().lower().startswith(prefix.strip().lower())
    violation = None if is_match else f"Expected prefix '{prefix}', got '{extracted}'"
    return is_match, f"starts with '{prefix}'", violation


def _match_one_of(extracted: str, rule: dict) -> tuple[bool, str, Optional[str]]:
    """Check if extracted value is one of the allowed values (case-insensitive).

    Args:
        extracted: The extracted value as a string.
        rule: Rule dict with key "expected" containing a list of allowed strings.

    Returns:
        Tuple of (is_match, expected_display_value, violation_message_or_None).
    """
    allowed = rule.get("expected", [])
    if not isinstance(allowed, list):
        allowed = [allowed]

    # Case-insensitive comparison against each allowed value
    extracted_lower = extracted.strip().lower()
    is_match = any(extracted_lower == str(a).strip().lower() for a in allowed)

    expected_display = f"one of {allowed}"
    violation = None if is_match else f"Expected one of {allowed}, got '{extracted}'"
    return is_match, expected_display, violation


def _match_tolerance(extracted: str, rule: dict) -> tuple[bool, str, Optional[str]]:
    """Check if a numeric extracted value is within tolerance of a reference.

    The tolerance is specified as a percentage. For example, if the reference
    is 15000 and max_deviation_percent is 5, then 14250 to 15750 are acceptable.

    Args:
        extracted: The extracted value as a string (will be parsed to float).
        rule: Rule dict with keys:
            - reference_value (float): The expected numeric value.
            - max_deviation_percent (float): Maximum allowed deviation as %.

    Returns:
        Tuple of (is_match, expected_display_value, violation_message_or_None).
    """
    reference = float(rule.get("reference_value", 0))
    max_deviation = float(rule.get("max_deviation_percent", 0))

    try:
        # Strip units and commas from the extracted value before parsing
        cleaned = extracted.replace(",", "").strip()
        # Remove common weight units (kg, lbs, etc.)
        for unit in ["kg", "lbs", "lb", "mt", "tons", "cbm", "m3"]:
            cleaned = cleaned.lower().replace(unit, "").strip()
        extracted_num = float(cleaned)
    except (ValueError, TypeError):
        # Cannot parse the extracted value as a number
        return False, f"{reference} +/- {max_deviation}%", \
            f"Cannot parse '{extracted}' as number for tolerance check"

    # Calculate the allowed range
    tolerance = reference * (max_deviation / 100.0)
    lower_bound = reference - tolerance
    upper_bound = reference + tolerance

    is_match = lower_bound <= extracted_num <= upper_bound

    expected_display = f"{reference} +/- {max_deviation}%"
    violation = None if is_match else \
        f"Value {extracted_num} outside tolerance range [{lower_bound:.1f}, {upper_bound:.1f}]"

    return is_match, expected_display, violation


def _match_contains_any(extracted: str, rule: dict) -> tuple[bool, str, Optional[str]]:
    """Check if extracted value contains at least one of the expected keywords.

    Args:
        extracted: The extracted value as a string.
        rule: Rule dict with key "expected_contains" containing a list of keywords.

    Returns:
        Tuple of (is_match, expected_display_value, violation_message_or_None).
    """
    keywords = rule.get("expected_contains", [])
    if not isinstance(keywords, list):
        keywords = [keywords]

    # Case-insensitive keyword search
    extracted_lower = extracted.strip().lower()
    is_match = any(str(kw).lower() in extracted_lower for kw in keywords)

    expected_display = f"contains any of {keywords}"
    violation = None if is_match else \
        f"Expected to contain any of {keywords}, got '{extracted}'"

    return is_match, expected_display, violation


# Registry mapping match_type strings to their handler functions.
# This replaces an if/elif chain for cleaner dispatch and easier extension.
_MATCH_REGISTRY = {
    "exact": _match_exact,
    "prefix": _match_prefix,
    "one_of": _match_one_of,
    "tolerance": _match_tolerance,
    "contains_any": _match_contains_any,
}


# ── Public API ──────────────────────────────────────────────────────────

def compare_fields(
    extracted_fields: dict,
    confidence_scores: dict,
    customer_rules: dict,
    confidence_threshold: float = 0.7,
) -> list[FieldComparison]:
    """Compare extracted document fields against customer verification rules.

    This is the main comparison function. For each extracted field, it:
    1. Checks if there is a customer rule for that field
    2. If the extraction confidence is below threshold, marks as "uncertain"
    3. Otherwise, runs the appropriate match function (exact, prefix, etc.)
    4. Returns a FieldComparison with the result

    Fields with no customer rule get status="no_rule" (informational only).
    Fields with low confidence get status="uncertain" regardless of match.

    Args:
        extracted_fields: Dict mapping field names to extracted values.
            These come from the vision pipeline's ExtractionResult.
        confidence_scores: Dict mapping field names to confidence floats
            (0.0 to 1.0). These come from the vision pipeline.
        customer_rules: The 'rules' dict from the customer's YAML file.
            Maps field names to rule dicts with 'match_type' and criteria.
            Pass an empty dict if no rules are configured.
        confidence_threshold: Minimum confidence to trust an extraction.
            Below this threshold, the comparison is "uncertain".
            Default 0.7; loaded from config/settings.yaml in production.

    Returns:
        List of FieldComparison objects, one per extracted field.
        The list includes all extracted fields, even those without rules.
    """
    comparisons = []

    logger.info(
        "Comparing %d extracted fields against %d customer rules (threshold=%.2f)",
        len(extracted_fields), len(customer_rules), confidence_threshold,
    )

    for field_name, extracted_value in extracted_fields.items():
        # Get the extraction confidence for this field (default to 0.0 if missing)
        confidence = float(confidence_scores.get(field_name, 0.0))

        # Convert extracted value to string for comparison (handle None, lists, etc.)
        extracted_str = str(extracted_value) if extracted_value is not None else ""

        # Check if there is a customer rule for this field
        rule = customer_rules.get(field_name)

        if rule is None:
            # No rule for this field — informational only
            comparisons.append(FieldComparison(
                field_name=field_name,
                extracted_value=extracted_str if extracted_str else None,
                expected_value=None,
                status="no_rule",
                confidence=confidence,
                rule_type=None,
                rule_violated=None,
            ))
            continue

        # Check confidence threshold BEFORE comparing values.
        # If the extraction itself is unreliable, we cannot trust the comparison.
        if confidence < confidence_threshold:
            # Determine the expected display value for the UI
            match_type = rule.get("match_type", "exact")
            expected_display = _get_expected_display(rule, match_type)

            comparisons.append(FieldComparison(
                field_name=field_name,
                extracted_value=extracted_str if extracted_str else None,
                expected_value=expected_display,
                status="uncertain",
                confidence=confidence,
                rule_type=match_type,
                rule_violated=f"Confidence {confidence:.2f} below threshold {confidence_threshold}",
            ))
            logger.debug(
                "Field '%s': uncertain (confidence %.2f < %.2f)",
                field_name, confidence, confidence_threshold,
            )
            continue

        # Run the appropriate match function based on rule match_type
        match_type = rule.get("match_type", "exact")
        match_func = _MATCH_REGISTRY.get(match_type)

        if match_func is None:
            # Unknown match type in the rule — treat as no_rule with a warning
            logger.warning(
                "Unknown match_type '%s' for field '%s', treating as no_rule",
                match_type, field_name,
            )
            comparisons.append(FieldComparison(
                field_name=field_name,
                extracted_value=extracted_str if extracted_str else None,
                expected_value=None,
                status="no_rule",
                confidence=confidence,
                rule_type=match_type,
                rule_violated=f"Unknown match_type '{match_type}'",
            ))
            continue

        # Execute the match function
        try:
            is_match, expected_display, violation = match_func(extracted_str, rule)
        except Exception as e:
            # Match function raised an unexpected error — mark as uncertain
            logger.error(
                "Match function '%s' failed for field '%s': %s",
                match_type, field_name, e, exc_info=True,
            )
            comparisons.append(FieldComparison(
                field_name=field_name,
                extracted_value=extracted_str if extracted_str else None,
                expected_value=None,
                status="uncertain",
                confidence=confidence,
                rule_type=match_type,
                rule_violated=f"Comparison error: {e}",
            ))
            continue

        # Determine the status based on the match result
        status = "match" if is_match else "mismatch"

        comparisons.append(FieldComparison(
            field_name=field_name,
            extracted_value=extracted_str if extracted_str else None,
            expected_value=expected_display,
            status=status,
            confidence=confidence,
            rule_type=match_type,
            rule_violated=violation,
        ))

        logger.debug(
            "Field '%s': %s (type=%s, confidence=%.2f)",
            field_name, status, match_type, confidence,
        )

    logger.info(
        "Comparison complete: %d fields — %d match, %d mismatch, %d uncertain, %d no_rule",
        len(comparisons),
        sum(1 for c in comparisons if c.status == "match"),
        sum(1 for c in comparisons if c.status == "mismatch"),
        sum(1 for c in comparisons if c.status == "uncertain"),
        sum(1 for c in comparisons if c.status == "no_rule"),
    )

    return comparisons


def determine_overall_status(comparisons: list[FieldComparison]) -> str:
    """Determine the overall verification status from individual field comparisons.

    The overall status follows a strict priority:
    1. Any mismatch        -> "amendment_required" (document needs correction)
    2. No mismatch but uncertain -> "uncertain" (needs human review)
    3. All match or no_rule   -> "approved" (document is compliant)

    Args:
        comparisons: List of FieldComparison objects from compare_fields().

    Returns:
        One of: "amendment_required", "uncertain", "approved".
    """
    if not comparisons:
        # No fields to compare — cannot verify, treat as uncertain
        return "uncertain"

    has_mismatch = any(c.status == "mismatch" for c in comparisons)
    has_uncertain = any(c.status == "uncertain" for c in comparisons)

    if has_mismatch:
        return "amendment_required"
    elif has_uncertain:
        return "uncertain"
    else:
        return "approved"


# ── Internal Helpers ────────────────────────────────────────────────────

def _get_expected_display(rule: dict, match_type: str) -> str:
    """Build a human-readable expected value string for display in the UI.

    This is used when a field is marked "uncertain" (low confidence) to
    show the user what the rule expects, even though we cannot trust
    the comparison.

    Args:
        rule: The customer rule dict.
        match_type: The match type string.

    Returns:
        A human-readable string describing the expected value.
    """
    if match_type == "exact":
        return str(rule.get("expected", ""))
    elif match_type == "prefix":
        return f"starts with '{rule.get('expected_prefix', '')}'"
    elif match_type == "one_of":
        return f"one of {rule.get('expected', [])}"
    elif match_type == "tolerance":
        ref = rule.get("reference_value", 0)
        dev = rule.get("max_deviation_percent", 0)
        return f"{ref} +/- {dev}%"
    elif match_type == "contains_any":
        return f"contains any of {rule.get('expected_contains', [])}"
    else:
        return str(rule.get("expected", ""))
