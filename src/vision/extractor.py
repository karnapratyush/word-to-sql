"""Vision document extractor -- extract structured fields from PDF/image files.

This module implements the vision extraction pipeline. It uses a vision-capable
LLM (e.g., Gemini 2.5 Flash, GPT-4o-mini) to extract structured data (field
names, values, confidence scores) from uploaded logistics documents like
invoices, bills of lading, and packing lists.

Pipeline:
1. Validate the file (type, size)
2. Save to db/uploads/ temporarily for processing
3. Convert PDF pages to images using PyMuPDF (fitz), or use image directly
4. Quick language check -- reject clearly non-English documents
5. Classify document type using the vision LLM (if no hint provided)
6. Extract fields using the vision LLM with a document-type-specific prompt
7. Parse the structured JSON response (with retry on parse failure)
8. Build and return an ExtractionResult with fields and confidence scores

Guardrails applied:
- File type: only .pdf, .png, .jpg, .jpeg
- File size: max 10 MB (from settings.yaml extraction.max_file_size_mb)
- Language: basic ASCII/Latin heuristic on raw text
- Unrecognizable: if overall confidence < 0.3, flag as unrecognizable
"""

import base64
import json
import logging
import os
import re
import uuid
from typing import Optional

from langchain_core.messages import HumanMessage

from src.common.config_loader import load_prompts, load_settings, load_model_config
from src.common.exceptions import ExtractionError, UnsupportedFileError
from src.common.schemas import ExtractionResult, FieldExtraction
from src.models.llm_factory import _create_model_instance
from src.tracing import get_callbacks

# ── Logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────
# Navigate from src/vision/ up two levels to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Temporary upload directory for files being processed
UPLOAD_DIR = os.path.join(BASE_DIR, "db", "uploads")

# Supported file extensions (must match settings.yaml extraction.supported_file_types)
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

# Known document types for classification
KNOWN_DOCUMENT_TYPES = {"invoice", "bill_of_lading", "packing_list", "customs_declaration", "unknown"}

# Minimum overall confidence below which we flag the document as unrecognizable
UNRECOGNIZABLE_THRESHOLD = 0.3


# ── Private Helpers ─────────────────────────────────────────────────────

def _validate_file(file_bytes: bytes, file_name: str) -> str:
    """Validate the uploaded file's extension and size.

    Checks that the file extension is one of the supported types (.pdf, .png,
    .jpg, .jpeg) and that the file size does not exceed the maximum configured
    in settings.yaml (extraction.max_file_size_mb).

    Args:
        file_bytes: Raw bytes of the uploaded file.
        file_name: Original filename (used to detect extension).

    Returns:
        The lowercase file extension (e.g., ".pdf", ".png").

    Raises:
        UnsupportedFileError: If the file extension is not supported.
        UnsupportedFileError: If the file exceeds the maximum size.
    """
    # Extract and normalize the file extension
    _, ext = os.path.splitext(file_name.lower())

    # Load supported extensions from settings, falling back to hardcoded set
    settings = load_settings()
    extraction_settings = settings.get("extraction", {})
    supported = set(extraction_settings.get("supported_file_types", SUPPORTED_EXTENSIONS))

    if ext not in supported:
        raise UnsupportedFileError(
            f"Unsupported file type '{ext}'. "
            f"Supported types: {', '.join(sorted(supported))}"
        )

    # Check file size against configured maximum
    max_size_mb = extraction_settings.get("max_file_size_mb", 10)
    max_size_bytes = max_size_mb * 1024 * 1024
    if len(file_bytes) > max_size_bytes:
        raise UnsupportedFileError(
            f"File size ({len(file_bytes) / (1024 * 1024):.1f} MB) exceeds "
            f"maximum allowed size ({max_size_mb} MB)."
        )

    return ext


def _save_temp_file(file_bytes: bytes, file_name: str) -> str:
    """Save the uploaded file to db/uploads/ with a UUID prefix.

    Creates the db/uploads/ directory if it does not exist. The file is saved
    as {uuid}_{original_filename} to avoid name collisions.

    Args:
        file_bytes: Raw bytes of the file to save.
        file_name: Original filename (preserved in the saved name).

    Returns:
        Absolute path to the saved temporary file.
    """
    # Ensure the upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Generate a unique filename: {uuid}_{original_name}
    unique_name = f"{uuid.uuid4()}_{file_name}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    logger.info("Saved temporary file: %s (%d bytes)", file_path, len(file_bytes))
    return file_path


def _pdf_to_images(file_bytes: bytes) -> list[bytes]:
    """Convert PDF bytes to a list of PNG image bytes, one per page.

    Uses PyMuPDF (fitz) to render each page of the PDF as a PNG image at
    a resolution suitable for OCR (150 DPI). This allows the vision LLM
    to process PDF documents as images.

    Args:
        file_bytes: Raw bytes of the PDF file.

    Returns:
        List of PNG image bytes, one entry per page.

    Raises:
        ExtractionError: If the PDF cannot be opened or rendered.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ExtractionError(
            "PyMuPDF (fitz) is required for PDF processing. "
            "Install it with: pip install PyMuPDF"
        )

    try:
        # Open the PDF from bytes (no need to save to disk first)
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        images = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render at 150 DPI (default is 72 DPI; 150/72 ~ 2.08x zoom)
            zoom = 150 / 72
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix)
            # Convert the pixmap to PNG bytes
            png_bytes = pixmap.tobytes("png")
            images.append(png_bytes)
            logger.debug("Rendered PDF page %d/%d as PNG (%d bytes)",
                         page_num + 1, len(doc), len(png_bytes))

        doc.close()

        if not images:
            raise ExtractionError("PDF has no pages to render.")

        logger.info("Converted PDF to %d page image(s)", len(images))
        return images

    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"Failed to convert PDF to images: {e}")


def _invoke_vision_model(messages: list[HumanMessage], trace_name: str = "vision") -> tuple:
    """Invoke a vision-capable LLM with multimodal (image + text) messages.

    This function handles the fallback chain for multimodal input. Unlike
    get_model_with_fallback() which expects string messages, this function
    works with LangChain HumanMessage objects that contain image_url content
    blocks for the vision LLM.

    Walks the vision model chain from config/models.yaml and tries each
    model until one succeeds. Returns the response content and model name.

    Args:
        messages: List of LangChain HumanMessage objects with multimodal content
            (text + image_url blocks).
        trace_name: Name for LangFuse tracing (default: "vision").

    Returns:
        Tuple of (response_content_string, model_name_string).

    Raises:
        ExtractionError: If all models in the vision fallback chain fail.
    """
    config = load_model_config()
    task_config = config["task_routing"]["vision"]

    chain = task_config.get("chain", [])
    temperature = task_config.get("temperature", 0.1)
    max_tokens = task_config.get("max_tokens", 4000)
    timeout = task_config.get("timeout_seconds", 60)

    # Get LangFuse callbacks for tracing (empty list if not configured)
    cb_config = get_callbacks(trace_name=trace_name)

    errors = []
    for entry in chain:
        provider = entry["provider"]
        model = entry["model"]
        model_name = f"{provider}/{model}"
        try:
            # Create the model instance
            instance = _create_model_instance(provider, model, temperature, max_tokens, timeout)

            # Invoke with multimodal messages directly (not string-based)
            response = instance.invoke(messages, config=cb_config)

            # Extract string content from the LangChain response object
            content = response.content if hasattr(response, "content") else str(response)
            logger.info("Vision model %s responded successfully (%d chars)",
                        model_name, len(content))
            return content, model_name

        except Exception as e:
            error_msg = f"{model_name}: {type(e).__name__}: {e}"
            errors.append(error_msg)
            logger.warning("Vision model %s failed: %s", model_name, e)
            continue

    raise ExtractionError(
        f"All vision models failed. Errors: {'; '.join(errors)}"
    )


def _classify_document(image_bytes: bytes) -> str:
    """Classify document language and type in a single vision LLM call.

    Sends the first page image to the vision LLM with a prompt that asks
    for both language and document type. The LLM responds with:
        "english,invoice" or "non_english,unknown"

    Rejects non-English documents immediately (before extraction) to
    avoid wasting an expensive extraction LLM call.

    Args:
        image_bytes: PNG image bytes of the first page of the document.

    Returns:
        The classified document type string (one of the known types).

    Raises:
        UnsupportedFileError: If the document is not in English.
    """
    prompts = load_prompts()
    classifier_prompt = prompts.get("vision", {}).get("classifier", "What type of document is this?")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(content=[
        {"type": "text", "text": classifier_prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
    ])

    try:
        response_text, model_name = _invoke_vision_model([message], trace_name="vision_classify")
        response_clean = response_text.strip().lower().replace(" ", "")

        # Parse "language,type" format (e.g., "english,invoice")
        parts = [p.strip() for p in response_clean.split(",")]

        # Extract language (first part)
        language = parts[0] if parts else "unknown"

        # Extract document type (second part, or try the whole response)
        doc_type_raw = parts[1] if len(parts) > 1 else response_clean

        # Check language — reject non-English before extraction
        if "non_english" in language or (language != "english" and "english" not in language):
            # Double check: maybe the LLM didn't follow the format.
            # If we find a known doc type, it might still be English.
            has_known_type = any(t in doc_type_raw for t in KNOWN_DOCUMENT_TYPES - {"unknown"})
            if not has_known_type:
                logger.warning("Document rejected: non-English (language=%s, model=%s)",
                              language, model_name)
                raise UnsupportedFileError(
                    "This document appears to be in a non-English language. "
                    "Currently only English documents are supported."
                )

        # Match document type to known types
        for known_type in KNOWN_DOCUMENT_TYPES:
            if known_type in doc_type_raw:
                logger.info("Classified: language=%s, type=%s (model=%s)",
                           language, known_type, model_name)
                return known_type

        logger.warning("Unrecognized classification: '%s', defaulting to 'unknown'",
                        response_text.strip())
        return "unknown"

    except UnsupportedFileError:
        raise  # Re-raise language rejection
    except Exception as e:
        logger.error("Document classification failed: %s", e)
        return "unknown"


def _get_expected_fields(document_type: str) -> list[str]:
    """Get the expected fields list for a given document type from prompts.yaml.

    Reads the document_types section of prompts.yaml to find the expected
    fields for extraction prompts. This tells the LLM which fields to look
    for in the document.

    Args:
        document_type: The type of document (e.g., "invoice", "bill_of_lading").

    Returns:
        List of expected field name strings. Returns an empty list if the
        document type is not found in the configuration.
    """
    prompts = load_prompts()
    doc_types_config = prompts.get("document_types", {})
    type_config = doc_types_config.get(document_type, {})
    return type_config.get("expected_fields", [])


def _extract_fields_from_images(
    image_bytes_list: list[bytes],
    document_type: str,
) -> tuple[dict, str, str]:
    """Send images to the vision LLM to extract structured fields.

    Constructs a multimodal message containing all page images and a
    document-type-specific extraction prompt from config/prompts.yaml.
    The LLM is asked to return JSON with field values and confidence scores.

    For multi-page documents, all pages are sent in a single message so the
    LLM can cross-reference information across pages.

    Args:
        image_bytes_list: List of PNG image bytes (one per page).
        document_type: The classified document type for prompt selection.

    Returns:
        Tuple of (parsed_json_dict, model_name, raw_response_text).
        The parsed_json_dict has keys: "fields", "confidence_scores",
        "document_type", "notes".

    Raises:
        ExtractionError: If extraction fails after all retries.
    """
    prompts = load_prompts()
    settings = load_settings()

    # Get the extractor prompt template and fill in document-type details
    extractor_prompt_template = prompts.get("vision", {}).get(
        "extractor",
        "Extract all fields from this {document_type} document as JSON."
    )
    expected_fields = _get_expected_fields(document_type)
    expected_fields_str = "\n".join(f"- {f}" for f in expected_fields) if expected_fields else "- (extract all visible fields)"

    extractor_prompt = extractor_prompt_template.format(
        document_type=document_type,
        expected_fields=expected_fields_str,
    )

    # Build the multimodal message content: text prompt + all page images
    content_parts = [{"type": "text", "text": extractor_prompt}]
    for i, img_bytes in enumerate(image_bytes_list):
        image_b64 = base64.b64encode(img_bytes).decode("utf-8")
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        })

    message = HumanMessage(content=content_parts)

    # Get retry settings
    max_retries = settings.get("extraction", {}).get("max_retries", 2)
    retry_prompt = prompts.get("vision", {}).get(
        "extractor_retry",
        "Your response was not valid JSON. Please try again with only JSON."
    )

    last_error = None
    last_error_msg = ""
    raw_response = ""
    model_name = ""

    # Attempt extraction with retries on parse/validation failure.
    # Each retry includes the specific error message so the LLM can self-correct.
    for attempt in range(1 + max_retries):
        try:
            if attempt == 0:
                # First attempt: use the full extraction prompt with images
                response_text, model_name = _invoke_vision_model(
                    [message], trace_name="vision_extract"
                )
            else:
                # Retry: build a prompt that includes the specific error
                logger.info("Extraction retry %d/%d — error: %s",
                            attempt, max_retries, last_error_msg)

                # Dynamic retry prompt with the actual error
                error_specific_prompt = (
                    f"Your previous response had an error: {last_error_msg}\n\n"
                    f"{retry_prompt}"
                )

                retry_content = [{"type": "text", "text": error_specific_prompt}]
                for img_bytes in image_bytes_list:
                    image_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    retry_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    })
                retry_message = HumanMessage(content=retry_content)
                response_text, model_name = _invoke_vision_model(
                    [retry_message], trace_name="vision_extract_retry"
                )

            raw_response = response_text

            # Parse and validate the JSON response
            parsed = _parse_extraction_response(response_text)
            logger.info("Successfully parsed extraction response on attempt %d", attempt + 1)
            return parsed, model_name, raw_response

        except (json.JSONDecodeError, ValueError) as e:
            # JSONDecodeError: response isn't valid JSON
            # ValueError: valid JSON but missing keys, wrong types, or empty fields
            last_error = e
            last_error_msg = str(e)
            logger.warning("Parse/validation failed on attempt %d: %s", attempt + 1, e)
            continue
        except ExtractionError:
            raise
        except Exception as e:
            last_error = e
            logger.warning("Extraction attempt %d failed: %s", attempt + 1, e)
            continue

    # All attempts exhausted -- raise with context
    raise ExtractionError(
        f"Failed to extract fields after {1 + max_retries} attempts. "
        f"Last error: {last_error}"
    )


def _parse_extraction_response(response_text: str) -> dict:
    """Parse the JSON response from the extraction LLM.

    The LLM should respond with JSON in the format:
    {
        "fields": {"field_name": "value", ...},
        "confidence_scores": {"field_name": 0.95, ...},
        "document_type": "invoice",
        "notes": "any observations"
    }

    This function handles common LLM quirks like markdown code fences
    around the JSON, extra whitespace, and missing optional keys.

    Args:
        response_text: Raw text response from the LLM.

    Returns:
        Parsed dict with at least "fields" and "confidence_scores" keys.

    Raises:
        json.JSONDecodeError: If the response cannot be parsed as JSON.
    """
    text = response_text.strip()

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Try to find a JSON object in the text if direct parsing fails
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from surrounding text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            parsed = json.loads(match.group())
        else:
            raise

    # Validate required keys — raise so retry can include the specific error
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

    if "fields" not in parsed:
        raise ValueError("Missing required key 'fields' in response JSON")

    if not isinstance(parsed.get("fields"), dict):
        raise ValueError(f"'fields' must be a dict, got {type(parsed.get('fields')).__name__}")

    if "confidence_scores" not in parsed:
        raise ValueError("Missing required key 'confidence_scores' in response JSON")

    if not isinstance(parsed.get("confidence_scores"), dict):
        raise ValueError(f"'confidence_scores' must be a dict, got {type(parsed.get('confidence_scores')).__name__}")

    if not parsed["fields"]:
        raise ValueError("'fields' is empty — no data was extracted from the document")

    return parsed


def _build_extraction_result(
    parsed_response: dict,
    document_type: str,
    model_name: str,
    raw_text: str,
) -> ExtractionResult:
    """Build an ExtractionResult from the parsed LLM response.

    Converts the flat dicts of field values and confidence scores into
    FieldExtraction objects, calculates overall confidence, and assembles
    the final ExtractionResult.

    Args:
        parsed_response: Dict with "fields", "confidence_scores", "notes" keys.
        document_type: The classified document type.
        model_name: Name of the model that performed the extraction.
        raw_text: The raw LLM response text (for debugging).

    Returns:
        ExtractionResult ready for validation and user review.
    """
    fields_dict = parsed_response.get("fields", {})
    confidence_dict = parsed_response.get("confidence_scores", {})
    notes = parsed_response.get("notes", "")

    # Build FieldExtraction objects for each extracted field
    field_extractions = {}
    for field_name, value in fields_dict.items():
        # Get the confidence score for this field; default to 0.5 if not provided
        confidence = confidence_dict.get(field_name, 0.5)

        # LLMs sometimes return unexpected types for confidence:
        #   - list: [0.8, 0.9] → take the average
        #   - dict: {"value": 0.8} → extract the value
        #   - str: "0.85" → convert to float
        #   - None → default to 0.5
        if isinstance(confidence, list):
            # Average of list elements (e.g., per-line-item confidences)
            numeric = [float(x) for x in confidence if isinstance(x, (int, float, str))]
            confidence = sum(numeric) / len(numeric) if numeric else 0.5
        elif isinstance(confidence, dict):
            # Extract a numeric value from dict
            confidence = confidence.get("value", confidence.get("score", 0.5))
        elif confidence is None:
            confidence = 0.5

        # Clamp confidence to [0.0, 1.0]
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (ValueError, TypeError):
            confidence = 0.5

        field_extractions[field_name] = FieldExtraction(
            value=value,
            confidence=confidence,
            needs_review=False,  # Will be set by validator
        )

    # Calculate overall confidence as the average of all field confidences
    if field_extractions:
        overall_confidence = sum(
            f.confidence for f in field_extractions.values()
        ) / len(field_extractions)
    else:
        overall_confidence = 0.0

    # Check if overall confidence is below the unrecognizable threshold
    if overall_confidence < UNRECOGNIZABLE_THRESHOLD and field_extractions:
        notes = f"WARNING: Very low overall confidence ({overall_confidence:.2f}). " \
                f"Document may be unrecognizable. {notes}"

    return ExtractionResult(
        document_type=document_type,
        fields=field_extractions,
        overall_confidence=round(overall_confidence, 4),
        model_used=model_name,
        notes=notes,
        raw_text=raw_text,
    )


# ── Public API ──────────────────────────────────────────────────────────

def extract_from_document(
    file_bytes: bytes,
    file_name: str,
    document_type_hint: Optional[str] = None,
) -> ExtractionResult:
    """Extract structured fields from a PDF or image using a vision LLM.

    This is the main entry point for the extraction pipeline. It validates
    the input file, converts PDFs to images, classifies the document type,
    extracts fields using a vision-capable LLM, and returns an ExtractionResult
    with all extracted fields and confidence scores.

    The returned ExtractionResult is NOT yet validated (needs_review flags
    are not set). Call validate_extraction() after this for quality checks.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        file_name: Original filename (used to detect file type via extension).
        document_type_hint: Optional hint about document type (e.g., "invoice")
            to skip the classification step and use a type-specific prompt.

    Returns:
        ExtractionResult with extracted fields and confidence scores.

    Raises:
        UnsupportedFileError: If the file type or size is invalid.
        ExtractionError: If extraction fails completely (all retries exhausted).
    """
    logger.info("Starting extraction for file: %s (%d bytes)",
                file_name, len(file_bytes))

    # Step 1: Validate file type and size
    ext = _validate_file(file_bytes, file_name)
    logger.info("File validation passed: extension=%s", ext)

    # Step 2: Save to db/uploads/ temporarily
    temp_path = _save_temp_file(file_bytes, file_name)
    logger.info("Temporary file saved: %s", temp_path)

    try:
        # Step 3: Convert PDF pages to images, or use the image directly
        if ext == ".pdf":
            image_bytes_list = _pdf_to_images(file_bytes)
        else:
            # For image files, use the raw bytes directly as a single "page"
            image_bytes_list = [file_bytes]

        # Step 4: Classify document type AND check language.
        # Always run the classifier — it checks both language and type in one call.
        # Non-English documents are rejected here before the expensive extraction.
        logger.info("Classifying document language and type using vision LLM...")
        document_type = _classify_document(image_bytes_list[0])
        logger.info("Document classified as: %s", document_type)

        # Reject unknown document types — we don't know what fields to extract.
        # This catches non-logistics documents (selfies, screenshots, random files).
        if document_type == "unknown":
            raise UnsupportedFileError(
                "Could not identify this as a supported logistics document. "
                "Supported types: invoice, bill of lading, packing list, customs declaration."
            )

        # Step 5: Extract fields using vision LLM
        logger.info("Extracting fields for document type: %s", document_type)
        parsed_response, model_name, raw_response = _extract_fields_from_images(
            image_bytes_list, document_type
        )

        # Step 6: Build the ExtractionResult
        result = _build_extraction_result(
            parsed_response, document_type, model_name, raw_response
        )

        logger.info(
            "Extraction complete: %d fields, overall confidence=%.2f, model=%s",
            len(result.fields), result.overall_confidence, result.model_used,
        )
        return result

    except (UnsupportedFileError, ExtractionError):
        # Re-raise domain exceptions as-is
        raise
    except Exception as e:
        # Wrap unexpected errors so they become ExtractionError
        logger.error("Unexpected extraction error: %s", e, exc_info=True)
        raise ExtractionError(f"Extraction failed: {e}")
    finally:
        # Clean up the temp file only if we want to -- but per the flow,
        # the temp file is deleted after user approval/rejection, not here.
        # We leave it for the agent to clean up.
        pass


def delete_temp_file(file_path: str) -> None:
    """Delete a temporary file from db/uploads/.

    Called after the user approves or rejects the extraction to clean up
    the temporary upload. Silently ignores missing files.

    Args:
        file_path: Absolute path to the file to delete.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Deleted temp file: %s", file_path)
        else:
            logger.debug("Temp file already deleted or not found: %s", file_path)
    except OSError as e:
        logger.warning("Failed to delete temp file %s: %s", file_path, e)


def get_temp_file_path(file_name: str) -> Optional[str]:
    """Find the temporary file path for a given original filename.

    Searches db/uploads/ for files matching the pattern *_{file_name}.
    Returns the first match, or None if no matching file is found.

    Args:
        file_name: The original filename to search for.

    Returns:
        Absolute path to the temporary file, or None if not found.
    """
    if not os.path.exists(UPLOAD_DIR):
        return None

    for entry in os.listdir(UPLOAD_DIR):
        # Files are saved as {uuid}_{original_filename}
        if entry.endswith(f"_{file_name}") or entry == file_name:
            return os.path.join(UPLOAD_DIR, entry)

    return None
