"""Simple folder watcher trigger for automated document verification.

Polls the db/inbox/ directory every 2 seconds for new document files
(PDF, PNG, JPG). When a file is detected, it is processed through the
verification pipeline and then moved to db/inbox/processed/.

This provides a simple CLI-based entry point for automated verification
without requiring the API server. Useful for batch processing or
integration with file-drop workflows.

Usage:
    # Run from the project root:
    python -m src.verification.trigger --customer sample_customer

    # With custom polling interval and inbox path:
    python -m src.verification.trigger \\
        --customer toyota_japan \\
        --inbox db/inbox/ \\
        --interval 5

The watcher runs until interrupted with Ctrl+C. Processed files are
moved (not deleted) so they can be reviewed or reprocessed later.

Directory structure:
    db/inbox/                  <- Drop files here
    db/inbox/processed/        <- Files move here after verification
"""

import argparse
import logging
import os
import shutil
import sys
import time
from typing import Optional

# ── Path Setup ──────────────────────────────────────────────────────────
# Ensure the project root is in sys.path so imports work when running
# this module directly (python -m src.verification.trigger)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

# ── Logger ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────
# Default inbox directory relative to the project root
DEFAULT_INBOX = os.path.join(_BASE_DIR, "db", "inbox")
# Supported file extensions for verification
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
# Default polling interval in seconds
DEFAULT_INTERVAL = 2


def watch_inbox(
    customer_id: str,
    inbox_dir: str = DEFAULT_INBOX,
    interval: float = DEFAULT_INTERVAL,
    shipment_ref: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Poll the inbox directory for new documents and verify them.

    This is the main loop of the folder watcher. It:
    1. Scans inbox_dir for files with supported extensions
    2. Processes each file through the verification pipeline
    3. Moves processed files to inbox_dir/processed/
    4. Sleeps for `interval` seconds before scanning again
    5. Repeats until KeyboardInterrupt (Ctrl+C)

    Files that fail verification are also moved to processed/ but with
    a log warning. This prevents the watcher from getting stuck retrying
    bad files.

    Args:
        customer_id: Customer identifier for rule loading (e.g., "sample_customer").
        inbox_dir: Path to the inbox directory to watch.
        interval: Polling interval in seconds (default 2).
        shipment_ref: Optional shipment reference to attach to all verifications.
        db_path: Optional database path override.
    """
    # Ensure the inbox directory exists
    os.makedirs(inbox_dir, exist_ok=True)

    # Create the processed subdirectory for completed files
    processed_dir = os.path.join(inbox_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    logger.info(
        "Starting inbox watcher: dir=%s, customer=%s, interval=%ds",
        inbox_dir, customer_id, interval,
    )
    logger.info("Drop PDF/PNG/JPG files into %s to verify them", inbox_dir)
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            # Scan for files with supported extensions
            files_found = _scan_inbox(inbox_dir)

            if files_found:
                logger.info("Found %d file(s) in inbox", len(files_found))

                for file_path in files_found:
                    file_name = os.path.basename(file_path)
                    logger.info("Processing: %s", file_name)

                    try:
                        # Read the file bytes
                        with open(file_path, "rb") as f:
                            file_bytes = f.read()

                        # Run the verification pipeline
                        from src.verification.agent import run_verification

                        result = run_verification(
                            file_bytes=file_bytes,
                            file_name=file_name,
                            customer_id=customer_id,
                            shipment_ref=shipment_ref,
                            db_path=db_path,
                        )

                        # Log the outcome
                        status = result.get("overall_status", "unknown")
                        error = result.get("error")

                        if error:
                            logger.warning(
                                "Verification completed with error for %s: %s (status=%s)",
                                file_name, error, status,
                            )
                        else:
                            logger.info(
                                "Verification complete for %s: status=%s, id=%s",
                                file_name, status,
                                result.get("verification_id", "?"),
                            )

                    except Exception as e:
                        logger.error(
                            "Failed to process %s: %s",
                            file_name, e, exc_info=True,
                        )

                    # Move the file to processed/ regardless of outcome
                    # This prevents infinite re-processing of bad files
                    dest_path = os.path.join(processed_dir, file_name)

                    # Handle filename conflicts by appending a timestamp
                    if os.path.exists(dest_path):
                        name, ext = os.path.splitext(file_name)
                        timestamp = int(time.time())
                        dest_path = os.path.join(
                            processed_dir, f"{name}_{timestamp}{ext}"
                        )

                    try:
                        shutil.move(file_path, dest_path)
                        logger.info("Moved to processed: %s", dest_path)
                    except Exception as e:
                        logger.error(
                            "Failed to move %s to processed: %s",
                            file_path, e,
                        )

            # Sleep before next scan
            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Inbox watcher stopped by user (Ctrl+C)")


def _scan_inbox(inbox_dir: str) -> list[str]:
    """Scan the inbox directory for files with supported extensions.

    Only returns files (not directories or the 'processed' subdirectory).
    Files are sorted by name for deterministic processing order.

    Args:
        inbox_dir: Path to the inbox directory.

    Returns:
        Sorted list of absolute file paths with supported extensions.
    """
    if not os.path.isdir(inbox_dir):
        return []

    files = []
    for entry in sorted(os.listdir(inbox_dir)):
        # Skip the processed subdirectory and hidden files
        if entry.startswith(".") or entry == "processed":
            continue

        file_path = os.path.join(inbox_dir, entry)

        # Only process regular files (not directories)
        if not os.path.isfile(file_path):
            continue

        # Check if the file has a supported extension
        _, ext = os.path.splitext(entry)
        if ext.lower() in SUPPORTED_EXTENSIONS:
            files.append(file_path)

    return files


# ── CLI Entry Point ─────────────────────────────────────────────────────

def main():
    """CLI entry point for the inbox folder watcher.

    Parses command-line arguments and starts the polling loop.
    """
    parser = argparse.ArgumentParser(
        description="Watch db/inbox/ for new documents and verify them against customer rules.",
    )
    parser.add_argument(
        "--customer",
        required=True,
        help="Customer ID for rule loading (e.g., 'sample_customer', 'toyota_japan')",
    )
    parser.add_argument(
        "--inbox",
        default=DEFAULT_INBOX,
        help=f"Path to the inbox directory (default: {DEFAULT_INBOX})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"Polling interval in seconds (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--shipment-ref",
        default=None,
        help="Optional shipment reference to attach to all verifications",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional database path override (default uses config)",
    )

    args = parser.parse_args()

    watch_inbox(
        customer_id=args.customer,
        inbox_dir=args.inbox,
        interval=args.interval,
        shipment_ref=args.shipment_ref,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
