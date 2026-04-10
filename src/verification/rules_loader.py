"""Customer rules loader for document verification.

Loads customer-specific verification rules from YAML files located in
config/customer_rules/{customer_id}.yaml. Each YAML file defines the
expected values and match types for fields in shipping documents.

The rules tell the comparator what to check. For example, a customer
might require that consignee_name exactly matches "ACME Corp", while
port_of_discharge must be one of ["Rotterdam", "Amsterdam"].

Rule files are discovered automatically — dropping a new YAML file in
config/customer_rules/ makes that customer available without any code
changes.

File format (see config/customer_rules/sample_customer.yaml for example):
    customer_id: "ACME-GLOBAL"
    customer_name: "ACME Global Trading Corp"
    rules:
      consignee_name:
        expected: "ACME Global Trading Corp, Rotterdam"
        match_type: exact
      hs_code:
        expected_prefix: "8471"
        match_type: prefix

Public functions:
    load_customer_rules(customer_id) -> dict | None
    list_available_customers() -> list[dict]
    get_rules_directory() -> str
"""

import logging
import os
from typing import Optional

import yaml

# ── Logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Path Resolution ─────────────────────────────────────────────────────
# Navigate from src/verification/ up two levels to the project root
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Customer rules YAML files live in config/customer_rules/
_RULES_DIR = os.path.join(_BASE_DIR, "config", "customer_rules")


def get_rules_directory() -> str:
    """Return the absolute path to the customer rules directory.

    Returns:
        Absolute path string to config/customer_rules/.
    """
    return _RULES_DIR


def load_customer_rules(customer_id: str) -> Optional[dict]:
    """Load verification rules for a specific customer from their YAML file.

    Looks for a file named config/customer_rules/{customer_id}.yaml.
    If the file does not exist, returns None (graceful handling for
    unknown customers). If the file exists but is malformed, logs a
    warning and returns None.

    Args:
        customer_id: The customer identifier matching the YAML filename
            (e.g., "sample_customer" loads sample_customer.yaml).

    Returns:
        Dict with keys: customer_id, customer_name, rules.
        The 'rules' value is a dict mapping field names to rule dicts.
        Returns None if the customer's rule file is not found or invalid.
    """
    # Sanitize customer_id to prevent path traversal attacks
    safe_id = os.path.basename(customer_id)
    if safe_id != customer_id:
        logger.warning(
            "Customer ID sanitized from '%s' to '%s' (possible path traversal)",
            customer_id, safe_id,
        )
        customer_id = safe_id

    # Build the path to the customer's YAML rule file
    rules_path = os.path.join(_RULES_DIR, f"{customer_id}.yaml")

    if not os.path.exists(rules_path):
        logger.info(
            "No rules file found for customer '%s' at %s",
            customer_id, rules_path,
        )
        return None

    try:
        with open(rules_path, "r") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            logger.warning(
                "Rules file for customer '%s' is empty or malformed: %s",
                customer_id, rules_path,
            )
            return None

        # Validate the required structure
        if "rules" not in data:
            logger.warning(
                "Rules file for customer '%s' is missing 'rules' key: %s",
                customer_id, rules_path,
            )
            return None

        logger.info(
            "Loaded %d rules for customer '%s' (%s)",
            len(data.get("rules", {})),
            data.get("customer_name", customer_id),
            customer_id,
        )

        return data

    except yaml.YAMLError as e:
        logger.error(
            "Failed to parse YAML rules for customer '%s': %s",
            customer_id, e,
        )
        return None
    except Exception as e:
        logger.error(
            "Unexpected error loading rules for customer '%s': %s",
            customer_id, e, exc_info=True,
        )
        return None


def list_available_customers() -> list[dict]:
    """List all customers that have verification rules configured.

    Scans the config/customer_rules/ directory for YAML files and
    returns basic info (customer_id, customer_name) for each one.
    This is used by the API and Streamlit UI to populate the customer
    selector dropdown.

    Returns:
        List of dicts, each with keys:
            - customer_id: The customer identifier (YAML filename stem)
            - customer_name: Human-readable name from the YAML file
            - file_name: The YAML filename
            - rule_count: Number of rules defined for this customer

        Returns empty list if the rules directory is missing or empty.
    """
    if not os.path.isdir(_RULES_DIR):
        logger.warning("Customer rules directory not found: %s", _RULES_DIR)
        return []

    customers = []

    # Scan for .yaml and .yml files in the rules directory
    for filename in sorted(os.listdir(_RULES_DIR)):
        if not filename.endswith((".yaml", ".yml")):
            continue

        # The customer_id is the filename without the extension
        customer_id = os.path.splitext(filename)[0]

        # Load the full rules to get the customer_name
        rules_data = load_customer_rules(customer_id)
        if rules_data is None:
            continue

        customers.append({
            "customer_id": customer_id,
            "customer_name": rules_data.get("customer_name", customer_id),
            "file_name": filename,
            "rule_count": len(rules_data.get("rules", {})),
        })

    logger.info("Found %d customers with verification rules", len(customers))
    return customers
