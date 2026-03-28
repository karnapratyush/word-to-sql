"""Generate 25 sample logistics documents (PDFs and images) for testing the vision/OCR pipeline.

This script produces four types of documents commonly found in international freight logistics:

  1. Invoices        (10 documents)  -- freight invoices with line-item charges
  2. Bills of Lading  (8 documents)  -- ocean/air transport receipts
  3. Packing Lists    (4 documents)  -- itemised descriptions of cargo contents
  4. Customs Declarations (3 documents) -- import/export customs forms

Quality tiers:
  - clear   : 12pt font, generous spacing, clean layout   (simulates high-res digital PDF)
  - medium  : 10pt font, tighter spacing, denser layout    (simulates a compact but legible PDF)
  - low     : 8pt font, very dense, compressed feel        (simulates a scanned/degraded copy)
  - image   : PDF rendered to PNG at low DPI with noise     (simulates a photo of a printed page)

Libraries required (all listed in requirements.txt):
  - reportlab  : PDF generation
  - Pillow     : image manipulation / noise injection
  - PyMuPDF    : PDF-to-image rasterisation (imported as fitz)

Usage:
    python db/sample_documents.py

All output is written to  db/samples/
"""

import os
import io
import random
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER

# ---------------------------------------------------------------------------
# Image manipulation
# ---------------------------------------------------------------------------
from PIL import Image, ImageFilter
import numpy as np

# ---------------------------------------------------------------------------
# PDF-to-image rasterisation
# ---------------------------------------------------------------------------
import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Reproducibility -- fix the random seed so the script produces the same set
# of documents on every run.
# ---------------------------------------------------------------------------
random.seed(2024)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "db", "samples")

# ============================================================================
# REFERENCE DATA  (mirrors the values used in seed_data.py so our sample
# documents feel consistent with the database)
# ============================================================================

CARRIER_NAMES = [
    "Maersk Line",
    "Mediterranean Shipping Company",
    "COSCO Shipping",
    "CMA CGM Group",
    "Hapag-Lloyd",
    "Evergreen Marine",
    "Ocean Network Express",
    "DHL Express",
    "DHL Global Forwarding",
    "FedEx Express",
    "UPS Supply Chain",
    "Emirates SkyCargo",
    "DB Schenker",
    "Kuehne+Nagel",
]

# Mapping from carrier name to abbreviated code used in B/L numbers, etc.
CARRIER_CODES = {
    "Maersk Line": "MAERSK",
    "Mediterranean Shipping Company": "MSC",
    "COSCO Shipping": "COSCO",
    "CMA CGM Group": "CMA",
    "Hapag-Lloyd": "HAPAG",
    "Evergreen Marine": "EVERGREEN",
    "Ocean Network Express": "ONE",
    "DHL Express": "DHL",
    "DHL Global Forwarding": "DHL_FWD",
    "FedEx Express": "FEDEX",
    "UPS Supply Chain": "UPS",
    "Emirates SkyCargo": "EMIRATES",
    "DB Schenker": "DBS",
    "Kuehne+Nagel": "KN",
}

# Vessel names per carrier (realistic)
VESSEL_NAMES = {
    "Maersk Line": ["MAERSK EDINBURGH", "MAERSK ELBA", "EMMA MAERSK"],
    "Mediterranean Shipping Company": ["MSC GULSUN", "MSC OSCAR", "MSC DIANA"],
    "COSCO Shipping": ["COSCO SHIPPING UNIVERSE", "COSCO FAITH", "COSCO GLORY"],
    "CMA CGM Group": ["CMA CGM JACQUES SAADE", "CMA CGM MARCO POLO", "CMA CGM KERGUELEN"],
    "Hapag-Lloyd": ["HAPAG BERLIN EXPRESS", "HAPAG COLOMBO EXPRESS", "HAPAG TOKYO EXPRESS"],
    "Evergreen Marine": ["EVER GIVEN", "EVER ACE", "EVER GOLDEN"],
    "Ocean Network Express": ["ONE COLUMBA", "ONE CYGNUS", "ONE AQUILA"],
}

# Customer names -- drawn from the seed_data naming pattern:
# "{city} {industry} Co #{id}"
CUSTOMER_NAMES = [
    "Busan Food Beverage Co #7585",
    "New York Textiles Co #3021",
    "Jebel Ali Textiles Co #4412",
    "Shanghai Electronics Co #1052",
    "Mumbai Pharmaceuticals Co #6891",
    "Rotterdam Chemicals Co #2234",
    "Hamburg Automotive Co #5567",
    "Singapore Machinery Co #8899",
    "Tokyo Consumer Goods Co #9312",
    "Santos Furniture Co #1778",
    "London Apparel Co #6234",
    "Chennai Raw Materials Co #4501",
]

CUSTOMER_CODES = {name: f"CUST-{name.split('#')[1].strip()}" if "#" in name else "CUST-00001" for name in CUSTOMER_NAMES}

# Ports -- mirrors PORTS dict in seed_data.py
COUNTRY_PORTS = {
    "China": ["Shanghai Port", "Shenzhen Port", "Ningbo-Zhoushan Port"],
    "India": ["Nhava Sheva (JNPT)", "Mundra Port", "Chennai Port"],
    "USA": ["Port of Los Angeles", "Port of Long Beach", "Port of New York"],
    "Germany": ["Port of Hamburg", "Port of Bremerhaven"],
    "Japan": ["Port of Tokyo", "Port of Yokohama"],
    "South Korea": ["Port of Busan", "Port of Incheon"],
    "UAE": ["Jebel Ali Port", "Port of Fujairah"],
    "Singapore": ["Port of Singapore", "Jurong Port"],
    "Netherlands": ["Port of Rotterdam", "Port of Amsterdam"],
    "Brazil": ["Port of Santos", "Port of Paranagua"],
    "UK": ["Port of Felixstowe", "Port of Southampton"],
    "France": ["Port of Le Havre", "Port of Marseille"],
}

# Commodities
COMMODITIES = [
    "Electronics",
    "Textiles",
    "Automotive Parts",
    "Pharmaceuticals",
    "Frozen Food",
    "Chemicals",
    "Industrial Machinery",
    "Furniture",
    "Apparel",
    "Consumer Goods",
    "Raw Materials",
    "Medical Equipment",
]

INCOTERMS = ["FOB", "CIF", "EXW", "DDP", "CFR", "FCA", "DAP", "CPT"]
CONTAINER_TYPES = ["20ft", "40ft", "40ft HC"]
PAYMENT_TERMS = ["Net 15", "Net 30", "Net 45", "Net 60"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _random_date(start_year: int = 2024, end_year: int = 2025) -> datetime:
    """Return a random datetime between start_year-01-01 and end_year-12-31."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def _random_route():
    """Pick two distinct countries and a port from each.

    Returns a dict with origin_country, origin_port, dest_country, dest_port.
    """
    countries = list(COUNTRY_PORTS.keys())
    origin_country = random.choice(countries)
    dest_country = random.choice([c for c in countries if c != origin_country])
    return {
        "origin_country": origin_country,
        "origin_port": random.choice(COUNTRY_PORTS[origin_country]),
        "dest_country": dest_country,
        "dest_port": random.choice(COUNTRY_PORTS[dest_country]),
    }


def _country_code(country: str) -> str:
    """Return a two-letter abbreviation for a country (used in B/L numbers)."""
    mapping = {
        "China": "CN",
        "India": "IN",
        "USA": "US",
        "Germany": "DE",
        "Japan": "JP",
        "South Korea": "KR",
        "UAE": "AE",
        "Singapore": "SG",
        "Netherlands": "NL",
        "Brazil": "BR",
        "UK": "GB",
        "France": "FR",
    }
    return mapping.get(country, "XX")


def _generate_shipment_id(booking_date: datetime, seq: int) -> str:
    """Produce a shipment ID in the format SHP-YYYY-NNNNNN."""
    return f"SHP-{booking_date.year}-{seq:06d}"


def _generate_invoice_number(invoice_date: datetime, seq: int) -> str:
    """Produce an invoice number in the format INV-YYYY-NNNNNN."""
    return f"INV-{invoice_date.year}-{seq:06d}"


def _generate_bl_number(origin_country: str, dest_country: str, seq: int) -> str:
    """Produce a B/L number like BL-2024-SG-KR-0734."""
    oc = _country_code(origin_country)
    dc = _country_code(dest_country)
    return f"BL-2024-{oc}-{dc}-{seq:04d}"


def _generate_po_number() -> str:
    """Produce a PO number in the format PO-YYYY-XXXXX."""
    year = random.choice([2024, 2025])
    return f"PO-{year}-{random.randint(1, 99999):05d}"


def _generate_container_numbers(carrier_name: str, count: int) -> list[str]:
    """Generate realistic container number strings, e.g. MSCU-1234567."""
    prefix = CARRIER_CODES.get(carrier_name, "UNKN")[:4].upper() + "U"
    return [f"{prefix}-{random.randint(1000000, 9999999)}" for _ in range(count)]


def _make_charges(base_freight: float) -> list[tuple[str, float]]:
    """Build a list of (description, amount) charge lines for an invoice.

    Always includes Ocean/Air Freight.  Other charges are randomly added so
    that totals end up in the $5K-$50K range.
    """
    charges = [("Ocean Freight", round(base_freight, 2))]

    # Customs duty -- 5-12 % of freight
    if random.random() < 0.75:
        charges.append(("Customs Duty", round(base_freight * random.uniform(0.05, 0.12), 2)))

    # Insurance -- 1-3 % of freight
    if random.random() < 0.65:
        charges.append(("Insurance", round(base_freight * random.uniform(0.01, 0.03), 2)))

    # Handling -- fixed-ish
    if random.random() < 0.60:
        charges.append(("Handling", round(random.uniform(150, 800), 2)))

    # Documentation fee
    if random.random() < 0.50:
        charges.append(("Documentation", round(random.uniform(50, 200), 2)))

    # Demurrage (occasional)
    if random.random() < 0.15:
        charges.append(("Demurrage", round(random.uniform(200, 1500), 2)))

    return charges


# ============================================================================
# DATA GENERATION -- one function per document type creates a list of dicts,
# each dict containing all the fields needed to render that document.
# ============================================================================


def generate_invoice_data(count: int = 10) -> list[dict]:
    """Create *count* invoice data records with realistic logistics content."""
    records = []
    # Use a range of sequence numbers that overlap with the golden shipments
    # referenced in seed_data.py (SHP-2024-000001 .. SHP-2025-010000).
    seq_start = 734  # chosen to match the example in the spec

    for i in range(count):
        seq = seq_start + i
        booking_date = _random_date()
        invoice_date = booking_date + timedelta(days=random.randint(1, 14))
        due_date = invoice_date + timedelta(days=random.choice([15, 30, 45, 60]))
        carrier = random.choice(CARRIER_NAMES)
        customer = random.choice(CUSTOMER_NAMES)
        route = _random_route()
        commodity = random.choice(COMMODITIES)

        # Base freight keeps totals in the $5K-$50K band
        base_freight = round(random.uniform(4000, 35000), 2)
        charges = _make_charges(base_freight)
        subtotal = round(sum(amt for _, amt in charges), 2)
        tax_pct = round(random.uniform(0.05, 0.18), 4)
        tax = round(subtotal * tax_pct, 2)
        total = round(subtotal + tax, 2)

        container_type = random.choice(CONTAINER_TYPES)
        container_count = random.randint(1, 6)
        weight_kg = round(random.uniform(2000, 25000), 1)

        records.append(
            {
                "invoice_number": _generate_invoice_number(invoice_date, seq),
                "invoice_date": invoice_date.strftime("%Y-%m-%d"),
                "due_date": due_date.strftime("%Y-%m-%d"),
                "carrier": carrier,
                "customer": customer,
                "customer_code": CUSTOMER_CODES.get(customer, "CUST-00001"),
                "shipment_id": _generate_shipment_id(booking_date, seq),
                "po_number": _generate_po_number(),
                "origin_country": route["origin_country"],
                "dest_country": route["dest_country"],
                "origin_port": route["origin_port"],
                "dest_port": route["dest_port"],
                "charges": charges,
                "subtotal": subtotal,
                "tax_pct": tax_pct,
                "tax": tax,
                "total": total,
                "payment_terms": random.choice(PAYMENT_TERMS),
                "currency": "USD",
                "container_type": container_type,
                "container_count": container_count,
                "weight_kg": weight_kg,
                "commodity": commodity,
                "incoterm": random.choice(INCOTERMS),
            }
        )
    return records


def generate_bol_data(count: int = 8) -> list[dict]:
    """Create *count* Bill of Lading data records."""
    records = []
    seq_start = 734

    for i in range(count):
        seq = seq_start + 100 + i  # offset so BL numbers differ from invoices
        route = _random_route()
        carrier = random.choice(
            [c for c in CARRIER_NAMES if c in VESSEL_NAMES]  # only carriers with vessels
        )
        booking_date = _random_date()
        issue_date = booking_date + timedelta(days=random.randint(0, 5))

        customer = random.choice(CUSTOMER_NAMES)
        shipper_name = f"{route['origin_country']} {random.choice(COMMODITIES)} Exports Ltd"
        notify_party = f"{route['dest_country']} Import Agency"

        vessel = random.choice(VESSEL_NAMES.get(carrier, ["UNKNOWN VESSEL"]))
        voyage = (
            f"{_country_code(route['origin_country'])}-"
            f"{_country_code(route['dest_country'])}-"
            f"{booking_date.year}-{random.randint(100, 999)}"
        )

        container_count = random.randint(1, 4)
        containers = _generate_container_numbers(carrier, container_count)

        weight_kg = round(random.uniform(2000, 25000), 1)
        volume_cbm = round(weight_kg / random.uniform(200, 400), 1)
        packages = random.randint(50, 2000)
        package_type = random.choice(["cartons", "pallets", "crates", "drums", "bags"])
        commodity = random.choice(COMMODITIES)

        records.append(
            {
                "bl_number": _generate_bl_number(route["origin_country"], route["dest_country"], seq),
                "issue_date": issue_date.strftime("%Y-%m-%d"),
                "shipper": shipper_name,
                "consignee": customer,
                "notify_party": notify_party,
                "carrier": carrier,
                "vessel": vessel,
                "voyage": voyage,
                "origin_port": route["origin_port"],
                "dest_port": route["dest_port"],
                "origin_country": route["origin_country"],
                "dest_country": route["dest_country"],
                "containers": containers,
                "commodity": commodity,
                "weight_kg": weight_kg,
                "volume_cbm": volume_cbm,
                "packages": packages,
                "package_type": package_type,
                "shipment_id": _generate_shipment_id(booking_date, seq),
            }
        )
    return records


def generate_packing_list_data(count: int = 4) -> list[dict]:
    """Create *count* Packing List data records."""
    records = []
    seq_start = 900

    for i in range(count):
        seq = seq_start + i
        route = _random_route()
        booking_date = _random_date()
        carrier = random.choice(CARRIER_NAMES)
        customer = random.choice(CUSTOMER_NAMES)
        commodity = random.choice(COMMODITIES)

        # Build a list of line items (3-8 items)
        num_items = random.randint(3, 8)
        items = []
        for j in range(num_items):
            desc = f"{commodity} - Grade {chr(65 + j)}"
            qty = random.randint(10, 500)
            unit_weight = round(random.uniform(0.5, 25.0), 2)
            total_weight = round(qty * unit_weight, 2)
            dimensions = f"{random.randint(20,120)}x{random.randint(20,80)}x{random.randint(10,60)} cm"
            items.append(
                {
                    "item_no": j + 1,
                    "description": desc,
                    "quantity": qty,
                    "unit_weight_kg": unit_weight,
                    "total_weight_kg": total_weight,
                    "dimensions": dimensions,
                    "package_type": random.choice(["carton", "pallet", "crate", "drum"]),
                }
            )

        total_weight = round(sum(it["total_weight_kg"] for it in items), 2)
        total_packages = sum(it["quantity"] for it in items)

        container_count = random.randint(1, 3)
        containers = _generate_container_numbers(carrier, container_count)

        records.append(
            {
                "packing_list_number": f"PL-{booking_date.year}-{seq:06d}",
                "date": booking_date.strftime("%Y-%m-%d"),
                "shipper": f"{route['origin_country']} {commodity} Exports Ltd",
                "consignee": customer,
                "carrier": carrier,
                "origin_port": route["origin_port"],
                "dest_port": route["dest_port"],
                "origin_country": route["origin_country"],
                "dest_country": route["dest_country"],
                "shipment_id": _generate_shipment_id(booking_date, seq),
                "po_number": _generate_po_number(),
                "items": items,
                "total_weight_kg": total_weight,
                "total_packages": total_packages,
                "containers": containers,
                "commodity": commodity,
            }
        )
    return records


def generate_customs_data(count: int = 3) -> list[dict]:
    """Create *count* Customs Declaration data records."""
    records = []
    seq_start = 950

    for i in range(count):
        seq = seq_start + i
        route = _random_route()
        booking_date = _random_date()
        carrier = random.choice(CARRIER_NAMES)
        customer = random.choice(CUSTOMER_NAMES)
        commodity = random.choice(COMMODITIES)

        declared_value = round(random.uniform(5000, 50000), 2)
        duty_rate = round(random.uniform(0.02, 0.15), 4)
        duty_amount = round(declared_value * duty_rate, 2)
        tax_rate = round(random.uniform(0.05, 0.18), 4)
        tax_amount = round(declared_value * tax_rate, 2)
        total_payable = round(duty_amount + tax_amount, 2)

        weight_kg = round(random.uniform(2000, 25000), 1)
        packages = random.randint(50, 1500)

        # HS code (Harmonized System) -- 6-digit, realistic ranges
        hs_code = f"{random.randint(10, 99)}.{random.randint(10, 99)}.{random.randint(10, 99)}"

        records.append(
            {
                "declaration_number": f"CD-{booking_date.year}-{seq:06d}",
                "date": booking_date.strftime("%Y-%m-%d"),
                "customs_office": f"{route['dest_country']} Customs Authority",
                "importer": customer,
                "exporter": f"{route['origin_country']} {commodity} Exports Ltd",
                "carrier": carrier,
                "origin_country": route["origin_country"],
                "dest_country": route["dest_country"],
                "origin_port": route["origin_port"],
                "dest_port": route["dest_port"],
                "shipment_id": _generate_shipment_id(booking_date, seq),
                "commodity": commodity,
                "hs_code": hs_code,
                "declared_value_usd": declared_value,
                "currency": "USD",
                "duty_rate": duty_rate,
                "duty_amount": duty_amount,
                "tax_rate": tax_rate,
                "tax_amount": tax_amount,
                "total_payable": total_payable,
                "weight_kg": weight_kg,
                "packages": packages,
                "package_type": random.choice(["cartons", "pallets", "crates", "drums"]),
                "incoterm": random.choice(INCOTERMS),
                "transport_mode": random.choice(["Ocean", "Air", "Road"]),
            }
        )
    return records


# ============================================================================
# PDF RENDERING  -- each document type has a dedicated render function that
# takes a data dict and a quality tier, and returns a reportlab
# SimpleDocTemplate written to a BytesIO buffer.
# ============================================================================

# Quality-tier font-size and spacing presets
QUALITY_PRESETS = {
    "clear": {
        "title_size": 18,
        "heading_size": 13,
        "body_size": 12,
        "small_size": 10,
        "leading": 16,           # line height
        "spacer_after_title": 14,
        "spacer_between": 10,
        "table_padding": 8,
        "page_margin": 20 * mm,
    },
    "medium": {
        "title_size": 15,
        "heading_size": 11,
        "body_size": 10,
        "small_size": 8,
        "leading": 13,
        "spacer_after_title": 8,
        "spacer_between": 6,
        "table_padding": 5,
        "page_margin": 15 * mm,
    },
    "low": {
        "title_size": 12,
        "heading_size": 9,
        "body_size": 8,
        "small_size": 7,
        "leading": 10,
        "spacer_after_title": 4,
        "spacer_between": 3,
        "table_padding": 3,
        "page_margin": 10 * mm,
    },
}


def _build_styles(quality: str) -> dict:
    """Return a dict of ParagraphStyles tuned for the given quality tier.

    Keys returned: title, heading, body, body_right, body_center, small, small_right.
    """
    p = QUALITY_PRESETS[quality]
    base = getSampleStyleSheet()

    # We create fresh styles to avoid mutating the shared stylesheet
    styles = {
        "title": ParagraphStyle(
            "doc_title",
            parent=base["Title"],
            fontSize=p["title_size"],
            leading=p["title_size"] + 4,
            alignment=TA_LEFT,
            spaceAfter=p["spacer_after_title"],
            textColor=colors.HexColor("#1a1a2e"),
        ),
        "heading": ParagraphStyle(
            "doc_heading",
            parent=base["Heading2"],
            fontSize=p["heading_size"],
            leading=p["heading_size"] + 3,
            spaceAfter=p["spacer_between"],
            textColor=colors.HexColor("#16213e"),
        ),
        "body": ParagraphStyle(
            "doc_body",
            parent=base["Normal"],
            fontSize=p["body_size"],
            leading=p["leading"],
            alignment=TA_LEFT,
        ),
        "body_right": ParagraphStyle(
            "doc_body_right",
            parent=base["Normal"],
            fontSize=p["body_size"],
            leading=p["leading"],
            alignment=TA_RIGHT,
        ),
        "body_center": ParagraphStyle(
            "doc_body_center",
            parent=base["Normal"],
            fontSize=p["body_size"],
            leading=p["leading"],
            alignment=TA_CENTER,
        ),
        "small": ParagraphStyle(
            "doc_small",
            parent=base["Normal"],
            fontSize=p["small_size"],
            leading=p["small_size"] + 3,
            textColor=colors.HexColor("#555555"),
        ),
        "small_right": ParagraphStyle(
            "doc_small_right",
            parent=base["Normal"],
            fontSize=p["small_size"],
            leading=p["small_size"] + 3,
            alignment=TA_RIGHT,
            textColor=colors.HexColor("#555555"),
        ),
    }
    return styles


def _format_usd(value: float) -> str:
    """Format a float as a USD string like '12,345.67'."""
    return f"{value:,.2f}"


# ---------------------------------------------------------------------------
# Invoice PDF
# ---------------------------------------------------------------------------

def render_invoice_pdf(data: dict, quality: str) -> bytes:
    """Render a freight invoice to a PDF and return the raw bytes.

    Parameters
    ----------
    data : dict
        A single record from generate_invoice_data().
    quality : str
        One of 'clear', 'medium', 'low'.

    Returns
    -------
    bytes
        The complete PDF file content.
    """
    buf = io.BytesIO()
    p = QUALITY_PRESETS[quality]
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=p["page_margin"],
        rightMargin=p["page_margin"],
        topMargin=p["page_margin"],
        bottomMargin=p["page_margin"],
    )
    styles = _build_styles(quality)
    story = []

    # ── Title ──
    story.append(Paragraph("FREIGHT INVOICE", styles["title"]))
    story.append(
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a1a2e"))
    )
    story.append(Spacer(1, p["spacer_after_title"]))

    # ── Header info (two-column layout via a table) ──
    left_info = [
        f"<b>Invoice No:</b> {data['invoice_number']}",
        f"<b>Date:</b> {data['invoice_date']}",
        f"<b>Due Date:</b> {data['due_date']}",
    ]
    right_info = [
        f"<b>Shipment Ref:</b> {data['shipment_id']}",
        f"<b>PO Number:</b> {data['po_number']}",
        f"<b>Incoterm:</b> {data['incoterm']}",
    ]
    header_data = [
        [
            Paragraph("<br/>".join(left_info), styles["body"]),
            Paragraph("<br/>".join(right_info), styles["body_right"]),
        ]
    ]
    header_table = Table(header_data, colWidths=["50%", "50%"])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, p["spacer_between"]))

    # ── From / To ──
    story.append(Paragraph(f"<b>From (Carrier):</b> {data['carrier']}", styles["body"]))
    story.append(Paragraph(f"<b>To (Customer):</b> {data['customer']}", styles["body"]))
    story.append(Paragraph(f"<b>Customer Code:</b> {data['customer_code']}", styles["body"]))
    story.append(Spacer(1, p["spacer_between"]))

    # ── Route ──
    story.append(
        Paragraph(
            f"<b>Route:</b> {data['origin_port']} ({data['origin_country']}) "
            f"&rarr; {data['dest_port']} ({data['dest_country']})",
            styles["body"],
        )
    )
    story.append(Spacer(1, p["spacer_between"]))

    # ── Charges table ──
    # Header row
    table_data = [
        [
            Paragraph("<b>Description</b>", styles["body"]),
            Paragraph("<b>Amount (USD)</b>", styles["body_right"]),
        ]
    ]
    # Charge rows
    for desc, amt in data["charges"]:
        table_data.append([
            Paragraph(desc, styles["body"]),
            Paragraph(_format_usd(amt), styles["body_right"]),
        ])
    # Separator + subtotal / tax / total
    table_data.append([
        Paragraph("<b>Subtotal</b>", styles["body"]),
        Paragraph(f"<b>{_format_usd(data['subtotal'])}</b>", styles["body_right"]),
    ])
    tax_pct_display = f"{data['tax_pct'] * 100:.1f}%"
    table_data.append([
        Paragraph(f"<b>Tax ({tax_pct_display})</b>", styles["body"]),
        Paragraph(f"<b>{_format_usd(data['tax'])}</b>", styles["body_right"]),
    ])
    table_data.append([
        Paragraph("<b>TOTAL</b>", styles["body"]),
        Paragraph(f"<b>{_format_usd(data['total'])}</b>", styles["body_right"]),
    ])

    charges_table = Table(table_data, colWidths=["65%", "35%"])
    # Calculate the index where summary rows start (after header + charge rows)
    num_charge_rows = len(data["charges"])
    summary_start_row = num_charge_rows + 1  # +1 for header

    charges_table.setStyle(
        TableStyle(
            [
                # Header row
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), p["body_size"]),
                # Grid
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                # Padding
                ("TOPPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("LEFTPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("RIGHTPADDING", (0, 0), (-1, -1), p["table_padding"]),
                # Alternating row backgrounds for charge lines
                *[
                    ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f5f5f5"))
                    for r in range(1, summary_start_row)
                    if r % 2 == 0
                ],
                # Summary rows background
                ("BACKGROUND", (0, summary_start_row), (-1, summary_start_row), colors.HexColor("#e8e8e8")),
                ("BACKGROUND", (0, summary_start_row + 1), (-1, summary_start_row + 1), colors.HexColor("#e8e8e8")),
                ("BACKGROUND", (0, summary_start_row + 2), (-1, summary_start_row + 2), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, summary_start_row + 2), (-1, summary_start_row + 2), colors.white),
                # Line above subtotal
                ("LINEABOVE", (0, summary_start_row), (-1, summary_start_row), 1.5, colors.HexColor("#1a1a2e")),
            ]
        )
    )
    story.append(charges_table)
    story.append(Spacer(1, p["spacer_between"] * 2))

    # ── Footer details ──
    footer_lines = [
        f"<b>Payment Terms:</b> {data['payment_terms']}",
        f"<b>Currency:</b> {data['currency']}",
        f"<b>Container:</b> {data['container_type']} x {data['container_count']}",
        f"<b>Weight:</b> {data['weight_kg']:,.1f} kg",
        f"<b>Commodity:</b> {data['commodity']}",
    ]
    for line in footer_lines:
        story.append(Paragraph(line, styles["body"]))

    story.append(Spacer(1, p["spacer_between"] * 2))
    story.append(
        Paragraph(
            "This is a computer-generated document. No signature required.",
            styles["small"],
        )
    )

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Bill of Lading PDF
# ---------------------------------------------------------------------------

def render_bol_pdf(data: dict, quality: str) -> bytes:
    """Render a Bill of Lading to PDF and return the raw bytes.

    Parameters
    ----------
    data : dict
        A single record from generate_bol_data().
    quality : str
        One of 'clear', 'medium', 'low'.

    Returns
    -------
    bytes
        The complete PDF file content.
    """
    buf = io.BytesIO()
    p = QUALITY_PRESETS[quality]
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=p["page_margin"],
        rightMargin=p["page_margin"],
        topMargin=p["page_margin"],
        bottomMargin=p["page_margin"],
    )
    styles = _build_styles(quality)
    story = []

    # ── Title ──
    story.append(Paragraph("BILL OF LADING", styles["title"]))
    story.append(
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a1a2e"))
    )
    story.append(Spacer(1, p["spacer_after_title"]))

    # ── B/L number and date ──
    header_data = [
        [
            Paragraph(f"<b>B/L Number:</b> {data['bl_number']}", styles["body"]),
            Paragraph(f"<b>Date of Issue:</b> {data['issue_date']}", styles["body_right"]),
        ]
    ]
    header_table = Table(header_data, colWidths=["50%", "50%"])
    header_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(header_table)
    story.append(Spacer(1, p["spacer_between"]))

    # ── Parties ──
    story.append(Paragraph("<b>SHIPPER / EXPORTER</b>", styles["heading"]))
    story.append(Paragraph(data["shipper"], styles["body"]))
    story.append(Spacer(1, p["spacer_between"]))

    story.append(Paragraph("<b>CONSIGNEE</b>", styles["heading"]))
    story.append(Paragraph(data["consignee"], styles["body"]))
    story.append(Spacer(1, p["spacer_between"]))

    story.append(Paragraph("<b>NOTIFY PARTY</b>", styles["heading"]))
    story.append(Paragraph(data["notify_party"], styles["body"]))
    story.append(Spacer(1, p["spacer_between"]))

    # ── Vessel / Voyage ──
    story.append(
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa"))
    )
    story.append(Spacer(1, p["spacer_between"]))

    vessel_data = [
        [
            Paragraph(f"<b>Carrier:</b> {data['carrier']}", styles["body"]),
            Paragraph(f"<b>Vessel:</b> {data['vessel']}", styles["body_right"]),
        ],
        [
            Paragraph(f"<b>Voyage No:</b> {data['voyage']}", styles["body"]),
            Paragraph(f"<b>Shipment Ref:</b> {data['shipment_id']}", styles["body_right"]),
        ],
    ]
    vessel_table = Table(vessel_data, colWidths=["50%", "50%"])
    vessel_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(vessel_table)
    story.append(Spacer(1, p["spacer_between"]))

    # ── Ports ──
    port_data = [
        [
            Paragraph("<b>Port of Loading</b>", styles["body_center"]),
            Paragraph("<b>Port of Discharge</b>", styles["body_center"]),
        ],
        [
            Paragraph(data["origin_port"], styles["body_center"]),
            Paragraph(data["dest_port"], styles["body_center"]),
        ],
    ]
    port_table = Table(port_data, colWidths=["50%", "50%"])
    port_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(port_table)
    story.append(Spacer(1, p["spacer_between"]))

    # ── Container details ──
    story.append(Paragraph("<b>CONTAINER DETAILS</b>", styles["heading"]))
    container_rows = [
        [
            Paragraph("<b>Container No.</b>", styles["body"]),
            Paragraph("<b>Type</b>", styles["body_center"]),
            Paragraph("<b>Seal No.</b>", styles["body_center"]),
        ]
    ]
    for cnum in data["containers"]:
        seal = f"SL-{random.randint(100000, 999999)}"
        container_rows.append([
            Paragraph(cnum, styles["body"]),
            Paragraph("40ft HC", styles["body_center"]),
            Paragraph(seal, styles["body_center"]),
        ])
    container_table = Table(container_rows, colWidths=["40%", "30%", "30%"])
    container_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("TOPPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), p["table_padding"]),
            ]
        )
    )
    story.append(container_table)
    story.append(Spacer(1, p["spacer_between"]))

    # ── Cargo description ──
    story.append(Paragraph("<b>CARGO DESCRIPTION</b>", styles["heading"]))
    cargo_data = [
        [Paragraph("<b>Field</b>", styles["body"]), Paragraph("<b>Value</b>", styles["body"])],
        [Paragraph("Description of Goods", styles["body"]), Paragraph(f"{data['commodity']} Products", styles["body"])],
        [Paragraph("Gross Weight", styles["body"]), Paragraph(f"{data['weight_kg']:,.1f} KG", styles["body"])],
        [Paragraph("Measurement", styles["body"]), Paragraph(f"{data['volume_cbm']:,.1f} CBM", styles["body"])],
        [Paragraph("Number of Packages", styles["body"]), Paragraph(f"{data['packages']} {data['package_type']}", styles["body"])],
    ]
    cargo_table = Table(cargo_data, colWidths=["40%", "60%"])
    cargo_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f5f5f5")),
            ]
        )
    )
    story.append(cargo_table)
    story.append(Spacer(1, p["spacer_between"] * 2))

    # ── Footer ──
    story.append(
        Paragraph(
            "Shipped on board in apparent good order and condition. "
            "This Bill of Lading is issued subject to the carrier's standard terms and conditions.",
            styles["small"],
        )
    )
    story.append(Spacer(1, p["spacer_between"]))
    story.append(
        Paragraph(
            f"<b>Place and Date of Issue:</b> {data['origin_port']}, {data['issue_date']}",
            styles["body"],
        )
    )

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Packing List PDF
# ---------------------------------------------------------------------------

def render_packing_list_pdf(data: dict, quality: str) -> bytes:
    """Render a Packing List to PDF and return the raw bytes.

    Parameters
    ----------
    data : dict
        A single record from generate_packing_list_data().
    quality : str
        One of 'clear', 'medium', 'low'.

    Returns
    -------
    bytes
        The complete PDF file content.
    """
    buf = io.BytesIO()
    p = QUALITY_PRESETS[quality]
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=p["page_margin"],
        rightMargin=p["page_margin"],
        topMargin=p["page_margin"],
        bottomMargin=p["page_margin"],
    )
    styles = _build_styles(quality)
    story = []

    # ── Title ──
    story.append(Paragraph("PACKING LIST", styles["title"]))
    story.append(
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a1a2e"))
    )
    story.append(Spacer(1, p["spacer_after_title"]))

    # ── Header ──
    header_data = [
        [
            Paragraph(f"<b>Packing List No:</b> {data['packing_list_number']}", styles["body"]),
            Paragraph(f"<b>Date:</b> {data['date']}", styles["body_right"]),
        ],
        [
            Paragraph(f"<b>Shipment Ref:</b> {data['shipment_id']}", styles["body"]),
            Paragraph(f"<b>PO Number:</b> {data['po_number']}", styles["body_right"]),
        ],
    ]
    header_table = Table(header_data, colWidths=["50%", "50%"])
    header_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(header_table)
    story.append(Spacer(1, p["spacer_between"]))

    # ── Parties ──
    story.append(Paragraph(f"<b>Shipper:</b> {data['shipper']}", styles["body"]))
    story.append(Paragraph(f"<b>Consignee:</b> {data['consignee']}", styles["body"]))
    story.append(Paragraph(f"<b>Carrier:</b> {data['carrier']}", styles["body"]))
    story.append(Spacer(1, p["spacer_between"]))

    # ── Route ──
    story.append(
        Paragraph(
            f"<b>From:</b> {data['origin_port']} ({data['origin_country']}) &rarr; "
            f"<b>To:</b> {data['dest_port']} ({data['dest_country']})",
            styles["body"],
        )
    )
    story.append(Spacer(1, p["spacer_between"]))

    # ── Container numbers ──
    story.append(Paragraph(f"<b>Containers:</b> {', '.join(data['containers'])}", styles["body"]))
    story.append(Spacer(1, p["spacer_between"]))

    # ── Items table ──
    item_header = [
        Paragraph("<b>#</b>", styles["body_center"]),
        Paragraph("<b>Description</b>", styles["body"]),
        Paragraph("<b>Qty</b>", styles["body_center"]),
        Paragraph("<b>Pkg Type</b>", styles["body_center"]),
        Paragraph("<b>Unit Wt (kg)</b>", styles["body_right"]),
        Paragraph("<b>Total Wt (kg)</b>", styles["body_right"]),
        Paragraph("<b>Dimensions</b>", styles["body_center"]),
    ]
    item_rows = [item_header]
    for it in data["items"]:
        item_rows.append([
            Paragraph(str(it["item_no"]), styles["body_center"]),
            Paragraph(it["description"], styles["body"]),
            Paragraph(str(it["quantity"]), styles["body_center"]),
            Paragraph(it["package_type"], styles["body_center"]),
            Paragraph(f"{it['unit_weight_kg']:.2f}", styles["body_right"]),
            Paragraph(f"{it['total_weight_kg']:,.2f}", styles["body_right"]),
            Paragraph(it["dimensions"], styles["body_center"]),
        ])

    # Totals row
    item_rows.append([
        Paragraph("", styles["body"]),
        Paragraph("<b>TOTAL</b>", styles["body"]),
        Paragraph(f"<b>{data['total_packages']}</b>", styles["body_center"]),
        Paragraph("", styles["body"]),
        Paragraph("", styles["body"]),
        Paragraph(f"<b>{data['total_weight_kg']:,.2f}</b>", styles["body_right"]),
        Paragraph("", styles["body"]),
    ])

    items_table = Table(
        item_rows,
        colWidths=["5%", "22%", "8%", "12%", "14%", "14%", "25%"],
    )
    num_item_rows = len(data["items"])
    items_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), p["table_padding"]),
                # Alternating row colours for item rows
                *[
                    ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f5f5f5"))
                    for r in range(1, num_item_rows + 1)
                    if r % 2 == 0
                ],
                # Totals row
                ("BACKGROUND", (0, num_item_rows + 1), (-1, num_item_rows + 1), colors.HexColor("#e8e8e8")),
                ("LINEABOVE", (0, num_item_rows + 1), (-1, num_item_rows + 1), 1.5, colors.HexColor("#1a1a2e")),
            ]
        )
    )
    story.append(items_table)
    story.append(Spacer(1, p["spacer_between"] * 2))

    # ── Commodity summary ──
    story.append(Paragraph(f"<b>Commodity:</b> {data['commodity']}", styles["body"]))
    story.append(
        Paragraph(
            f"<b>Total Gross Weight:</b> {data['total_weight_kg']:,.2f} kg",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            f"<b>Total Number of Packages:</b> {data['total_packages']}",
            styles["body"],
        )
    )
    story.append(Spacer(1, p["spacer_between"] * 2))
    story.append(
        Paragraph(
            "Packed and verified by shipper. Contents as described above.",
            styles["small"],
        )
    )

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Customs Declaration PDF
# ---------------------------------------------------------------------------

def render_customs_pdf(data: dict, quality: str) -> bytes:
    """Render a Customs Declaration to PDF and return the raw bytes.

    Parameters
    ----------
    data : dict
        A single record from generate_customs_data().
    quality : str
        One of 'clear', 'medium', 'low'.

    Returns
    -------
    bytes
        The complete PDF file content.
    """
    buf = io.BytesIO()
    p = QUALITY_PRESETS[quality]
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=p["page_margin"],
        rightMargin=p["page_margin"],
        topMargin=p["page_margin"],
        bottomMargin=p["page_margin"],
    )
    styles = _build_styles(quality)
    story = []

    # ── Title ──
    story.append(Paragraph("CUSTOMS DECLARATION", styles["title"]))
    story.append(
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a1a2e"))
    )
    story.append(Spacer(1, p["spacer_after_title"]))

    # ── Header info ──
    header_data = [
        [
            Paragraph(f"<b>Declaration No:</b> {data['declaration_number']}", styles["body"]),
            Paragraph(f"<b>Date:</b> {data['date']}", styles["body_right"]),
        ],
        [
            Paragraph(f"<b>Customs Office:</b> {data['customs_office']}", styles["body"]),
            Paragraph(f"<b>Shipment Ref:</b> {data['shipment_id']}", styles["body_right"]),
        ],
    ]
    header_table = Table(header_data, colWidths=["55%", "45%"])
    header_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(header_table)
    story.append(Spacer(1, p["spacer_between"]))

    # ── Parties ──
    story.append(Paragraph("<b>IMPORTER / CONSIGNEE</b>", styles["heading"]))
    story.append(Paragraph(data["importer"], styles["body"]))
    story.append(Spacer(1, p["spacer_between"]))

    story.append(Paragraph("<b>EXPORTER / SHIPPER</b>", styles["heading"]))
    story.append(Paragraph(data["exporter"], styles["body"]))
    story.append(Spacer(1, p["spacer_between"]))

    story.append(Paragraph(f"<b>Carrier:</b> {data['carrier']}", styles["body"]))
    story.append(Paragraph(f"<b>Transport Mode:</b> {data['transport_mode']}", styles["body"]))
    story.append(Paragraph(f"<b>Incoterm:</b> {data['incoterm']}", styles["body"]))
    story.append(Spacer(1, p["spacer_between"]))

    # ── Route ──
    route_data = [
        [
            Paragraph("<b>Country of Origin</b>", styles["body_center"]),
            Paragraph("<b>Country of Destination</b>", styles["body_center"]),
        ],
        [
            Paragraph(f"{data['origin_country']}<br/>{data['origin_port']}", styles["body_center"]),
            Paragraph(f"{data['dest_country']}<br/>{data['dest_port']}", styles["body_center"]),
        ],
    ]
    route_table = Table(route_data, colWidths=["50%", "50%"])
    route_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), p["table_padding"]),
            ]
        )
    )
    story.append(route_table)
    story.append(Spacer(1, p["spacer_between"]))

    # ── Goods description ──
    story.append(Paragraph("<b>GOODS DESCRIPTION</b>", styles["heading"]))
    goods_data = [
        [Paragraph("<b>Field</b>", styles["body"]), Paragraph("<b>Value</b>", styles["body"])],
        [Paragraph("Commodity", styles["body"]), Paragraph(data["commodity"], styles["body"])],
        [Paragraph("HS Code", styles["body"]), Paragraph(data["hs_code"], styles["body"])],
        [Paragraph("Gross Weight", styles["body"]), Paragraph(f"{data['weight_kg']:,.1f} KG", styles["body"])],
        [Paragraph("Number of Packages", styles["body"]), Paragraph(f"{data['packages']} {data['package_type']}", styles["body"])],
        [Paragraph("Declared Value", styles["body"]), Paragraph(f"USD {_format_usd(data['declared_value_usd'])}", styles["body"])],
    ]
    goods_table = Table(goods_data, colWidths=["35%", "65%"])
    goods_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("TOPPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f5f5f5")),
            ]
        )
    )
    story.append(goods_table)
    story.append(Spacer(1, p["spacer_between"]))

    # ── Duty and tax calculation ──
    story.append(Paragraph("<b>DUTY AND TAX ASSESSMENT</b>", styles["heading"]))
    duty_data = [
        [Paragraph("<b>Item</b>", styles["body"]), Paragraph("<b>Rate</b>", styles["body_center"]), Paragraph("<b>Amount (USD)</b>", styles["body_right"])],
        [
            Paragraph("Import Duty", styles["body"]),
            Paragraph(f"{data['duty_rate'] * 100:.2f}%", styles["body_center"]),
            Paragraph(_format_usd(data["duty_amount"]), styles["body_right"]),
        ],
        [
            Paragraph("Value Added Tax / GST", styles["body"]),
            Paragraph(f"{data['tax_rate'] * 100:.2f}%", styles["body_center"]),
            Paragraph(_format_usd(data["tax_amount"]), styles["body_right"]),
        ],
        [
            Paragraph("<b>TOTAL PAYABLE</b>", styles["body"]),
            Paragraph("", styles["body"]),
            Paragraph(f"<b>{_format_usd(data['total_payable'])}</b>", styles["body_right"]),
        ],
    ]
    duty_table = Table(duty_data, colWidths=["40%", "25%", "35%"])
    duty_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), p["table_padding"]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), p["table_padding"]),
                # Total row emphasis
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8e8e8")),
                ("LINEABOVE", (0, -1), (-1, -1), 1.5, colors.HexColor("#1a1a2e")),
            ]
        )
    )
    story.append(duty_table)
    story.append(Spacer(1, p["spacer_between"] * 2))

    # ── Declaration statement ──
    story.append(
        Paragraph(
            "I hereby declare that the information provided in this document is true, "
            "complete, and correct to the best of my knowledge. All goods are in "
            "compliance with the import/export regulations of the declaring country.",
            styles["small"],
        )
    )
    story.append(Spacer(1, p["spacer_between"]))
    story.append(
        Paragraph(
            f"<b>Declared at:</b> {data['dest_port']}, {data['date']}",
            styles["body"],
        )
    )

    doc.build(story)
    return buf.getvalue()


# ============================================================================
# IMAGE CONVERSION  -- convert a PDF (bytes) to a PNG image, optionally
# degrading quality to simulate a photograph of a printed page.
# ============================================================================

def pdf_to_degraded_image(pdf_bytes: bytes, dpi: int = 120, add_noise: bool = True) -> bytes:
    """Rasterise the first page of a PDF to a PNG and optionally degrade it.

    Steps:
      1. Open the PDF with PyMuPDF and render page 0 at the requested DPI.
      2. Convert the pixmap to a Pillow Image.
      3. (Optional) Add Gaussian noise to simulate a camera photo of a printout.
      4. Apply a slight blur to soften sharp edges (mimics camera defocus).
      5. Reduce and re-encode as PNG.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file content.
    dpi : int
        Render resolution. Lower = more degraded.  100-150 is realistic for
        a phone photo; 300 would be a clean scan.
    add_noise : bool
        If True, sprinkle Gaussian noise onto the image.

    Returns
    -------
    bytes
        PNG image bytes.
    """
    # Step 1 -- rasterise with PyMuPDF
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = pdf_doc[0]
    # The zoom factor relates DPI to the default 72 DPI
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    # Step 2 -- convert pixmap to Pillow Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pdf_doc.close()

    # Step 3 -- add Gaussian noise
    if add_noise:
        img_array = np.array(img, dtype=np.float32)
        # Standard deviation of noise -- higher = grainier
        noise_sigma = random.uniform(8, 18)
        noise = np.random.normal(0, noise_sigma, img_array.shape).astype(np.float32)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

    # Step 4 -- slight Gaussian blur (radius 0.5-1.0)
    blur_radius = random.uniform(0.4, 0.9)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Step 5 -- slight brightness/contrast shift to mimic lighting
    # We darken the image very slightly (multiply by 0.92-0.98)
    img_array = np.array(img, dtype=np.float32)
    brightness_factor = random.uniform(0.92, 0.98)
    img_array = np.clip(img_array * brightness_factor, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    # Encode as PNG
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# ============================================================================
# DOCUMENT PLAN -- defines exactly which documents to generate, their type,
# quality tier, and output format.  The list has 25 entries.
# ============================================================================

# Each entry: (doc_type, quality, output_format, sequence_index_within_type)
# doc_type:       'invoice' | 'bol' | 'packing' | 'customs'
# quality:        'clear' | 'medium' | 'low'
# output_format:  'pdf' | 'image'

DOCUMENT_PLAN = [
    # ── Invoices (10) ──────────────────────────────────────────────────
    # 4 clear PDFs
    ("invoice", "clear",  "pdf",   0),
    ("invoice", "clear",  "pdf",   1),
    ("invoice", "clear",  "pdf",   2),
    ("invoice", "clear",  "pdf",   3),
    # 3 medium PDFs
    ("invoice", "medium", "pdf",   4),
    ("invoice", "medium", "pdf",   5),
    ("invoice", "medium", "pdf",   6),
    # 2 low quality PDFs
    ("invoice", "low",    "pdf",   7),
    ("invoice", "low",    "pdf",   8),
    # 1 image (PNG)
    ("invoice", "low",    "image", 9),

    # ── Bills of Lading (8) ───────────────────────────────────────────
    # 3 clear PDFs
    ("bol",     "clear",  "pdf",   0),
    ("bol",     "clear",  "pdf",   1),
    ("bol",     "clear",  "pdf",   2),
    # 3 medium PDFs
    ("bol",     "medium", "pdf",   3),
    ("bol",     "medium", "pdf",   4),
    ("bol",     "medium", "pdf",   5),
    # 1 low quality PDF
    ("bol",     "low",    "pdf",   6),
    # 1 image (PNG)
    ("bol",     "low",    "image", 7),

    # ── Packing Lists (4) ─────────────────────────────────────────────
    # 2 clear PDFs
    ("packing", "clear",  "pdf",   0),
    ("packing", "clear",  "pdf",   1),
    # 2 medium PDFs
    ("packing", "medium", "pdf",   2),
    ("packing", "medium", "pdf",   3),

    # ── Customs Declarations (3) ──────────────────────────────────────
    # 2 clear PDFs
    ("customs", "clear",  "pdf",   0),
    ("customs", "clear",  "pdf",   1),
    # 1 image (PNG)
    ("customs", "low",    "image", 2),
]


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate all 25 sample logistics documents and save them to db/samples/."""

    # Ensure the output directory exists
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    # ── Step 1: Generate all data records ──────────────────────────────
    # We pre-generate the full batch of data for each document type, then
    # pick from the list by index when rendering.
    print("Generating document data...")
    invoice_data = generate_invoice_data(count=10)
    bol_data = generate_bol_data(count=8)
    packing_data = generate_packing_list_data(count=4)
    customs_data = generate_customs_data(count=3)

    # Map from doc_type to (data list, render function)
    type_registry = {
        "invoice": (invoice_data, render_invoice_pdf),
        "bol":     (bol_data, render_bol_pdf),
        "packing": (packing_data, render_packing_list_pdf),
        "customs": (customs_data, render_customs_pdf),
    }

    # ── Step 2: Render and save each document ──────────────────────────
    generated_files = []  # collect (filename, doc_type, quality, format) for summary
    seq_counters = {}     # track per-type sequential numbering for filenames

    print(f"Rendering {len(DOCUMENT_PLAN)} documents...\n")

    for doc_type, quality, output_format, data_idx in DOCUMENT_PLAN:
        # Increment per-type counter for descriptive filename
        seq_counters.setdefault(doc_type, 0)
        seq_counters[doc_type] += 1
        seq_num = seq_counters[doc_type]

        # Look up the data record and render function
        data_list, render_fn = type_registry[doc_type]
        record = data_list[data_idx]

        # Render to PDF bytes
        pdf_bytes = render_fn(record, quality)

        # Determine file extension and apply image conversion if needed
        if output_format == "image":
            ext = "png"
            # Convert the PDF to a degraded PNG
            dpi = random.randint(100, 150)
            file_bytes = pdf_to_degraded_image(pdf_bytes, dpi=dpi, add_noise=True)
        else:
            ext = "pdf"
            file_bytes = pdf_bytes

        # Build the filename:  invoice_clear_001.pdf, bol_medium_002.pdf, etc.
        filename = f"{doc_type}_{quality}_{seq_num:03d}.{ext}"
        filepath = os.path.join(SAMPLES_DIR, filename)

        # Write to disk
        with open(filepath, "wb") as f:
            f.write(file_bytes)

        generated_files.append((filename, doc_type, quality, output_format))

        # Per-file progress
        size_kb = len(file_bytes) / 1024
        print(f"  [{len(generated_files):2d}/25]  {filename:<35s}  ({size_kb:7.1f} KB)")

    # ── Step 3: Print summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GENERATION COMPLETE")
    print("=" * 60)
    print(f"\n  Output directory: {SAMPLES_DIR}")
    print(f"  Total documents:  {len(generated_files)}\n")

    # Count by type
    from collections import Counter
    type_counts = Counter(doc_type for _, doc_type, _, _ in generated_files)
    quality_counts = Counter(quality for _, _, quality, _ in generated_files)
    format_counts = Counter(fmt for _, _, _, fmt in generated_files)

    print("  By document type:")
    for dtype in ["invoice", "bol", "packing", "customs"]:
        label = {
            "invoice": "Invoices",
            "bol": "Bills of Lading",
            "packing": "Packing Lists",
            "customs": "Customs Declarations",
        }[dtype]
        print(f"    {label:<25s}  {type_counts.get(dtype, 0)}")

    print("\n  By quality tier:")
    for q in ["clear", "medium", "low"]:
        print(f"    {q:<25s}  {quality_counts.get(q, 0)}")

    print("\n  By output format:")
    for fmt in ["pdf", "image"]:
        label = "PDF" if fmt == "pdf" else "PNG (image)"
        print(f"    {label:<25s}  {format_counts.get(fmt, 0)}")

    # List all files
    print("\n  All generated files:")
    print("  " + "-" * 56)
    for filename, doc_type, quality, fmt in generated_files:
        print(f"    {filename}")
    print("  " + "-" * 56)
    print()


if __name__ == "__main__":
    main()
