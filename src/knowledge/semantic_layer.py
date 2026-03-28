"""Semantic Layer — rich table/column descriptions for LLM context.

Instead of feeding raw DDL to the LLM, this module provides business-
friendly descriptions with:
- What each table represents in plain English
- What each column means (not just its SQL type)
- Sample values for enum-like columns (e.g., status, mode, tier)
- Relationships explained with real data examples
- 3 golden customers whose data is traced across all tables

The golden customers provide consistent examples across all descriptions,
making it easier for the LLM to understand how tables relate:
- CUST-07585: Busan Food Beverage Co (has delays, ocean+air shipments)
- CUST-00612: New York Textiles Co (all on-time, ocean only)
- CUST-03786: Jebel Ali Textiles Co (varied modes: rail, air, ocean, road)

These descriptions are embedded in ChromaDB (see vector_store.py) and
retrieved at query time via semantic similarity search. This means the
LLM only sees the 3-5 most relevant table descriptions for a given
question, not all 7+ tables.
"""

# ── Table Descriptions ───────────────────────────────────────────────
# Each entry is a document that gets embedded in ChromaDB.
# The "text" field is what gets embedded for similarity search.
# The "metadata" field carries structured info injected into the LLM prompt.

TABLE_DESCRIPTIONS = [
    # ── Customers Table ──────────────────────────────────────────────
    {
        "id": "table_customers",
        "text": (
            "customers table: Companies that ship goods through our logistics platform. "
            "Each customer has an industry (electronics, textiles, automotive, pharmaceuticals, "
            "food_beverage, chemicals, machinery, furniture, apparel, consumer_goods), "
            "a country, a city, and a tier (gold=top 15%, silver=35%, bronze=50% by volume). "
            "Use this table when questions mention company, customer, buyer, client, or industry."
        ),
        "metadata": {
            "table": "customers",
            "columns": "id, customer_code, company_name, industry, country, city, tier, contact_email, annual_volume_teu",
            "row_count": 10000,
            "key_values": "industry: electronics|textiles|automotive|pharmaceuticals|food_beverage|chemicals|machinery|furniture|apparel|consumer_goods. tier: gold|silver|bronze. country: India|China|USA|Germany|Japan|UK|UAE|Singapore|Netherlands|Brazil|South Korea|France",
            "example": "CUST-07585 = 'Busan Food Beverage Co #7585' (South Korea, food_beverage, silver tier)",
        },
    },
    # ── Carriers Table ───────────────────────────────────────────────
    {
        "id": "table_carriers",
        "text": (
            "carriers table: Shipping and logistics companies that transport goods. "
            "30 real carriers including Maersk Line, DHL Express, FedEx Express, COSCO Shipping, "
            "UPS Supply Chain, etc. Each has a type (ocean, air, road, rail, multimodal), "
            "reliability_score (0-1), and cost_per_kg_usd. "
            "Use when questions mention carrier, shipping company, logistics provider, or transport company."
        ),
        "metadata": {
            "table": "carriers",
            "columns": "id, carrier_code, carrier_name, carrier_type, headquarters_country, reliability_score, avg_transit_days, cost_per_kg_usd",
            "row_count": 30,
            "key_values": "carrier_type: ocean|air|road|rail|multimodal. carrier_name examples: Maersk Line, DHL Express, FedEx Express, COSCO Shipping, CMA CGM Group, Hapag-Lloyd, UPS Supply Chain, DB Schenker, Kuehne+Nagel, TransContainer",
            "example": "ZIM Integrated Shipping (ocean, reliability 0.78, $1.70/kg), Cathay Cargo (air, reliability 0.86, $7.80/kg)",
        },
    },
    # ── Shipments Table (CORE) ───────────────────────────────────────
    {
        "id": "table_shipments",
        "text": (
            "shipments table: Every physical shipment from origin to destination. The CORE table — "
            "most analytics questions start here. Contains booking dates, origin/destination ports and countries, "
            "transport mode (ocean/air/road/rail), current status, weight, commodity type, delay info. "
            "Links to customers (who ships), carriers (who transports), and has child records in "
            "shipment_charges, tracking_events, and invoices. "
            "Use for questions about shipments, deliveries, delays, routes, transit times, status, weight, volume."
        ),
        "metadata": {
            "table": "shipments",
            "columns": "id, shipment_id, customer_id, carrier_id, origin_port, origin_country, destination_port, destination_country, mode, status, booking_date, estimated_departure, actual_departure, estimated_arrival, actual_arrival, weight_kg, volume_cbm, container_type, container_count, commodity, incoterm, po_number, delay_days, delay_reason, priority",
            "row_count": 10000,
            "key_values": "mode: ocean(55%)|air(25%)|road(12%)|rail(8%). status: delivered(45%)|in_transit(19%)|delayed(13%)|booked(8%)|picked_up(5%)|cancelled(5%)|customs_hold(4%). delay_reason: port_congestion|customs_hold|weather|vessel_breakdown|documentation_error|labor_strike|equipment_shortage|security_inspection. commodity: electronics|textiles|automotive_parts|pharmaceuticals|frozen_food|chemicals|industrial_machinery|furniture|apparel|consumer_goods|raw_materials|medical_equipment. priority: standard(60%)|express(20%)|economy(20%)",
            "joins": "JOIN carriers ON shipments.carrier_id = carriers.id; JOIN customers ON shipments.customer_id = customers.id",
            "example": "SHP-2024-000734: ocean shipment via ZIM from Singapore→South Korea, delivered with 2-day delay. SHP-2024-003382: air shipment via Cathay Cargo from China→UK, delayed 14 days (exception event logged).",
        },
    },
    # ── Shipment Charges Table ───────────────────────────────────────
    {
        "id": "table_shipment_charges",
        "text": (
            "shipment_charges table: Cost breakdown per shipment. Each shipment has 1-6 charge line items. "
            "EVERY shipment has a 'freight' charge. Additional charges: customs_duty (60% of shipments), "
            "insurance (50%), handling (40%), documentation (30%), demurrage (10%). "
            "Amount is in USD. To get total shipping cost, SUM(amount_usd) grouped by shipment_id. "
            "Use when questions mention cost, expense, charge, freight, customs, insurance, price, spend."
        ),
        "metadata": {
            "table": "shipment_charges",
            "columns": "id, shipment_id, charge_type, description, amount_usd, currency, exchange_rate, amount_original, invoice_reference",
            "row_count": 28936,
            "key_values": "charge_type: freight|customs_duty|insurance|handling|documentation|demurrage. Cost ranges by mode — ocean freight: $1.5K-$15K, air freight: $5K-$50K, road freight: $500-$5K",
            "joins": "JOIN shipments ON shipment_charges.shipment_id = shipments.id",
            "example": "SHP-2024-000734 charges: freight=$9692, customs_duty=$997, insurance=$229, handling=$435, documentation=$96 (total ~$11,449). SHP-2024-004547 charges: freight=$32,336, customs_duty=$4,655, handling=$125 (air shipment, much more expensive).",
        },
    },
    # ── Tracking Events Table ────────────────────────────────────────
    {
        "id": "table_tracking_events",
        "text": (
            "tracking_events table: Status milestones for each shipment over time. "
            "Events follow a sequence: booked → picked_up → departed_origin → in_transit → "
            "arrived_port → customs_cleared → out_for_delivery → delivered. "
            "Delayed shipments also have an 'exception' event with the delay reason. "
            "Each event has a timestamp and location. "
            "Use when questions mention tracking, timeline, events, milestones, transit time, "
            "delivery time, when something happened."
        ),
        "metadata": {
            "table": "tracking_events",
            "columns": "id, shipment_id, event_type, event_timestamp, location, location_country, description, created_by, is_milestone",
            "row_count": 56181,
            "key_values": "event_type: booked|picked_up|departed_origin|in_transit|arrived_port|customs_cleared|out_for_delivery|delivered|exception. created_by: system|carrier_api|manual|edi",
            "joins": "JOIN shipments ON tracking_events.shipment_id = shipments.id",
            "example": "SHP-2024-000734 tracking: booked(2024-02-19) → picked_up(2024-02-21) → departed_origin(Singapore) → in_transit → arrived_port(Port of Busan) → ... → delivered(2024-05-11). SHP-2024-003382 had an 'exception' event at Port of Tilbury due to delay.",
        },
    },
    # ── Invoices Table ───────────────────────────────────────────────
    {
        "id": "table_invoices",
        "text": (
            "invoices table: Billing records, one per shipment. Contains invoice amounts (subtotal + tax = total), "
            "payment status, and dates. The total_usd = sum of all shipment_charges + tax. "
            "Use when questions mention invoice, billing, payment, overdue, revenue, amount due, "
            "accounts receivable, paid/unpaid."
        ),
        "metadata": {
            "table": "invoices",
            "columns": "id, invoice_number, shipment_id, customer_id, carrier_id, invoice_date, due_date, subtotal_usd, tax_usd, total_usd, currency, payment_status, payment_date",
            "row_count": 10000,
            "key_values": "payment_status: paid(50%)|pending(25%)|overdue(15%)|disputed(7%)|cancelled(3%). Invoice range: $564-$69,258, avg $14,450. Tax: 5-18% of subtotal",
            "joins": "JOIN shipments ON invoices.shipment_id = shipments.id; JOIN customers ON invoices.customer_id = customers.id; JOIN carriers ON invoices.carrier_id = carriers.id",
            "example": "INV-2024-000734: $12,839.54, paid (for SHP-2024-000734). INV-2024-003190: $8,530.81, DISPUTED (for rail shipment from India→Netherlands). INV-2024-006279: $30,414.10, OVERDUE (air shipment stuck in customs_hold).",
        },
    },
    # ── Extracted Documents Table ─────────────────────────────────────
    {
        "id": "table_extracted_documents",
        "text": (
            "extracted_documents table: Data extracted from uploaded PDF/image documents by the vision AI agent. "
            "Stores structured fields as JSON (flexible schema for different document types). "
            "Fields have confidence scores. Documents can be linked to shipments. "
            "Use when questions mention uploaded documents, extracted data, invoices from PDFs, "
            "bills of lading, document confidence."
        ),
        "metadata": {
            "table": "extracted_documents",
            "columns": "id, document_id, document_type, file_name, upload_timestamp, extraction_model, overall_confidence, review_status, reviewed_at, extracted_fields (JSON), confidence_scores (JSON), raw_text, linked_shipment_id, notes",
            "row_count": 0,
            "key_values": "document_type: invoice|bill_of_lading|packing_list|customs_declaration. review_status: pending|approved|rejected|corrected. Query JSON fields with: json_extract(extracted_fields, '$.field_name')",
            "joins": "LEFT JOIN shipments ON extracted_documents.linked_shipment_id = shipments.shipment_id",
            "example": "This table starts empty. Documents are added when users upload PDFs via the vision agent. Example: an uploaded Maersk invoice might have extracted_fields = {\"invoice_number\": \"INV-EXT-001\", \"vendor\": \"Maersk\", \"total_amount\": 5000.00}",
        },
    },
]

# ── Cross-Table Relationship Descriptions ────────────────────────────
# These are embedded alongside table descriptions in ChromaDB.
# When a user asks a JOIN-related question (e.g., "carrier with most delays"),
# the relationship description helps the LLM understand which tables to join
# and what join keys to use.

RELATIONSHIP_DESCRIPTIONS = [
    {
        "id": "rel_customer_shipment",
        "text": (
            "Relationship: customers → shipments. A customer can have many shipments. "
            "JOIN: shipments.customer_id = customers.id. "
            "Example: CUST-07585 (Busan Food Beverage Co) has 5 shipments including "
            "SHP-2024-000734 (ocean via ZIM) and SHP-2024-003382 (air via Cathay Cargo, delayed 14 days). "
            "Use this JOIN when you need customer name, industry, tier, or country alongside shipment data."
        ),
        "metadata": {"from_table": "customers", "to_table": "shipments", "join_key": "shipments.customer_id = customers.id"},
    },
    {
        "id": "rel_carrier_shipment",
        "text": (
            "Relationship: carriers → shipments. A carrier handles many shipments. "
            "JOIN: shipments.carrier_id = carriers.id. "
            "Example: ZIM Integrated Shipping (ocean carrier) handles SHP-2024-000734. "
            "Cathay Cargo (air carrier) handles SHP-2024-003382. "
            "Use this JOIN when you need carrier name, type, reliability score alongside shipment data."
        ),
        "metadata": {"from_table": "carriers", "to_table": "shipments", "join_key": "shipments.carrier_id = carriers.id"},
    },
    {
        "id": "rel_shipment_charges",
        "text": (
            "Relationship: shipments → shipment_charges. Each shipment has 1-6 cost line items. "
            "JOIN: shipment_charges.shipment_id = shipments.id. "
            "Example: SHP-2024-000734 has 5 charges totaling ~$11,449 (freight $9692 + customs $997 + insurance $229 + handling $435 + docs $96). "
            "To get total cost per shipment: SELECT shipment_id, SUM(amount_usd) FROM shipment_charges GROUP BY shipment_id. "
            "Use this JOIN for any cost, expense, or financial analysis."
        ),
        "metadata": {"from_table": "shipments", "to_table": "shipment_charges", "join_key": "shipment_charges.shipment_id = shipments.id"},
    },
    {
        "id": "rel_shipment_tracking",
        "text": (
            "Relationship: shipments → tracking_events. Each shipment has 1-8 tracking milestones. "
            "JOIN: tracking_events.shipment_id = shipments.id. "
            "Example: SHP-2024-000734 has events from booked(2024-02-19) through delivered(2024-05-11). "
            "To calculate transit time: date difference between 'departed_origin' and 'delivered' events. "
            "Use this JOIN for timeline analysis, transit time calculations, milestone tracking."
        ),
        "metadata": {"from_table": "shipments", "to_table": "tracking_events", "join_key": "tracking_events.shipment_id = shipments.id"},
    },
    {
        "id": "rel_shipment_invoices",
        "text": (
            "Relationship: shipments → invoices. One invoice per shipment. "
            "JOIN: invoices.shipment_id = shipments.id. "
            "Example: SHP-2024-000734 → INV-2024-000734 ($12,839.54, paid). "
            "SHP-2024-003190 → INV-2024-003190 ($8,530.81, DISPUTED). "
            "Use this JOIN for billing, payment status, revenue, overdue analysis."
        ),
        "metadata": {"from_table": "shipments", "to_table": "invoices", "join_key": "invoices.shipment_id = shipments.id"},
    },
    {
        "id": "rel_customer_invoices",
        "text": (
            "Relationship: customers → invoices (direct). "
            "JOIN: invoices.customer_id = customers.id. "
            "Use for customer billing analysis without going through shipments table. "
            "Example: CUST-03786 has invoice INV-2024-006279 ($30,414.10, OVERDUE)."
        ),
        "metadata": {"from_table": "customers", "to_table": "invoices", "join_key": "invoices.customer_id = customers.id"},
    },
]
