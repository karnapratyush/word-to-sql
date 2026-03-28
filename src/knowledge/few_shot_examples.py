"""Few-shot examples — curated question-to-SQL pairs for accuracy improvement.

These examples are embedded in ChromaDB alongside schema descriptions
(see vector_store.py). At query time, the 3 most semantically similar
examples are retrieved and injected into the SQL generator prompt.

The LLM sees "here's how similar questions were answered" before
generating SQL. This technique improves accuracy by ~7% (measured
in research: nilenso blog, 2025) because:
- The LLM learns the correct table/column names from examples
- JOIN patterns and GROUP BY conventions are demonstrated
- Edge cases (HAVING, CASE WHEN, subqueries) are shown explicitly

Examples use our golden customers for consistency:
- CUST-07585: Busan Food Beverage Co (delays, ocean+air)
- CUST-00612: New York Textiles Co (all on-time)
- CUST-03786: Jebel Ali Textiles Co (all 4 transport modes)

Each example has:
- id: Unique identifier for ChromaDB storage
- question: Natural language question (embedded for similarity search)
- sql: The correct SQL answer
- tables: Which tables are used (for metadata filtering)
- complexity: simple/medium/complex (for analytics)
"""

FEW_SHOT_EXAMPLES = [
    # ── Simple counts and aggregations ────────────────────────────
    {
        "id": "fs_count_shipments",
        "question": "How many shipments are there?",
        "sql": "SELECT COUNT(*) as total_shipments FROM shipments",
        "tables": ["shipments"],
        "complexity": "simple",
    },
    {
        "id": "fs_count_delayed",
        "question": "How many shipments are delayed?",
        "sql": "SELECT COUNT(*) as delayed_shipments FROM shipments WHERE status = 'delayed'",
        "tables": ["shipments"],
        "complexity": "simple",
    },
    {
        "id": "fs_count_by_status",
        "question": "Count shipments by status",
        "sql": "SELECT status, COUNT(*) as count FROM shipments GROUP BY status ORDER BY count DESC",
        "tables": ["shipments"],
        "complexity": "simple",
    },
    {
        "id": "fs_count_by_mode",
        "question": "How many shipments per transport mode?",
        "sql": "SELECT mode, COUNT(*) as shipment_count FROM shipments GROUP BY mode ORDER BY shipment_count DESC",
        "tables": ["shipments"],
        "complexity": "simple",
    },

    # ── JOINs with carriers ───────────────────────────────────────
    # These teach the LLM the correct join pattern for carrier data
    {
        "id": "fs_top_carriers",
        "question": "Which carriers have the most shipments?",
        "sql": (
            "SELECT c.carrier_name, c.carrier_type, COUNT(*) as shipment_count "
            "FROM shipments s "
            "JOIN carriers c ON s.carrier_id = c.id "
            "GROUP BY c.id, c.carrier_name, c.carrier_type "
            "ORDER BY shipment_count DESC LIMIT 10"
        ),
        "tables": ["shipments", "carriers"],
        "complexity": "simple",
    },
    {
        "id": "fs_carrier_delays",
        "question": "Which carrier has the most delayed shipments?",
        "sql": (
            "SELECT c.carrier_name, COUNT(*) as delayed_count "
            "FROM shipments s "
            "JOIN carriers c ON s.carrier_id = c.id "
            "WHERE s.delay_days > 0 "
            "GROUP BY c.carrier_name "
            "ORDER BY delayed_count DESC LIMIT 10"
        ),
        "tables": ["shipments", "carriers"],
        "complexity": "simple",
    },
    {
        "id": "fs_delay_rate_by_carrier",
        "question": "What is the delay rate by carrier?",
        "sql": (
            "SELECT c.carrier_name, "
            "COUNT(*) as total, "
            "SUM(CASE WHEN s.delay_days > 0 THEN 1 ELSE 0 END) as delayed, "
            "ROUND(100.0 * SUM(CASE WHEN s.delay_days > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as delay_rate_pct "
            "FROM shipments s "
            "JOIN carriers c ON s.carrier_id = c.id "
            "GROUP BY c.carrier_name "
            "ORDER BY delay_rate_pct DESC LIMIT 10"
        ),
        "tables": ["shipments", "carriers"],
        "complexity": "medium",
    },

    # ── Cost analysis (requires shipment_charges) ─────────────────
    # These teach the LLM to use the shipment_charges table for cost queries
    {
        "id": "fs_avg_freight_by_carrier",
        "question": "What is the average freight cost by carrier?",
        "sql": (
            "SELECT c.carrier_name, ROUND(AVG(sc.amount_usd), 2) as avg_freight "
            "FROM shipment_charges sc "
            "JOIN shipments s ON sc.shipment_id = s.id "
            "JOIN carriers c ON s.carrier_id = c.id "
            "WHERE sc.charge_type = 'freight' "
            "GROUP BY c.carrier_name "
            "ORDER BY avg_freight DESC LIMIT 10"
        ),
        "tables": ["shipment_charges", "shipments", "carriers"],
        "complexity": "medium",
    },
    {
        "id": "fs_total_cost_by_mode",
        "question": "What is the total shipping cost by transport mode?",
        "sql": (
            "SELECT s.mode, ROUND(SUM(sc.amount_usd), 2) as total_cost "
            "FROM shipment_charges sc "
            "JOIN shipments s ON sc.shipment_id = s.id "
            "GROUP BY s.mode "
            "ORDER BY total_cost DESC"
        ),
        "tables": ["shipment_charges", "shipments"],
        "complexity": "simple",
    },
    {
        "id": "fs_cost_breakdown",
        "question": "Show the cost breakdown by charge type",
        "sql": (
            "SELECT charge_type, COUNT(*) as count, "
            "ROUND(AVG(amount_usd), 2) as avg_amount, "
            "ROUND(SUM(amount_usd), 2) as total_amount "
            "FROM shipment_charges "
            "GROUP BY charge_type "
            "ORDER BY total_amount DESC"
        ),
        "tables": ["shipment_charges"],
        "complexity": "simple",
    },
    {
        "id": "fs_cost_per_kg",
        "question": "What is the average freight cost per kg by mode?",
        "sql": (
            "SELECT s.mode, "
            "ROUND(SUM(sc.amount_usd) / SUM(s.weight_kg), 2) as cost_per_kg "
            "FROM shipment_charges sc "
            "JOIN shipments s ON sc.shipment_id = s.id "
            "WHERE sc.charge_type = 'freight' "
            "GROUP BY s.mode "
            "ORDER BY cost_per_kg DESC"
        ),
        "tables": ["shipment_charges", "shipments"],
        "complexity": "medium",
    },

    # ── Route analysis ────────────────────────────────────────────
    {
        "id": "fs_top_routes",
        "question": "What are the top routes by shipment volume?",
        "sql": (
            "SELECT origin_country, destination_country, COUNT(*) as shipment_count "
            "FROM shipments "
            "GROUP BY origin_country, destination_country "
            "ORDER BY shipment_count DESC LIMIT 10"
        ),
        "tables": ["shipments"],
        "complexity": "simple",
    },
    {
        "id": "fs_route_delay",
        "question": "Which routes have the highest average delay?",
        "sql": (
            "SELECT origin_country, destination_country, "
            "ROUND(AVG(delay_days), 1) as avg_delay, COUNT(*) as total "
            "FROM shipments "
            "WHERE delay_days > 0 "
            "GROUP BY origin_country, destination_country "
            "HAVING total >= 5 "
            "ORDER BY avg_delay DESC LIMIT 10"
        ),
        "tables": ["shipments"],
        "complexity": "medium",
    },

    # ── Time-based queries ────────────────────────────────────────
    # These teach the LLM SQLite's strftime() for date grouping
    {
        "id": "fs_monthly_shipments",
        "question": "Show monthly shipment count trend",
        "sql": (
            "SELECT strftime('%Y-%m', booking_date) as month, COUNT(*) as count "
            "FROM shipments "
            "GROUP BY month "
            "ORDER BY month"
        ),
        "tables": ["shipments"],
        "complexity": "simple",
    },
    {
        "id": "fs_quarterly_revenue",
        "question": "What is the quarterly revenue from invoices?",
        "sql": (
            "SELECT strftime('%Y', invoice_date) || '-Q' || ((CAST(strftime('%m', invoice_date) AS INTEGER) - 1) / 3 + 1) as quarter, "
            "ROUND(SUM(total_usd), 2) as revenue "
            "FROM invoices "
            "WHERE payment_status != 'cancelled' "
            "GROUP BY quarter "
            "ORDER BY quarter"
        ),
        "tables": ["invoices"],
        "complexity": "medium",
    },

    # ── Invoice / payment queries ─────────────────────────────────
    {
        "id": "fs_payment_status",
        "question": "Show invoice count by payment status",
        "sql": (
            "SELECT payment_status, COUNT(*) as count, ROUND(SUM(total_usd), 2) as total_amount "
            "FROM invoices "
            "GROUP BY payment_status "
            "ORDER BY count DESC"
        ),
        "tables": ["invoices"],
        "complexity": "simple",
    },
    {
        "id": "fs_overdue_invoices",
        "question": "Which customers have the most overdue invoices?",
        "sql": (
            "SELECT cu.company_name, cu.country, COUNT(*) as overdue_count, "
            "ROUND(SUM(i.total_usd), 2) as overdue_amount "
            "FROM invoices i "
            "JOIN customers cu ON i.customer_id = cu.id "
            "WHERE i.payment_status = 'overdue' "
            "GROUP BY cu.id, cu.company_name, cu.country "
            "ORDER BY overdue_amount DESC LIMIT 10"
        ),
        "tables": ["invoices", "customers"],
        "complexity": "medium",
    },

    # ── Customer analysis ─────────────────────────────────────────
    {
        "id": "fs_top_customers_spend",
        "question": "Who are the top customers by total shipping spend?",
        "sql": (
            "SELECT cu.company_name, cu.country, cu.tier, "
            "ROUND(SUM(i.total_usd), 2) as total_spend, "
            "COUNT(DISTINCT s.id) as shipment_count "
            "FROM customers cu "
            "JOIN shipments s ON cu.id = s.customer_id "
            "JOIN invoices i ON s.id = i.shipment_id "
            "GROUP BY cu.id, cu.company_name, cu.country, cu.tier "
            "ORDER BY total_spend DESC LIMIT 10"
        ),
        "tables": ["customers", "shipments", "invoices"],
        "complexity": "complex",
    },
    {
        "id": "fs_customer_by_industry",
        "question": "How many customers per industry?",
        "sql": (
            "SELECT industry, COUNT(*) as customer_count "
            "FROM customers "
            "GROUP BY industry "
            "ORDER BY customer_count DESC"
        ),
        "tables": ["customers"],
        "complexity": "simple",
    },

    # ── Complex multi-table queries ───────────────────────────────
    {
        "id": "fs_delay_reasons",
        "question": "What are the most common delay reasons?",
        "sql": (
            "SELECT delay_reason, COUNT(*) as count "
            "FROM shipments "
            "WHERE delay_reason IS NOT NULL "
            "GROUP BY delay_reason "
            "ORDER BY count DESC"
        ),
        "tables": ["shipments"],
        "complexity": "simple",
    },
    {
        "id": "fs_carrier_performance",
        "question": "Compare carrier performance: on-time delivery rate and average cost",
        "sql": (
            "SELECT c.carrier_name, c.carrier_type, "
            "COUNT(*) as total_shipments, "
            "ROUND(100.0 * SUM(CASE WHEN s.delay_days = 0 AND s.status = 'delivered' THEN 1 ELSE 0 END) / "
            "NULLIF(SUM(CASE WHEN s.status = 'delivered' THEN 1 ELSE 0 END), 0), 1) as ontime_pct, "
            "ROUND(AVG(sc_total.total_cost), 2) as avg_cost "
            "FROM carriers c "
            "JOIN shipments s ON c.id = s.carrier_id "
            "LEFT JOIN (SELECT shipment_id, SUM(amount_usd) as total_cost FROM shipment_charges GROUP BY shipment_id) sc_total "
            "ON s.id = sc_total.shipment_id "
            "GROUP BY c.id, c.carrier_name, c.carrier_type "
            "ORDER BY total_shipments DESC LIMIT 10"
        ),
        "tables": ["carriers", "shipments", "shipment_charges"],
        "complexity": "complex",
    },
    {
        "id": "fs_commodity_analysis",
        "question": "Which commodities are shipped most and by what mode?",
        "sql": (
            "SELECT commodity, mode, COUNT(*) as count "
            "FROM shipments "
            "GROUP BY commodity, mode "
            "ORDER BY count DESC LIMIT 20"
        ),
        "tables": ["shipments"],
        "complexity": "simple",
    },

    # ── Filtering examples ────────────────────────────────────────
    # These teach the LLM how to apply WHERE filters for specific countries/dates
    {
        "id": "fs_filter_country",
        "question": "Show all delayed shipments from India",
        "sql": (
            "SELECT s.shipment_id, s.origin_port, s.destination_country, "
            "c.carrier_name, s.commodity, s.delay_days, s.delay_reason "
            "FROM shipments s "
            "JOIN carriers c ON s.carrier_id = c.id "
            "WHERE s.origin_country = 'India' AND s.delay_days > 0 "
            "ORDER BY s.delay_days DESC LIMIT 20"
        ),
        "tables": ["shipments", "carriers"],
        "complexity": "simple",
    },
    {
        "id": "fs_filter_date",
        "question": "Shipments booked in January 2025",
        "sql": (
            "SELECT shipment_id, mode, status, origin_country, destination_country, booking_date "
            "FROM shipments "
            "WHERE booking_date >= '2025-01-01' AND booking_date < '2025-02-01' "
            "ORDER BY booking_date LIMIT 20"
        ),
        "tables": ["shipments"],
        "complexity": "simple",
    },

    # ── Extracted Documents / JSON queries ────────────────────────────
    # These examples teach the LLM how to query the extracted_documents table
    # using json_extract() for fields stored in the JSON column.

    {
        "id": "fs_extracted_bl_lookup",
        "question": "Give me details about BL number BL-2024-KR-US-0841",
        "sql": (
            "SELECT document_type, overall_confidence, review_status, "
            "bl_number, shipper_name, consignee_name, vessel_name, "
            "port_of_loading, port_of_discharge, gross_weight "
            "FROM extracted_document_fields "
            "WHERE bl_number = 'BL-2024-KR-US-0841' "
            "LIMIT 20"
        ),
        "tables": ["extracted_document_fields"],
        "complexity": "medium",
    },
    {
        "id": "fs_extracted_all_docs",
        "question": "Show all extracted documents",
        "sql": (
            "SELECT document_id, document_type, file_name, overall_confidence, "
            "review_status, upload_timestamp "
            "FROM extracted_documents "
            "ORDER BY upload_timestamp DESC LIMIT 20"
        ),
        "tables": ["extracted_documents"],
        "complexity": "simple",
    },
    {
        "id": "fs_extracted_invoice_amounts",
        "question": "What is the total amount from extracted invoices?",
        "sql": (
            "SELECT invoice_number, total_amount, vendor_name "
            "FROM extracted_document_fields "
            "WHERE document_type = 'invoice' "
            "LIMIT 20"
        ),
        "tables": ["extracted_document_fields"],
        "complexity": "medium",
    },
    {
        "id": "fs_extracted_linked_shipments",
        "question": "Which shipments have linked documents?",
        "sql": (
            "SELECT s.shipment_id, s.status, s.mode, s.origin_country, s.destination_country, "
            "edf.document_type, edf.overall_confidence, edf.review_status "
            "FROM extracted_document_fields edf "
            "JOIN shipments s ON edf.linked_shipment_id = s.shipment_id "
            "LIMIT 20"
        ),
        "tables": ["extracted_document_fields", "shipments"],
        "complexity": "medium",
    },
    {
        "id": "fs_extracted_low_confidence",
        "question": "Show documents with confidence below 0.8",
        "sql": (
            "SELECT document_type, file_name, overall_confidence, review_status "
            "FROM extracted_document_fields "
            "WHERE overall_confidence < 0.8 "
            "ORDER BY overall_confidence ASC LIMIT 20"
        ),
        "tables": ["extracted_document_fields"],
        "complexity": "simple",
    },
]
