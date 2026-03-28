CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_code TEXT UNIQUE NOT NULL,
    company_name TEXT NOT NULL,
    industry TEXT NOT NULL,
    country TEXT NOT NULL,
    city TEXT NOT NULL,
    tier TEXT NOT NULL,
    contact_email TEXT,
    annual_volume_teu INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS carriers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    carrier_code TEXT UNIQUE NOT NULL,
    carrier_name TEXT NOT NULL,
    carrier_type TEXT NOT NULL,
    headquarters_country TEXT,
    reliability_score REAL,
    avg_transit_days INTEGER,
    cost_per_kg_usd REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS shipments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_id TEXT UNIQUE NOT NULL,
    customer_id INTEGER NOT NULL,
    carrier_id INTEGER NOT NULL,
    origin_port TEXT NOT NULL,
    origin_country TEXT NOT NULL,
    destination_port TEXT NOT NULL,
    destination_country TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    booking_date DATE NOT NULL,
    estimated_departure DATE,
    actual_departure DATE,
    estimated_arrival DATE,
    actual_arrival DATE,
    weight_kg REAL NOT NULL,
    volume_cbm REAL,
    container_type TEXT,
    container_count INTEGER DEFAULT 1,
    commodity TEXT NOT NULL,
    incoterm TEXT,
    po_number TEXT,
    delay_days INTEGER DEFAULT 0,
    delay_reason TEXT,
    priority TEXT DEFAULT 'standard',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (carrier_id) REFERENCES carriers(id)
);

CREATE TABLE IF NOT EXISTS shipment_charges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_id INTEGER NOT NULL,
    charge_type TEXT NOT NULL,
    description TEXT,
    amount_usd REAL NOT NULL,
    currency TEXT DEFAULT 'USD',
    exchange_rate REAL DEFAULT 1.0,
    amount_original REAL,
    invoice_reference TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (shipment_id) REFERENCES shipments(id)
);

CREATE TABLE IF NOT EXISTS tracking_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_timestamp TIMESTAMP NOT NULL,
    location TEXT NOT NULL,
    location_country TEXT,
    description TEXT,
    created_by TEXT DEFAULT 'system',
    is_milestone BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (shipment_id) REFERENCES shipments(id)
);

CREATE TABLE IF NOT EXISTS invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_number TEXT UNIQUE NOT NULL,
    shipment_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    carrier_id INTEGER NOT NULL,
    invoice_date DATE NOT NULL,
    due_date DATE NOT NULL,
    subtotal_usd REAL NOT NULL,
    tax_usd REAL DEFAULT 0,
    total_usd REAL NOT NULL,
    currency TEXT DEFAULT 'USD',
    payment_status TEXT DEFAULT 'pending',
    payment_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (shipment_id) REFERENCES shipments(id),
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (carrier_id) REFERENCES carriers(id)
);

CREATE TABLE IF NOT EXISTS extracted_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT UNIQUE NOT NULL,
    document_type TEXT NOT NULL,
    file_name TEXT NOT NULL,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    extraction_model TEXT,
    overall_confidence REAL,
    review_status TEXT DEFAULT 'pending',
    reviewed_at TIMESTAMP,
    extracted_fields JSON NOT NULL,
    confidence_scores JSON NOT NULL,
    raw_text TEXT,
    linked_shipment_id TEXT,
    notes TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_customers_industry ON customers(industry);
CREATE INDEX IF NOT EXISTS idx_customers_country ON customers(country);
CREATE INDEX IF NOT EXISTS idx_customers_tier ON customers(tier);
CREATE INDEX IF NOT EXISTS idx_shipments_customer ON shipments(customer_id);
CREATE INDEX IF NOT EXISTS idx_shipments_carrier ON shipments(carrier_id);
CREATE INDEX IF NOT EXISTS idx_shipments_status ON shipments(status);
CREATE INDEX IF NOT EXISTS idx_shipments_mode ON shipments(mode);
CREATE INDEX IF NOT EXISTS idx_shipments_origin ON shipments(origin_country);
CREATE INDEX IF NOT EXISTS idx_shipments_dest ON shipments(destination_country);
CREATE INDEX IF NOT EXISTS idx_shipments_booking ON shipments(booking_date);
CREATE INDEX IF NOT EXISTS idx_shipments_commodity ON shipments(commodity);
CREATE INDEX IF NOT EXISTS idx_charges_shipment ON shipment_charges(shipment_id);
CREATE INDEX IF NOT EXISTS idx_charges_type ON shipment_charges(charge_type);
CREATE INDEX IF NOT EXISTS idx_tracking_shipment ON tracking_events(shipment_id);
CREATE INDEX IF NOT EXISTS idx_tracking_type ON tracking_events(event_type);
CREATE INDEX IF NOT EXISTS idx_tracking_timestamp ON tracking_events(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_invoices_shipment ON invoices(shipment_id);
CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices(customer_id);
CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(payment_status);
CREATE INDEX IF NOT EXISTS idx_invoices_date ON invoices(invoice_date);
CREATE INDEX IF NOT EXISTS idx_extracted_type ON extracted_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_extracted_review ON extracted_documents(review_status);
CREATE INDEX IF NOT EXISTS idx_extracted_linked ON extracted_documents(linked_shipment_id);
