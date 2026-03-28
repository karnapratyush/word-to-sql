"""Seed the logistics database with realistic data.

Tables seeded:
  - carriers: 30 rows
  - customers: 10,000 rows
  - shipments: 10,000 rows
  - shipment_charges: ~30,000 rows
  - tracking_events: ~40,000 rows
  - invoices: 10,000 rows
"""

import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "logistics.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "db", "schema.sql")

# ── Reference data ──────────────────────────────────────────────────

CARRIERS = [
    ("MAERSK", "Maersk Line", "ocean", "Denmark", 0.87, 28, 2.10),
    ("MSC", "Mediterranean Shipping Company", "ocean", "Switzerland", 0.82, 30, 1.95),
    ("COSCO", "COSCO Shipping", "ocean", "China", 0.80, 32, 1.80),
    ("CMA_CGM", "CMA CGM Group", "ocean", "France", 0.85, 29, 2.05),
    ("HAPAG", "Hapag-Lloyd", "ocean", "Germany", 0.88, 27, 2.20),
    ("EVERGREEN", "Evergreen Marine", "ocean", "Taiwan", 0.81, 31, 1.90),
    ("ONE", "Ocean Network Express", "ocean", "Japan", 0.83, 29, 2.00),
    ("YANG_MING", "Yang Ming Marine", "ocean", "Taiwan", 0.79, 33, 1.75),
    ("ZIM", "ZIM Integrated Shipping", "ocean", "Israel", 0.78, 34, 1.70),
    ("HMM", "HMM Co Ltd", "ocean", "South Korea", 0.80, 31, 1.85),
    ("PIL", "Pacific International Lines", "ocean", "Singapore", 0.76, 35, 1.65),
    ("WAN_HAI", "Wan Hai Lines", "ocean", "Taiwan", 0.77, 34, 1.72),
    ("DHL_EXPRESS", "DHL Express", "air", "Germany", 0.91, 3, 8.50),
    ("DHL_FREIGHT", "DHL Global Forwarding", "multimodal", "Germany", 0.86, 14, 3.20),
    ("FEDEX", "FedEx Express", "air", "USA", 0.90, 3, 9.00),
    ("UPS", "UPS Supply Chain", "air", "USA", 0.89, 4, 8.00),
    ("TNT", "TNT Express", "air", "Netherlands", 0.85, 4, 7.50),
    ("EMIRATES_SC", "Emirates SkyCargo", "air", "UAE", 0.88, 3, 8.80),
    ("CATHAY_CARGO", "Cathay Cargo", "air", "Hong Kong", 0.86, 4, 7.80),
    ("SINGAPORE_AIR", "Singapore Airlines Cargo", "air", "Singapore", 0.87, 3, 8.20),
    ("DB_SCHENKER", "DB Schenker", "multimodal", "Germany", 0.84, 12, 3.50),
    ("KUEHNE", "Kuehne+Nagel", "multimodal", "Switzerland", 0.86, 11, 3.80),
    ("PANALPINA", "DSV Panalpina", "multimodal", "Denmark", 0.83, 13, 3.30),
    ("EXPEDITORS", "Expeditors International", "multimodal", "USA", 0.82, 14, 3.10),
    ("XPO", "XPO Logistics", "road", "USA", 0.82, 5, 1.50),
    ("JB_HUNT", "J.B. Hunt Transport", "road", "USA", 0.80, 6, 1.40),
    ("SCHNEIDER", "Schneider National", "road", "USA", 0.79, 6, 1.35),
    ("RYDER", "Ryder System", "road", "USA", 0.78, 7, 1.30),
    ("TRANSPLACE", "Uber Freight (Transplace)", "road", "USA", 0.77, 7, 1.25),
    ("INTERMODAL_RUS", "TransContainer", "rail", "Russia", 0.72, 18, 0.90),
]

INDUSTRIES = [
    "electronics", "textiles", "automotive", "pharmaceuticals",
    "food_beverage", "chemicals", "machinery", "furniture",
    "apparel", "consumer_goods",
]

COUNTRIES_CITIES = {
    "India": ["Mumbai", "Delhi", "Chennai", "Kolkata", "Bengaluru", "Ahmedabad"],
    "China": ["Shanghai", "Shenzhen", "Guangzhou", "Ningbo", "Qingdao", "Tianjin"],
    "USA": ["Los Angeles", "New York", "Chicago", "Houston", "Seattle", "Miami"],
    "Germany": ["Hamburg", "Frankfurt", "Munich", "Bremen", "Duisburg", "Berlin"],
    "Japan": ["Tokyo", "Yokohama", "Osaka", "Kobe", "Nagoya", "Fukuoka"],
    "UK": ["London", "Liverpool", "Southampton", "Manchester", "Birmingham", "Glasgow"],
    "UAE": ["Dubai", "Abu Dhabi", "Sharjah", "Jebel Ali", "Fujairah", "Ajman"],
    "Singapore": ["Singapore", "Jurong", "Tuas", "Changi", "Pasir Panjang", "Sembawang"],
    "Netherlands": ["Rotterdam", "Amsterdam", "The Hague", "Utrecht", "Eindhoven", "Tilburg"],
    "Brazil": ["Santos", "Sao Paulo", "Rio de Janeiro", "Paranagua", "Itajai", "Manaus"],
    "South Korea": ["Busan", "Seoul", "Incheon", "Ulsan", "Gwangyang", "Pyeongtaek"],
    "France": ["Le Havre", "Marseille", "Paris", "Lyon", "Dunkirk", "Bordeaux"],
}

PORTS = {
    "India": ["Nhava Sheva (JNPT)", "Mundra Port", "Chennai Port", "Kolkata Port", "Visakhapatnam Port", "Cochin Port"],
    "China": ["Shanghai Port", "Shenzhen Port", "Ningbo-Zhoushan Port", "Guangzhou Port", "Qingdao Port", "Tianjin Port"],
    "USA": ["Port of Los Angeles", "Port of Long Beach", "Port of New York", "Port of Houston", "Port of Seattle", "Port of Miami"],
    "Germany": ["Port of Hamburg", "Port of Bremerhaven", "Port of Wilhelmshaven", "Duisburg Inland Port", "Port of Rostock", "Port of Lubeck"],
    "Japan": ["Port of Tokyo", "Port of Yokohama", "Port of Kobe", "Port of Osaka", "Port of Nagoya", "Port of Hakata"],
    "UK": ["Port of Felixstowe", "Port of Southampton", "Port of London Gateway", "Port of Liverpool", "Port of Tilbury", "Port of Immingham"],
    "UAE": ["Jebel Ali Port", "Port of Fujairah", "Port of Abu Dhabi", "Sharjah Port", "Khor Fakkan Port", "Ras Al Khaimah Port"],
    "Singapore": ["Port of Singapore", "Jurong Port", "Tuas Port", "Pasir Panjang Terminal", "Keppel Terminal", "Brani Terminal"],
    "Netherlands": ["Port of Rotterdam", "Port of Amsterdam", "Port of Moerdijk", "Port of Vlissingen", "Port of Delfzijl", "Port of Terneuzen"],
    "Brazil": ["Port of Santos", "Port of Paranagua", "Port of Itajai", "Port of Rio Grande", "Port of Suape", "Port of Manaus"],
    "South Korea": ["Port of Busan", "Port of Incheon", "Port of Gwangyang", "Port of Ulsan", "Port of Pyeongtaek", "Port of Mokpo"],
    "France": ["Port of Le Havre", "Port of Marseille", "Port of Dunkirk", "Port of Nantes", "Port of Bordeaux", "Port of Rouen"],
}

TIERS_WEIGHTS = {"gold": 15, "silver": 35, "bronze": 50}

COMMODITIES = [
    "electronics", "textiles", "automotive_parts", "pharmaceuticals",
    "frozen_food", "chemicals", "industrial_machinery", "furniture",
    "apparel", "consumer_goods", "raw_materials", "medical_equipment",
]

INCOTERMS = ["FOB", "CIF", "EXW", "DDP", "CFR", "FCA", "DAP", "CPT"]

STATUSES = ["booked", "picked_up", "in_transit", "customs_hold", "delivered", "delayed", "cancelled"]
STATUS_WEIGHTS = [8, 5, 20, 4, 45, 13, 5]

MODES = ["ocean", "air", "road", "rail"]
MODE_WEIGHTS = [55, 25, 12, 8]

DELAY_REASONS = [
    "port_congestion", "customs_hold", "weather", "vessel_breakdown",
    "documentation_error", "labor_strike", "equipment_shortage",
    "security_inspection",
]

CONTAINER_TYPES = ["20ft", "40ft", "40ft_hc", "LCL"]

CHARGE_TYPES = ["freight", "customs_duty", "insurance", "handling", "documentation", "demurrage", "detention"]

TRACKING_EVENTS_SEQ = [
    "booked", "picked_up", "departed_origin", "in_transit",
    "arrived_port", "customs_cleared", "out_for_delivery", "delivered",
]

PAYMENT_STATUSES = ["pending", "paid", "overdue", "disputed", "cancelled"]
PAYMENT_WEIGHTS = [25, 50, 15, 7, 3]

DATE_START = datetime(2023, 6, 1)
DATE_END = datetime(2025, 3, 27)
DATE_RANGE_DAYS = (DATE_END - DATE_START).days


def random_date(start=DATE_START, days_range=DATE_RANGE_DAYS):
    return start + timedelta(days=random.randint(0, days_range))


def random_email(company_name: str) -> str:
    domain = company_name.lower().replace(" ", "").replace(".", "")[:12]
    return f"contact@{domain}.com"


def seed_carriers(cursor):
    print("Seeding carriers...")
    for c in CARRIERS:
        cursor.execute(
            "INSERT INTO carriers (carrier_code, carrier_name, carrier_type, "
            "headquarters_country, reliability_score, avg_transit_days, cost_per_kg_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            c,
        )
    print(f"  -> {len(CARRIERS)} carriers")


def seed_customers(cursor, n=10_000):
    print(f"Seeding {n} customers...")
    tiers = random.choices(
        list(TIERS_WEIGHTS.keys()),
        weights=list(TIERS_WEIGHTS.values()),
        k=n,
    )
    countries = list(COUNTRIES_CITIES.keys())
    for i in range(1, n + 1):
        country = random.choice(countries)
        city = random.choice(COUNTRIES_CITIES[country])
        industry = random.choice(INDUSTRIES)
        company = f"{city} {industry.replace('_', ' ').title()} Co #{i}"
        cursor.execute(
            "INSERT INTO customers (customer_code, company_name, industry, country, "
            "city, tier, contact_email, annual_volume_teu) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"CUST-{i:05d}",
                company,
                industry,
                country,
                city,
                tiers[i - 1],
                random_email(company),
                random.randint(10, 5000),
            ),
        )
    print(f"  -> {n} customers")


def _carrier_ids_by_mode(cursor) -> dict[str, list[int]]:
    cursor.execute("SELECT id, carrier_type FROM carriers")
    by_mode = {}
    for row in cursor.fetchall():
        ctype = row[1]
        by_mode.setdefault(ctype, []).append(row[0])
    # Map shipment modes to carrier types
    return {
        "ocean": by_mode.get("ocean", []),
        "air": by_mode.get("air", []),
        "road": by_mode.get("road", []),
        "rail": by_mode.get("rail", []) or by_mode.get("multimodal", []),
        "multimodal": by_mode.get("multimodal", []),
    }


def seed_shipments(cursor, n=10_000):
    print(f"Seeding {n} shipments...")
    carrier_map = _carrier_ids_by_mode(cursor)
    all_carrier_ids = []
    for v in carrier_map.values():
        all_carrier_ids.extend(v)
    all_carrier_ids = list(set(all_carrier_ids))

    countries = list(PORTS.keys())

    for i in range(1, n + 1):
        mode = random.choices(MODES, weights=MODE_WEIGHTS, k=1)[0]
        status = random.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]

        carrier_pool = carrier_map.get(mode) or carrier_map.get("multimodal", all_carrier_ids)
        carrier_id = random.choice(carrier_pool)
        customer_id = random.randint(1, 10_000)

        origin_country = random.choice(countries)
        dest_country = random.choice([c for c in countries if c != origin_country])
        origin_port = random.choice(PORTS[origin_country])
        dest_port = random.choice(PORTS[dest_country])

        booking_date = random_date()
        transit_base = {"ocean": 25, "air": 3, "road": 5, "rail": 15}[mode]
        transit_days = transit_base + random.randint(-3, 10)

        etd = booking_date + timedelta(days=random.randint(1, 7))
        atd = etd + timedelta(days=random.randint(-1, 2)) if status not in ("booked",) else None
        eta = etd + timedelta(days=transit_days)

        delay_days = 0
        delay_reason = None
        if status == "delayed":
            delay_days = random.randint(1, 30)
            delay_reason = random.choice(DELAY_REASONS)
        elif status == "delivered" and random.random() < 0.2:
            delay_days = random.randint(1, 10)
            delay_reason = random.choice(DELAY_REASONS)

        ata = None
        if status == "delivered":
            ata = eta + timedelta(days=delay_days + random.randint(-1, 2))

        weight_kg = round(random.uniform(50, 25000), 1)
        volume_cbm = round(weight_kg / random.uniform(150, 400), 2)

        container_type = None
        container_count = 1
        if mode == "ocean":
            container_type = random.choice(CONTAINER_TYPES)
            container_count = random.randint(1, 8) if container_type != "LCL" else 0

        commodity = random.choice(COMMODITIES)
        incoterm = random.choice(INCOTERMS)
        po_number = f"PO-{random.randint(2023, 2025)}-{random.randint(1, 99999):05d}"
        priority = random.choices(["standard", "express", "economy"], weights=[60, 20, 20], k=1)[0]

        cursor.execute(
            "INSERT INTO shipments (shipment_id, customer_id, carrier_id, origin_port, "
            "origin_country, destination_port, destination_country, mode, status, "
            "booking_date, estimated_departure, actual_departure, estimated_arrival, "
            "actual_arrival, weight_kg, volume_cbm, container_type, container_count, "
            "commodity, incoterm, po_number, delay_days, delay_reason, priority) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"SHP-{booking_date.year}-{i:06d}",
                customer_id, carrier_id,
                origin_port, origin_country, dest_port, dest_country,
                mode, status,
                booking_date.strftime("%Y-%m-%d"),
                etd.strftime("%Y-%m-%d"),
                atd.strftime("%Y-%m-%d") if atd else None,
                eta.strftime("%Y-%m-%d"),
                ata.strftime("%Y-%m-%d") if ata else None,
                weight_kg, volume_cbm,
                container_type, container_count,
                commodity, incoterm, po_number,
                delay_days, delay_reason, priority,
            ),
        )
    print(f"  -> {n} shipments")


def seed_charges(cursor, shipment_count=10_000):
    print("Seeding shipment charges...")
    count = 0
    for ship_id in range(1, shipment_count + 1):
        # Every shipment has freight
        cursor.execute("SELECT mode FROM shipments WHERE id = ?", (ship_id,))
        row = cursor.fetchone()
        mode = row[0] if row else "ocean"

        cost_ranges = {
            "ocean": (1500, 15000),
            "air": (5000, 50000),
            "road": (500, 5000),
            "rail": (800, 8000),
        }
        lo, hi = cost_ranges.get(mode, (1000, 10000))
        freight = round(random.uniform(lo, hi), 2)

        charges = [("freight", freight, f"Freight charges - {mode}")]

        # Random additional charges
        if random.random() < 0.6:
            customs = round(freight * random.uniform(0.02, 0.15), 2)
            charges.append(("customs_duty", customs, "Import customs duty"))
        if random.random() < 0.5:
            insurance = round(freight * random.uniform(0.005, 0.03), 2)
            charges.append(("insurance", insurance, "Cargo insurance"))
        if random.random() < 0.4:
            handling = round(random.uniform(50, 500), 2)
            charges.append(("handling", handling, "Port handling charges"))
        if random.random() < 0.3:
            doc = round(random.uniform(25, 150), 2)
            charges.append(("documentation", doc, "Documentation and processing fee"))
        if random.random() < 0.1:
            demurrage = round(random.uniform(100, 2000), 2)
            charges.append(("demurrage", demurrage, "Container demurrage charges"))

        for charge_type, amount, desc in charges:
            cursor.execute(
                "INSERT INTO shipment_charges (shipment_id, charge_type, description, "
                "amount_usd, currency, exchange_rate, amount_original) "
                "VALUES (?, ?, ?, ?, 'USD', 1.0, ?)",
                (ship_id, charge_type, desc, amount, amount),
            )
            count += 1

    print(f"  -> {count} charges")


def seed_tracking(cursor, shipment_count=10_000):
    print("Seeding tracking events...")
    count = 0
    for ship_id in range(1, shipment_count + 1):
        cursor.execute(
            "SELECT status, booking_date, actual_departure, actual_arrival, "
            "origin_port, origin_country, destination_port, destination_country, delay_reason "
            "FROM shipments WHERE id = ?",
            (ship_id,),
        )
        row = cursor.fetchone()
        if not row:
            continue

        status, booking_str, atd_str, ata_str, orig_port, orig_country, dest_port, dest_country, delay_reason = row
        booking = datetime.strptime(booking_str, "%Y-%m-%d")

        # Determine how far along the tracking sequence to go
        status_to_events = {
            "booked": 1,
            "picked_up": 2,
            "in_transit": 4,
            "customs_hold": 5,
            "delivered": 8,
            "delayed": random.randint(3, 6),
            "cancelled": random.randint(1, 3),
        }
        n_events = status_to_events.get(status, 4)

        locations = [
            (orig_port, orig_country),
            (orig_port, orig_country),
            (orig_port, orig_country),
            ("In Transit", None),
            (dest_port, dest_country),
            (dest_port, dest_country),
            (dest_port, dest_country),
            (dest_port, dest_country),
        ]

        ts = booking
        for j in range(min(n_events, len(TRACKING_EVENTS_SEQ))):
            event_type = TRACKING_EVENTS_SEQ[j]
            ts = ts + timedelta(hours=random.randint(6, 72))
            loc, loc_country = locations[j]

            desc = f"{event_type.replace('_', ' ').title()}"
            if event_type == "departed_origin":
                desc = f"Departed from {orig_port}"
            elif event_type == "arrived_port":
                desc = f"Arrived at {dest_port}"
            elif event_type == "delivered":
                desc = f"Delivered to consignee at {dest_port}"

            is_milestone = event_type in ("booked", "departed_origin", "arrived_port", "delivered")
            created_by = random.choice(["system", "carrier_api", "manual", "edi"])

            cursor.execute(
                "INSERT INTO tracking_events (shipment_id, event_type, event_timestamp, "
                "location, location_country, description, created_by, is_milestone) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ship_id, event_type,
                    ts.strftime("%Y-%m-%d %H:%M:%S"),
                    loc, loc_country, desc, created_by, is_milestone,
                ),
            )
            count += 1

        # Add exception event for delayed shipments
        if status == "delayed" and delay_reason:
            ts = ts + timedelta(hours=random.randint(1, 24))
            cursor.execute(
                "INSERT INTO tracking_events (shipment_id, event_type, event_timestamp, "
                "location, location_country, description, created_by, is_milestone) "
                "VALUES (?, 'exception', ?, ?, ?, ?, 'system', 0)",
                (ship_id, ts.strftime("%Y-%m-%d %H:%M:%S"), dest_port, dest_country,
                 f"Delay: {delay_reason.replace('_', ' ')}"),
            )
            count += 1

    print(f"  -> {count} tracking events")


def seed_invoices(cursor, shipment_count=10_000):
    print("Seeding invoices...")
    for ship_id in range(1, shipment_count + 1):
        # Sum all charges for this shipment
        cursor.execute(
            "SELECT SUM(amount_usd) FROM shipment_charges WHERE shipment_id = ?",
            (ship_id,),
        )
        total_charges = cursor.fetchone()[0] or 0

        cursor.execute(
            "SELECT customer_id, carrier_id, booking_date FROM shipments WHERE id = ?",
            (ship_id,),
        )
        row = cursor.fetchone()
        if not row:
            continue
        customer_id, carrier_id, booking_str = row
        booking = datetime.strptime(booking_str, "%Y-%m-%d")

        subtotal = round(total_charges, 2)
        tax = round(subtotal * random.uniform(0.05, 0.18), 2)
        total = round(subtotal + tax, 2)

        invoice_date = booking + timedelta(days=random.randint(1, 14))
        due_date = invoice_date + timedelta(days=random.choice([15, 30, 45, 60]))

        payment_status = random.choices(PAYMENT_STATUSES, weights=PAYMENT_WEIGHTS, k=1)[0]
        payment_date = None
        if payment_status == "paid":
            payment_date = invoice_date + timedelta(days=random.randint(5, 50))

        cursor.execute(
            "INSERT INTO invoices (invoice_number, shipment_id, customer_id, carrier_id, "
            "invoice_date, due_date, subtotal_usd, tax_usd, total_usd, currency, "
            "payment_status, payment_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'USD', ?, ?)",
            (
                f"INV-{invoice_date.year}-{ship_id:06d}",
                ship_id, customer_id, carrier_id,
                invoice_date.strftime("%Y-%m-%d"),
                due_date.strftime("%Y-%m-%d"),
                subtotal, tax, total,
                payment_status,
                payment_date.strftime("%Y-%m-%d") if payment_date else None,
            ),
        )
    print(f"  -> {shipment_count} invoices")


def main():
    # Remove existing DB
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Enable WAL mode
    cursor.execute("PRAGMA journal_mode=WAL")

    # Create schema
    with open(SCHEMA_PATH) as f:
        cursor.executescript(f.read())

    # Seed in order (respecting FK dependencies)
    seed_carriers(cursor)
    conn.commit()

    seed_customers(cursor)
    conn.commit()

    seed_shipments(cursor)
    conn.commit()

    seed_charges(cursor)
    conn.commit()

    seed_tracking(cursor)
    conn.commit()

    seed_invoices(cursor)
    conn.commit()

    # Print summary
    print("\n=== Database Summary ===")
    for table in ["carriers", "customers", "shipments", "shipment_charges", "tracking_events", "invoices", "extracted_documents"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"  {table}: {cursor.fetchone()[0]} rows")

    conn.close()
    print(f"\nDatabase saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
