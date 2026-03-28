"""Tests for the seeded database — verify data integrity, relationships, distributions.

These tests validate the seed data quality, not the code that generated it.
"""

import sqlite3

import pytest


class TestDataIntegrity:
    """Verify row counts and non-null constraints."""

    def test_carriers_count(self, db_connection):
        row = db_connection.execute("SELECT COUNT(*) FROM carriers").fetchone()
        assert row[0] == 30

    def test_customers_count(self, db_connection):
        row = db_connection.execute("SELECT COUNT(*) FROM customers").fetchone()
        assert row[0] == 10000

    def test_shipments_count(self, db_connection):
        row = db_connection.execute("SELECT COUNT(*) FROM shipments").fetchone()
        assert row[0] == 10000

    def test_charges_exist(self, db_connection):
        row = db_connection.execute("SELECT COUNT(*) FROM shipment_charges").fetchone()
        assert row[0] > 20000  # ~3 per shipment

    def test_tracking_events_exist(self, db_connection):
        row = db_connection.execute("SELECT COUNT(*) FROM tracking_events").fetchone()
        assert row[0] > 30000  # ~4 per shipment

    def test_invoices_count(self, db_connection):
        row = db_connection.execute("SELECT COUNT(*) FROM invoices").fetchone()
        assert row[0] == 10000

    def test_extracted_documents_exists(self, db_connection):
        # Table exists and is queryable (may have data from vision pipeline usage)
        row = db_connection.execute("SELECT COUNT(*) FROM extracted_documents").fetchone()
        assert row[0] >= 0


class TestRelationships:
    """Verify FK relationships are valid."""

    def test_all_shipments_have_valid_customer(self, db_connection):
        row = db_connection.execute(
            "SELECT COUNT(*) FROM shipments s "
            "LEFT JOIN customers c ON s.customer_id = c.id "
            "WHERE c.id IS NULL"
        ).fetchone()
        assert row[0] == 0, "Found shipments with invalid customer_id"

    def test_all_shipments_have_valid_carrier(self, db_connection):
        row = db_connection.execute(
            "SELECT COUNT(*) FROM shipments s "
            "LEFT JOIN carriers c ON s.carrier_id = c.id "
            "WHERE c.id IS NULL"
        ).fetchone()
        assert row[0] == 0, "Found shipments with invalid carrier_id"

    def test_all_charges_have_valid_shipment(self, db_connection):
        row = db_connection.execute(
            "SELECT COUNT(*) FROM shipment_charges sc "
            "LEFT JOIN shipments s ON sc.shipment_id = s.id "
            "WHERE s.id IS NULL"
        ).fetchone()
        assert row[0] == 0, "Found charges with invalid shipment_id"

    def test_all_tracking_events_have_valid_shipment(self, db_connection):
        row = db_connection.execute(
            "SELECT COUNT(*) FROM tracking_events te "
            "LEFT JOIN shipments s ON te.shipment_id = s.id "
            "WHERE s.id IS NULL"
        ).fetchone()
        assert row[0] == 0, "Found tracking events with invalid shipment_id"

    def test_all_invoices_have_valid_shipment(self, db_connection):
        row = db_connection.execute(
            "SELECT COUNT(*) FROM invoices i "
            "LEFT JOIN shipments s ON i.shipment_id = s.id "
            "WHERE s.id IS NULL"
        ).fetchone()
        assert row[0] == 0, "Found invoices with invalid shipment_id"

    def test_every_shipment_has_at_least_one_charge(self, db_connection):
        row = db_connection.execute(
            "SELECT COUNT(*) FROM shipments s "
            "LEFT JOIN shipment_charges sc ON s.id = sc.shipment_id "
            "WHERE sc.id IS NULL"
        ).fetchone()
        assert row[0] == 0, "Found shipments with no charges"

    def test_every_shipment_has_invoice(self, db_connection):
        row = db_connection.execute(
            "SELECT COUNT(*) FROM shipments s "
            "LEFT JOIN invoices i ON s.id = i.shipment_id "
            "WHERE i.id IS NULL"
        ).fetchone()
        assert row[0] == 0, "Found shipments with no invoice"


class TestDistributions:
    """Verify data distributions are realistic."""

    def test_modes_distribution(self, db_connection):
        rows = db_connection.execute(
            "SELECT mode, COUNT(*) as cnt FROM shipments GROUP BY mode ORDER BY cnt DESC"
        ).fetchall()
        modes = {r[0]: r[1] for r in rows}
        # Ocean should be most common (~55%)
        assert modes.get("ocean", 0) > modes.get("air", 0)
        assert modes.get("air", 0) > modes.get("road", 0)

    def test_status_distribution(self, db_connection):
        rows = db_connection.execute(
            "SELECT status, COUNT(*) as cnt FROM shipments GROUP BY status"
        ).fetchall()
        statuses = {r[0]: r[1] for r in rows}
        # Delivered should be most common (~45%)
        assert statuses.get("delivered", 0) > statuses.get("booked", 0)

    def test_carrier_types(self, db_connection):
        rows = db_connection.execute(
            "SELECT DISTINCT carrier_type FROM carriers"
        ).fetchall()
        types = {r[0] for r in rows}
        assert "ocean" in types
        assert "air" in types
        assert "road" in types

    def test_customer_tiers(self, db_connection):
        rows = db_connection.execute(
            "SELECT tier, COUNT(*) FROM customers GROUP BY tier"
        ).fetchall()
        tiers = {r[0]: r[1] for r in rows}
        assert "gold" in tiers
        assert "silver" in tiers
        assert "bronze" in tiers
        # Bronze should be most (~50%)
        assert tiers["bronze"] > tiers["gold"]

    def test_delayed_shipments_have_reasons(self, db_connection):
        row = db_connection.execute(
            "SELECT COUNT(*) FROM shipments WHERE status = 'delayed' AND delay_reason IS NULL"
        ).fetchone()
        assert row[0] == 0, "Delayed shipments should always have a reason"

    def test_delivered_shipments_have_arrival(self, db_connection):
        row = db_connection.execute(
            "SELECT COUNT(*) FROM shipments WHERE status = 'delivered' AND actual_arrival IS NULL"
        ).fetchone()
        assert row[0] == 0, "Delivered shipments should have actual_arrival date"

    def test_payment_statuses(self, db_connection):
        rows = db_connection.execute(
            "SELECT payment_status, COUNT(*) FROM invoices GROUP BY payment_status"
        ).fetchall()
        statuses = {r[0]: r[1] for r in rows}
        assert "paid" in statuses
        assert "pending" in statuses
        # Paid should be most common
        assert statuses["paid"] > statuses.get("disputed", 0)

    def test_shipment_ids_unique(self, db_connection):
        row = db_connection.execute(
            "SELECT COUNT(DISTINCT shipment_id) FROM shipments"
        ).fetchone()
        assert row[0] == 10000

    def test_invoice_numbers_unique(self, db_connection):
        row = db_connection.execute(
            "SELECT COUNT(DISTINCT invoice_number) FROM invoices"
        ).fetchone()
        assert row[0] == 10000


class TestQueryPatterns:
    """Test that common analytics queries work on the seeded data."""

    def test_top_carriers_by_shipment_count(self, db_connection):
        rows = db_connection.execute(
            "SELECT c.carrier_name, COUNT(*) as cnt "
            "FROM shipments s JOIN carriers c ON s.carrier_id = c.id "
            "GROUP BY c.carrier_name ORDER BY cnt DESC LIMIT 5"
        ).fetchall()
        assert len(rows) == 5
        assert rows[0][1] > 0

    def test_avg_freight_cost_by_mode(self, db_connection):
        rows = db_connection.execute(
            "SELECT s.mode, AVG(sc.amount_usd) as avg_cost "
            "FROM shipments s JOIN shipment_charges sc ON s.id = sc.shipment_id "
            "WHERE sc.charge_type = 'freight' "
            "GROUP BY s.mode"
        ).fetchall()
        assert len(rows) >= 4  # ocean, air, road, rail

    def test_monthly_shipment_volume(self, db_connection):
        rows = db_connection.execute(
            "SELECT strftime('%Y-%m', booking_date) as month, COUNT(*) "
            "FROM shipments GROUP BY month ORDER BY month"
        ).fetchall()
        assert len(rows) > 12  # at least 12 months of data

    def test_customer_spend_with_joins(self, db_connection):
        rows = db_connection.execute(
            "SELECT cu.company_name, SUM(i.total_usd) as total_spend "
            "FROM customers cu "
            "JOIN invoices i ON cu.id = i.customer_id "
            "GROUP BY cu.id ORDER BY total_spend DESC LIMIT 5"
        ).fetchall()
        assert len(rows) == 5
        assert rows[0][1] > 0
