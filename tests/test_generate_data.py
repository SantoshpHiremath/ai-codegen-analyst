"""Tests for src/generate_data.py -- the synthetic claims data generator."""
from __future__ import annotations

from src.generate_data import generate_claims, CLAIM_TYPES, REGIONS, STATUSES, CHANNELS


class TestGenerateClaims:
    def test_returns_requested_row_count_or_more(self):
        # Duplicate-row injection can push the count slightly above n.
        rows = generate_claims(n=500, seed=1)
        assert len(rows) >= 500

    def test_deterministic_for_same_seed(self):
        rows_a = generate_claims(n=200, seed=7)
        rows_b = generate_claims(n=200, seed=7)
        assert rows_a == rows_b

    def test_different_seeds_produce_different_data(self):
        rows_a = generate_claims(n=200, seed=1)
        rows_b = generate_claims(n=200, seed=2)
        assert rows_a != rows_b

    def test_all_expected_columns_present(self):
        rows = generate_claims(n=50, seed=1)
        expected_cols = {
            "claim_id", "claim_type", "region", "status", "channel",
            "days_open", "claim_amount_eur", "customer_age", "reported_date",
        }
        for row in rows:
            assert expected_cols.issubset(row.keys())

    def test_claim_type_values_are_from_known_set(self):
        rows = generate_claims(n=500, seed=1)
        for row in rows:
            assert row["claim_type"] in CLAIM_TYPES

    def test_region_values_are_from_known_set_ignoring_case_injection(self):
        rows = generate_claims(n=500, seed=1)
        for row in rows:
            assert row["region"].upper() in {r.upper() for r in REGIONS}

    def test_status_values_are_from_known_set(self):
        rows = generate_claims(n=500, seed=1)
        for row in rows:
            assert row["status"] in STATUSES

    def test_channel_values_are_from_known_set(self):
        rows = generate_claims(n=500, seed=1)
        for row in rows:
            assert row["channel"] in CHANNELS

    def test_injects_missing_claim_amounts(self):
        rows = generate_claims(n=2000, seed=42)
        none_count = sum(1 for r in rows if r["claim_amount_eur"] is None)
        assert none_count > 0

    def test_injects_negative_claim_amounts(self):
        rows = generate_claims(n=2000, seed=42)
        negative_count = sum(
            1 for r in rows if r["claim_amount_eur"] is not None and r["claim_amount_eur"] < 0
        )
        assert negative_count > 0

    def test_injects_uppercase_region_casing_inconsistency(self):
        rows = generate_claims(n=2000, seed=42)
        upper_count = sum(1 for r in rows if r["region"] == r["region"].upper() and r["region"].isalpha() is False or r["region"].isupper())
        # At least one region string should be all-uppercase (the injected case).
        assert any(r["region"].isupper() and len(r["region"]) > 3 for r in rows)

    def test_injects_sentinel_invalid_age(self):
        rows = generate_claims(n=2000, seed=42)
        sentinel_count = sum(1 for r in rows if r["customer_age"] == -1)
        assert sentinel_count > 0

    def test_injects_duplicate_rows(self):
        rows = generate_claims(n=2000, seed=42)
        ids = [r["claim_id"] for r in rows]
        assert len(ids) != len(set(ids))

    def test_claim_ids_follow_expected_format(self):
        rows = generate_claims(n=50, seed=1)
        for row in rows:
            assert row["claim_id"].startswith("C")
            assert len(row["claim_id"]) == 7

    def test_days_open_within_expected_range(self):
        rows = generate_claims(n=500, seed=1)
        for row in rows:
            assert 1 <= row["days_open"] <= 120
