"""
Generates a synthetic motor/personal-injury insurance claims dataset --
not real Allianz or any other insurer's data, which I have no access to.
Modeled on the posting's own domain ("Schadensmanagement Kraft-
Personenschaden" -- motor/personal-injury claims management), with
realistic claims-relevant columns and deliberately injected data-quality
problems, since a real claims-analytics workflow has to handle messy
source data, not a clean textbook table.
"""
from __future__ import annotations

import random

CLAIM_TYPES = ["Kfz-Haftpflicht", "Kfz-Kasko", "Personenschaden", "Glasschaden", "Wildschaden"]
REGIONS = ["Bayern", "Nordrhein-Westfalen", "Baden-Württemberg", "Niedersachsen", "Hessen", "Sachsen"]
STATUSES = ["Offen", "In Bearbeitung", "Abgeschlossen", "Abgelehnt"]
CHANNELS = ["Online-Portal", "Telefon", "Makler", "App"]


def _severity_base(claim_type: str) -> float:
    return {
        "Kfz-Haftpflicht": 3200.0,
        "Kfz-Kasko": 2100.0,
        "Personenschaden": 8500.0,
        "Glasschaden": 450.0,
        "Wildschaden": 1600.0,
    }[claim_type]


def generate_claims(n: int = 2000, seed: int = 42) -> list:
    rng = random.Random(seed)
    rows = []

    for i in range(1, n + 1):
        claim_type = rng.choice(CLAIM_TYPES)
        region = rng.choice(REGIONS)
        status = rng.choices(STATUSES, weights=[0.15, 0.25, 0.50, 0.10])[0]
        channel = rng.choice(CHANNELS)

        days_open = rng.randint(1, 120)
        base = _severity_base(claim_type)
        claim_amount_eur = round(max(50.0, rng.gauss(base, base * 0.4)), 2)
        customer_age = rng.randint(18, 85)

        row = {
            "claim_id": f"C{i:06d}",
            "claim_type": claim_type,
            "region": region,
            "status": status,
            "channel": channel,
            "days_open": days_open,
            "claim_amount_eur": claim_amount_eur,
            "customer_age": customer_age,
            "reported_date": f"2026-{rng.randint(1, 8):02d}-{rng.randint(1, 28):02d}",
        }

        # --- deliberately injected data-quality problems ---
        issue = rng.random()
        if issue < 0.03:
            row["claim_amount_eur"] = None  # missing amount
        elif issue < 0.05:
            row["claim_amount_eur"] = -abs(row["claim_amount_eur"])  # sign-entry error
        elif issue < 0.07:
            row["region"] = row["region"].upper()  # inconsistent casing from a source feed
        elif issue < 0.08:
            row["customer_age"] = -1  # sentinel/invalid age from a legacy system

        rows.append(row)
        if issue < 0.02:
            rows.append(dict(row))  # duplicate row (upstream retry)

    return rows


def write_csv(path: str, n: int = 2000, seed: int = 42) -> int:
    import csv
    rows = generate_claims(n=n, seed=seed)
    fieldnames = [
        "claim_id", "claim_type", "region", "status", "channel",
        "days_open", "claim_amount_eur", "customer_age", "reported_date",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    count = write_csv("data/claims.csv")
    print(f"Generated {count} claim rows.")
