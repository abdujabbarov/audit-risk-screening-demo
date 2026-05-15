"""Generate the synthetic procurement dataset with planted irregularities."""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import RANDOM_SEED

MINISTRIES = [
    "Ministry of Health",
    "Ministry of Education",
    "Ministry of Transport",
    "Ministry of Public Works",
    "Ministry of Justice",
    "Ministry of Digital Affairs",
]

CATEGORIES = [
    "Construction",
    "IT services",
    "Office supplies",
    "Consulting",
    "Medical equipment",
    "Transport",
]

PROCUREMENT_METHODS = ["Open tender", "Limited tender", "Single-source", "Simplified"]

PAYMENT_TERMS = [14, 30, 45, 60, 90]
PAYMENT_TERM_WEIGHTS = [0.10, 0.60, 0.10, 0.15, 0.05]

# Per-category log-normal amount parameters (mu and sigma of underlying normal).
CATEGORY_AMOUNT_PARAMS = {
    "Construction":      (10.8, 0.9),
    "IT services":       (10.2, 0.8),
    "Office supplies":    (8.8, 0.7),
    "Consulting":         (9.8, 0.7),
    "Medical equipment": (10.5, 0.9),
    "Transport":          (9.6, 0.8),
}

ADJACENT_CATEGORIES = {
    "Construction":      ["Transport", "Office supplies"],
    "IT services":       ["Consulting", "Office supplies"],
    "Office supplies":   ["IT services", "Consulting"],
    "Consulting":        ["IT services", "Office supplies"],
    "Medical equipment": ["Construction", "Office supplies"],
    "Transport":         ["Construction", "Office supplies"],
}

# Method probabilities by amount tier.
METHOD_TIERS = {
    "low":  {"Simplified": 0.80, "Limited tender": 0.15, "Open tender": 0.05},
    "mid":  {"Limited tender": 0.60, "Open tender": 0.30, "Simplified": 0.10},
    "high": {"Open tender": 0.75, "Limited tender": 0.20, "Single-source": 0.05},
}

VENDOR_FIRST_WORDS = [
    "Acme", "BetaTech", "Citadel", "Delta", "Evergreen", "Fortis", "Granite",
    "Helios", "Indigo", "Juno", "Keystone", "Lumen", "Meridian", "Nexus",
    "Orion", "Pioneer", "Quanta", "Rivera", "Summit", "Trident", "Unity",
    "Vertex", "Westwind", "Xenon", "Yara", "Zenith", "Apex", "Bluefin",
    "Cascade", "Drift", "Ember", "Forge", "Glacier", "Harbor", "Ivory",
    "Junction", "Kelvin", "Legacy", "Mosaic", "Northbound",
]

CATEGORY_SECOND_WORDS = {
    "Construction":      ["Construction", "Builders", "Contracting", "Engineering", "Works"],
    "IT services":       ["Tech", "Systems", "Software", "Digital", "Solutions"],
    "Office supplies":   ["Supplies", "Stationery", "Office Group", "Trading", "Distributors"],
    "Consulting":        ["Consulting", "Advisory", "Partners", "Strategy", "Group"],
    "Medical equipment": ["Medical", "Health Systems", "MedTech", "Diagnostics", "Care"],
    "Transport":         ["Logistics", "Transport", "Freight", "Movers", "Cargo"],
}

DESCRIPTION_TEMPLATES = {
    "Construction":      ["office renovation", "site preparation", "structural repairs", "roof refurbishment", "facade works"],
    "IT services":       ["cloud migration", "ERP rollout", "endpoint refresh", "network upgrade", "support contract"],
    "Office supplies":   ["paper and toner order", "ergonomic chairs batch", "stationery resupply", "printer cartridges", "desk and shelving"],
    "Consulting":        ["strategy advisory", "process review", "risk assessment", "compliance audit", "transformation programme"],
    "Medical equipment": ["MRI maintenance", "diagnostic kits", "ICU monitors", "surgical instruments", "imaging consumables"],
    "Transport":         ["fleet maintenance", "fuel supply", "vehicle leasing", "logistics services", "delivery contract"],
}

DESCRIPTION_SUFFIXES = ["Q1 — north wing", "Q2 — south wing", "Q3 — east wing", "Q4 — west wing",
                        "phase 1", "phase 2", "annual contract", "framework lot A", "framework lot B"]

NORMAL_VENDOR_COUNT = 200
TOTAL_TRANSACTIONS = 10_000

DATE_START = date(2024, 1, 1)
DATE_END = date(2025, 12, 31)


def _make_vendor_pool(rng: random.Random) -> list[dict]:
    """Build the normal vendor pool with names and primary categories."""
    pool: list[dict] = []
    seen_names: set[str] = set()
    for idx in range(1, NORMAL_VENDOR_COUNT + 1):
        category = rng.choice(CATEGORIES)
        for _ in range(50):
            name = f"{rng.choice(VENDOR_FIRST_WORDS)} {rng.choice(CATEGORY_SECOND_WORDS[category])}"
            if name not in seen_names:
                seen_names.add(name)
                break
        pool.append({
            "vendor_id": f"V-{idx:03d}",
            "vendor_name": name,
            "primary_category": category,
        })
    return pool


def _amount_tier(amount: int) -> str:
    """Return low/mid/high tier label for an amount."""
    if amount < 10_000:
        return "low"
    if amount < 50_000:
        return "mid"
    return "high"


def _sample_method(amount: int, rng: random.Random) -> str:
    """Sample a procurement method from the amount-tier distribution."""
    weights = METHOD_TIERS[_amount_tier(amount)]
    methods = list(weights.keys())
    probs = list(weights.values())
    return rng.choices(methods, weights=probs, k=1)[0]


def _sample_amount(category: str, rng_np: np.random.Generator) -> int:
    """Sample an amount in the 5,000–500,000 range, rounded to the nearest 100."""
    mu, sigma = CATEGORY_AMOUNT_PARAMS[category]
    for _ in range(20):
        value = float(rng_np.lognormal(mean=mu, sigma=sigma))
        if 5_000 <= value <= 500_000:
            return int(round(value / 100.0) * 100)
    return int(round(min(max(value, 5_000), 500_000) / 100.0) * 100)


def _sample_date(rng_np: np.random.Generator) -> date:
    """Sample a date with a mild end-of-fiscal-quarter bump."""
    total_days = (DATE_END - DATE_START).days
    while True:
        offset = int(rng_np.integers(0, total_days + 1))
        candidate = DATE_START + timedelta(days=offset)
        weight = 1.0
        month, day = candidate.month, candidate.day
        if month in (3, 6, 9, 12) and day >= 20:
            weight = 1.7
        if rng_np.random() < weight / 1.7:
            return candidate


def _sample_category(vendor: dict, rng: random.Random) -> str:
    """Pick category for a transaction: 92% primary, 8% adjacent."""
    if rng.random() < 0.92:
        return vendor["primary_category"]
    return rng.choice(ADJACENT_CATEGORIES[vendor["primary_category"]])


def _make_description(category: str, rng: random.Random) -> str:
    """Compose a short human-readable contract description."""
    base = rng.choice(DESCRIPTION_TEMPLATES[category])
    suffix = rng.choice(DESCRIPTION_SUFFIXES)
    return f"{base} — {suffix}"


def _generate_normal_rows(vendors: list[dict],
                          rng: random.Random,
                          rng_np: np.random.Generator) -> list[dict]:
    """Generate the 10,000 normal-shaped rows."""
    rows: list[dict] = []
    for i in range(1, TOTAL_TRANSACTIONS + 1):
        vendor = rng.choice(vendors)
        category = _sample_category(vendor, rng)
        amount = _sample_amount(category, rng_np)
        method = _sample_method(amount, rng)
        terms = rng.choices(PAYMENT_TERMS, weights=PAYMENT_TERM_WEIGHTS, k=1)[0]
        rows.append({
            "transaction_id": f"T-{i:05d}",
            "date": _sample_date(rng_np).isoformat(),
            "ministry": rng.choice(MINISTRIES),
            "vendor_id": vendor["vendor_id"],
            "vendor_name": vendor["vendor_name"],
            "category": category,
            "amount": amount,
            "payment_terms_days": terms,
            "procurement_method": method,
            "contract_description": _make_description(category, rng),
            "_planted_pattern": "",
        })
    return rows


def _plant_pattern_a(rows: list[dict], rng: random.Random) -> None:
    """Threshold-avoidance: 10 transactions just below 100,000, single-source."""
    pattern_a_vendors = [
        {"vendor_id": "V-NEW-A01", "vendor_name": "Pinnacle Contracting"},
        {"vendor_id": "V-NEW-A02", "vendor_name": "Summit Engineering"},
        {"vendor_id": "V-NEW-A03", "vendor_name": "Beacon Services"},
    ]
    target_ids = [f"T-{i:05d}" for i in range(1001, 1011)]
    ministry_cycle = list(MINISTRIES)
    for idx, tid in enumerate(target_ids):
        row = next(r for r in rows if r["transaction_id"] == tid)
        v = pattern_a_vendors[idx % len(pattern_a_vendors)]
        amount = int(round(rng.uniform(98_000, 99_500) / 100.0) * 100)
        row.update({
            "vendor_id": v["vendor_id"],
            "vendor_name": v["vendor_name"],
            "category": rng.choice(["Construction", "IT services", "Consulting"]),
            "amount": amount,
            "procurement_method": "Single-source",
            "ministry": ministry_cycle[idx % len(ministry_cycle)],
            "contract_description": "advisory services — phase B",
            "_planted_pattern": "A_threshold_avoidance",
        })


def _plant_pattern_b(rows: list[dict], rng: random.Random) -> None:
    """New-vendor concentration: 8 contracts to V-NEW-001 within final 6 weeks of FY 2025."""
    target_ids = [f"T-{i:05d}" for i in (4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008)]
    end = date(2025, 12, 31)
    start_window = end - timedelta(days=42)
    for idx, tid in enumerate(target_ids):
        row = next(r for r in rows if r["transaction_id"] == tid)
        offset = int(rng.uniform(0, 42))
        d = start_window + timedelta(days=offset)
        if tid == "T-04004":
            amount = 99_500
        else:
            amount = int(round(rng.uniform(60_000, 140_000) / 100.0) * 100)
        row.update({
            "vendor_id": "V-NEW-001",
            "vendor_name": "NewVendor LLC",
            "ministry": "Ministry of Public Works",
            "category": "Construction",
            "amount": amount,
            "procurement_method": "Single-source",
            "date": d.isoformat(),
            "contract_description": "facilities works — year-end programme",
            "_planted_pattern": "B_new_vendor_concentration",
        })


def _plant_pattern_c(rows: list[dict], rng: random.Random) -> None:
    """End-of-period burst: 7 contracts to one vendor in Dec 22-31 2025."""
    target_ids = [f"T-{i:05d}" for i in range(7001, 7008)]
    days = [date(2025, 12, d) for d in (22, 23, 24, 26, 28, 30, 31)]
    for idx, tid in enumerate(target_ids):
        row = next(r for r in rows if r["transaction_id"] == tid)
        amount = int(round(rng.uniform(48_000, 52_000) / 100.0) * 100)
        row.update({
            "vendor_id": "V-NEW-C01",
            "vendor_name": "Rapid Procurement Group",
            "ministry": "Ministry of Education",
            "category": "Office supplies",
            "amount": amount,
            "procurement_method": "Limited tender",
            "date": days[idx].isoformat(),
            "contract_description": "year-end office equipment lot",
            "_planted_pattern": "C_end_of_period_burst",
        })


def _plant_pattern_d(rows: list[dict]) -> None:
    """Round-number bias: 5 round-amount single-source contracts to one vendor."""
    target_ids = [f"T-{i:05d}" for i in range(9001, 9006)]
    amounts = [50_000, 75_000, 100_000, 125_000, 150_000]
    categories = ["IT services", "Consulting", "Medical equipment", "Construction", "Transport"]
    descriptions = [
        "strategic IT review",
        "advisory engagement",
        "diagnostic equipment lot",
        "renovation services",
        "logistics framework",
    ]
    for idx, tid in enumerate(target_ids):
        row = next(r for r in rows if r["transaction_id"] == tid)
        row.update({
            "vendor_id": "V-NEW-D01",
            "vendor_name": "Omnibus Holdings",
            "ministry": "Ministry of Digital Affairs",
            "category": categories[idx],
            "amount": amounts[idx],
            "procurement_method": "Single-source",
            "date": date(2025, 11, 15 + idx).isoformat(),
            "contract_description": descriptions[idx],
            "_planted_pattern": "D_round_number_bias",
        })


def generate_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build the full 10,000-row synthetic procurement dataset."""
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)

    vendors = _make_vendor_pool(rng)
    rows = _generate_normal_rows(vendors, rng, rng_np)

    _plant_pattern_a(rows, rng)
    _plant_pattern_b(rows, rng)
    _plant_pattern_c(rows, rng)
    _plant_pattern_d(rows)

    df = pd.DataFrame(rows)
    df = df[[
        "transaction_id", "date", "ministry", "vendor_id", "vendor_name",
        "category", "amount", "payment_terms_days", "procurement_method",
        "contract_description", "_planted_pattern",
    ]]
    return df


def write_dataset(output_path: Path, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate the dataset and write it to CSV. Skips the write when content is unchanged."""
    df = generate_dataset(seed=seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_csv = df.to_csv(index=False)
    if output_path.exists() and output_path.read_text() == new_csv:
        return df
    output_path.write_text(new_csv)
    return df


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic procurement CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_procurement.csv"),
        help="Path to write the generated CSV.",
    )
    args = parser.parse_args()
    df = write_dataset(args.output)
    planted = (df["_planted_pattern"] != "").sum()
    print(f"Wrote {len(df):,} transactions to {args.output} ({planted} planted).")


if __name__ == "__main__":
    _cli()
