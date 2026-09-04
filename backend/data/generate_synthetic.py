"""Generate a reproducible, imbalanced payment-transaction dataset.

The generator exists only to exercise a defensive detector.  It creates broad
risk correlations for model development; it is not a simulator for testing how
to evade a fraud system.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260903
COUNTRIES = np.array(["IN", "US", "GB", "SG", "AE", "DE", "AU", "CA"])
COUNTRY_P = np.array([0.52, 0.15, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03])
CATEGORIES = np.array(
    ["grocery", "fuel", "restaurant", "retail", "travel", "electronics", "digital_goods", "utilities"]
)
CATEGORY_P = np.array([0.22, 0.12, 0.17, 0.18, 0.06, 0.07, 0.08, 0.10])


def _different_country(rng: np.random.Generator, billing: str) -> str:
    candidates = COUNTRIES[COUNTRIES != billing]
    return str(rng.choice(candidates))


def generate_transactions(n_rows: int = 50_000, seed: int = SEED) -> pd.DataFrame:
    if n_rows < 1_000:
        raise ValueError("n_rows must be at least 1,000 so temporal patterns are meaningful")

    rng = np.random.default_rng(seed)
    n_users = max(800, n_rows // 12)
    users = np.array([f"usr_{i:06d}" for i in range(n_users)])
    user_weights = rng.pareto(1.8, n_users) + 0.2
    user_weights /= user_weights.sum()
    user_ids = rng.choice(users, size=n_rows, p=user_weights)

    user_country = dict(zip(users, rng.choice(COUNTRIES, size=n_users, p=COUNTRY_P)))
    user_typical = dict(zip(users, rng.lognormal(mean=np.log(1_400), sigma=0.75, size=n_users)))
    user_card = {u: f"card_{i:06d}" for i, u in enumerate(users)}
    user_device = {u: f"dev_{i:06d}_0" for i, u in enumerate(users)}

    start = pd.Timestamp("2026-01-01T00:00:00Z")
    offsets = np.sort(rng.integers(0, 90 * 24 * 3600, size=n_rows))
    timestamp_ns = start.value + offsets * 1_000_000_000
    cards = np.array([user_card[u] for u in user_ids], dtype=object)
    devices = np.array([user_device[u] for u in user_ids], dtype=object)
    billing = np.array([user_country[u] for u in user_ids], dtype=object)
    ip_country = billing.copy()
    normal_mismatch = rng.random(n_rows) < 0.035
    for i in np.flatnonzero(normal_mismatch):
        ip_country[i] = _different_country(rng, str(billing[i]))

    base_amount = np.array([user_typical[u] for u in user_ids])
    amounts = np.maximum(10.0, base_amount * rng.lognormal(0.0, 0.48, n_rows))
    categories = rng.choice(CATEGORIES, size=n_rows, p=CATEGORY_P)
    latent_fraud = np.zeros(n_rows, dtype=np.int8)

    # Pattern mix: no single rule defines fraud.  This leaves deliberate overlap
    # with legitimate activity and makes threshold/cost selection meaningful.
    # Start with a larger latent incident pool because the observation process
    # below deliberately hides many real incidents while preserving a realistic
    # observed positive rate.
    target_fraud = max(1, int(round(n_rows * 0.035)))
    candidates = rng.choice(np.arange(20, n_rows - 1), size=target_fraud, replace=False)
    pattern = rng.choice(4, size=target_fraud, p=[0.36, 0.23, 0.24, 0.17])

    for idx, kind in zip(candidates, pattern):
        latent_fraud[idx] = 1
        u = user_ids[idx]
        if kind == 0:  # velocity burst immediately before the labelled transaction
            burst_size = int(rng.integers(5, 12))
            prior = np.arange(max(0, idx - burst_size), idx)
            cards[prior] = cards[idx]
            devices[prior] = devices[idx]
            seconds = np.sort(rng.integers(20, 45 * 60, size=len(prior)))[::-1]
            timestamp_ns[prior] = timestamp_ns[idx] - seconds * 1_000_000_000
        elif kind == 1:  # geographic inconsistency
            ip_country[idx] = _different_country(rng, str(billing[idx]))
        elif kind == 2:  # sharp deviation from the customer's usual amount
            amounts[idx] = base_amount[idx] * rng.uniform(7.0, 16.0)
        else:  # first-seen device paired with a high amount
            devices[idx] = f"newdev_{idx:07d}"
            amounts[idx] = base_amount[idx] * rng.uniform(5.0, 12.0)

        # Some cases combine signals, but not all: explanations remain varied.
        if rng.random() < 0.43:
            ip_country[idx] = _different_country(rng, str(billing[idx]))
        if rng.random() < 0.34:
            amounts[idx] = max(amounts[idx], base_amount[idx] * rng.uniform(5.0, 11.0))

    frame = pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:08d}" for i in range(n_rows)],
            "timestamp": pd.to_datetime(timestamp_ns, utc=True),
            "user_id": user_ids,
            "device_id": devices,
            "card_id": cards,
            "amount": np.round(amounts, 2),
            "billing_country": billing,
            "ip_country": ip_country,
            "merchant_category": categories,
            "is_fraud": latent_fraud,
        }
    ).sort_values(["timestamp", "transaction_id"], kind="mergesort", ignore_index=True)

    # Exactly 2.5% of labels are wrong: delayed/missed reports create false
    # negatives (2.1 percentage points) and disputes create false positives
    # (0.4 points). The asymmetric mix keeps the observed label near 1.8%.
    positive_positions = frame.index[frame["is_fraud"].eq(1)].to_numpy()
    negative_positions = frame.index[frame["is_fraud"].eq(0)].to_numpy()
    false_negative_count = min(len(positive_positions), int(round(n_rows * 0.021)))
    false_positive_count = min(len(negative_positions), int(round(n_rows * 0.004)))
    false_negatives = rng.choice(positive_positions, false_negative_count, replace=False)
    false_positives = rng.choice(negative_positions, false_positive_count, replace=False)
    frame.loc[false_negatives, "is_fraud"] = 0
    frame.loc[false_positives, "is_fraud"] = 1
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "transactions.csv",
    )
    args = parser.parse_args()
    transactions = generate_transactions(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(args.output, index=False)
    print(f"wrote {len(transactions):,} rows to {args.output}")
    print(f"observed fraud rate: {transactions['is_fraud'].mean():.3%}")


if __name__ == "__main__":
    main()
