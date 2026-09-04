"""Point-in-time-correct feature engineering for transactions."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "transaction_id", "timestamp", "user_id", "device_id", "card_id", "amount",
    "billing_country", "ip_country", "merchant_category",
}

MODEL_FEATURES = [
    "amount", "amount_log", "card_txn_count_1h", "card_txn_count_24h",
    "device_txn_count_1h", "device_txn_count_24h", "geo_mismatch",
    "user_amount_zscore", "user_amount_mean", "user_amount_std",
    "is_new_device", "seconds_since_user_last_txn", "hour_sin", "hour_cos",
]


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    @property
    def std(self) -> float:
        return sqrt(self.m2 / (self.count - 1)) if self.count > 1 else 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)


@dataclass
class FeatureState:
    """Mutable history used for chronological batch or online scoring."""

    card_1h: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    card_24h: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    device_1h: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    device_24h: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    user_stats: dict[str, RunningStats] = field(default_factory=dict)
    user_devices: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    user_last_timestamp: dict[str, pd.Timestamp] = field(default_factory=dict)
    latest_timestamp: pd.Timestamp | None = None


def _prune(queue: deque, cutoff: pd.Timestamp) -> None:
    while queue and queue[0] < cutoff:
        queue.popleft()


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    result["amount"] = pd.to_numeric(result["amount"], errors="raise")
    if (result["amount"] < 0).any():
        raise ValueError("amount must be non-negative")
    return result


def engineer_features(frame: pd.DataFrame, state: FeatureState | None = None) -> pd.DataFrame:
    """Return original rows plus features based strictly on earlier timestamps.

    Rows sharing a timestamp are evaluated against the same prior state and are
    only inserted into history after the entire timestamp group is scored.
    """
    state = state or FeatureState()
    data = _validate(frame)
    if data.empty:
        raise ValueError("at least one transaction is required")
    if state.latest_timestamp is not None and data["timestamp"].min() <= state.latest_timestamp:
        raise ValueError(
            "transactions must be later than previously scored data; submit equal-timestamp rows in one batch"
        )
    data["input_order_internal"] = np.arange(len(data))
    data = data.sort_values(["timestamp", "transaction_id"], kind="mergesort")
    feature_rows: list[dict[str, float]] = []

    for timestamp, group in data.groupby("timestamp", sort=True):
        pending: list[tuple[str, str, str, float, pd.Timestamp]] = []
        for row in group.itertuples(index=False):
            card = str(row.card_id)
            device = str(row.device_id)
            user = str(row.user_id)
            amount = float(row.amount)
            for q, delta in (
                (state.card_1h[card], pd.Timedelta(hours=1)),
                (state.card_24h[card], pd.Timedelta(hours=24)),
                (state.device_1h[device], pd.Timedelta(hours=1)),
                (state.device_24h[device], pd.Timedelta(hours=24)),
            ):
                _prune(q, timestamp - delta)

            stats = state.user_stats.get(user, RunningStats())
            std = stats.std
            zscore = (amount - stats.mean) / std if std > 1e-9 else 0.0
            last = state.user_last_timestamp.get(user)
            since_last = (timestamp - last).total_seconds() if last is not None else -1.0
            hour = timestamp.hour + timestamp.minute / 60.0
            feature_rows.append(
                {
                    "input_order_internal": row.input_order_internal,
                    "amount_log": np.log1p(amount),
                    "card_txn_count_1h": len(state.card_1h[card]),
                    "card_txn_count_24h": len(state.card_24h[card]),
                    "device_txn_count_1h": len(state.device_1h[device]),
                    "device_txn_count_24h": len(state.device_24h[device]),
                    "geo_mismatch": float(row.billing_country != row.ip_country),
                    "user_amount_zscore": float(np.clip(zscore, -20.0, 20.0)),
                    "user_amount_mean": stats.mean if stats.count else 0.0,
                    "user_amount_std": std,
                    "is_new_device": float(device not in state.user_devices[user]),
                    "seconds_since_user_last_txn": since_last,
                    "hour_sin": np.sin(2 * np.pi * hour / 24),
                    "hour_cos": np.cos(2 * np.pi * hour / 24),
                }
            )
            pending.append((card, device, user, amount, timestamp))

        for card, device, user, amount, ts in pending:
            state.card_1h[card].append(ts)
            state.card_24h[card].append(ts)
            state.device_1h[device].append(ts)
            state.device_24h[device].append(ts)
            state.user_devices[user].add(device)
            state.user_stats.setdefault(user, RunningStats()).update(amount)
            state.user_last_timestamp[user] = ts

    state.latest_timestamp = data["timestamp"].max()

    features = pd.DataFrame(feature_rows).sort_values("input_order_internal")
    original = data.sort_values("input_order_internal").drop(columns="input_order_internal").reset_index(drop=True)
    return pd.concat([original, features.drop(columns="input_order_internal").reset_index(drop=True)], axis=1)


def model_matrix(featured: pd.DataFrame) -> pd.DataFrame:
    return featured.loc[:, MODEL_FEATURES].astype(float)
