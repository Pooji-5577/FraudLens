"""Real, held-out global feature importance for the dashboard's signal panel.

Computes mean(|SHAP value|) per model feature on the same untouched test
window reported in backend/reports/metrics/evaluation.json, then groups the 21 model
features into the six signal categories the dashboard names, so the panel
shows a measured statistic instead of an illustrative guess.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.src.explain import shap_contributions
from backend.src.features import MODEL_FEATURES, engineer_features, model_matrix

SIGNAL_GROUPS: dict[str, list[str]] = {
    "Transaction velocity": [
        "card_txn_count_1h", "card_txn_count_24h",
        "device_txn_count_1h", "device_txn_count_24h",
        "card_velocity_1h_log", "device_velocity_1h_log",
    ],
    "Geography mismatch": ["geo_mismatch", "geo_new_device", "geo_amount_ratio"],
    "New device": ["is_new_device", "new_device_amount_ratio"],
    "Amount deviation": [
        "amount", "amount_log", "user_amount_zscore", "user_amount_mean", "user_amount_std",
        "amount_to_user_mean_ratio",
    ],
    "Time of day": ["hour_sin", "hour_cos"],
    "Transaction recency": ["seconds_since_user_last_txn", "rapid_repeat"],
}


def _assert_groups_partition_features() -> None:
    grouped_features = {f for group in SIGNAL_GROUPS.values() for f in group}
    if grouped_features != set(MODEL_FEATURES):
        missing = set(MODEL_FEATURES) - grouped_features
        extra = grouped_features - set(MODEL_FEATURES)
        raise ValueError(f"SIGNAL_GROUPS must partition MODEL_FEATURES exactly; missing={missing} extra={extra}")


def group_and_normalize(mean_abs_shap: pd.Series) -> dict[str, float]:
    """Sum a per-feature mean(|SHAP|) series into signal groups, normalized to percent."""
    _assert_groups_partition_features()
    group_totals = {name: float(mean_abs_shap[feats].sum()) for name, feats in SIGNAL_GROUPS.items()}
    grand_total = sum(group_totals.values())
    return {
        name: round(100 * value / grand_total, 1) if grand_total else 0.0
        for name, value in group_totals.items()
    }


def compute_featured_importance(featured: pd.DataFrame, explanation_model) -> dict[str, float]:
    """Return grouped mean absolute SHAP influence for a scored dataset."""
    if featured.empty:
        return {name: 0.0 for name in SIGNAL_GROUPS}
    contributions = shap_contributions(explanation_model, model_matrix(featured))
    mean_abs = pd.Series(np.abs(contributions).mean(axis=0), index=MODEL_FEATURES)
    return group_and_normalize(mean_abs)


def compute_global_importance(
    root: Path = Path(__file__).resolve().parents[1],
) -> dict:
    """Mean absolute SHAP contribution per signal group, normalized to percent."""
    _assert_groups_partition_features()

    artifact = joblib.load(root / "models" / "fraud_detector.joblib")
    raw = pd.read_csv(root / "data" / "transactions.csv")
    featured = engineer_features(raw)
    test_started = pd.Timestamp(artifact["test_started"])
    test = featured[featured["timestamp"] >= test_started]
    matrix = model_matrix(test)

    return {
        "held_out_test_start": str(test_started),
        "held_out_rows": int(len(test)),
        "method": (
            "mean(|SHAP value|) per model feature over the held-out test set, "
            "grouped into dashboard signal names and normalized to sum to 100%"
        ),
        "signal_importance_percent": compute_featured_importance(test, artifact["explanation_model"]),
    }


def write_global_importance_artifact(
    root: Path = Path(__file__).resolve().parents[1],
) -> dict:
    result = compute_global_importance(root)
    output_path = root / "reports" / "metrics" / "global_feature_importance.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    result = write_global_importance_artifact()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
