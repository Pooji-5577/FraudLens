"""SHAP-backed feature attribution and reviewer-friendly reason codes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import MODEL_FEATURES


def shap_contributions(model, matrix: pd.DataFrame) -> np.ndarray:
    import shap

    if hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        transformed = model.named_steps["standardscaler"].transform(matrix)
        explainer = shap.LinearExplainer(model.named_steps["logisticregression"], transformed)
        values = explainer.shap_values(transformed)
    else:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _reason(feature: str, row: pd.Series) -> str:
    value = row[feature]
    reasons = {
        "card_txn_count_1h": f"{int(value)} earlier transactions on this card in the last hour",
        "card_txn_count_24h": f"{int(value)} earlier transactions on this card in the last 24 hours",
        "device_txn_count_1h": f"{int(value)} earlier transactions from this device in the last hour",
        "device_txn_count_24h": f"{int(value)} earlier transactions from this device in the last 24 hours",
        "geo_mismatch": (
            f"Billing country {row['billing_country']}, but IP country {row['ip_country']}"
            if value else f"Billing and IP countries both {row['billing_country']} contributed in this model context"
        ),
        "user_amount_zscore": (
            f"Amount {row['amount']:,.0f} is {abs(value):.1f} standard deviations "
            f"{'above' if value >= 0 else 'below'} this customer's prior average"
        ),
        "user_amount_mean": f"Customer's prior average amount was {value:,.0f}",
        "user_amount_std": f"Customer's prior spending range is unusually {'variable' if value > 0 else 'unknown'}",
        "is_new_device": (
            "This device has not previously been seen for this customer"
            if value else "This device was previously seen, but its model interaction increased risk"
        ),
        "seconds_since_user_last_txn": (
            "No earlier transaction was available for this customer"
            if value < 0 else
            f"Transaction followed the customer's previous payment by {value / 60:.1f} minutes"
        ),
        "amount": (
            f"Amount {row['amount']:,.0f} versus prior customer average {row['user_amount_mean']:,.0f}"
            if row["user_amount_mean"] > 0 else
            f"Amount {row['amount']:,.0f} with no prior customer spending baseline"
        ),
        "amount_log": f"Transaction amount {row['amount']:,.0f} was a strong model signal",
        "hour_sin": f"Transaction time ({row['timestamp']}) contributed to the risk score",
        "hour_cos": f"Transaction time ({row['timestamp']}) contributed to the risk score",
    }
    return reasons.get(feature, f"{feature}={value:.3g} increased risk")


def explain_flagged(model, featured: pd.DataFrame, probabilities, threshold: float, top_n: int = 3) -> list[list[str]]:
    result: list[list[str]] = [[] for _ in range(len(featured))]
    flagged_positions = np.flatnonzero(np.asarray(probabilities) >= threshold)
    if len(flagged_positions) == 0:
        return result
    matrix = featured.iloc[flagged_positions][MODEL_FEATURES].astype(float)
    contributions = shap_contributions(model, matrix)
    for local_i, position in enumerate(flagged_positions):
        positive = np.flatnonzero(contributions[local_i] > 0)
        ranked = positive[np.argsort(contributions[local_i, positive])[::-1]][:top_n]
        if len(ranked) == 0:
            ranked = np.argsort(np.abs(contributions[local_i]))[::-1][:top_n]
        result[position] = [_reason(MODEL_FEATURES[j], featured.iloc[position]) for j in ranked]
    return result
