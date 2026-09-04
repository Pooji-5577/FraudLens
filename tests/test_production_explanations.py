import pandas as pd
import pytest

from backend.src.features import engineer_features
from backend.src.score import FraudScorer

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _velocity_burst():
    base = pd.Timestamp("2026-06-02T10:00:00Z")
    return pd.DataFrame([
        {
            "transaction_id": f"velocity-regression-{i:02d}",
            "timestamp": (base + pd.Timedelta(minutes=2 * i)).isoformat(),
            "user_id": "velocity-regression-user",
            "device_id": "velocity-regression-device",
            "card_id": "velocity-regression-card",
            "amount": 100,
            "billing_country": "IN",
            "ip_country": "IN",
            "merchant_category": "grocery",
        }
        for i in range(21)
    ])


def test_production_shap_top_reason_tracks_extreme_velocity():
    transactions = _velocity_burst()
    featured = engineer_features(transactions)
    assert featured.iloc[-1]["card_txn_count_1h"] == 20
    assert featured.iloc[-1]["device_txn_count_1h"] == 20

    scorer = FraudScorer()
    assert scorer.artifact["model_name"] == "tuned_xgboost_uncalibrated"
    result = scorer.score_batch(transactions).iloc[-1]
    assert result["flagged"]
    assert result["blocked"]
    assert "transactions" in result["reasons"][0]
    assert "last hour" in result["reasons"][0]
