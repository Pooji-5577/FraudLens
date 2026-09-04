import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from backend.src.features import MODEL_FEATURES, FeatureState, engineer_features


def _rows():
    return pd.DataFrame(
        [
            ["t1", "2026-01-01T10:00:00Z", "u1", "d1", "c1", 100, "IN", "IN", "grocery"],
            ["t2", "2026-01-01T10:30:00Z", "u1", "d1", "c1", 120, "IN", "IN", "grocery"],
            ["t3", "2026-01-01T12:00:00Z", "u1", "d2", "c1", 500, "IN", "US", "retail"],
        ],
        columns=["transaction_id", "timestamp", "user_id", "device_id", "card_id", "amount", "billing_country", "ip_country", "merchant_category"],
    )


def test_future_rows_cannot_change_earlier_features():
    original = _rows()
    early_only = engineer_features(original.iloc[:2])
    with_future = engineer_features(original)
    assert_frame_equal(
        early_only[MODEL_FEATURES].reset_index(drop=True),
        with_future.iloc[:2][MODEL_FEATURES].reset_index(drop=True),
    )


def test_same_timestamp_rows_do_not_see_each_other():
    rows = _rows().iloc[:2].copy()
    rows.loc[1, "timestamp"] = rows.loc[0, "timestamp"]
    featured = engineer_features(rows)
    assert featured["card_txn_count_1h"].tolist() == [0, 0]
    assert featured["is_new_device"].tolist() == [1.0, 1.0]


def test_expected_prior_history_values():
    featured = engineer_features(_rows())
    assert featured.loc[1, "card_txn_count_1h"] == 1
    assert featured.loc[1, "is_new_device"] == 0
    assert featured.loc[2, "card_txn_count_1h"] == 0
    assert featured.loc[2, "card_txn_count_24h"] == 2
    assert featured.loc[2, "geo_mismatch"] == 1
    assert featured.loc[2, "is_new_device"] == 1


def test_state_rejects_out_of_order_or_repeated_timestamps():
    state = FeatureState()
    engineer_features(_rows().iloc[:2], state)
    with pytest.raises(ValueError, match="later than"):
        engineer_features(_rows().iloc[[0]], state)
