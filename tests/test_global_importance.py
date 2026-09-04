import pandas as pd
import pytest

from backend.src.features import MODEL_FEATURES
from backend.src.global_importance import SIGNAL_GROUPS, group_and_normalize


def test_signal_groups_partition_model_features_exactly():
    grouped_features = {f for group in SIGNAL_GROUPS.values() for f in group}
    assert grouped_features == set(MODEL_FEATURES)
    total_slots = sum(len(group) for group in SIGNAL_GROUPS.values())
    assert total_slots == len(MODEL_FEATURES)


def test_group_and_normalize_sums_to_one_hundred_percent():
    mean_abs = pd.Series({feature: 1.0 for feature in MODEL_FEATURES})

    percentages = group_and_normalize(mean_abs)

    assert set(percentages) == set(SIGNAL_GROUPS)
    assert sum(percentages.values()) == pytest.approx(100.0, abs=0.5)


def test_group_and_normalize_weighs_groups_by_their_features_shap_mass():
    mean_abs = pd.Series({feature: 0.0 for feature in MODEL_FEATURES})
    mean_abs["is_new_device"] = 1.0
    mean_abs["geo_mismatch"] = 3.0

    percentages = group_and_normalize(mean_abs)

    assert percentages["New device"] == pytest.approx(25.0)
    assert percentages["Geography mismatch"] == pytest.approx(75.0)
    assert percentages["Transaction velocity"] == 0.0


def test_group_and_normalize_handles_all_zero_shap_without_dividing_by_zero():
    mean_abs = pd.Series({feature: 0.0 for feature in MODEL_FEATURES})

    percentages = group_and_normalize(mean_abs)

    assert all(value == 0.0 for value in percentages.values())
