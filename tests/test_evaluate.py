import numpy as np
import pytest
import json
from pathlib import Path

from backend.src.evaluate import (
    best_cost_threshold,
    metrics_at_threshold,
    sweep_thresholds,
    write_evaluation_artifacts,
)


def test_metrics_and_cost_are_counted_exactly():
    y = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.4, 0.8, 0.1])
    result = metrics_at_threshold(y, probabilities, 0.5, cost_fp=5, cost_fn=100)
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 1, 1, 1)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["false_positive_rate"] == pytest.approx(0.5)
    assert result["false_negative_rate"] == pytest.approx(0.5)
    assert result["specificity"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["total_cost"] == 105


def test_cost_sensitive_search_can_prefer_lower_threshold():
    y = np.array([1, 1, 0, 0])
    probabilities = np.array([0.8, 0.45, 0.6, 0.1])
    curve = sweep_thresholds(y, probabilities, [0.4, 0.7], cost_fp=2, cost_fn=100)
    chosen = best_cost_threshold(curve)
    assert chosen["threshold"] == pytest.approx(0.4)
    assert chosen["total_cost"] == pytest.approx(2)


def test_negative_cost_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        sweep_thresholds([0, 1], [0.1, 0.9], cost_fp=-1)


def test_final_evaluation_uses_locked_validation_threshold(tmp_path):
    y_test = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.4, 0.8, 0.1])
    validation_selection = {
        "threshold": 0.7,
        "rows": 20,
        "metrics": {"threshold": 0.7, "total_cost": 10.0},
    }

    result = write_evaluation_artifacts(
        y_test,
        probabilities,
        tmp_path,
        decision_threshold=0.7,
        validation_selection=validation_selection,
        cost_fp=5,
        cost_fn=100,
    )

    assert result["chosen"]["threshold"] == pytest.approx(0.7)
    assert result["chosen"]["total_cost"] == pytest.approx(105)
    assert result["threshold_selection"]["test_set_used_for_selection"] is False


def test_readme_cost_matches_generated_evaluation():
    root = Path(__file__).resolve().parents[1]
    evaluation = json.loads((root / "backend/reports/metrics/evaluation.json").read_text())
    chosen = evaluation["chosen"]
    expected = chosen["fn"] * evaluation["cost_fn"] + chosen["fp"] * evaluation["cost_fp"]
    assert chosen["total_cost"] == pytest.approx(expected)
    assert f"${expected:,.0f}" in (root / "README.md").read_text()
