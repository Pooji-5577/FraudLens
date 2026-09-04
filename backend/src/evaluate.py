"""Held-out classification, calibration, and cost-sensitive evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def metrics_at_threshold(y_true, probabilities, threshold: float, cost_fp: float, cost_fn: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = p >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    negatives = fp + tn
    positives = tp + fn
    predicted_positives = tp + fp
    predicted_negatives = tn + fn
    false_positive_rate = fp / negatives if negatives else 0.0
    false_negative_rate = fn / positives if positives else 0.0
    specificity = tn / negatives if negatives else 0.0
    negative_predictive_value = tn / predicted_negatives if predicted_negatives else 0.0
    return {
        "threshold": float(threshold), "precision": float(precision), "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "specificity": float(specificity),
        "negative_predictive_value": float(negative_predictive_value),
        "accuracy": float((tp + tn) / len(y)) if len(y) else 0.0,
        "balanced_accuracy": float((recall + specificity) / 2),
        "fraud_prevalence": float(positives / len(y)) if len(y) else 0.0,
        "predicted_positive_rate": float(predicted_positives / len(y)) if len(y) else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total_cost": float(fp * cost_fp + fn * cost_fn),
    }


def sweep_thresholds(
    y_true, probabilities, thresholds=None, cost_fp: float = 5.0, cost_fn: float = 500.0
) -> pd.DataFrame:
    if cost_fp < 0 or cost_fn < 0:
        raise ValueError("costs must be non-negative")
    if thresholds is None:
        # Calibrated rare-event models often have useful operating points below
        # 0.05, so search that region more finely than the rest of the range.
        thresholds = np.unique(
            np.concatenate([np.linspace(0.001, 0.05, 50), np.linspace(0.06, 0.99, 94)])
        )
    rows = [metrics_at_threshold(y_true, probabilities, float(t), cost_fp, cost_fn) for t in thresholds]
    return pd.DataFrame(rows)


def best_cost_threshold(curve: pd.DataFrame) -> dict:
    if curve.empty:
        raise ValueError("threshold curve is empty")
    # Stable tie-break: lower cost, then higher recall, then higher threshold.
    row = curve.sort_values(["total_cost", "recall", "threshold"], ascending=[True, False, False]).iloc[0]
    return row.to_dict()


def calibration_summary(y_true, probabilities, n_bins: int = 10) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "bins": [
            {"mean_predicted_risk": float(pred), "observed_fraud_rate": float(actual)}
            for pred, actual in zip(mean_pred, frac_pos)
        ],
    }


def write_evaluation_artifacts(
    y_true,
    probabilities,
    output_dir: Path,
    decision_threshold: float,
    validation_selection: dict,
    cost_fp: float = 5.0,
    cost_fn: float = 500.0,
) -> dict:
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    curve = sweep_thresholds(y_true, probabilities, cost_fp=cost_fp, cost_fn=cost_fn)
    chosen = metrics_at_threshold(
        y_true, probabilities, decision_threshold, cost_fp=cost_fp, cost_fn=cost_fn
    )
    recomputed_cost = chosen["fn"] * cost_fn + chosen["fp"] * cost_fp
    if not np.isclose(chosen["total_cost"], recomputed_cost):
        raise AssertionError("reported threshold cost does not match FP/FN cost arithmetic")
    calibration = calibration_summary(y_true, probabilities)
    summary = {
        "evaluation_split": "most recent 30% by timestamp",
        "threshold_selection": {
            "source": "validation window inside the earlier 70% training period",
            "objective": "minimum illustrative false-positive/false-negative cost",
            "test_set_used_for_selection": False,
            **validation_selection,
        },
        "cost_fp": cost_fp, "cost_fn": cost_fn,
        "chosen": chosen, "calibration": calibration,
        "ranking": {
            "average_precision": float(average_precision_score(y_true, probabilities)),
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
        },
        "selected_thresholds": [
            metrics_at_threshold(y_true, probabilities, t, cost_fp, cost_fn)
            for t in sorted({0.005, 0.01, 0.025, 0.05, 0.1, 0.25, float(decision_threshold)})
        ],
    }
    curve.to_csv(metrics_dir / "threshold_curve.csv", index=False)
    (metrics_dir / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_plots(
        y_true, probabilities, curve, calibration, figures_dir, float(decision_threshold)
    )
    return summary


def _write_plots(
    y_true,
    probabilities,
    curve: pd.DataFrame,
    calibration: dict,
    figures_dir: Path,
    decision_threshold: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["threshold"], curve["total_cost"])
    ax.axvline(decision_threshold, color="black", linestyle="--", label="validation-selected")
    ax.set(
        xlabel="Decision threshold",
        ylabel="Illustrative error cost ($)",
        title="Held-out threshold sensitivity (descriptive only)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "cost_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["threshold"], curve["precision"], label="precision")
    ax.plot(curve["threshold"], curve["recall"], label="recall")
    ax.plot(curve["threshold"], curve["f1"], label="F1")
    ax.plot(curve["threshold"], curve["false_positive_rate"], label="false-positive rate")
    ax.axvline(decision_threshold, color="black", linestyle="--", label="validation-selected")
    ax.set(xlabel="Decision threshold", ylabel="Metric", ylim=(0, 1), title="Held-out threshold trade-offs")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "threshold_metrics.png", dpi=150)
    plt.close(fig)

    bins = calibration["bins"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot([b["mean_predicted_risk"] for b in bins], [b["observed_fraud_rate"] for b in bins], "o-")
    ax.set(xlabel="Mean predicted risk", ylabel="Observed fraud rate", title="Held-out reliability plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "calibration.png", dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(recall, precision)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Held-out precision-recall curve")
    axes[1].plot(false_positive_rate, true_positive_rate)
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Held-out ROC curve")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "ranking_curves.png", dpi=150)
    plt.close(fig)
