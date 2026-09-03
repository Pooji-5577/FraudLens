"""Time-respecting XGBoost hyperparameter search optimized for PR-AUC."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

MAX_DEPTHS = (3, 4, 5, 6, 8)
N_ESTIMATORS = (100, 300, 500)
LEARNING_RATES = (0.3, 0.1, 0.05, 0.01)
SCALE_POS_WEIGHT_MULTIPLIERS = (0.5, 1.0, 2.0)


def make_xgboost(params: dict, seed: int) -> XGBClassifier:
    return XGBClassifier(
        **params,
        subsample=0.85,
        colsample_bytree=0.9,
        min_child_weight=4,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def temporal_xgboost_search(
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    output_dir: Path,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    positives = int(y_fit.sum())
    if positives == 0:
        raise ValueError("XGBoost tuning requires positive training labels")
    class_ratio = float((len(y_fit) - positives) / positives)
    rows: list[dict] = []
    grid = list(itertools.product(
        SCALE_POS_WEIGHT_MULTIPLIERS, MAX_DEPTHS, N_ESTIMATORS, LEARNING_RATES
    ))
    for index, (multiplier, max_depth, n_estimators, learning_rate) in enumerate(grid, start=1):
        params = {
            "scale_pos_weight": class_ratio * multiplier,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
        }
        model = make_xgboost(params, seed)
        model.fit(x_fit, y_fit)
        probabilities = model.predict_proba(x_validation)[:, 1]
        rows.append({
            **params,
            "scale_pos_weight_multiplier": multiplier,
            "validation_pr_auc": float(average_precision_score(y_validation, probabilities)),
            "validation_roc_auc": float(roc_auc_score(y_validation, probabilities)),
        })
        if index % 20 == 0 or index == len(grid):
            print(f"XGBoost temporal search: {index}/{len(grid)} candidates")

    results = pd.DataFrame(rows).sort_values(
        ["validation_pr_auc", "validation_roc_auc", "max_depth", "n_estimators"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    best_row = results.iloc[0]
    best = {
        "scale_pos_weight_multiplier": float(best_row["scale_pos_weight_multiplier"]),
        "scale_pos_weight": float(best_row["scale_pos_weight"]),
        "max_depth": int(best_row["max_depth"]),
        "n_estimators": int(best_row["n_estimators"]),
        "learning_rate": float(best_row["learning_rate"]),
        "validation_pr_auc": float(best_row["validation_pr_auc"]),
        "validation_roc_auc": float(best_row["validation_roc_auc"]),
        "candidates_evaluated": len(results),
    }
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(metrics_dir / "xgboost_temporal_search.csv", index=False)
    (metrics_dir / "xgboost_temporal_search.json").write_text(
        json.dumps({"best": best, "candidates": results.to_dict(orient="records")}, indent=2) + "\n"
    )
    return best, results
