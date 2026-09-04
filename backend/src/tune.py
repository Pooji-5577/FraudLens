"""Time-respecting XGBoost search optimized for stable rare-event ranking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier

SEARCH_ITERATIONS = 144
SEARCH_SPACE = {
    "scale_pos_weight_multiplier": (0.25, 0.4, 0.55, 0.75, 1.0),
    "max_depth": (2, 3, 4, 5),
    "n_estimators": (180, 300, 450, 650),
    "learning_rate": (0.025, 0.04, 0.06, 0.09),
    "min_child_weight": (2, 4, 8, 12),
    "subsample": (0.75, 0.85, 0.95, 1.0),
    "colsample_bytree": (0.7, 0.82, 0.92, 1.0),
    "gamma": (0.0, 0.1, 0.3, 0.6),
    "reg_alpha": (0.0, 0.05, 0.2, 0.75),
    "reg_lambda": (1.5, 3.0, 6.0, 10.0),
    "max_delta_step": (0, 1, 3),
}


def make_xgboost(params: dict, seed: int) -> XGBClassifier:
    defaults = dict(
        subsample=0.85,
        colsample_bytree=0.9,
        min_child_weight=4,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    return XGBClassifier(**{**defaults, **params})


def _temporal_folds(
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """Create expanding-window folds while preserving the supplied final validation window."""
    inner_cutoff = int(len(x_fit) * 0.75)
    if inner_cutoff < 1 or inner_cutoff >= len(x_fit):
        raise ValueError("XGBoost tuning requires enough rows for two temporal windows")
    return [
        (
            x_fit.iloc[:inner_cutoff],
            y_fit.iloc[:inner_cutoff],
            x_fit.iloc[inner_cutoff:],
            y_fit.iloc[inner_cutoff:],
        ),
        (x_fit, y_fit, x_validation, y_validation),
    ]


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
    folds = _temporal_folds(x_fit, y_fit, x_validation, y_validation)
    candidates = list(ParameterSampler(SEARCH_SPACE, n_iter=SEARCH_ITERATIONS, random_state=seed))
    # Always retain the previous winner as a benchmark against the expanded search.
    candidates.append({
        "scale_pos_weight_multiplier": 0.5,
        "max_depth": 3,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "min_child_weight": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
        "max_delta_step": 0,
    })
    for index, candidate in enumerate(candidates, start=1):
        candidate = dict(candidate)
        multiplier = float(candidate.pop("scale_pos_weight_multiplier"))
        fold_pr_auc: list[float] = []
        fold_roc_auc: list[float] = []
        for fold_x_fit, fold_y_fit, fold_x_validation, fold_y_validation in folds:
            fold_positives = int(fold_y_fit.sum())
            fold_ratio = float((len(fold_y_fit) - fold_positives) / fold_positives)
            model = make_xgboost(
                {**candidate, "scale_pos_weight": fold_ratio * multiplier}, seed
            )
            model.fit(fold_x_fit, fold_y_fit)
            probabilities = model.predict_proba(fold_x_validation)[:, 1]
            fold_pr_auc.append(float(average_precision_score(fold_y_validation, probabilities)))
            fold_roc_auc.append(float(roc_auc_score(fold_y_validation, probabilities)))
        mean_pr_auc = float(np.mean(fold_pr_auc))
        std_pr_auc = float(np.std(fold_pr_auc))
        rows.append({
            **candidate,
            "scale_pos_weight": class_ratio * multiplier,
            "scale_pos_weight_multiplier": multiplier,
            "validation_pr_auc": mean_pr_auc,
            "validation_pr_auc_std": std_pr_auc,
            "validation_pr_auc_min": float(np.min(fold_pr_auc)),
            "validation_roc_auc": float(np.mean(fold_roc_auc)),
            "selection_score": mean_pr_auc - 0.15 * std_pr_auc,
        })
        if index % 20 == 0 or index == len(candidates):
            print(f"XGBoost temporal search: {index}/{len(candidates)} candidates")

    results = pd.DataFrame(rows).sort_values(
        ["selection_score", "validation_pr_auc_min", "validation_roc_auc", "max_depth"],
        ascending=[False, False, False, True],
        ignore_index=True,
    )
    best_row = results.iloc[0]
    best = {
        "scale_pos_weight_multiplier": float(best_row["scale_pos_weight_multiplier"]),
        "scale_pos_weight": float(best_row["scale_pos_weight"]),
        "max_depth": int(best_row["max_depth"]),
        "n_estimators": int(best_row["n_estimators"]),
        "learning_rate": float(best_row["learning_rate"]),
        "min_child_weight": float(best_row["min_child_weight"]),
        "subsample": float(best_row["subsample"]),
        "colsample_bytree": float(best_row["colsample_bytree"]),
        "gamma": float(best_row["gamma"]),
        "reg_alpha": float(best_row["reg_alpha"]),
        "reg_lambda": float(best_row["reg_lambda"]),
        "max_delta_step": int(best_row["max_delta_step"]),
        "validation_pr_auc": float(best_row["validation_pr_auc"]),
        "validation_pr_auc_std": float(best_row["validation_pr_auc_std"]),
        "validation_pr_auc_min": float(best_row["validation_pr_auc_min"]),
        "validation_roc_auc": float(best_row["validation_roc_auc"]),
        "selection_score": float(best_row["selection_score"]),
        "temporal_folds": len(folds),
        "candidates_evaluated": len(results),
    }
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(metrics_dir / "xgboost_temporal_search.csv", index=False)
    (metrics_dir / "xgboost_temporal_search.json").write_text(
        json.dumps({"best": best, "candidates": results.to_dict(orient="records")}, indent=2) + "\n"
    )
    return best, results
