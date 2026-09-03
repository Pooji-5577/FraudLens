"""Fair model comparisons on the same point-in-time features and test window."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_baselines(x_train: pd.DataFrame, y_train: pd.Series, seed: int) -> dict:
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=1_000, random_state=seed, solver="liblinear"
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
            model.fit(x_train, y_train)
        predict_probabilities(model, x_train.iloc[:10], name)
    return models


def predict_probabilities(model, matrix: pd.DataFrame, name: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
        probabilities = model.predict_proba(matrix)[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"{name} produced non-finite probabilities")
    return probabilities


def compare_probability_sets(y_test, probability_sets: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, probabilities in probability_sets.items():
        probabilities = np.asarray(probabilities, dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{name} contains non-finite probabilities")
        rows.append({
            "model": name,
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        })
    return pd.DataFrame(rows).sort_values("average_precision", ascending=False, ignore_index=True)


def write_benchmark_artifacts(results: pd.DataFrame, output_dir: Path) -> None:
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    old_csv = metrics_dir / "model_comparison.csv"
    old_json = metrics_dir / "model_comparison.json"
    old_plot = figures_dir / "model_comparison.png"
    if old_csv.exists() and not (metrics_dir / "model_comparison_initial.csv").exists():
        shutil.copy2(old_csv, metrics_dir / "model_comparison_initial.csv")
    if old_json.exists() and not (metrics_dir / "model_comparison_initial.json").exists():
        shutil.copy2(old_json, metrics_dir / "model_comparison_initial.json")
    if old_plot.exists() and not (figures_dir / "model_comparison_initial.png").exists():
        shutil.copy2(old_plot, figures_dir / "model_comparison_initial.png")

    results.to_csv(metrics_dir / "model_comparison_tuned.csv", index=False)
    results.to_csv(old_csv, index=False)
    payload = json.dumps(results.to_dict(orient="records"), indent=2) + "\n"
    (metrics_dir / "model_comparison_tuned.json").write_text(payload)
    old_json.write_text(payload)

    plot = results.set_index("model")[["average_precision", "roc_auc"]]
    ax = plot.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=10)
    ax.set(title="Temporal held-out model comparison after XGBoost tuning", xlabel="", ylabel="Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["PR-AUC (average precision)", "ROC-AUC"])
    ax.figure.tight_layout()
    ax.figure.savefig(figures_dir / "model_comparison_tuned.png", dpi=150)
    ax.figure.savefig(old_plot, dpi=150)
    plt.close(ax.figure)
