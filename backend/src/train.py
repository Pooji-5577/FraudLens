"""Regenerate synthetic data, tune temporally, lock policy, and evaluate once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.isotonic import IsotonicRegression

from backend.data.generate_synthetic import SEED, generate_transactions
from backend.src.benchmark import (
    compare_probability_sets,
    fit_baselines,
    predict_probabilities,
    write_benchmark_artifacts,
)
from backend.src.calibration import TemporalCalibratedClassifier
from backend.src.evaluate import best_cost_threshold, sweep_thresholds, write_evaluation_artifacts
from backend.src.features import MODEL_FEATURES, engineer_features, model_matrix
from backend.src.tune import make_xgboost, temporal_xgboost_search


def _split_before_timestamp(frame, fraction: float):
    position = int(len(frame) * fraction)
    cutoff = frame.iloc[position]["timestamp"]
    return frame[frame["timestamp"] < cutoff], frame[frame["timestamp"] >= cutoff], cutoff


def train_pipeline(
    rows: int = 50_000,
    seed: int = SEED,
    root: Path = Path(__file__).resolve().parents[1],
) -> dict:
    data_path = root / "data" / "transactions.csv"
    model_path = root / "models" / "fraud_detector.joblib"
    report_path = root / "reports"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    raw = generate_transactions(rows, seed)
    raw.to_csv(data_path, index=False)
    featured = engineer_features(raw)
    train, test, test_cutoff = _split_before_timestamp(featured, 0.70)
    search_fit, validation, validation_cutoff = _split_before_timestamp(train, 0.80)
    x_search_fit = model_matrix(search_fit)
    y_search_fit = search_fit["is_fraud"].astype(int)
    x_validation = model_matrix(validation)
    y_validation = validation["is_fraud"].astype(int)
    x_train = model_matrix(train)
    y_train = train["is_fraud"].astype(int)
    x_test = model_matrix(test)
    y_test = test["is_fraud"].astype(int)

    best, _ = temporal_xgboost_search(
        x_search_fit, y_search_fit, x_validation, y_validation, report_path, seed
    )
    search_params = {
        "scale_pos_weight": best["scale_pos_weight"],
        "max_depth": best["max_depth"],
        "n_estimators": best["n_estimators"],
        "learning_rate": best["learning_rate"],
        "min_child_weight": best["min_child_weight"],
        "subsample": best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "gamma": best["gamma"],
        "reg_alpha": best["reg_alpha"],
        "reg_lambda": best["reg_lambda"],
        "max_delta_step": best["max_delta_step"],
    }

    calibration_base = make_xgboost(search_params, seed)
    calibration_base.fit(x_search_fit, y_search_fit)
    validation_raw = calibration_base.predict_proba(x_validation)[:, 1]
    validation_curve = sweep_thresholds(
        y_validation, validation_raw, cost_fp=5.0, cost_fn=500.0
    )
    validation_policy = best_cost_threshold(validation_curve)
    decision_threshold = float(validation_policy["threshold"])
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(validation_raw, y_validation)

    full_positives = int(y_train.sum())
    full_ratio = float((len(y_train) - full_positives) / full_positives)
    final_params = {
        **search_params,
        "scale_pos_weight": full_ratio * best["scale_pos_weight_multiplier"],
    }
    tuned_xgboost = make_xgboost(final_params, seed)
    tuned_xgboost.fit(x_train, y_train)
    xgboost_raw_probabilities = tuned_xgboost.predict_proba(x_test)[:, 1]
    calibrated_xgboost = TemporalCalibratedClassifier(tuned_xgboost, calibrator)
    xgboost_calibrated_probabilities = calibrated_xgboost.predict_proba(x_test)[:, 1]

    baselines = fit_baselines(x_train, y_train, seed)
    probability_sets = {
        "logistic_regression": predict_probabilities(
            baselines["logistic_regression"], x_test, "logistic_regression"
        ),
        "random_forest": predict_probabilities(
            baselines["random_forest"], x_test, "random_forest"
        ),
        "tuned_xgboost_uncalibrated": xgboost_raw_probabilities,
        "tuned_xgboost_calibrated": xgboost_calibrated_probabilities,
    }
    comparison = compare_probability_sets(y_test, probability_sets)
    write_benchmark_artifacts(comparison, report_path)

    # The test-set ranking above is descriptive. It must not choose the deployed
    # artifact. The production family is the temporally tuned, uncalibrated
    # XGBoost model, and its operating threshold was locked on validation data.
    production_model_name = "tuned_xgboost_uncalibrated"
    validation_selection = {
        "threshold": decision_threshold,
        "rows": int(len(validation)),
        "started": str(validation["timestamp"].min()),
        "ended": str(validation["timestamp"].max()),
        "metrics": validation_policy,
    }
    summary = write_evaluation_artifacts(
        y_test,
        xgboost_raw_probabilities,
        report_path,
        decision_threshold=decision_threshold,
        validation_selection=validation_selection,
    )
    summary["evaluation_period"] = {
        "started": str(test["timestamp"].min()),
        "ended": str(test["timestamp"].max()),
        "cost_unit": "illustrative dollars; not observed loss",
    }
    summary["model_comparison"] = comparison.to_dict(orient="records")
    summary["model_comparison_use"] = (
        "descriptive held-out benchmark only; not used to select the saved model"
    )
    summary["production_model"] = production_model_name
    summary["production_model_selection"] = (
        "XGBoost hyperparameters selected by expanding temporal validation; "
        "the held-out test ranking did not select the saved model"
    )
    summary["xgboost_search_best"] = best
    (report_path / "metrics" / "evaluation.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    artifact = {
        "model": tuned_xgboost,
        "explanation_model": tuned_xgboost,
        "model_name": production_model_name,
        "feature_names": MODEL_FEATURES,
        "threshold": decision_threshold,
        "threshold_selected_on": "validation",
        "cost_fp": summary["cost_fp"],
        "cost_fn": summary["cost_fn"],
        "training_ended": str(train["timestamp"].max()),
        "validation_started": str(validation_cutoff),
        "test_started": str(test_cutoff),
        "xgboost_best_params": final_params,
        "seed": seed,
    }
    joblib.dump(artifact, model_path)
    print(json.dumps(summary, indent=2))
    print(f"production model: {production_model_name}")
    print(f"validation-selected threshold: {decision_threshold:.3f}")
    print(f"saved model to {model_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train_pipeline(rows=args.rows, seed=args.seed)


if __name__ == "__main__":
    main()
