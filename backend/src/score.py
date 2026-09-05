"""Stateful single-transaction and chronological batch scoring."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd

from backend.src.explain import explain_flagged
from backend.src.features import FeatureState, engineer_features, model_matrix
from backend.src.report import generate_report


class FraudScorer:
    def __init__(
        self,
        model_path: str | Path = Path(__file__).resolve().parents[1]
        / "models"
        / "fraud_detector.joblib",
    ) -> None:
        self.artifact = joblib.load(model_path)
        self.model = self.artifact["model"]
        self.explanation_model = self.artifact.get("explanation_model", self.model)
        self.threshold = float(self.artifact["threshold"])
        self.state = FeatureState()
        self._lock = Lock()

    def score_batch(self, transactions: pd.DataFrame) -> pd.DataFrame:
        with self._lock:
            data = transactions.copy()
            # ``velocity`` is accepted as a friendly upload field as well as
            # the internal ``uploaded_velocity_per_hour`` name.  Keep the
            # original value for the result contract, while letting the
            # existing feature pipeline use it as an optional prior signal.
            if "velocity" in data.columns:
                velocity = pd.to_numeric(data["velocity"], errors="coerce")
                if "uploaded_velocity_per_hour" not in data.columns:
                    data["uploaded_velocity_per_hour"] = velocity
                else:
                    existing_velocity = pd.to_numeric(
                        data["uploaded_velocity_per_hour"], errors="coerce"
                    )
                    data["uploaded_velocity_per_hour"] = existing_velocity.fillna(velocity)

            featured = engineer_features(data, self.state)
            if "uploaded_velocity_per_hour" in featured.columns:
                velocity = pd.to_numeric(
                    featured["uploaded_velocity_per_hour"], errors="coerce"
                ).fillna(0.0)
                for column in ("card_txn_count_1h", "device_txn_count_1h"):
                    featured[column] = featured[column].where(
                        featured[column] >= velocity, velocity
                    )
                for column in ("card_txn_count_24h", "device_txn_count_24h"):
                    featured[column] = featured[column].where(
                        featured[column] >= velocity, velocity
                    )
                featured["card_velocity_1h_log"] = np.log1p(featured["card_txn_count_1h"])
                featured["device_velocity_1h_log"] = np.log1p(featured["device_txn_count_1h"])
            probabilities = self.model.predict_proba(model_matrix(featured))[:, 1]
            reasons = explain_flagged(self.explanation_model, featured, probabilities, self.threshold)
            for index, probability in enumerate(probabilities):
                decision = (
                    f"Score {probability:.3f} met the {self.threshold:.3f} review threshold"
                    if probability >= self.threshold
                    else f"Score {probability:.3f} stayed below the {self.threshold:.3f} review threshold"
                )
                reasons[index] = [*reasons[index], decision]
        result = featured.copy().reset_index(drop=True)
        result["score"] = probabilities
        result["flagged"] = probabilities >= self.threshold
        result["blocked"] = result["flagged"]
        result["reasons"] = reasons

        # These are the stable, reviewer-facing parameters returned by the
        # batch API and persisted for uploaded datasets.  The raw engineered
        # columns remain in the frame too, so downstream consumers can inspect
        # exactly what the model saw.
        computed_velocity = result[["card_txn_count_1h", "device_txn_count_1h"]].max(axis=1)
        if "uploaded_velocity_per_hour" in result.columns:
            provided_velocity = pd.to_numeric(
                result["uploaded_velocity_per_hour"], errors="coerce"
            )
            result["velocity"] = provided_velocity.where(
                provided_velocity.notna(), computed_velocity
            )
        else:
            result["velocity"] = computed_velocity

        computed_ip_billing = pd.Series(
            np.where(result["geo_mismatch"].astype(bool), "Mismatch", "Match"),
            index=result.index,
        )
        if "ip_billing" in result.columns:
            provided_ip_billing = result["ip_billing"].astype("string").str.strip()
            result["ip_billing"] = provided_ip_billing.where(
                provided_ip_billing.notna() & provided_ip_billing.ne(""),
                computed_ip_billing,
            )
        else:
            result["ip_billing"] = computed_ip_billing

        computed_device = pd.Series(
            np.where(result["is_new_device"].astype(bool), "New", "Known"),
            index=result.index,
        )
        if "device" in result.columns:
            provided_device = result["device"].astype("string").str.strip()
            result["device"] = provided_device.where(
                provided_device.notna() & provided_device.ne(""), computed_device
            )
        else:
            result["device"] = computed_device

        # The compact reviewer table expresses deviation as percentage versus
        # the user's prior mean. Keep the model's exact z-score available as
        # ``user_amount_zscore`` in the raw result for evidence/reporting.
        computed_amount_deviation = (
            pd.to_numeric(result["amount_to_user_mean_ratio"], errors="coerce") - 1.0
        ) * 100.0
        if "amount_deviation" in result.columns:
            provided_amount_deviation = pd.to_numeric(
                result["amount_deviation"], errors="coerce"
            )
            result["amount_deviation"] = provided_amount_deviation.where(
                provided_amount_deviation.notna(), computed_amount_deviation
            )
        else:
            result["amount_deviation"] = computed_amount_deviation

        computed_hour = pd.to_datetime(result["timestamp"], utc=True).dt.hour
        if "hour" in result.columns:
            provided_hour = pd.to_numeric(result["hour"], errors="coerce")
            result["hour"] = provided_hour.where(provided_hour.notna(), computed_hour)
        else:
            result["hour"] = computed_hour
        result["hour"] = result["hour"].round().astype(int)
        result["status"] = np.where(result["flagged"], "Flagged", "Not flagged")
        if "actual" not in result.columns:
            result["actual"] = None
        return result

    def score_one(self, transaction: dict, include_report: bool = False) -> dict:
        result = self.score_batch(pd.DataFrame([transaction])).iloc[0]
        response = {
            "score": float(result["score"]),
            "flagged": bool(result["flagged"]),
            "blocked": bool(result["blocked"]),
            "reasons": result["reasons"],
        }
        if include_report and response["blocked"]:
            report = generate_report(
                result.to_dict(), response["score"], response["blocked"], self.threshold, response["reasons"]
            )
            if report is not None:
                response["report"] = report
        return response
