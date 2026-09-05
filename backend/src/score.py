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
            featured = engineer_features(transactions, self.state)
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
