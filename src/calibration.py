"""Temporal probability calibration wrappers."""

from __future__ import annotations

import numpy as np


class TemporalCalibratedClassifier:
    """Apply a pre-fitted one-dimensional calibrator to classifier scores."""

    def __init__(self, base_model, calibrator) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, matrix) -> np.ndarray:
        raw = self.base_model.predict_proba(matrix)[:, 1]
        positive = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])
