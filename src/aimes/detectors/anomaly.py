"""Layer 4 - anomaly detection.

Isolation Forest fitted on legitimate mail only.  A high anomaly score means
"this message does not look like the legitimate mail the model has seen"; it is
NOT by itself evidence that the message is malicious, and the ensemble gives it
a correspondingly small weight.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Detector, clamp01
from ..schema import LayerResult, ParsedEmail

BUNDLE = "anomaly_model.joblib"


class AnomalyDetector(Detector):
    name = "anomaly"

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []
        self.lo = -0.5
        self.hi = 0.5

    def load(self, model_dir) -> "AnomalyDetector":
        path = Path(model_dir) / BUNDLE
        if not path.exists():
            return self
        import joblib

        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_names = list(bundle["feature_names"])
        self.lo = float(bundle.get("score_p01", -0.5))
        self.hi = float(bundle.get("score_p99", 0.5))
        self.model_backed = True
        return self

    def score_vector(self, vector: dict[str, float]) -> float:
        row = np.array([[float(vector.get(k, 0.0)) for k in self.feature_names]])
        row = self.scaler.transform(row)
        # decision_function: higher = more normal.  Map to 0-1 where 1 = odd.
        raw = float(self.model.decision_function(row)[0])
        span = max(1e-6, self.hi - self.lo)
        return clamp01(1.0 - (raw - self.lo) / span)

    def _analyze(self, email: ParsedEmail) -> LayerResult:
        if self.model is None:
            return LayerResult(name=self.name, score=None, available=False, confidence=0.0,
                               error="No anomaly model. Run scripts/train_all.py first.")
        from ..ml.features import tabular_features

        vector = tabular_features(email)
        score = self.score_vector(vector)
        note = ("Message sits outside the distribution of known legitimate mail"
                if score > 0.7 else
                "Message is structurally similar to known legitimate mail")
        return LayerResult(
            name=self.name,
            score=score,
            confidence=0.5,
            features={"n_features": len(self.feature_names),
                      "note": "Deviation from normal, not proof of malice"},
            evidence=[f"{note} (deviation {score * 100:.0f}/100)"],
        )
