"""Layer 1 - lexical machine learning (TF-IDF + classical classifier).

The trained bundle is produced by aimes.ml.training and holds the fitted
vectoriser, the winning classifier and the class order.  Nothing is hard-coded:
if no bundle exists on disk the layer reports itself unavailable.
"""
from __future__ import annotations

from pathlib import Path

from .base import Detector
from ..schema import LayerResult, ParsedEmail

BUNDLE = "lexical_model.joblib"


class LexicalDetector(Detector):
    name = "lexical"

    def __init__(self) -> None:
        self.pipeline = None
        self.classes: list[str] = []
        self.algorithm = ""

    def load(self, model_dir) -> "LexicalDetector":
        path = Path(model_dir) / BUNDLE
        if not path.exists():
            return self
        import joblib

        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.classes = list(bundle["classes"])
        self.algorithm = bundle.get("algorithm", "unknown")
        self.model_backed = True
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        probs = self.pipeline.predict_proba([text])[0]
        return {cls: float(p) for cls, p in zip(self.classes, probs)}

    def _analyze(self, email: ParsedEmail) -> LayerResult:
        if self.pipeline is None:
            return LayerResult(name=self.name, score=None, available=False, confidence=0.0,
                               error="No trained lexical model. Run scripts/train_all.py first.")
        text = email.combined_text
        if not text.strip():
            return LayerResult(name=self.name, score=None, available=False,
                               confidence=0.0, error="Empty message body")

        probs = self.predict_proba(text)
        ham = probs.get("HAM", 0.0)
        score = 1.0 - ham
        top = max(probs, key=probs.get)
        margin = sorted(probs.values(), reverse=True)
        confidence = float(margin[0] - (margin[1] if len(margin) > 1 else 0.0))

        return LayerResult(
            name=self.name,
            score=score,
            confidence=confidence,
            features={"algorithm": self.algorithm,
                      "probs": {k: round(v, 4) for k, v in probs.items()},
                      "predicted_class": top},
            evidence=[f"{self.algorithm} on TF-IDF features predicts {top} "
                      f"({probs[top] * 100:.1f}% of the probability mass)"],
        )
