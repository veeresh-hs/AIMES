"""Layer 2 - transformer classifier (ADVANCED mode only).

Loads a fine-tuned DistilBERT-style model from models/<version>/transformer/.
If transformers/torch are not installed, or no fine-tuned model exists, the
layer reports itself unavailable and the ensemble renormalises without it.
Nothing is simulated.
"""
from __future__ import annotations

from pathlib import Path

from .base import Detector
from ..config import settings
from ..schema import LayerResult, ParsedEmail

SUBDIR = "transformer"


class TransformerDetector(Detector):
    name = "transformer"

    def __init__(self) -> None:
        self.pipe = None
        self.classes: list[str] = []
        self.reason = "not loaded"
        self.device = "cpu"

    def load(self, model_dir) -> "TransformerDetector":
        if settings.mode != "ADVANCED":
            self.reason = "FAST mode: transformer layer disabled"
            return self
        path = Path(model_dir) / SUBDIR
        if not path.exists():
            self.reason = "No fine-tuned transformer in this model version"
            return self
        try:
            import torch  # noqa: F401
            from transformers import (AutoModelForSequenceClassification,  # type: ignore
                                      AutoTokenizer)
        except Exception as exc:
            self.reason = f"transformers/torch unavailable: {type(exc).__name__}"
            return self
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(str(path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(self.device)
            self.model.eval()
            label_map = self.model.config.id2label
            self.classes = [label_map[i] for i in sorted(label_map)]
            self.pipe = True
            self.model_backed = True
        except Exception as exc:  # pragma: no cover
            self.reason = f"load failed: {type(exc).__name__}: {exc}"
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        import torch

        batch = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**batch).logits[0]
            probs = torch.softmax(logits, dim=-1).tolist()
        return {cls: float(p) for cls, p in zip(self.classes, probs)}

    def _analyze(self, email: ParsedEmail) -> LayerResult:
        if self.pipe is None:
            return LayerResult(name=self.name, score=None, available=False,
                               confidence=0.0, error=self.reason)
        text = email.combined_text
        if not text.strip():
            return LayerResult(name=self.name, score=None, available=False,
                               confidence=0.0, error="Empty message body")
        probs = self.predict_proba(text)
        ham = probs.get("HAM", 0.0)
        top = max(probs, key=probs.get)
        ordered = sorted(probs.values(), reverse=True)
        return LayerResult(
            name=self.name,
            score=1.0 - ham,
            confidence=float(ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)),
            features={"model": settings.transformer_model,
                      "probs": {k: round(v, 4) for k, v in probs.items()},
                      "predicted_class": top},
            evidence=[f"Fine-tuned {settings.transformer_model} predicts {top} "
                      f"({probs[top] * 100:.1f}%)"],
        )