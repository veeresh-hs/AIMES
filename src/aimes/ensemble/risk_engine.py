"""The ensemble risk engine.

Takes the nine layer results and produces:
  * risk_score 0-100 (fused evidence)
  * base_class HAM / PROMOTIONAL_SPAM / PHISHING  (supervised, real labels)
  * threat tags derived from layer evidence, each with its trigger recorded
  * confidence, based on agreement between layers and their own confidences

Layers that abstain (score = None) are excluded and the remaining weights are
renormalised.  Nothing is substituted for a missing layer.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..config import DEFAULT_WEIGHTS, LAYER_NAMES, settings
from ..logging_conf import get_logger
from ..schema import AnalysisResult, LayerResult
from .thresholds import band_for

log = get_logger(__name__)

TAG_RULES = {
    "SUSPICIOUS_URL": ("url", 0.55),
    "SUSPICIOUS_ATTACHMENT": ("attachment", 0.5),
    "IMPERSONATION": ("impersonation", 0.45),
    "SOCIAL_ENGINEERING": ("social_engineering", 0.45),
    "ANOMALOUS": ("anomaly", 0.75),
}
SCAM_TERMS = ("payment_pressure", "reward")


class RiskEngine:
    def __init__(self) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        self.meta = None
        self.meta_layers: list[str] = LAYER_NAMES
        self.class_head = None
        self.fusion_note = "default prior weights (no fitted fusion model loaded)"
        self.fitted = False

    # ---------------------------------------------------------------- load
    def load(self, model_dir) -> "RiskEngine":
        path = Path(model_dir) / "fusion.joblib"
        if path.exists():
            import joblib

            bundle = joblib.load(path)
            self.meta = bundle.get("meta")
            self.meta_layers = list(bundle.get("layer_names", LAYER_NAMES))
            self.weights = dict(bundle.get("weights", self.weights))
            self.fusion_note = bundle.get("note", "")
            self.fitted = True
        head_path = Path(model_dir) / "class_head.joblib"
        if head_path.exists():
            import joblib

            self.class_head = joblib.load(head_path)
        return self

    # ------------------------------------------------------------- fusion
    def vectorise(self, layers: dict[str, LayerResult]) -> tuple[np.ndarray, np.ndarray]:
        scores, mask = [], []
        for name in self.meta_layers:
            result = layers.get(name)
            if result is not None and result.score is not None:
                scores.append(float(result.score))
                mask.append(1.0)
            else:
                scores.append(0.0)
                mask.append(0.0)
        return np.array(scores, dtype=float), np.array(mask, dtype=float)

    def weighted_score(self, layers: dict[str, LayerResult]) -> float:
        total, weight_sum = 0.0, 0.0
        for name, weight in self.weights.items():
            result = layers.get(name)
            if result is None or result.score is None:
                continue
            total += weight * result.score
            weight_sum += weight
        if weight_sum == 0:
            return 0.0
        return total / weight_sum

    def stacked_score(self, layers: dict[str, LayerResult]) -> float | None:
        if self.meta is None:
            return None
        scores, mask = self.vectorise(layers)
        features = np.hstack([scores, mask]).reshape(1, -1)
        return float(self.meta.predict_proba(features)[0, 1])

    # --------------------------------------------------------------- tags
    def derive_tags(self, layers: dict[str, LayerResult], base_class: str) -> tuple[list[str], list[str]]:
        tags, reasons = [], []
        for tag, (layer_name, cutoff) in TAG_RULES.items():
            result = layers.get(layer_name)
            if result is None or result.score is None or result.score < cutoff:
                continue
            tags.append(tag)
            top = result.evidence[0] if result.evidence else f"{layer_name} score {result.score:.2f}"
            reasons.append(top)

        se = layers.get("social_engineering")
        if se is not None and se.features:
            money = sum(se.features.get(k, 0) for k in SCAM_TERMS)
            if money >= 2 and base_class != "HAM":
                tags.append("FINANCIAL_SCAM")
                reasons.append("Money-transfer or prize language combined with a non-legitimate classification")
        return tags, reasons

    # ------------------------------------------------------------ predict
    def combine(self, layers: dict[str, LayerResult], message_id: str = "",
                model_version: str = "") -> AnalysisResult:
        available = {k: v for k, v in layers.items() if v.score is not None}
        if not available:
            return AnalysisResult(message_id=message_id, risk_score=0, band="SAFE",
                                  base_class="HAM", confidence=0.0, layers=layers,
                                  reasons=["No detection layer could analyse this message"],
                                  fusion="none", model_version=model_version)

        weighted = self.weighted_score(layers)
        stacked = self.stacked_score(layers)
        if settings.fusion == "stacking" and stacked is not None:
            fused, strategy = stacked, "stacking"
        else:
            fused, strategy = weighted, "weighted"

        risk = int(round(min(100.0, max(0.0, fused * 100))))

        # base class: lexical layer is the supervised source of truth
        lexical = layers.get("lexical")
        transformer = layers.get("transformer")
        probs: dict[str, float] = {}
        if transformer is not None and transformer.features.get("probs") and transformer.score is not None:
            probs = dict(transformer.features["probs"])
        if lexical is not None and lexical.features.get("probs"):
            lex = lexical.features["probs"]
            probs = ({k: 0.5 * lex.get(k, 0.0) + 0.5 * probs.get(k, 0.0) for k in set(lex) | set(probs)}
                     if probs else dict(lex))
        if probs:
            base_class = max(probs, key=probs.get)
        else:
            # No supervised layer available: fall back to a band-derived label and
            # say so, rather than presenting a rule threshold as a classification.
            base_class = "PHISHING" if risk >= settings.band_suspicious else "HAM"

        tags, tag_reasons = self.derive_tags(layers, base_class)

        # confidence: layer agreement (inverse spread) blended with own confidences
        values = [v.score for v in available.values()]
        spread = float(np.std(values)) if len(values) > 1 else 0.25
        agreement = math.exp(-3.0 * spread)
        own = float(np.mean([max(0.0, v.confidence) for v in available.values()]))
        coverage = len(available) / max(1, len(LAYER_NAMES))
        confidence = float(min(1.0, 0.45 * agreement + 0.35 * own + 0.20 * coverage))

        reasons = self.explain(layers, base_class, risk, tag_reasons)
        if not probs:
            reasons.append("Class shown is derived from the risk band: no trained "
                           "lexical or transformer layer was available to classify this message")

        return AnalysisResult(
            message_id=message_id,
            risk_score=risk,
            band=band_for(risk),
            base_class=base_class,
            base_class_probs={k: round(v, 4) for k, v in probs.items()},
            tags=tags,
            confidence=round(confidence, 4),
            layers=layers,
            reasons=reasons,
            fusion=strategy if self.fitted else f"{strategy} (unfitted prior weights)",
            model_version=model_version,
        )

    def explain(self, layers: dict[str, LayerResult], base_class: str,
                risk: int, tag_reasons: list[str]) -> list[str]:
        """Ranks layers by their actual contribution to this decision."""
        contributions = []
        for name, weight in self.weights.items():
            result = layers.get(name)
            if result is None or result.score is None:
                continue
            contributions.append((weight * result.score, name, result))
        contributions.sort(reverse=True, key=lambda t: t[0])

        reasons = []
        for contribution, name, result in contributions[:5]:
            if contribution < 0.01 and reasons:
                break
            head = result.evidence[0] if result.evidence else f"{name} score {result.score:.2f}"
            reasons.append(f"{head} [{name}, contributed {contribution * 100:.0f} of {risk}]")
        for reason in tag_reasons:
            if reason not in " ".join(reasons):
                reasons.append(reason)

        missing = [n for n in LAYER_NAMES if n not in layers or layers[n].score is None]
        if missing:
            reasons.append("Layers that could not run on this message: " + ", ".join(missing))
        return reasons
