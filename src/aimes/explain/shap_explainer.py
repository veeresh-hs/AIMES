"""SHAP explanations.

Two places where SHAP is meaningful in this system:
  * the fusion meta-model - which layer pushed this email over the line
  * the lexical model - which terms drove the base class

For the transformer layer we deliberately do NOT present attention weights as
explanations; attention is not a causal attribution.  Token attributions there
are optional and labelled as supporting evidence only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..logging_conf import get_logger

log = get_logger(__name__)


def _shap():
    try:
        import shap  # type: ignore

        return shap
    except Exception:
        return None


def explain_fusion(model_dir: Path, layer_scores: dict[str, float | None]) -> list[dict]:
    """Per-layer contribution to the fused probability.

    Falls back to weight x score when shap is not installed, which is the exact
    decomposition the weighted ensemble uses anyway.
    """
    import joblib

    path = Path(model_dir) / "fusion.joblib"
    if not path.exists():
        return []
    bundle = joblib.load(path)
    names = list(bundle["layer_names"])
    meta = bundle["meta"]
    scores = np.array([[layer_scores.get(n) or 0.0 for n in names]])
    mask = np.array([[1.0 if layer_scores.get(n) is not None else 0.0 for n in names]])
    row = np.hstack([scores, mask])

    shap = _shap()
    if shap is not None:
        try:
            background = np.zeros_like(row)
            explainer = shap.LinearExplainer(meta, background)
            values = np.asarray(explainer.shap_values(row)).reshape(-1)[: len(names)]
            return [{"layer": n, "contribution": float(v), "score": layer_scores.get(n),
                     "method": "shap"}
                    for n, v in sorted(zip(names, values), key=lambda t: -abs(t[1]))]
        except Exception as exc:
            log.warning("shap explanation failed (%s), using coefficient decomposition",
                        type(exc).__name__)

    coefs = meta.coef_[0][: len(names)]
    values = coefs * scores[0]
    return [{"layer": n, "contribution": float(v), "score": layer_scores.get(n),
             "method": "coefficient x score"}
            for n, v in sorted(zip(names, values), key=lambda t: -abs(t[1]))]


def explain_lexical(model_dir: Path, text: str, top_k: int = 12) -> list[dict]:
    """Terms that pushed the lexical model towards its prediction."""
    import joblib

    path = Path(model_dir) / "lexical_model.joblib"
    if not path.exists():
        return []
    bundle = joblib.load(path)
    pipeline = bundle["pipeline"]
    vectoriser = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]

    coef = getattr(clf, "coef_", None)
    if coef is None:  # calibrated or tree models expose no direct coefficients
        base = getattr(clf, "calibrated_classifiers_", None)
        if base:
            inner = getattr(base[0], "estimator", None)
            coef = getattr(inner, "coef_", None)
    if coef is None:
        return []

    classes = list(bundle["classes"])
    vector = vectoriser.transform([text])
    try:
        names = vectoriser.get_feature_names_out()
    except Exception:
        return []

    index = vector.nonzero()[1]
    out = []
    for cls_idx, cls in enumerate(classes):
        weights = coef[cls_idx] if coef.shape[0] > 1 else coef[0]
        pairs = [(names[i], float(weights[i] * vector[0, i])) for i in index]
        pairs.sort(key=lambda t: -abs(t[1]))
        out.append({"class": cls, "terms": [{"term": t, "weight": round(w, 4)}
                                            for t, w in pairs[:top_k]]})
    return out
