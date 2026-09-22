"""Turns layer output into sentences a non-specialist can act on."""
from __future__ import annotations

from ..schema import AnalysisResult

ACTIONS = {
    "CRITICAL": "Do not click anything in this email. Report it and delete it.",
    "HIGH_RISK": "Treat this as hostile until verified. Contact the sender through a channel you already trust.",
    "SUSPICIOUS": "Some signals are unusual. Check the sender address before acting on it.",
    "SAFE": "Nothing stood out. Normal caution applies.",
}
LAYER_LABEL = {
    "lexical": "Wording (statistical model)",
    "transformer": "Wording (language model)",
    "semantic": "Similarity to known messages",
    "anomaly": "Deviation from normal mail",
    "url": "Links",
    "header": "Sender authentication",
    "attachment": "Attachments",
    "impersonation": "Sender identity",
    "social_engineering": "Persuasion tactics",
}


def summarise(result: AnalysisResult) -> dict:
    lines = []
    for name, layer in result.layers.items():
        if layer.score is None:
            continue
        lines.append({
            "layer": name,
            "label": LAYER_LABEL.get(name, name),
            "score": round(layer.score, 3),
            "confidence": round(layer.confidence, 3),
            "evidence": layer.evidence,
            "latency_ms": round(layer.latency_ms, 2),
        })
    lines.sort(key=lambda d: -d["score"])
    return {
        "headline": f"{result.base_class.replace('_', ' ').title()} - risk {result.risk_score}/100",
        "action": ACTIONS.get(result.band, ""),
        "reasons": result.reasons,
        "tags": result.tags,
        "layers": lines,
        "skipped": [{"layer": n, "reason": l.error}
                    for n, l in result.layers.items() if l.score is None],
    }
