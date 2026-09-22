"""Plain dataclasses shared by every layer.  No third-party dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


@dataclass
class Attachment:
    filename: str = ""
    content_type: str = ""
    size: int = 0
    sha256: str = ""
    # raw bytes are deliberately NOT stored on the instance: attachments are
    # inspected in memory and discarded.  See detectors/attachment.py.
    has_macro_marker: bool = False


@dataclass
class ParsedEmail:
    message_id: str = ""
    thread_id: str = ""
    sender: str = ""
    sender_name: str = ""
    sender_domain: str = ""
    to: str = ""
    cc: str = ""
    reply_to: str = ""
    return_path: str = ""
    subject: str = ""
    date: Optional[datetime] = None
    body_text: str = ""
    body_html: str = ""
    clean_text: str = ""
    urls: list[str] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)
    received_chain: list[str] = field(default_factory=list)
    auth_results: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    size_bytes: int = 0
    body_sha256: str = ""
    has_headers: bool = True  # False for body-only corpora (e.g. Enron-Spam)

    @property
    def combined_text(self) -> str:
        return f"{self.subject}\n{self.clean_text}".strip()


@dataclass
class LayerResult:
    """Output of one detection layer.

    score       0.0 - 1.0, higher = more threat-like.  None means the layer
                could not run on this email (e.g. header layer on a body-only
                corpus).  Callers must renormalise weights over available
                layers rather than substituting a guess.
    confidence  0.0 - 1.0, how much the layer trusts its own score.
    """

    name: str
    score: Optional[float]
    confidence: float = 0.5
    features: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    available: bool = True
    error: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    message_id: str = ""
    risk_score: int = 0
    band: str = "SAFE"
    base_class: str = "HAM"
    base_class_probs: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    layers: dict[str, LayerResult] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    fusion: str = "weighted"
    model_version: str = ""
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["layers"] = {k: v.to_dict() if isinstance(v, LayerResult) else v
                       for k, v in self.layers.items()}
        return d
