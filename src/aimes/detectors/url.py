"""Layer 5 - URL analysis.

Pure lexical analysis of URL strings.  The detector NEVER resolves, fetches or
follows a URL; that is a deliberate safety property, not an omission.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl

from .base import Detector, clamp01
from ..schema import LayerResult, ParsedEmail
from ..textutils import defang, host_of, registered_domain

IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
HEX_IP_RE = re.compile(r"^0x[0-9a-f]+$", re.I)
SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "cutt.ly", "rb.gy", "shorturl.at", "rebrand.ly", "tiny.cc",
    "bl.ink", "trib.al", "lnkd.in", "t.ly", "shorte.st",
}
RISKY_TLDS = {
    "zip", "mov", "tk", "ml", "ga", "cf", "gq", "top", "xyz", "work", "click",
    "link", "country", "stream", "download", "racing", "loan", "date", "review",
    "kim", "men", "party", "science", "gdn", "bid", "win", "rest", "cam", "buzz",
}
SENSITIVE_WORDS = (
    "login", "signin", "verify", "verification", "secure", "security", "account",
    "update", "confirm", "password", "credential", "bank", "wallet", "invoice",
    "payment", "billing", "unlock", "suspend", "recover", "authenticate", "otp",
    "webscr", "ebayisapi", "support", "helpdesk",
)
BRAND_WORDS = (
    "paypal", "microsoft", "office365", "apple", "icloud", "google", "gmail",
    "amazon", "netflix", "facebook", "instagram", "whatsapp", "linkedin",
    "dhl", "fedex", "usps", "hsbc", "chase", "sbi", "hdfc", "icici", "axis",
    "coinbase", "binance", "metamask", "irs", "dropbox", "adobe", "steam",
)

FEATURE_NAMES = [
    "url_len", "host_len", "path_len", "query_len", "n_dots", "n_subdomains",
    "n_hyphens", "n_digits_host", "digit_ratio_host", "is_ip", "is_https",
    "has_at", "n_pct_encoded", "n_slashes", "n_query_params", "n_equals",
    "has_port", "tld_risky", "is_shortener", "sensitive_word_hits",
    "brand_in_subdomain", "brand_not_in_regdomain", "host_entropy",
    "longest_token", "n_underscores", "double_slash_in_path", "punycode",
]


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def url_features(url: str) -> dict[str, float]:
    if "://" not in url:
        url = "http://" + url
    try:
        parts = urlsplit(url)
    except ValueError:
        parts = urlsplit("http://invalid.invalid")
    host = (parts.hostname or "").lower()
    path, query = parts.path or "", parts.query or ""
    regdom = registered_domain(host)
    tld = regdom.rsplit(".", 1)[-1] if "." in regdom else ""
    sub = host[: -len(regdom)].strip(".") if regdom and host.endswith(regdom) else ""
    digits = sum(c.isdigit() for c in host)
    lowered = url.lower()
    tokens = re.split(r"[^a-zA-Z0-9]+", lowered)

    return {
        "url_len": len(url),
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "n_dots": host.count("."),
        "n_subdomains": max(0, len(sub.split(".")) if sub else 0),
        "n_hyphens": host.count("-"),
        "n_digits_host": digits,
        "digit_ratio_host": digits / len(host) if host else 0.0,
        "is_ip": 1.0 if (IP_RE.match(host) or HEX_IP_RE.match(host)) else 0.0,
        "is_https": 1.0 if parts.scheme == "https" else 0.0,
        "has_at": 1.0 if "@" in url.split("://", 1)[-1].split("/", 1)[0] else 0.0,
        "n_pct_encoded": lowered.count("%"),
        "n_slashes": url.count("/"),
        "n_query_params": len(parse_qsl(query)),
        "n_equals": url.count("="),
        "has_port": 1.0 if parts.port not in (None, 80, 443) else 0.0,
        "tld_risky": 1.0 if tld in RISKY_TLDS else 0.0,
        "is_shortener": 1.0 if regdom in SHORTENERS else 0.0,
        "sensitive_word_hits": sum(w in lowered for w in SENSITIVE_WORDS),
        "brand_in_subdomain": 1.0 if any(b in sub or b in path.lower() for b in BRAND_WORDS) else 0.0,
        "brand_not_in_regdomain": 1.0 if (any(b in lowered for b in BRAND_WORDS)
                                          and not any(b in regdom for b in BRAND_WORDS)) else 0.0,
        "host_entropy": shannon_entropy(host),
        "longest_token": max((len(t) for t in tokens), default=0),
        "n_underscores": url.count("_"),
        "double_slash_in_path": 1.0 if "//" in path else 0.0,
        "punycode": 1.0 if "xn--" in host else 0.0,
    }


def heuristic_url_score(f: dict[str, float]) -> tuple[float, list[str]]:
    """Transparent fallback used until models/<v>/url_model.joblib exists."""
    score, why = 0.0, []
    if f["is_ip"]:
        score += 0.30; why.append("Host is a bare IP address rather than a domain")
    if f["has_at"]:
        score += 0.20; why.append("'@' in the authority section can hide the real host")
    if f["punycode"]:
        score += 0.15; why.append("Punycode host may imitate a lookalike domain")
    if f["tld_risky"]:
        score += 0.15; why.append("Top-level domain is frequently abused")
    if f["is_shortener"]:
        score += 0.12; why.append("Shortened link hides its destination")
    if f["brand_not_in_regdomain"]:
        score += 0.18; why.append("Brand name appears outside the registered domain")
    if f["sensitive_word_hits"] >= 2:
        score += 0.12; why.append("Several credential-related words in the link")
    elif f["sensitive_word_hits"] == 1:
        score += 0.06
    if f["url_len"] > 110:
        score += 0.10; why.append("Unusually long URL")
    if f["n_subdomains"] >= 3:
        score += 0.10; why.append("Deeply nested subdomains")
    if f["n_pct_encoded"] >= 4:
        score += 0.08; why.append("Heavy percent-encoding in the link")
    if f["n_hyphens"] >= 4:
        score += 0.06; why.append("Many hyphens in the host name")
    if not f["is_https"]:
        score += 0.05
    if f["has_port"]:
        score += 0.05; why.append("Non-standard port")
    return clamp01(score), why


class URLDetector(Detector):
    name = "url"

    def __init__(self) -> None:
        self.model = None

    def load(self, model_dir) -> "URLDetector":
        path = Path(model_dir) / "url_model.joblib"
        if path.exists():
            import joblib

            bundle = joblib.load(path)
            self.model = bundle["model"]
            self.model_backed = True
        return self

    def score_url(self, url: str) -> tuple[float, dict[str, float], list[str]]:
        feats = url_features(url)
        heur, why = heuristic_url_score(feats)
        if self.model is not None:
            import numpy as np

            vec = np.array([[feats[k] for k in FEATURE_NAMES]], dtype=float)
            prob = float(self.model.predict_proba(vec)[0, 1])
            # keep the heuristic as a floor so obvious tricks are never missed
            return max(prob, heur * 0.6), feats, why
        return heur, feats, why

    def _analyze(self, email: ParsedEmail) -> LayerResult:
        if not email.urls:
            return LayerResult(name=self.name, score=0.0, confidence=0.6,
                               features={"n_urls": 0}, evidence=["No links in this email"])
        scored = []
        for url in email.urls[:50]:
            s, feats, why = self.score_url(url)
            scored.append((s, url, feats, why))
        scored.sort(reverse=True, key=lambda t: t[0])
        top = scored[0]
        mean = sum(s for s, *_ in scored) / len(scored)
        # worst link dominates, average keeps a mass-of-bad-links signal
        score = 0.75 * top[0] + 0.25 * mean

        evidence = [f"Riskiest link ({top[0] * 100:.0f}/100): {defang(top[1])}"]
        evidence += [f"  - {w}" for w in top[3][:4]]
        if len(scored) > 1:
            evidence.append(f"{len(scored)} links analysed, mean risk {mean * 100:.0f}/100")

        return LayerResult(
            name=self.name,
            score=score,
            confidence=0.8 if self.model_backed else 0.6,
            features={
                "n_urls": len(email.urls),
                "max_url_risk": round(top[0], 4),
                "mean_url_risk": round(mean, 4),
                "model_backed": self.model_backed,
                "top_url_defanged": defang(top[1]),
                "top_url_features": {k: round(v, 3) for k, v in top[2].items()},
                "all_urls_defanged": [defang(u) for _, u, _, _ in scored[:10]],
                "distinct_domains": len({registered_domain(host_of(u)) for _, u, _, _ in scored}),
            },
            evidence=evidence,
        )
