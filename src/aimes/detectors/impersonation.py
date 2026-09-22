"""Layer 8 - display-name / domain impersonation.

Keeps a small, editable organisation list (data/brands.txt overrides it) rather
than an enormous hard-coded database, and relies on edit distance and homoglyph
normalisation so unseen brands still trigger on lookalike domains.
"""
from __future__ import annotations

import re
from pathlib import Path

from .base import Detector, clamp01
from ..config import settings
from ..schema import LayerResult, ParsedEmail
from ..textutils import domain_of_address, registered_domain, similarity, levenshtein

DEFAULT_BRANDS = {
    "paypal": "paypal.com", "microsoft": "microsoft.com", "office365": "microsoft.com",
    "apple": "apple.com", "icloud": "apple.com", "google": "google.com",
    "amazon": "amazon.com", "netflix": "netflix.com", "facebook": "facebook.com",
    "instagram": "instagram.com", "linkedin": "linkedin.com", "dhl": "dhl.com",
    "fedex": "fedex.com", "dropbox": "dropbox.com", "adobe": "adobe.com",
    "whatsapp": "whatsapp.com", "coinbase": "coinbase.com", "binance": "binance.com",
    "sbi": "sbi.co.in", "hdfc": "hdfcbank.com", "icici": "icicibank.com",
}
HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
                            "7": "t", "$": "s", "|": "l", "!": "i"})
ROLE_WORDS = ("support", "security", "helpdesk", "service", "billing", "admin",
              "team", "alert", "notification", "no-reply", "noreply", "account")


def load_brands() -> dict[str, str]:
    """data/brands.txt lines: name,canonical-domain - keeps the list extensible."""
    brands = dict(DEFAULT_BRANDS)
    path = Path(settings.data_dir) / "brands.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "," not in line:
                continue
            name, domain = line.split(",", 1)
            brands[name.strip().lower()] = domain.strip().lower()
    return brands


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower().translate(HOMOGLYPHS))


class ImpersonationDetector(Detector):
    name = "impersonation"

    def __init__(self) -> None:
        self.brands = load_brands()

    def _analyze(self, email: ParsedEmail) -> LayerResult:
        if not email.sender and not email.sender_name:
            return LayerResult(name=self.name, score=None, available=False,
                               confidence=0.0, error="No sender information")

        display = email.sender_name or ""
        from_dom = registered_domain(email.sender_domain)
        reply_dom = registered_domain(domain_of_address(email.reply_to))
        norm_display = normalise(display)
        norm_from = normalise(from_dom.split(".")[0]) if from_dom else ""

        score, evidence, matched = 0.0, [], ""

        for brand, canonical in self.brands.items():
            if brand in norm_display:
                matched = brand
                if from_dom and registered_domain(canonical) != from_dom:
                    dist = levenshtein(from_dom, registered_domain(canonical))
                    if 0 < dist <= 3:
                        score += 0.55
                        evidence.append(
                            f"Sender domain {from_dom} is {dist} character(s) from the real {canonical}")
                    else:
                        score += 0.40
                        evidence.append(
                            f"Display name claims {brand.title()} but the mail comes from {from_dom}")
                break

        if not matched and from_dom:
            # unseen brand: lookalike of the *display* name itself
            if norm_display and norm_from and norm_display != norm_from:
                sim = similarity(norm_display[:20], norm_from[:20])
                if 0.6 <= sim < 0.95:
                    score += 0.25
                    evidence.append(
                        f"Display name '{display}' is similar to but not the same as the sending domain {from_dom}")

        if re.search(r"[a-z]@[a-z]", display, re.I) or "@" in display:
            other = domain_of_address(display)
            if other and from_dom and registered_domain(other) != from_dom:
                score += 0.35
                evidence.append(f"Display name shows an address at {other} but the mail was sent from {from_dom}")

        if from_dom and (norm_from != normalise(from_dom.split(".")[0])):
            pass
        if from_dom and any(ch.isdigit() for ch in from_dom.split(".")[0]):
            base = from_dom.split(".")[0]
            deglyphed = base.translate(HOMOGLYPHS)
            if deglyphed != base:
                for brand, canonical in self.brands.items():
                    if deglyphed.startswith(brand) or brand in deglyphed:
                        score += 0.45
                        evidence.append(
                            f"Sending domain {from_dom} substitutes digits for letters in '{brand}'")
                        break

        if any(w in display.lower() for w in ROLE_WORDS) and reply_dom and reply_dom != from_dom:
            score += 0.20
            evidence.append("Support-style display name with replies routed to a different domain")

        return LayerResult(
            name=self.name,
            score=clamp01(score),
            confidence=0.7,
            features={"display_name": display, "from_domain": from_dom,
                      "reply_to_domain": reply_dom, "brand_claimed": matched,
                      "brand_list_size": len(self.brands)},
            evidence=evidence or ["Display name is consistent with the sending domain"],
        )
