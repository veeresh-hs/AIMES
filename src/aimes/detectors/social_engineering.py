"""Layer 9 - social-engineering pressure features.

These are *features*, computed as normalised densities over the message and fed
to the ensemble, not stand-alone keyword rules that decide a verdict.
"""
from __future__ import annotations

import math
import re

from .base import Detector, clamp01
from ..schema import LayerResult, ParsedEmail

CATEGORIES: dict[str, list[str]] = {
    "urgency": [r"\burgent\b", r"\bimmediat\w*", r"right away", r"\bnow\b",
                r"within \d+ (hours?|minutes?|days?)", r"\bexpir\w+", r"\blast chance\b",
                r"\bact fast\b", r"\btoday only\b", r"\bdeadline\b", r"\basap\b"],
    "threat": [r"\bsuspend\w*", r"\bterminat\w*", r"\bclos\w+ your account\b",
               r"\block\w*\b", r"\bdisabl\w+", r"\blegal action\b", r"\bpenalt\w+",
               r"\bfine\b", r"\bunauthori[sz]ed\b", r"\bcompromis\w+", r"\bbreach\b"],
    "credential_request": [r"\bverify your (account|identity|email)\b", r"\bconfirm your (account|password|identity)\b",
                           r"\bre-?enter your password\b", r"\bupdate your (payment|billing|account) (details|information)\b",
                           r"\bsign ?in to (confirm|verify|restore)\b", r"\blogin credentials?\b",
                           r"\bsecurity question\b", r"\bvalidate your account\b"],
    "otp_request": [r"\botp\b", r"\bone[- ]time (password|code|pin)\b", r"\bverification code\b",
                    r"\b2fa code\b", r"\bshare the code\b", r"\bsend (me )?the code\b"],
    "payment_pressure": [r"\bwire transfer\b", r"\bpayment (is )?(due|required|pending)\b",
                         r"\boutstanding (invoice|balance)\b", r"\bgift ?card\b",
                         r"\bbitcoin\b", r"\bcrypto(currency)?\b", r"\busdt\b",
                         r"\bbank (details|transfer)\b", r"\binvoice attached\b", r"\bremittance\b"],
    "reward": [r"\byou (have )?won\b", r"\bcongratulations\b", r"\bprize\b", r"\blottery\b",
               r"\bfree gift\b", r"\bclaim your\b", r"\bselected as (a )?winner\b",
               r"\bcash reward\b", r"\binheritance\b", r"\bmillion (usd|dollars|euros)\b"],
    "authority": [r"\bsecurity team\b", r"\bit (department|support|helpdesk)\b",
                  r"\bsystem administrator\b", r"\bcompliance (team|department)\b",
                  r"\bgovernment\b", r"\bincome tax\b", r"\bpolice\b", r"\bcourt\b",
                  r"\bceo\b", r"\bmanaging director\b"],
    "secrecy": [r"\bconfidential\b", r"\bdo not (tell|inform|share)\b", r"\bkeep this between\b",
                r"\bdiscreet\b", r"\bprivate matter\b"],
    "emergency_request": [r"\bi need (a )?favou?r\b", r"\bare you (available|at your desk)\b",
                          r"\bstuck (abroad|overseas)\b", r"\bhospital\b", r"\bemergency\b"],
    "action_link": [r"\bclick (here|below|the link)\b", r"\bfollow this link\b",
                    r"\bopen the attachment\b", r"\bdownload (the )?attach\w*", r"\breview the document\b"],
    "impersonal_greeting": [r"\bdear (customer|user|client|member|sir/?madam|account holder)\b",
                            r"\bdear valued\b", r"\bhello dear\b"],
}
COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in CATEGORIES.items()}
FEATURE_NAMES = (list(CATEGORIES) +
                 ["exclamations", "all_caps_words", "caps_ratio", "money_mentions", "text_len"])

# Relative contribution of each category to the layer score.  Fitted weights
# from scripts/train_all.py replace these when a meta-model is available.
CATEGORY_WEIGHT = {
    "urgency": 0.10, "threat": 0.14, "credential_request": 0.22, "otp_request": 0.20,
    "payment_pressure": 0.14, "reward": 0.12, "authority": 0.08, "secrecy": 0.08,
    "emergency_request": 0.08, "action_link": 0.08, "impersonal_greeting": 0.06,
}


def extract_features(text: str) -> dict[str, float]:
    text = text or ""
    words = text.split()
    n_words = max(1, len(words))
    feats: dict[str, float] = {}
    for name, patterns in COMPILED.items():
        hits = sum(len(p.findall(text)) for p in patterns)
        feats[name] = float(hits)
    caps = [w for w in words if len(w) >= 3 and w.isupper()]
    feats["exclamations"] = float(text.count("!"))
    feats["all_caps_words"] = float(len(caps))
    feats["caps_ratio"] = len(caps) / n_words
    feats["money_mentions"] = float(len(re.findall(r"[$£€₹]\s?\d|\b\d+[,.]?\d*\s?(usd|eur|inr|gbp)\b", text, re.I)))
    feats["text_len"] = float(len(text))
    return feats


class SocialEngineeringDetector(Detector):
    name = "social_engineering"

    def _analyze(self, email: ParsedEmail) -> LayerResult:
        text = email.combined_text
        feats = extract_features(text)
        if not text.strip():
            return LayerResult(name=self.name, score=None, available=False,
                               confidence=0.0, error="Empty message body")

        # saturating contribution: two hits count, twenty do not count ten times
        score = sum(w * (1 - math.exp(-feats[c])) for c, w in CATEGORY_WEIGHT.items())
        if feats["exclamations"] >= 3:
            score += 0.04
        if feats["caps_ratio"] > 0.15:
            score += 0.05
        if feats["money_mentions"] >= 2:
            score += 0.04

        triggered = sorted(((c, feats[c]) for c in CATEGORY_WEIGHT if feats[c] > 0),
                           key=lambda t: -CATEGORY_WEIGHT[t[0]] * t[1])
        evidence = [f"{c.replace('_', ' ').capitalize()} language ({int(n)} match{'es' if n > 1 else ''})"
                    for c, n in triggered[:5]]

        return LayerResult(
            name=self.name,
            score=clamp01(score),
            confidence=0.65,
            features={k: round(v, 4) for k, v in feats.items()},
            evidence=evidence or ["No pressure or persuasion patterns detected"],
        )
