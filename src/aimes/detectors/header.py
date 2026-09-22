"""Layer 6 - header and authentication analysis.

A failed SPF/DKIM/DMARC check is evidence, not a verdict: forwarding and
mailing lists break authentication for perfectly legitimate mail.  The score
produced here is one input to the ensemble.
"""
from __future__ import annotations

import re

from .base import Detector, clamp01
from ..schema import LayerResult, ParsedEmail
from ..textutils import domain_of_address, registered_domain

SPF_RE = re.compile(r"\bspf=(\w+)", re.I)
DKIM_RE = re.compile(r"\bdkim=(\w+)", re.I)
DMARC_RE = re.compile(r"\bdmarc=(\w+)", re.I)
FREEMAIL = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com",
    "aol.com", "protonmail.com", "mail.ru", "yandex.ru", "gmx.com", "icloud.com",
    "rediffmail.com", "zoho.com", "inbox.lv", "163.com", "qq.com",
}
BULK_HINTS = ("list-unsubscribe", "precedence", "x-campaign", "x-mailer",
              "list-id", "feedback-id")


def _verdict(pattern: re.Pattern, text: str) -> str:
    match = pattern.search(text or "")
    return match.group(1).lower() if match else "none"


class HeaderDetector(Detector):
    name = "header"

    def _analyze(self, email: ParsedEmail) -> LayerResult:
        if not email.has_headers:
            return LayerResult(name=self.name, score=None, available=False,
                               confidence=0.0,
                               error="Corpus provides no headers for this sample")

        auth = email.auth_results or " ".join(email.headers.get(k, "") for k in
                                              ("received-spf", "dkim-signature", "arc-authentication-results"))
        spf = _verdict(SPF_RE, auth)
        if spf == "none" and email.headers.get("received-spf"):
            spf = email.headers["received-spf"].split()[0].lower()
        dkim = _verdict(DKIM_RE, auth)
        if dkim == "none" and "dkim-signature" in email.headers:
            dkim = "present-unverified"
        dmarc = _verdict(DMARC_RE, auth)

        from_dom = registered_domain(email.sender_domain)
        reply_dom = registered_domain(domain_of_address(email.reply_to))
        return_dom = registered_domain(domain_of_address(email.return_path))

        score, evidence = 0.0, []

        if spf in ("fail", "softfail", "permerror"):
            score += 0.22
            evidence.append(f"SPF result: {spf}")
        elif spf == "none":
            score += 0.06
            evidence.append("No SPF result recorded by the receiving server")
        if dkim in ("fail", "permerror"):
            score += 0.18
            evidence.append(f"DKIM result: {dkim}")
        elif dkim == "none":
            score += 0.06
        if dmarc in ("fail", "quarantine", "reject"):
            score += 0.22
            evidence.append(f"DMARC result: {dmarc}")

        if reply_dom and from_dom and reply_dom != from_dom:
            score += 0.20
            evidence.append(f"Reply-To domain ({reply_dom}) differs from From domain ({from_dom})")
        if return_dom and from_dom and return_dom != from_dom:
            score += 0.10
            evidence.append(f"Return-Path domain ({return_dom}) differs from From domain ({from_dom})")

        hops = len(email.received_chain)
        if hops == 0:
            score += 0.08
            evidence.append("No Received chain present")
        elif hops > 8:
            score += 0.08
            evidence.append(f"Unusually long relay chain ({hops} hops)")

        if not email.headers.get("message-id"):
            score += 0.08
            evidence.append("Missing Message-ID header")
        if not email.to and not email.cc:
            score += 0.06
            evidence.append("Recipient is not addressed directly (bcc-style bulk send)")

        bulk = sum(1 for h in BULK_HINTS if h in email.headers)
        if bulk >= 2:
            evidence.append("Bulk-mail headers present (typical of marketing mail)")

        freemail_sender = from_dom in FREEMAIL
        display_domainish = bool(re.search(r"\b(support|security|service|billing|admin|team)\b",
                                           email.sender_name, re.I))
        if freemail_sender and display_domainish:
            score += 0.14
            evidence.append("Corporate-sounding display name sent from a free webmail account")

        return LayerResult(
            name=self.name,
            score=clamp01(score),
            confidence=0.75,
            features={
                "spf": spf, "dkim": dkim, "dmarc": dmarc,
                "from_domain": from_dom, "reply_to_domain": reply_dom,
                "return_path_domain": return_dom,
                "reply_to_mismatch": int(bool(reply_dom and reply_dom != from_dom)),
                "received_hops": hops,
                "freemail_sender": int(freemail_sender),
                "bulk_headers": bulk,
                "has_message_id": int(bool(email.headers.get("message-id"))),
            },
            evidence=evidence or ["Header checks found nothing unusual"],
        )
