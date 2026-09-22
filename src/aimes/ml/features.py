"""Tabular feature vector shared by the anomaly layer and the experiments.

Deterministic, cheap, and computed without any network access.
"""
from __future__ import annotations

import math
import re

from ..detectors.attachment import (ARCHIVE_EXTS, EXECUTABLE_EXTS, MACRO_EXTS,
                                    extensions)
from ..detectors.header import FREEMAIL
from ..detectors.social_engineering import extract_features as se_features
from ..detectors.url import url_features
from ..schema import ParsedEmail
from ..textutils import domain_of_address, registered_domain

STRUCT_FEATURES = [
    "subject_len", "subject_words", "subject_caps_ratio", "subject_excl",
    "body_len", "body_words", "avg_word_len", "unique_word_ratio",
    "n_urls", "n_distinct_url_domains", "html_ratio", "n_attachments",
    "attach_total_kb", "has_exec_attach", "has_macro_attach", "has_archive_attach",
    "n_recipients", "reply_to_mismatch", "freemail_sender", "received_hops",
    "n_digits_subject", "n_nonascii", "url_max_len", "url_mean_dots",
    "url_any_ip", "url_any_shortener", "url_any_risky_tld", "url_max_sensitive_hits",
]
SE_FEATURES = list(se_features("").keys())
FEATURE_NAMES = STRUCT_FEATURES + [f"se_{name}" for name in SE_FEATURES]


def tabular_features(email: ParsedEmail) -> dict[str, float]:
    subject = email.subject or ""
    body = email.clean_text or ""
    words = body.split()
    subject_words = subject.split()
    caps = [w for w in subject_words if len(w) >= 3 and w.isupper()]

    url_feats = [url_features(u) for u in email.urls[:30]]
    exts = [extensions(a.filename)[-1] if extensions(a.filename) else "" for a in email.attachments]
    from_dom = registered_domain(email.sender_domain)
    reply_dom = registered_domain(domain_of_address(email.reply_to))

    feats: dict[str, float] = {
        "subject_len": len(subject),
        "subject_words": len(subject_words),
        "subject_caps_ratio": len(caps) / max(1, len(subject_words)),
        "subject_excl": subject.count("!"),
        "body_len": len(body),
        "body_words": len(words),
        "avg_word_len": sum(len(w) for w in words) / max(1, len(words)),
        "unique_word_ratio": len(set(w.lower() for w in words)) / max(1, len(words)),
        "n_urls": len(email.urls),
        "n_distinct_url_domains": len({registered_domain(u) for u in email.urls}),
        "html_ratio": len(email.body_html) / max(1, len(email.body_html) + len(email.body_text)),
        "n_attachments": len(email.attachments),
        "attach_total_kb": sum(a.size for a in email.attachments) / 1024.0,
        "has_exec_attach": float(any(e in EXECUTABLE_EXTS for e in exts)),
        "has_macro_attach": float(any(e in MACRO_EXTS for e in exts)
                                  or any(a.has_macro_marker for a in email.attachments)),
        "has_archive_attach": float(any(e in ARCHIVE_EXTS for e in exts)),
        "n_recipients": float(len(re.findall(r"@", f"{email.to},{email.cc}"))),
        "reply_to_mismatch": float(bool(reply_dom and from_dom and reply_dom != from_dom)),
        "freemail_sender": float(from_dom in FREEMAIL),
        "received_hops": float(len(email.received_chain)),
        "n_digits_subject": float(sum(c.isdigit() for c in subject)),
        "n_nonascii": float(sum(ord(c) > 127 for c in body[:5000])),
        "url_max_len": max((f["url_len"] for f in url_feats), default=0.0),
        "url_mean_dots": (sum(f["n_dots"] for f in url_feats) / len(url_feats)) if url_feats else 0.0,
        "url_any_ip": max((f["is_ip"] for f in url_feats), default=0.0),
        "url_any_shortener": max((f["is_shortener"] for f in url_feats), default=0.0),
        "url_any_risky_tld": max((f["tld_risky"] for f in url_feats), default=0.0),
        "url_max_sensitive_hits": max((f["sensitive_word_hits"] for f in url_feats), default=0.0),
    }
    for name, value in se_features(f"{subject}\n{body}").items():
        feats[f"se_{name}"] = value

    # log-compress the heavy-tailed counts so IsolationForest is not dominated
    for key in ("body_len", "subject_len", "attach_total_kb", "url_max_len", "se_text_len"):
        feats[key] = math.log1p(max(0.0, feats.get(key, 0.0)))
    return feats


def feature_row(email: ParsedEmail) -> list[float]:
    feats = tabular_features(email)
    return [float(feats.get(name, 0.0)) for name in FEATURE_NAMES]
