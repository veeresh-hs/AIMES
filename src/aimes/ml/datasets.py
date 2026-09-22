"""Dataset loading.

Supported public corpora (all free, all used under their own licences):

  data/raw/spamassassin/{easy_ham,hard_ham,spam,spam_2}/   raw RFC-822 files
  data/raw/enron_spam/{ham,spam}/*.txt                     body-only samples
  data/raw/nazario/*.mbox                                  phishing, full headers
  data/raw/phishing_csv/*.csv                              csv with text + label
  data/raw/url_dataset/*.csv                               URL + label, layer 5 only

scripts/download_data.py prints each licence and fetches what it can.  Anything
missing is skipped with a warning; the pipeline never invents data.

A synthetic generator is included for smoke-testing the plumbing on a machine
with no downloads.  Samples it produces carry source="SYNTHETIC" and every
report generated from them is stamped accordingly - synthetic numbers are not
experimental results and must never be presented as such.
"""
from __future__ import annotations

import hashlib
import mailbox
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

from ..config import settings
from ..logging_conf import get_logger
from ..connector.parser import from_plain_text, parse_bytes, parse_message
from ..schema import ParsedEmail

log = get_logger(__name__)

LABELS = ("HAM", "PROMOTIONAL_SPAM", "PHISHING")


@dataclass
class Sample:
    uid: str
    label: str
    source: str
    email: ParsedEmail
    date: Optional[datetime] = None
    meta: dict = field(default_factory=dict)


def _uid(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:20]


# --------------------------------------------------------------- loaders
def load_spamassassin(root: Path) -> list[Sample]:
    """SpamAssassin public corpus: full headers, the only source that exercises
    the header layer end to end."""
    out: list[Sample] = []
    folder_label = {"easy_ham": "HAM", "easy_ham_2": "HAM", "hard_ham": "HAM",
                    "spam": "PROMOTIONAL_SPAM", "spam_2": "PROMOTIONAL_SPAM"}
    for folder, label in folder_label.items():
        directory = root / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.name.startswith(".") or not path.is_file():
                continue
            try:
                email = parse_bytes(path.read_bytes(), message_id=path.name)
            except Exception:
                continue
            if not email.combined_text.strip():
                continue
            out.append(Sample(_uid(email.body_sha256 + path.name), label,
                              "spamassassin", email, email.date))
    log.info("spamassassin: %d samples", len(out))
    return out


def load_enron_spam(root: Path) -> list[Sample]:
    """Enron-Spam (Metsis et al.): body-only, so the header layer abstains."""
    out: list[Sample] = []
    for folder, label in (("ham", "HAM"), ("spam", "PROMOTIONAL_SPAM")):
        for path in sorted((root / folder).glob("**/*.txt")) if (root / folder).is_dir() else []:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            subject = ""
            if text.lower().startswith("subject:"):
                head, _, rest = text.partition("\n")
                subject, text = head[8:].strip(), rest
            email = from_plain_text(text, subject)
            if not email.combined_text.strip():
                continue
            out.append(Sample(_uid(email.body_sha256 + path.name), label, "enron_spam", email))
    log.info("enron_spam: %d samples", len(out))
    return out


def load_mbox(path: Path, label: str, source: str) -> list[Sample]:
    out: list[Sample] = []
    try:
        box = mailbox.mbox(str(path))
    except Exception as exc:
        log.warning("cannot open %s: %s", path, exc)
        return out
    for i, msg in enumerate(box):
        try:
            email = parse_message(msg, message_id=f"{path.stem}-{i}")
        except Exception:
            continue
        if not email.combined_text.strip():
            continue
        out.append(Sample(_uid(email.body_sha256 + str(i)), label, source, email, email.date))
    log.info("%s (%s): %d samples", source, path.name, len(out))
    return out


def load_nazario(root: Path) -> list[Sample]:
    out: list[Sample] = []
    for path in sorted(root.glob("*.mbox")) + sorted(root.glob("*.txt")):
        out.extend(load_mbox(path, "PHISHING", "nazario"))
    return out


_LABEL_MAP = {
    "1": "PHISHING", "phishing": "PHISHING", "phish": "PHISHING", "spam": "PROMOTIONAL_SPAM",
    "0": "HAM", "ham": "HAM", "legitimate": "HAM", "safe": "HAM", "not spam": "HAM",
}


def load_phishing_csv(root: Path) -> list[Sample]:
    """Any CSV with a text-ish column and a label column."""
    import pandas as pd

    out: list[Sample] = []
    for path in sorted(root.glob("*.csv")):
        try:
            frame = pd.read_csv(path, on_bad_lines="skip", engine="python")
        except Exception as exc:
            log.warning("cannot read %s: %s", path.name, exc)
            continue
        cols = {c.lower().strip(): c for c in frame.columns}
        body_col = next((cols[c] for c in ("body", "text", "email_text", "content", "message")
                         if c in cols), None)
        label_col = next((cols[c] for c in ("label", "class", "type", "target", "is_phishing")
                          if c in cols), None)
        subj_col = next((cols[c] for c in ("subject", "title") if c in cols), None)
        if body_col is None or label_col is None:
            log.warning("skipping %s: no usable text/label columns", path.name)
            continue
        for i, row in frame.iterrows():
            raw_label = str(row[label_col]).strip().lower()
            label = _LABEL_MAP.get(raw_label)
            if label is None:
                continue
            text = str(row[body_col])
            if len(text) < 20:
                continue
            subject = str(row[subj_col]) if subj_col and not pd.isna(row[subj_col]) else ""
            email = from_plain_text(text, subject)
            out.append(Sample(_uid(email.body_sha256 + f"{path.name}{i}"), label,
                              f"csv:{path.stem}", email))
    log.info("phishing csv: %d samples", len(out))
    return out


def load_url_dataset(root: Path):
    """URLs for layer 5 only.  Returns (urls, labels) or (None, None)."""
    import pandas as pd

    frames = []
    for path in sorted(root.glob("*.csv")):
        try:
            frames.append(pd.read_csv(path, on_bad_lines="skip", engine="python"))
        except Exception as exc:
            log.warning("cannot read %s: %s", path.name, exc)
    if not frames:
        return None, None
    frame = pd.concat(frames, ignore_index=True)
    cols = {c.lower().strip(): c for c in frame.columns}
    url_col = next((cols[c] for c in ("url", "urls", "domain", "link") if c in cols), None)
    label_col = next((cols[c] for c in ("label", "class", "type", "result", "status", "phishing")
                      if c in cols), None)
    if url_col is None or label_col is None:
        return None, None
    urls, labels = [], []
    for url, raw in zip(frame[url_col], frame[label_col]):
        text = str(raw).strip().lower()
        if text in ("1", "phishing", "bad", "malicious", "phish"):
            labels.append(1)
        elif text in ("0", "-1", "legitimate", "good", "benign", "safe"):
            labels.append(0)
        else:
            continue
        urls.append(str(url))
    log.info("url dataset: %d urls", len(urls))
    return urls, labels


# ------------------------------------------------------------- synthetic
_HAM_TOPICS = [
    ("Minutes from Tuesday's standup", "Hi {name},\n\nSharing the notes from this morning. {body}\nWe agreed to move the migration to next sprint and keep the current schema until then.\n\nThanks,\n{signer}"),
    ("Re: dataset for the term paper", "Hey {name},\n\n{body}\nI pushed the cleaned CSV to the shared drive, column names are normalised now. Shout if the encoding looks off on your machine.\n\n{signer}"),
    ("Lunch Friday?", "{name}, are you around Friday around 1? {body} There is a new place near the library that does decent thali.\n\n{signer}"),
    ("Invoice 8842 approved", "Hello {name},\n\nFinance approved invoice 8842 this morning. {body} No action needed from your side; the payment run goes out on the 30th as usual.\n\nRegards,\n{signer}"),
    ("Server maintenance window", "Team,\n\n{body} The staging cluster will be unavailable Saturday 02:00-04:00 IST for a kernel update. Production is unaffected.\n\n{signer}"),
    ("Paper review due", "Hi {name},\n\n{body} Reviews for the workshop submissions are due on the 14th. I have attached nothing this time, everything is in the review portal.\n\nBest,\n{signer}"),
]
_HAM_FILLER = [
    "The numbers from last quarter are attached to the shared sheet.",
    "I rebased the branch and the tests are green again.",
    "Let me know if the timing does not work for you.",
    "Nothing urgent, just keeping everyone in the loop.",
    "I will be out Thursday afternoon but reachable on chat.",
]
_PROMO = [
    ("Flat 50% off ends tonight", "Our biggest sale of the season is here. Shop over 4000 styles at half price. Free shipping on orders above 999. Use code SAVE50 at checkout.\n\nUnsubscribe: {url}"),
    ("Your weekly deals are here", "Handpicked offers for you this week: electronics, home and kitchen, and up to 40% off on bestsellers. Browse now: {url}\n\nYou receive this because you subscribed to our newsletter."),
    ("Last chance: webinar seats filling fast", "Only a few seats left for Thursday's webinar on growth marketing. Reserve yours: {url}\n\nManage preferences at the link below."),
    ("Upgrade and get 3 months free", "Move to the annual plan today and get three months at no cost. Compare plans here: {url}"),
]
_PHISH = [
    ("Urgent: your {brand} account will be suspended", "Dear customer,\n\nWe detected unauthorized activity on your {brand} account. Your account will be suspended within 24 hours unless you verify your identity immediately.\n\nVerify your account here: {url}\n\nFailure to confirm your password will result in permanent closure.\n\n{brand} Security Team"),
    ("Action required: confirm your payment details", "Dear valued customer,\n\nYour recent payment could not be processed. Update your billing information now to avoid service interruption: {url}\n\nThis is your final notice. Please do not ignore this message.\n\nAccounts Department"),
    ("You have won a cash reward", "Congratulations!\n\nYou have been selected as a winner of our annual draw. To claim your prize of 2,500,000 USD, confirm your identity and bank details at {url}.\n\nKeep this confidential until the transfer is complete."),
    ("Document shared with you", "A document has been shared with you on {brand} Drive.\n\nReview the document: {url}\n\nSign in with your email password to open the file. The link expires in 24 hours."),
    ("OTP verification pending", "We could not verify your login. Please send the one-time password you received to our security team or click here to authenticate: {url}\n\nUnauthorized access will be reported."),
]
_BRANDS = ["Microsoft", "PayPal", "Netflix", "Amazon", "Apple", "HDFC Bank", "DHL"]
_PHISH_HOSTS = ["secure-{b}-verify.tk", "{b}-account-update.xyz", "login.{b}.support-cdn.top",
                "{b}-billing.confirm-now.click", "192.168.{n}.{m}"]
_NAMES = ["Anita", "Rahul", "Priya", "Sam", "Meera", "Dev", "Karan", "Farah"]
_SIGNERS = ["Ravi", "Sneha", "the platform team", "Arjun", "Divya"]


def _synth_message(label: str, rng: random.Random, index: int) -> str:
    date = (datetime.now(timezone.utc) - timedelta(days=rng.randint(0, 400))).strftime(
        "%a, %d %b %Y %H:%M:%S %z")
    if label == "HAM":
        subject, template = rng.choice(_HAM_TOPICS)
        body = template.format(name=rng.choice(_NAMES), signer=rng.choice(_SIGNERS),
                               body=rng.choice(_HAM_FILLER))
        domain = rng.choice(["college.edu", "acme-works.com", "example.org", "labmail.in"])
        sender = f"{rng.choice(_SIGNERS).lower().replace(' ', '.')}@{domain}"
        headers = [f"From: {rng.choice(_NAMES)} <{sender}>", "To: student@example.com",
                   f"Subject: {subject}", f"Date: {date}",
                   f"Message-ID: <ham{index}@{domain}>",
                   "Authentication-Results: mx.example.com; spf=pass; dkim=pass; dmarc=pass",
                   f"Received: from mail.{domain} by mx.example.com; {date}",
                   "Content-Type: text/plain; charset=utf-8"]
    elif label == "PROMOTIONAL_SPAM":
        subject, template = rng.choice(_PROMO)
        brand = rng.choice(["shopnest", "dealbox", "megamart", "stylehub"])
        url = f"http://news.{brand}.com/c/{rng.randint(10000, 99999)}?utm_source=email"
        body = template.format(url=url)
        headers = [f"From: {brand.title()} Offers <offers@news.{brand}.com>",
                   "To: student@example.com", f"Subject: {subject}", f"Date: {date}",
                   f"Message-ID: <promo{index}@news.{brand}.com>",
                   "List-Unsubscribe: <http://news.%s.com/unsub>" % brand,
                   "Precedence: bulk",
                   "Authentication-Results: mx.example.com; spf=pass; dkim=pass; dmarc=pass",
                   "Content-Type: text/plain; charset=utf-8"]
    else:
        subject, template = rng.choice(_PHISH)
        brand = rng.choice(_BRANDS)
        slug = brand.lower().replace(" ", "")
        host = rng.choice(_PHISH_HOSTS).format(b=slug, n=rng.randint(1, 254), m=rng.randint(1, 254))
        url = f"http://{host}/{rng.choice(['login', 'verify', 'secure/account'])}/confirm?id={rng.randint(10**6, 10**7)}"
        body = template.format(url=url, brand=brand)
        subject = subject.format(brand=brand)
        fake_dom = rng.choice([f"{slug}-security.tk", f"{slug.replace('o', '0')}.com",
                               f"mail-{slug}.support-team.xyz"])
        headers = [f'From: "{brand} Support" <no-reply@{fake_dom}>',
                   "To: undisclosed-recipients:;", f"Subject: {subject}", f"Date: {date}",
                   f"Reply-To: recover{index}@mail.ru",
                   "Authentication-Results: mx.example.com; spf=fail; dkim=none; dmarc=fail",
                   "Content-Type: text/plain; charset=utf-8"]
    return "\n".join(headers) + "\n\n" + body


def generate_synthetic(n: int = 900, seed: int = 7) -> list[Sample]:
    """SMOKE TEST DATA ONLY - never report metrics measured on these samples."""
    rng = random.Random(seed)
    weights = [("HAM", 0.5), ("PROMOTIONAL_SPAM", 0.28), ("PHISHING", 0.22)]
    out: list[Sample] = []
    for i in range(n):
        roll, acc, label = rng.random(), 0.0, "HAM"
        for name, weight in weights:
            acc += weight
            if roll <= acc:
                label = name
                break
        raw = _synth_message(label, rng, i)
        email = parse_bytes(raw.encode("utf-8"), message_id=f"synthetic-{i}")
        out.append(Sample(f"syn{i}", label, "SYNTHETIC", email, email.date))
    log.warning("generated %d SYNTHETIC samples - for plumbing tests only", len(out))
    return out


# ------------------------------------------------------------- discovery
def load_all(raw_dir: Optional[Path] = None, allow_synthetic: bool = True) -> list[Sample]:
    raw_dir = Path(raw_dir or settings.data_dir / "raw")
    samples: list[Sample] = []
    if (raw_dir / "spamassassin").is_dir():
        samples += load_spamassassin(raw_dir / "spamassassin")
    if (raw_dir / "enron_spam").is_dir():
        samples += load_enron_spam(raw_dir / "enron_spam")
    if (raw_dir / "nazario").is_dir():
        samples += load_nazario(raw_dir / "nazario")
    if (raw_dir / "phishing_csv").is_dir():
        samples += load_phishing_csv(raw_dir / "phishing_csv")

    if not samples:
        if not allow_synthetic:
            raise SystemExit(
                "No corpora found under data/raw. Run: python scripts/download_data.py")
        log.warning("no real corpora found in %s - falling back to synthetic data", raw_dir)
        samples = generate_synthetic()
    return samples


def is_synthetic(samples: list[Sample]) -> bool:
    return bool(samples) and all(s.source == "SYNTHETIC" for s in samples)
