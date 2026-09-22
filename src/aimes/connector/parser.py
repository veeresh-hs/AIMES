"""RFC-822 / MIME parsing.

Treats every message as hostile input: size caps, part-count caps, no
rendering, no network, attachment bytes hashed in memory then dropped.
"""
from __future__ import annotations

import email
import email.policy
import hashlib
import re
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parseaddr, parsedate_to_datetime
from typing import Optional

from ..config import settings
from ..logging_conf import get_logger
from ..schema import Attachment, ParsedEmail
from ..textutils import clean_body, domain_of_address, extract_urls

log = get_logger(__name__)

MAX_PARTS = 60
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
MACRO_MARKERS = (b"vbaProject.bin", b"\x00A\x00t\x00t\x00r\x00i\x00b\x00u\x00t\x00e", b"Macros")
OFFICE_EXTS = (".doc", ".docm", ".xls", ".xlsm", ".ppt", ".pptm", ".dot", ".xlsb")


def _decode(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return re.sub(r"[\r\n]+", " ", value)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _looks_macro_enabled(filename: str, payload: bytes) -> bool:
    """Static check only.  Never opens or executes the document."""
    if filename.lower().endswith((".docm", ".xlsm", ".pptm", ".xlsb")):
        return True
    try:  # oletools is optional
        from oletools.olevba import VBA_Parser  # type: ignore

        if filename.lower().endswith(OFFICE_EXTS + (".docx", ".xlsx", ".pptx")):
            parser = VBA_Parser(filename, data=payload)
            try:
                return bool(parser.detect_vba_macros())
            finally:
                parser.close()
    except Exception:
        pass
    head = payload[:200_000]
    return any(marker in head for marker in MACRO_MARKERS)


def parse_message(msg: Message, message_id: str = "", thread_id: str = "") -> ParsedEmail:
    parsed = ParsedEmail(thread_id=thread_id)
    parsed.headers = {k.lower(): _decode(v) for k, v in msg.items()}

    parsed.message_id = message_id or parsed.headers.get("message-id", "").strip("<> ")
    parsed.subject = _decode(msg.get("Subject"))
    raw_from = _decode(msg.get("From"))
    parsed.sender_name, parsed.sender = parseaddr(raw_from)
    parsed.sender = parsed.sender.lower()
    parsed.sender_domain = domain_of_address(parsed.sender)
    parsed.to = _decode(msg.get("To"))
    parsed.cc = _decode(msg.get("Cc"))
    parsed.reply_to = parseaddr(_decode(msg.get("Reply-To")))[1].lower()
    parsed.return_path = parseaddr(_decode(msg.get("Return-Path")))[1].lower()
    parsed.received_chain = [_decode(v) for v in msg.get_all("Received", [])][:20]
    parsed.auth_results = " | ".join(
        _decode(v) for v in (msg.get_all("Authentication-Results", []) +
                             msg.get_all("ARC-Authentication-Results", []))
    )
    parsed.has_headers = bool(parsed.received_chain or parsed.sender or parsed.headers.get("date"))

    date_raw = msg.get("Date")
    if date_raw:
        try:
            parsed.date = parsedate_to_datetime(date_raw)
        except (TypeError, ValueError, IndexError):
            parsed.date = None
    if parsed.date is None:
        parsed.date = datetime.now(timezone.utc)

    text_parts: list[str] = []
    html_parts: list[str] = []
    part_count = 0

    for part in msg.walk():
        part_count += 1
        if part_count > MAX_PARTS:
            log.warning("MIME part cap reached for message %s", parsed.message_id)
            break
        if part.get_content_maintype() == "multipart":
            continue

        disposition = (part.get_content_disposition() or "").lower()
        filename = _decode(part.get_filename())
        ctype = (part.get_content_type() or "").lower()

        if disposition == "attachment" or (filename and ctype not in ("text/plain", "text/html")):
            try:
                payload = part.get_payload(decode=True) or b""
            except Exception:
                payload = b""
            if len(payload) > MAX_ATTACHMENT_BYTES:
                payload = payload[:MAX_ATTACHMENT_BYTES]
            parsed.attachments.append(
                Attachment(
                    filename=filename or "(unnamed)",
                    content_type=ctype,
                    size=len(payload),
                    sha256=_sha256(payload) if payload else "",
                    has_macro_marker=_looks_macro_enabled(filename or "", payload),
                )
            )
            del payload  # bytes are never persisted
            continue

        try:
            payload = part.get_payload(decode=True)
            charset = part.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="replace") if payload else ""
        except (LookupError, AttributeError, UnicodeDecodeError):
            body = str(part.get_payload())[:settings.max_body_chars]
        except Exception:
            body = ""

        if ctype == "text/html":
            html_parts.append(body)
        else:
            text_parts.append(body)

    parsed.body_text = "\n".join(text_parts)[: settings.max_body_chars * 2]
    parsed.body_html = "\n".join(html_parts)[: settings.max_body_chars * 4]
    parsed.clean_text = clean_body(parsed.body_text, parsed.body_html, settings.max_body_chars)
    parsed.urls = extract_urls(parsed.body_text, parsed.body_html)[:200]
    parsed.body_sha256 = _sha256(parsed.clean_text.encode("utf-8", "replace"))
    parsed.size_bytes = len(parsed.body_text) + len(parsed.body_html) + sum(a.size for a in parsed.attachments)
    return parsed


def parse_bytes(raw: bytes, message_id: str = "", thread_id: str = "") -> ParsedEmail:
    msg = email.message_from_bytes(raw, policy=email.policy.compat32)
    return parse_message(msg, message_id, thread_id)


def parse_string(raw: str, message_id: str = "", thread_id: str = "") -> ParsedEmail:
    return parse_bytes(raw.encode("utf-8", "replace"), message_id, thread_id)


def parse_file(path) -> ParsedEmail:
    with open(path, "rb") as fh:
        return parse_bytes(fh.read(), message_id=str(path))


def from_plain_text(text: str, label_subject: str = "") -> ParsedEmail:
    """Wrap a body-only sample (e.g. Enron-Spam) as a ParsedEmail."""
    parsed = ParsedEmail()
    parsed.subject = label_subject
    parsed.body_text = text
    parsed.clean_text = clean_body(text, "", settings.max_body_chars)
    parsed.urls = extract_urls(text)
    parsed.has_headers = False
    parsed.body_sha256 = _sha256(parsed.clean_text.encode("utf-8", "replace"))
    parsed.size_bytes = len(text)
    return parsed
