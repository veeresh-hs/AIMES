"""Gmail connector: delta polling via the history API, full message fetch,
optional labelling.  Never deletes, never trashes, never marks as read.
"""
from __future__ import annotations

import base64
import time
from typing import Iterator

from .. import database as db
from ..config import settings
from ..logging_conf import get_logger
from ..schema import ParsedEmail
from .oauth import get_credentials
from .parser import parse_bytes

log = get_logger(__name__)

HISTORY_KEY = "gmail_history_id"
LABEL_PREFIX = "AIMES"


class GmailClient:
    name = "gmail"

    def __init__(self) -> None:
        self.service = None
        self.profile: dict = {}

    # ------------------------------------------------------------ connect
    def connect(self, interactive: bool = True) -> "GmailClient":
        from googleapiclient.discovery import build  # type: ignore

        creds = get_credentials(interactive=interactive)
        self.service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        self.profile = self.service.users().getProfile(userId="me").execute()
        log.info("connected to Gmail as %s (%s messages in mailbox)",
                 self.profile.get("emailAddress"), self.profile.get("messagesTotal"))
        if not db.get_state(HISTORY_KEY):
            db.set_state(HISTORY_KEY, str(self.profile.get("historyId", "")))
        return self

    def _retry(self, request, attempts: int = 4):
        """Exponential backoff for the transient 429/5xx responses."""
        from googleapiclient.errors import HttpError  # type: ignore

        delay = 1.0
        for attempt in range(attempts):
            try:
                return request.execute()
            except HttpError as exc:
                status = getattr(exc.resp, "status", 0)
                if status in (403, 429, 500, 502, 503, 504) and attempt < attempts - 1:
                    log.warning("Gmail API %s, retrying in %.1fs", status, delay)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

    # -------------------------------------------------------------- fetch
    def _new_message_ids(self, limit: int) -> list[str]:
        """History delta when possible, list fallback when the cursor expired."""
        history_id = db.get_state(HISTORY_KEY)
        if history_id:
            try:
                ids, page, latest = [], None, history_id
                while True:
                    response = self._retry(self.service.users().history().list(
                        userId="me", startHistoryId=history_id,
                        historyTypes=["messageAdded"], pageToken=page, maxResults=100))
                    for record in response.get("history", []):
                        for added in record.get("messagesAdded", []):
                            ids.append(added["message"]["id"])
                    latest = response.get("historyId", latest)
                    page = response.get("nextPageToken")
                    if not page or len(ids) >= limit:
                        break
                db.set_state(HISTORY_KEY, str(latest))
                return list(dict.fromkeys(ids))[:limit]
            except Exception as exc:
                log.warning("history cursor unusable (%s); falling back to a listing", exc)

        response = self._retry(self.service.users().messages().list(
            userId="me", maxResults=limit, q="newer_than:2d"))
        profile = self._retry(self.service.users().getProfile(userId="me"))
        db.set_state(HISTORY_KEY, str(profile.get("historyId", "")))
        return [m["id"] for m in response.get("messages", [])]

    def fetch_message(self, gmail_id: str) -> ParsedEmail | None:
        response = self._retry(self.service.users().messages().get(
            userId="me", id=gmail_id, format="raw"))
        raw = base64.urlsafe_b64decode(response["raw"].encode("ASCII"))
        email = parse_bytes(raw, message_id=gmail_id, thread_id=response.get("threadId", ""))
        return email

    def fetch_new(self, limit: int | None = None) -> Iterator[ParsedEmail]:
        limit = limit or settings.max_messages_per_poll
        for gmail_id in self._new_message_ids(limit):
            if db.email_exists(gmail_id):
                continue
            try:
                email = self.fetch_message(gmail_id)
            except Exception as exc:
                log.error("could not fetch message %s: %s", gmail_id, exc)
                continue
            if email:
                yield email

    # -------------------------------------------------------------- label
    def ensure_label(self, name: str) -> str | None:
        if not settings.gmail_allow_modify:
            return None
        labels = self._retry(self.service.users().labels().list(userId="me")).get("labels", [])
        for label in labels:
            if label["name"] == name:
                return label["id"]
        created = self._retry(self.service.users().labels().create(
            userId="me", body={"name": name, "labelListVisibility": "labelShow",
                               "messageListVisibility": "show"}))
        return created["id"]

    def label(self, gmail_id: str, band: str) -> bool:
        """Adds a visible label such as 'AIMES/CRITICAL'.  Opt-in, non-destructive."""
        if not (settings.gmail_allow_modify and settings.gmail_label_high_risk):
            return False
        try:
            label_id = self.ensure_label(f"{LABEL_PREFIX}/{band}")
            if not label_id:
                return False
            self._retry(self.service.users().messages().modify(
                userId="me", id=gmail_id, body={"addLabelIds": [label_id]}))
            return True
        except Exception as exc:
            log.error("labelling failed for %s: %s", gmail_id, exc)
            return False
