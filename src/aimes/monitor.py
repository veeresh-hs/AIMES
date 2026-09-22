"""Polling monitor: fetch -> parse -> analyse -> store -> alert.

Interval is configurable (AIMES_POLL_SECONDS).  Delta polling through the Gmail
history API keeps quota usage negligible, and every failure backs off instead of
hammering the API.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from . import database as db
from .alerts.dispatcher import dispatch
from .config import settings
from .logging_conf import get_logger, hash_address
from .pipeline import get_pipeline

log = get_logger(__name__)


class Monitor:
    def __init__(self, provider=None) -> None:
        self.provider = provider
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_poll: str | None = None
        self.last_error: str = ""
        self.processed = 0

    # ----------------------------------------------------------- one pass
    def poll_once(self) -> int:
        if self.provider is None:
            raise RuntimeError("no mail provider connected")
        pipeline = get_pipeline()
        count = 0
        for email in self.provider.fetch_new():
            result = pipeline.analyze(email)
            email_id = db.save_analysis(email, result)
            dispatch(email, result, email_id)
            if result.band in ("HIGH_RISK", "CRITICAL"):
                self.provider.label(email.message_id, result.band)
            log.info("analysed %s from %s -> %s risk=%d", email.message_id[:12],
                     hash_address(email.sender), result.base_class, result.risk_score)
            count += 1
        self.last_poll = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.processed += count
        return count

    # ------------------------------------------------------------- thread
    def _loop(self) -> None:
        backoff = settings.poll_seconds
        while not self._stop.is_set():
            try:
                self.poll_once()
                self.last_error = ""
                backoff = settings.poll_seconds
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.error("poll failed: %s", self.last_error)
                backoff = min(backoff * 2, 600)
            self._stop.wait(backoff)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="aimes-monitor", daemon=True)
        self._thread.start()
        log.info("monitor started, polling every %ds", settings.poll_seconds)

    def stop(self) -> None:
        self._stop.set()
        log.info("monitor stopping")

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def status(self) -> dict:
        return {"running": self.running, "last_poll": self.last_poll,
                "processed": self.processed, "last_error": self.last_error,
                "interval_seconds": settings.poll_seconds,
                "provider": getattr(self.provider, "name", None)}


MONITOR = Monitor()


def connect_gmail(interactive: bool = False):
    from .connector.gmail_client import GmailClient

    client = GmailClient().connect(interactive=interactive)
    MONITOR.provider = client
    return client
