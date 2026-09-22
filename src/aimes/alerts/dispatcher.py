"""Alert delivery.

Free channels only: a desktop notification, the dashboard's live stream, the
local alert log, and optionally a Telegram bot (the Telegram Bot API is free).
No SMS, no paid notification service.
"""
from __future__ import annotations

import json
import queue
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from .. import database as db
from ..config import settings
from ..logging_conf import get_logger, hash_address
from ..schema import AnalysisResult, ParsedEmail

log = get_logger(__name__)

#: consumed by the dashboard's server-sent-events endpoint
STREAM: "queue.Queue[str]" = queue.Queue(maxsize=200)


def build_message(email: ParsedEmail, result: AnalysisResult) -> str:
    reasons = "\n".join(f"- {r.split(' [')[0]}" for r in result.reasons[:4])
    return (
        f"HIGH-RISK EMAIL DETECTED\n"
        f"Category: {result.base_class}"
        + (f" ({', '.join(result.tags)})" if result.tags else "") + "\n"
        f"Risk score: {result.risk_score}/100 (confidence {result.confidence * 100:.0f}%)\n"
        f"From: {email.sender or 'unknown'}\n"
        f"Subject: {email.subject[:120] or '(no subject)'}\n\n"
        f"Why:\n{reasons}\n\n"
        f"Open the dashboard for the full breakdown."
    )


def desktop_notify(title: str, message: str) -> tuple[bool, str]:
    if not settings.desktop_alerts:
        return False, "desktop alerts disabled"
    try:
        from plyer import notification  # type: ignore

        notification.notify(title=title, message=message[:250],
                            app_name="AIMES", timeout=20)
        return True, "plyer"
    except Exception as exc:
        log.info("plyer unavailable (%s); falling back to console", type(exc).__name__)
    try:  # Windows 10/11 without plyer
        import subprocess
        import sys

        if sys.platform.startswith("win"):
            safe = message.replace("'", "").replace("\n", " ")[:200]
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[reflection.assembly]::LoadWithPartialName('System.Windows.Forms')>$null;"
                 "$n=New-Object System.Windows.Forms.NotifyIcon;"
                 "$n.Icon=[System.Drawing.SystemIcons]::Warning;$n.Visible=$true;"
                 f"$n.ShowBalloonTip(15000,'{title}','{safe}',[System.Windows.Forms.ToolTipIcon]::Warning)"],
                check=False, capture_output=True, timeout=15)
            return True, "powershell"
    except Exception as exc:
        log.info("powershell notification failed: %s", exc)
    print("\n" + "=" * 60 + f"\n{title}\n{message}\n" + "=" * 60)
    return True, "console"


def telegram_notify(message: str) -> tuple[bool, str]:
    if not (settings.telegram_token and settings.telegram_chat_id):
        return False, "telegram not configured"
    url = f"https://api.telegram.org/bot{settings.telegram_token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": settings.telegram_chat_id,
                                      "text": message}).encode()
    try:
        with urllib.request.urlopen(url, data=payload, timeout=10) as response:
            return response.status == 200, f"http {response.status}"
    except Exception as exc:
        log.warning("telegram delivery failed: %s", exc)
        return False, str(exc)


def push_stream(email: ParsedEmail, result: AnalysisResult, email_id: int) -> None:
    event = json.dumps({
        "id": email_id,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sender": email.sender,
        "subject": email.subject[:120],
        "risk": result.risk_score,
        "band": result.band,
        "base_class": result.base_class,
        "tags": result.tags,
        "confidence": round(result.confidence * 100),
    })
    try:
        STREAM.put_nowait(event)
    except queue.Full:
        try:
            STREAM.get_nowait()
            STREAM.put_nowait(event)
        except queue.Empty:
            pass


def dispatch(email: ParsedEmail, result: AnalysisResult, email_id: int) -> list[str]:
    """Always streams to the dashboard; escalates above the alert threshold."""
    push_stream(email, result, email_id)
    channels: list[str] = ["dashboard"]

    if result.risk_score < settings.alert_threshold:
        return channels

    message = build_message(email, result)
    title = f"{result.band.replace('_', ' ')} email - risk {result.risk_score}/100"

    ok, detail = desktop_notify(title, message)
    db.record_alert(email_id, "desktop", ok, detail)
    if ok:
        channels.append("desktop")

    ok, detail = telegram_notify(message)
    if ok or settings.telegram_token:
        db.record_alert(email_id, "telegram", ok, detail)
        if ok:
            channels.append("telegram")

    log.warning("ALERT risk=%d class=%s sender=%s", result.risk_score,
                result.base_class, hash_address(email.sender))
    return channels
