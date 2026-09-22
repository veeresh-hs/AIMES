"""Signed-cookie sessions (stdlib hmac - no extra dependency, no server state)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from ..config import settings

COOKIE = "aimes_session"
MAX_AGE = 8 * 3600


def _sign(payload: bytes) -> str:
    return hmac.new(settings.session_secret.encode(), payload, hashlib.sha256).hexdigest()


def issue(username: str) -> str:
    payload = json.dumps({"u": username, "t": int(time.time())}, separators=(",", ":")).encode()
    body = base64.urlsafe_b64encode(payload).decode()
    return f"{body}.{_sign(payload)}"


def verify(token: str | None) -> str | None:
    if not token or "." not in token:
        return None
    body, signature = token.rsplit(".", 1)
    try:
        payload = base64.urlsafe_b64decode(body.encode())
    except Exception:
        return None
    if not hmac.compare_digest(signature, _sign(payload)):
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if time.time() - data.get("t", 0) > MAX_AGE:
        return None
    return data.get("u")
