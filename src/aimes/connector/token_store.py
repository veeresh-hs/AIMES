"""Encrypted OAuth-token storage.

Uses Fernet (from `cryptography`) with a key held in AIMES_TOKEN_KEY.  If the
package is missing the token is still never committed - it is written with
owner-only permissions and a loud warning, because a plaintext refresh token on
disk is a real risk, not a footnote.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from ..config import settings
from ..logging_conf import get_logger

log = get_logger(__name__)


def generate_key() -> str:
    try:
        from cryptography.fernet import Fernet  # type: ignore

        return Fernet.generate_key().decode()
    except Exception:
        return os.urandom(32).hex()


def _fernet():
    if not settings.token_key:
        return None
    try:
        from cryptography.fernet import Fernet  # type: ignore

        return Fernet(settings.token_key.encode())
    except Exception as exc:
        log.warning("token encryption unavailable (%s)", type(exc).__name__)
        return None


def save_token(data: dict, path: Path | None = None) -> None:
    path = Path(path or settings.gmail_token)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data).encode()
    box = _fernet()
    if box is not None:
        path.write_bytes(box.encrypt(payload))
    else:
        log.warning("storing the OAuth token UNENCRYPTED at %s. "
                    "Set AIMES_TOKEN_KEY in .env and install `cryptography`.", path)
        path.write_bytes(payload)
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def load_token(path: Path | None = None) -> dict | None:
    path = Path(path or settings.gmail_token)
    if not path.exists():
        return None
    raw = path.read_bytes()
    box = _fernet()
    if box is not None:
        try:
            return json.loads(box.decrypt(raw))
        except Exception as exc:
            log.error("could not decrypt %s (%s). Delete it and re-authorise.",
                      path, type(exc).__name__)
            return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.error("token file is encrypted but AIMES_TOKEN_KEY is not set")
        return None


def clear_token(path: Path | None = None) -> None:
    path = Path(path or settings.gmail_token)
    if path.exists():
        path.unlink()
