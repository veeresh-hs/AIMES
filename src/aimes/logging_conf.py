"""Structured-ish logging.  Sender addresses are hashed before they hit disk."""
from __future__ import annotations

import hashlib
import logging
import logging.handlers
from pathlib import Path

from .config import settings

_CONFIGURED = False


def hash_address(address: str) -> str:
    """One-way pseudonym so logs never leak a real mailbox."""
    if not address:
        return "-"
    return "addr:" + hashlib.sha256(address.strip().lower().encode()).hexdigest()[:12]


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings.ensure_dirs()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-28s %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = logging.handlers.RotatingFileHandler(
        Path(settings.log_dir) / "aimes.log", maxBytes=2_000_000,
        backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers[:] = [console, file_handler]
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
