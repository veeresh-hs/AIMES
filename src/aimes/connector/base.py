"""Provider interface so IMAP/Outlook can be added without touching detection."""
from __future__ import annotations

from typing import Iterator, Protocol

from ..schema import ParsedEmail


class MailProvider(Protocol):
    name: str

    def connect(self) -> None: ...

    def fetch_new(self, limit: int = 25) -> Iterator[ParsedEmail]: ...

    def label(self, message_id: str, label: str) -> bool: ...
