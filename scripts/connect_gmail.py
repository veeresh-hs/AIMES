"""One-time Gmail authorisation, then an optional monitoring loop.

    python scripts/connect_gmail.py            # authorise and fetch once
    python scripts/connect_gmail.py --watch    # keep polling in the foreground

Opens your browser for Google's consent screen.  You are never asked for your
password here, and the app requests read access only unless you explicitly
enabled labelling in .env.
"""
from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401

from aimes import database as db
from aimes.config import settings
from aimes.connector.oauth import required_scopes
from aimes.logging_conf import get_logger
from aimes.monitor import MONITOR, connect_gmail

log = get_logger("connect_gmail")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="keep polling until Ctrl+C")
    ap.add_argument("--once", action="store_true", help="fetch and analyse one batch")
    args = ap.parse_args()

    db.init_db()
    print(f"Requesting scopes: {required_scopes()}")
    client = connect_gmail(interactive=True)
    print(f"Connected as {client.profile.get('emailAddress')}")

    if args.watch:
        MONITOR.start()
        print(f"Polling every {settings.poll_seconds}s. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            MONITOR.stop()
            print("\nStopped.")
    else:
        count = MONITOR.poll_once()
        print(f"{count} new email(s) analysed. Open the dashboard to see them.")
