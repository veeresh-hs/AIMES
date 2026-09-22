"""Create the SQLite schema and the dashboard login."""
import argparse
import getpass

import _bootstrap  # noqa: F401

from aimes.database import init_db, create_user
from aimes.logging_conf import get_logger

log = get_logger("init_db")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Initialise the AIMES database")
    ap.add_argument("--user", default=None, help="dashboard username")
    ap.add_argument("--password", default=None, help="dashboard password (prompted if omitted)")
    args = ap.parse_args()

    init_db()
    print("Database schema created.")

    if args.user:
        password = args.password or getpass.getpass("Dashboard password: ")
        if len(password) < 8:
            raise SystemExit("Choose a password of at least 8 characters.")
        create_user(args.user, password)
        print(f"Dashboard user '{args.user}' created.")
    else:
        print("No dashboard user created. Re-run with --user <name> when you need one.")
