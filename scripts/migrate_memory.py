"""Opt-in SQLite migration; no .env loading, no implicit business DB selection.

python -m scripts.migrate_memory --database <explicit path>          # inspect only
python -m scripts.migrate_memory --database <explicit path> --apply  # add session_memory
"""

import argparse
import sqlite3
from pathlib import Path

from sqlalchemy.dialects.sqlite import dialect
from sqlalchemy.schema import CreateTable

from app.repositories.models import SessionMemoryRow


def create_session_memory(connection: sqlite3.Connection, *, apply: bool = False) -> bool:
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "sessions" not in tables:
        raise ValueError("sessions表不存在；请确认数据库路径。")
    if "session_memory" in tables:
        return False
    if apply:
        ddl = CreateTable(SessionMemoryRow.__table__, if_not_exists=True).compile(dialect=dialect())
        connection.execute(str(ddl))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    mode = "rw" if args.apply else "ro"
    uri = f"{args.database.resolve().as_uri()}?mode={mode}"
    with sqlite3.connect(uri, uri=True) as connection:
        needed = create_session_memory(connection, apply=args.apply)
    print(
        "Migration applied"
        if needed and args.apply
        else "Migration required (inspection only)"
        if needed
        else "Already compatible"
    )


if __name__ == "__main__":
    main()
