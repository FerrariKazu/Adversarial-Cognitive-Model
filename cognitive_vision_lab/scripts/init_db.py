#!/usr/bin/env python3
"""Initialize the PostgreSQL database schema."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cognitive_vision_lab.backend.database import SQL_SCHEMA, DATABASE_URL


def main():
    if not DATABASE_URL:
        print("DATABASE_URL not set. Skipping database initialization.")
        print("Benchmark results will be saved as JSON fallback.")
        return

    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(SQL_SCHEMA)
        conn.close()
        print(f"Database initialized at {DATABASE_URL}")
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        print("Benchmark results will use JSON fallback.")


if __name__ == "__main__":
    main()
