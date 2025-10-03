import sqlite3

DB_PATH = "people_count.db"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

tables = ["zones", "detections", "entry_exit_events"]

for t in tables:
    print(f"\n--- {t.upper()} ---")
    try:
        rows = cur.execute(f"SELECT * FROM {t} LIMIT 10").fetchall()
        if rows:
            for r in rows:
                print(dict(r))
        else:
            print("(empty)")
    except Exception as e:
        print(f"Error: {e}")

conn.close()
