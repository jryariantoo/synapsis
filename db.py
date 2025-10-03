# db.py
import sqlite3
import json
from contextlib import closing

def init_db(db_path="people_count.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS zones(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            coordinates TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS detections(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id INTEGER,
            timestamp TIMESTAMP,
            track_id INTEGER,
            bbox_x1 REAL, bbox_y1 REAL,
            bbox_x2 REAL, bbox_y2 REAL,
            confidence REAL,
            class_id INTEGER,
            FOREIGN KEY(zone_id) REFERENCES zones(id)
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS entry_exit_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            event_type TEXT CHECK(event_type IN ('entry','exit')),
            event_time TIMESTAMP,
            FOREIGN KEY(detection_id) REFERENCES detections(id)
        );
        """)
    return conn

def insert_zone(conn, name, coordinates):
    coords_json = json.dumps(coordinates)
    with conn:
        cur = conn.execute(
            "INSERT INTO zones (name, coordinates) VALUES (?, ?)",
            (name, coords_json)
        )
    return cur.lastrowid

def insert_detection(conn, zone_id, timestamp, track_id, bbox, confidence, class_id):
    with conn:
        cur = conn.execute(
            """INSERT INTO detections
               (zone_id, timestamp, track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, class_id)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (zone_id, timestamp, track_id, *bbox, confidence, class_id)
        )
    return cur.lastrowid

def insert_event(conn, detection_id, event_type, event_time):
    with conn:
        conn.execute(
            "INSERT INTO entry_exit_events (detection_id, event_type, event_time) VALUES (?,?,?)",
            (detection_id, event_type, event_time)
        )
