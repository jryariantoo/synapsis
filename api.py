# #!/usr/bin/env python3
# from fastapi import FastAPI, Query, HTTPException
# from pydantic import BaseModel, Field, validator
# from typing import List, Optional
# import sqlite3
# import json
# from datetime import datetime, timedelta

# DB_PATH = "people_count.db"

# app = FastAPI(
#     title="People Counter API",
#     description="Serves historical, live stats from people_count.db",
#     version="1.1.0"
# )

# # ---------- DB helpers ----------
# def get_conn():
#     conn = sqlite3.connect(DB_PATH, check_same_thread=False)
#     conn.row_factory = sqlite3.Row
#     return conn

# def get_default_zone_id(conn: sqlite3.Connection) -> int:
#     row = conn.execute("SELECT id FROM zones ORDER BY id DESC LIMIT 1").fetchone()
#     if not row:
#         raise HTTPException(status_code=404, detail="No zones found in database")
#     return int(row["id"])

# # ---------- Models ----------
# class ZoneCreate(BaseModel):
#     name: Optional[str] = Field(default=None, description="Optional zone name")
#     coordinates: List[List[float]] = Field(..., description="List of [x, y] points, at least 3")

#     @validator("coordinates")
#     def _check_coords(cls, v):
#         if not isinstance(v, list) or len(v) < 3:
#             raise ValueError("coordinates must have at least 3 points")
#         for i, pt in enumerate(v):
#             if not (isinstance(pt, list) or isinstance(pt, tuple)) or len(pt) != 2:
#                 raise ValueError(f"coordinates[{i}] must be [x, y]")
#             # floats/ints are fine
#         return v

# class StatsItem(BaseModel):
#     bucket_start: str  # ISO-like UTC
#     entry_count: int
#     exit_count: int

# class StatsResponse(BaseModel):
#     zone_id: int
#     bucket: str
#     start: Optional[str] = None
#     end: Optional[str] = None
#     totals: dict
#     items: List[StatsItem]

# class LiveStatsResponse(BaseModel):
#     zone_id: int
#     entries_total: int
#     exits_total: int
#     in_count: int
#     last_event_time: Optional[str] = None



# # ---------- Time helpers ----------
# BUCKET_SQL = {
#     "minute": "%Y-%m-%d %H:%M:00",
#     "hour":   "%Y-%m-%d %H:00:00",
#     "day":    "%Y-%m-%d 00:00:00",
# }

# def parse_iso(ts: Optional[str]) -> Optional[str]:
#     """
#     Accepts 'YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DD HH:MM:SS'. Returns normalized 'YYYY-MM-DD HH:MM:SS'.
#     """
#     if ts is None:
#         return None
#     ts = ts.strip().replace("T", " ")
#     if len(ts) == 16:  # allow seconds omitted
#         ts += ":00"
#     try:
#         datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
#         return ts
#     except ValueError:
#         raise HTTPException(status_code=400, detail=f"Invalid datetime format: {ts}. Use YYYY-MM-DDTHH:MM:SS")

# # ---------- Existing endpoints ----------
# @app.get("/api/zones")
# def list_zones():
#     with get_conn() as conn:
#         rows = conn.execute("SELECT id, name, coordinates, created_at FROM zones ORDER BY id").fetchall()
#         return [dict(r) for r in rows]

# @app.get("/api/stats/", response_model=StatsResponse)
# def get_stats(
#     zone_id: Optional[int] = Query(default=None, description="Zone ID; defaults to latest zone"),
#     start: Optional[str] = Query(default=None, description="Start time (UTC) YYYY-MM-DDTHH:MM:SS"),
#     end: Optional[str] = Query(default=None, description="End time (UTC) YYYY-MM-DDTHH:MM:SS"),
#     bucket: str = Query(default="minute", regex="^(minute|hour|day)$")
# ):
#     start_s = parse_iso(start)
#     end_s   = parse_iso(end)
#     if bucket not in BUCKET_SQL:
#         raise HTTPException(status_code=400, detail="bucket must be one of: minute, hour, day")

#     with get_conn() as conn:
#         zid = zone_id if zone_id is not None else get_default_zone_id(conn)
#         bucket_expr = f"strftime('{BUCKET_SQL[bucket]}', e.event_time)"
#         params = [zid]
#         where = "WHERE d.zone_id = ?"
#         if start_s:
#             where += " AND e.event_time >= ?"; params.append(start_s)
#         if end_s:
#             where += " AND e.event_time <= ?"; params.append(end_s)

#         q_items = f"""
#         SELECT
#             {bucket_expr} AS bucket_start,
#             SUM(CASE WHEN e.event_type='entry' THEN 1 ELSE 0 END) AS entry_count,
#             SUM(CASE WHEN e.event_type='exit'  THEN 1 ELSE 0 END) AS exit_count
#         FROM entry_exit_events e
#         JOIN detections d ON e.detection_id = d.id
#         {where}
#         GROUP BY bucket_start
#         ORDER BY bucket_start ASC;"""
#         items = [dict(r) for r in conn.execute(q_items, params).fetchall()]

#         q_tot = f"""
#         SELECT
#             SUM(CASE WHEN e.event_type='entry' THEN 1 ELSE 0 END) AS entries_total,
#             SUM(CASE WHEN e.event_type='exit'  THEN 1 ELSE 0 END) AS exits_total
#         FROM entry_exit_events e
#         JOIN detections d ON e.detection_id = d.id
#         {where};"""
#         tot = conn.execute(q_tot, params).fetchone()
#         totals = {"entries_total": int(tot["entries_total"] or 0),
#                   "exits_total":   int(tot["exits_total"]   or 0)}

#         return StatsResponse(
#             zone_id=zid, bucket=bucket, start=start_s, end=end_s,
#             totals=totals, items=[StatsItem(**it) for it in items]
#         )

# @app.get("/api/stats/live", response_model=LiveStatsResponse)
# def get_live_stats(zone_id: Optional[int] = Query(default=None)):
#     with get_conn() as conn:
#         zid = zone_id if zone_id is not None else get_default_zone_id(conn)
#         row = conn.execute("""
#             SELECT
#               SUM(CASE WHEN e.event_type='entry' THEN 1 ELSE 0 END) AS entries_total,
#               SUM(CASE WHEN e.event_type='exit'  THEN 1 ELSE 0 END) AS exits_total,
#               MAX(e.event_time) AS last_event_time
#             FROM entry_exit_events e
#             JOIN detections d ON e.detection_id = d.id
#             WHERE d.zone_id = ?;""", (zid,)).fetchone()
#         entries_total = int(row["entries_total"] or 0)
#         exits_total   = int(row["exits_total"]   or 0)
#         in_count      = max(0, entries_total - exits_total)
#         return LiveStatsResponse(
#             zone_id=zid,
#             entries_total=entries_total,
#             exits_total=exits_total,
#             in_count=in_count,
#             last_event_time=row["last_event_time"]
#         )

# # ---------- NEW: POST /api/config/area ----------
# @app.post("/api/config/area")
# def create_area(cfg: ZoneCreate):
#     with get_conn() as conn:
#         coords_json = json.dumps(cfg.coordinates)
#         cur = conn.execute(
#             "INSERT INTO zones (name, coordinates) VALUES (?, ?)",
#             (cfg.name, coords_json)
#         )
#         zid = int(cur.lastrowid)
#         row = conn.execute(
#             "SELECT id, name, coordinates, created_at FROM zones WHERE id = ?",
#             (zid,)
#         ).fetchone()
#         return dict(row)


# api.py (minimal)
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Any
import sqlite3, json, time

DB_PATH = "people_count.db"

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("PRAGMA foreign_keys = ON;")
    return conn

conn = get_conn()
app = FastAPI(title="People Count API (minimal)")

# ---- Models ----
class ZoneIn(BaseModel):
    name: str = Field(..., min_length=1)
    coordinates: List[Tuple[float, float]] = Field(..., min_items=3)  # polygon points

# ---- Endpoints ----

# 1) Historical stats
@app.get("/api/stats/")
def get_stats(
    start: Optional[int] = Query(None, description="Start unix epoch (inclusive)"),
    end: Optional[int] = Query(None, description="End unix epoch (inclusive)"),
    zone_id: Optional[int] = Query(None, description="Filter by zone id"),
):
    q = """
    SELECT z.id AS zone_id, z.name AS zone_name,
           SUM(CASE WHEN e.event_type='entry' THEN 1 ELSE 0 END) AS entries,
           SUM(CASE WHEN e.event_type='exit'  THEN 1 ELSE 0 END) AS exits
    FROM entry_exit_events e
    JOIN detections d ON d.id = e.detection_id
    JOIN zones z ON z.id = d.zone_id
    WHERE 1=1
    """
    args: List[Any] = []
    if start is not None:
        q += " AND e.event_time >= ?"; args.append(start)
    if end is not None:
        q += " AND e.event_time <= ?"; args.append(end)
    if zone_id is not None:
        q += " AND d.zone_id = ?"; args.append(zone_id)
    q += " GROUP BY z.id, z.name ORDER BY z.id"
    rows = conn.execute(q, args).fetchall()
    return [dict(r) for r in rows]

# 2) Live stats (last N seconds; default 60)
@app.get("/api/stats/live")
def get_stats_live(
    within_seconds: int = Query(60, ge=1, le=86400, description="Window in seconds (default 60)"),
    zone_id: Optional[int] = Query(None, description="Filter by zone id"),
):
    q = """
    SELECT z.id AS zone_id, z.name AS zone_name,
           SUM(CASE WHEN e.event_type='entry' THEN 1 ELSE 0 END) AS entries_recent,
           SUM(CASE WHEN e.event_type='exit'  THEN 1 ELSE 0 END) AS exits_recent
    FROM entry_exit_events e
    JOIN detections d ON d.id = e.detection_id
    JOIN zones z ON z.id = d.zone_id
    WHERE e.event_time >= strftime('%s','now') - ?
    """
    args: List[Any] = [within_seconds]
    if zone_id is not None:
        q += " AND d.zone_id = ?"; args.append(zone_id)
    q += " GROUP BY z.id, z.name ORDER BY z.id"
    rows = conn.execute(q, args).fetchall()
    return [dict(r) for r in rows]

# 3) Insert polygon (zone)
@app.post("/api/config/area")
def create_zone(zone: ZoneIn):
    with conn:
        cur = conn.execute(
            "INSERT INTO zones (name, coordinates) VALUES (?, ?)",
            (zone.name, json.dumps(zone.coordinates)),
        )
    return {"message": "zone created", "zone_id": cur.lastrowid}
