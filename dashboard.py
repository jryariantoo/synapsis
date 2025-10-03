import os
import time
import json
import datetime as dt
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import requests
from PIL import Image, ImageDraw
import streamlit as st

# Optional autorefresh for live stats
try:
    from streamlit_autorefresh import st_autorefresh
    AUTORELOAD_AVAILABLE = True
except Exception:
    AUTORELOAD_AVAILABLE = False

# Click-to-get (x,y) on image
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    CLICKER_AVAILABLE = True
except Exception:
    CLICKER_AVAILABLE = False

# For snapshot capture (single frame from MP4 or .m3u8)
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

# ------------------------------
# Config
# ------------------------------
API_BASE = "http://localhost:8000"
st.set_page_config(page_title="People Counter Dashboard", layout="wide")

# ------------------------------
# Helpers
# ------------------------------
def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except requests.RequestException:
        return None

def api_post(path: str, payload: Dict[str, Any]) -> Optional[Any]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=8)
        if r.status_code in (200, 201):
            return r.json()
        else:
            try:
                return {"error": r.text, "status": r.status_code}
            except Exception:
                return {"error": "Request failed", "status": r.status_code}
    except requests.RequestException as e:
        return {"error": str(e)}

def unix_epoch(dt_obj: dt.datetime) -> int:
    return int(dt_obj.timestamp())

def combine_date_time(d: dt.date, t: dt.time) -> dt.datetime:
    return dt.datetime.combine(d, t)

def draw_polygon(img: Image.Image, pts: List[Tuple[float, float]], color=(255, 0, 0), width=2, fill_alpha=60) -> Image.Image:
    if not pts:
        return img
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    if len(pts) >= 3:
        draw.polygon(pts, fill=(color[0], color[1], color[2], fill_alpha))
    draw.line(pts + [pts[0]] if len(pts) >= 2 else pts, fill=(color[0], color[1], color[2], 255), width=width)
    for (x, y) in pts:
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(color[0], color[1], color[2], 255))
    return out

def grab_snapshot(source_mode: str, mp4_file, stream_url: str) -> Optional[Image.Image]:
    if not CV2_AVAILABLE:
        return None
    if stream_url and (stream_url.startswith(("http://", "https://", "rtsp://"))):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|reorder_queue_size;0|buffer_size;102400"
    if source_mode == "MP4 file" and mp4_file is not None:
        tmp_path = f"/tmp/_st_video_{int(time.time()*1000)}.mp4"
        with open(tmp_path, "wb") as f:
            f.write(mp4_file.getvalue())
        cap = cv2.VideoCapture(tmp_path)
    elif source_mode == "Stream URL" and stream_url:
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    else:
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

def summarize_live_rows(rows: Optional[List[Dict[str, Any]]], zone_id: Optional[int]) -> Tuple[int, int]:
    if not rows:
        return (0, 0)
    if zone_id is not None:
        for r in rows:
            if r.get("zone_id") == zone_id:
                return (int(r.get("entries_recent", 0)), int(r.get("exits_recent", 0)))
        return (0, 0)
    e = sum(int(r.get("entries_recent", 0)) for r in rows)
    x = sum(int(r.get("exits_recent", 0)) for r in rows)
    return (e, x)

def summarize_hist_rows(rows: Optional[List[Dict[str, Any]]], zone_id: Optional[int]) -> Tuple[int, int]:
    if not rows:
        return (0, 0)
    if zone_id is not None:
        for r in rows:
            if r.get("zone_id") == zone_id:
                return (int(r.get("entries", 0)), int(r.get("exits", 0)))
        return (0, 0)
    e = sum(int(r.get("entries", 0)) for r in rows)
    x = sum(int(r.get("exits", 0)) for r in rows)
    return (e, x)

def ui_pros_cons():
    st.info(
        "🔌 **Source options**\n\n"
        "**MP4 file (local):** smooth playback, easy snapshots, works offline — but *not live*.\n"
        "**Stream URL (.m3u8 CCTV):** live, but may buffer/disconnect and snapshots can fail in some setups.\n",
        icon="ℹ️",
    )

# ------------------------------
# Session defaults
# ------------------------------
st.session_state.setdefault("source_mode", "MP4 file")
st.session_state.setdefault("mp4_file", None)
st.session_state.setdefault("stream_url", "")
st.session_state.setdefault("polygon_coords", [])
st.session_state.setdefault("zone_id", None)
st.session_state.setdefault("edit_points", [])
st.session_state.setdefault("last_snapshot", None)

# ------------------------------
# UI: Tabs
# ------------------------------
tab_home, tab_stats, tab_edit = st.tabs(["🏠 Home", "📈 Stats", "✏️ Update Coordinate"])

# =========================================
# Tab 1: Home
# =========================================
with tab_home:
    st.subheader("Video & Live Counters")
    ui_pros_cons()

    source_mode = st.radio("Select source:", ["MP4 file", "Stream URL"], horizontal=True, key="source_mode")
    col_a, col_b = st.columns([2, 3])

    with col_a:
        if source_mode == "MP4 file":
            st.file_uploader("Upload a .mp4 file", type=["mp4"], key="mp4_file")
            st.caption("Tip: short clips loop well for demos.")
        else:
            st.text_input("Stream URL (.m3u8)", value=st.session_state.get("stream_url", ""), key="stream_url")
            st.caption("If the video doesn't play in your browser, it may need Safari or a custom HLS player.")

        st.divider()
        zid = st.number_input("Zone ID (optional filter for stats)", min_value=1, step=1, value=st.session_state.get("zone_id") or 1)
        apply_zone = st.checkbox("Apply zone filter to live stats", value=True)
        st.session_state["zone_id"] = int(zid) if apply_zone else None

        within = st.slider("Live window (seconds)", 10, 900, 120, step=10)
        if AUTORELOAD_AVAILABLE:
            st_autorefresh(interval=5000, key="live_refresh_home")

    with col_b:
        if source_mode == "MP4 file" and st.session_state["mp4_file"] is not None:
            st.video(st.session_state["mp4_file"])
        elif source_mode == "Stream URL" and st.session_state["stream_url"]:
            st.video(st.session_state["stream_url"])
        else:
            st.info("Provide a source on the left to preview the video.")

        snap = grab_snapshot(source_mode, st.session_state["mp4_file"], st.session_state["stream_url"])
        poly = st.session_state.get("polygon_coords", [])
        if snap is not None and poly:
            st.image(draw_polygon(snap, poly, color=(255, 0, 0)), caption="Snapshot with polygon overlay", use_column_width=True)
        elif snap is not None:
            st.image(snap, caption="Snapshot (no polygon configured)", use_column_width=True)

    st.subheader("Live Stats")
    params = {"within_seconds": within}
    if st.session_state["zone_id"]:
        params["zone_id"] = st.session_state["zone_id"]
    live_rows = api_get("/api/stats/live", params=params)
    e_recent, x_recent = summarize_live_rows(live_rows, st.session_state["zone_id"])
    c1, c2 = st.columns(2)
    c1.metric("Entries (recent)", e_recent)
    c2.metric("Exits (recent)", x_recent)
    if not live_rows:
        st.caption("No recent events in the selected window. Make sure the counter is running and the polygon is in a busy area.")

# =========================================
# Tab 2: Stats
# =========================================
with tab_stats:
    st.subheader("Historical & Live")
    sf1, sf2, sf3 = st.columns(3)
    with sf1:
        use_zone = st.checkbox("Filter by Zone ID", value=(st.session_state.get("zone_id") is not None))
        zid_hist = st.number_input("Zone ID", min_value=1, step=1, value=int(st.session_state.get("zone_id") or 1))
    with sf2:
        start_date = st.date_input("Start date", value=dt.date.today() - dt.timedelta(days=1))
        start_time = st.time_input("Start time", value=dt.time(0, 0))
    with sf3:
        end_date = st.date_input("End date", value=dt.date.today())
        end_time = st.time_input("End time", value=dt.time(23, 59))

    start_dt = combine_date_time(start_date, start_time)
    end_dt = combine_date_time(end_date, end_time)
    if end_dt < start_dt:
        st.error("End must be after start.")
    else:
        qparams = {"start": unix_epoch(start_dt), "end": unix_epoch(end_dt)}
        if use_zone and zid_hist:
            qparams["zone_id"] = int(zid_hist)
        rows = api_get("/api/stats/", params=qparams)
        e_tot, x_tot = summarize_hist_rows(rows, int(zid_hist) if use_zone else None)

        c1, c2 = st.columns(2)
        c1.metric("Entries (range)", e_tot)
        c2.metric("Exits (range)", x_tot)

        if rows:
            st.caption("Grouped results by zone")
            st.table(rows)
        else:
            st.info("No historical events for the selected range.")

    st.divider()
    st.subheader("Live (quick view)")
    win_quick = st.slider("Window (seconds)", 10, 600, 60, key="live2")
    lp = {"within_seconds": win_quick}
    if use_zone and zid_hist:
        lp["zone_id"] = int(zid_hist)
    live_rows2 = api_get("/api/stats/live", params=lp)
    e2, x2 = summarize_live_rows(live_rows2, int(zid_hist) if use_zone else None)
    c3, c4 = st.columns(2)
    c3.metric("Entries (recent)", e2)
    c4.metric("Exits (recent)", x2)
    if AUTORELOAD_AVAILABLE:
        st_autorefresh(interval=5000, key="live_refresh_stats")

# =========================================
# Tab 3: Update Coordinate
# =========================================
with tab_edit:
    st.subheader("Define / Save Polygon (click-to-place points)")
    st.caption("We use a single frozen frame so clicking is precise. No drag — click to add vertices in order.")

    if not CLICKER_AVAILABLE:
        st.error("Please install `streamlit-image-coordinates`:\n\npip install streamlit-image-coordinates")
    elif not CV2_AVAILABLE:
        st.error("Please install OpenCV:\n\npip install opencv-python")
    else:
        ed1, ed2 = st.columns([2, 3])
        with ed1:
            emode = st.radio("Source for snapshot:", ["Current selection", "Upload MP4", "Enter stream URL"], horizontal=False)
            up_file = None
            url_in = ""
            if emode == "Upload MP4":
                up_file = st.file_uploader("Upload .mp4 for snapshot", type=["mp4"], key="edit_mp4")
            elif emode == "Enter stream URL":
                url_in = st.text_input("Stream URL (.m3u8)", value="")
            snap_btn = st.button("Capture snapshot")

        with ed2:
            snapshot = None
            if snap_btn:
                if emode == "Current selection":
                    snapshot = grab_snapshot(st.session_state["source_mode"], st.session_state["mp4_file"], st.session_state["stream_url"])
                elif emode == "Upload MP4":
                    snapshot = grab_snapshot("MP4 file", up_file, "")
                else:
                    snapshot = grab_snapshot("Stream URL", None, url_in)
                if snapshot is None:
                    st.error("Could not capture a snapshot from the selected source.")
                else:
                    st.session_state["last_snapshot"] = snapshot
                    st.session_state["edit_points"] = []  # reset when new snapshot captured

            snapshot = st.session_state.get("last_snapshot", None)
            if snapshot is None:
                st.info("Capture a snapshot first.")
            else:
                st.success("Snapshot ready. Click on the image to add polygon points.")
                pts = st.session_state.setdefault("edit_points", [])

                # 1) Click on RAW snapshot (keeps component active)
                click = streamlit_image_coordinates(
                    snapshot,
                    key="coord_img",
                    width=min(960, snapshot.width),
                )
                if click and "x" in click and "y" in click:
                    pts.append((float(click["x"]), float(click["y"])))

                # 2) Separate PREVIEW with polygon
                preview = draw_polygon(snapshot, pts, color=(255, 0, 0), width=2, fill_alpha=40)
                st.image(preview, caption="Polygon preview", use_column_width=True)

                # Controls
                cc1, cc2, cc3 = st.columns([1, 1, 2])
                with cc1:
                    if st.button("Undo last point", use_container_width=True):
                        if pts:
                            pts.pop()
                with cc2:
                    if st.button("Clear all", use_container_width=True):
                        pts.clear()
                with cc3:
                    zone_name = st.text_input("Zone name", value=f"Zone-{int(time.time())}")

                st.write("**Current points (x,y):**")
                st.code(json.dumps(pts, indent=2))

                # Save
                can_save = len(pts) >= 3
                if not can_save:
                    st.warning("Add at least 3 points to form a polygon.")
                if st.button("Save polygon to backend", disabled=not can_save, type="primary"):
                    coords = [[float(x), float(y)] for (x, y) in pts]   # LISTS for JSON
                    payload = {"name": zone_name, "coordinates": coords}
                    resp = api_post("/api/config/area", payload)
                    if resp and "zone_id" in resp:
                        zid = resp["zone_id"]
                        st.success(f"Saved ✅  (zone_id={zid})")
                        st.session_state["polygon_coords"] = coords   # Home can draw immediately
                        st.session_state["zone_id"] = zid
                    else:
                        st.error(f"Save failed: {resp}")

# Footer
st.divider()
st.caption(
    "Notes: The video player itself cannot be directly overlaid with shapes in Streamlit. "
    "We show a snapshot with the polygon for clarity. Live stats poll the backend; "
    "make sure your detector is running and sending events."
)
