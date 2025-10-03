# #!/usr/bin/env python3
# import numpy as np
# import supervision as sv
# from ultralytics import YOLO
# import cv2
# from datetime import datetime
# from db import init_db, insert_zone, insert_detection, insert_event

# # Initialise database and zone record

# conn = init_db("people_count.db")
# ## coordintaes test_1
# # polygon = np.array([
# #     [50, 50],
# #     [200, 50],
# #     [200, 250],
# #     [50, 250]
# # ])

# #coordinates sample 3
# polygon = np.array([
#     [400, 250],   # top-left
#     [600, 250],   # top-right
#     [600, 400],   # bottom-right
#     [400, 400]    # bottom-left
# ])

# zone_id = insert_zone(conn, name="SampleZone", coordinates=polygon.tolist())

# # YOLO model and tracker
# model = YOLO("yolov8s.pt")
# tracker = sv.ByteTrack()
# zone = sv.PolygonZone(polygon=polygon)
# box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color(0,255,0))
# zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color(255,0,0),
#                                          thickness=3, text_thickness=1, text_scale=0.6)

# prev_inside_ids = set()
# entries_total = 0
# exits_total = 0

# def process_frame(frame: np.ndarray, frame_index: int) -> np.ndarray:
#     global prev_inside_ids, entries_total, exits_total
#     timestamp = datetime.utcnow()

#     # Detect people
#     results = model(frame, imgsz=640, conf=0.2)[0]
#     detections = sv.Detections.from_ultralytics(results)
#     detections = detections[detections.class_id == 0]
#     # Track IDs
#     detections = tracker.update_with_detections(detections)
#     zone.trigger(detections=detections)

#     # Find IDs whose box centres lie in polygon
#     current_inside_ids = set()
#     for xyxy, track_id, conf, cls in zip(detections.xyxy, detections.tracker_id,
#                                          detections.confidence, detections.class_id):
#         if track_id is None:
#             continue
#         x1, y1, x2, y2 = xyxy
#         cx, cy = int((x1+x2)/2), int((y1+y2)/2)
#         if cv2.pointPolygonTest(polygon, (cx,cy), False) >= 0:
#             current_inside_ids.add(int(track_id))
#         # Log every detection
#         det_id = insert_detection(
#             conn, zone_id, timestamp, int(track_id), (x1,y1,x2,y2), float(conf), int(cls)
#         )

#     # Check for new entries/exits and record events
#     new_entries = current_inside_ids - prev_inside_ids
#     for _ in new_entries:
#         insert_event(conn, det_id, "entry", timestamp)
#     entries_total += len(new_entries)

#     exited = prev_inside_ids - current_inside_ids
#     for _ in exited:
#         insert_event(conn, det_id, "exit", timestamp)
#     exits_total += len(exited)

#     prev_inside_ids = current_inside_ids

#     # Annotate frame
#     frame = box_annotator.annotate(scene=frame, detections=detections)
#     frame = zone_annotator.annotate(scene=frame)
#     cv2.putText(frame, f"In zone: {len(current_inside_ids)}", (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#     cv2.putText(frame, f"Entries: {entries_total}", (20,80),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#     cv2.putText(frame, f"Exits: {exits_total}", (20,120),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#     return frame


# # video_path = "https://cctvjss.jogjakota.go.id/malioboro/NolKm_Utara.stream/playlist.m3u8"

# # Process video
# sv.process_video(
#     source_path="./datas/samples/sample3.mp4",
#     # source_path = video_path,
#     target_path="./datas/results/result_cctv.mp4",
#     callback=process_frame
# )
# print("✅ Done processing; results saved to MP4 and database.")
# print("Using device:", model.device)

#!/usr/bin/env python3
import os, argparse, time, json, sys
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

from db import init_db, insert_zone, insert_detection, insert_event

# ---------- Utilities ----------
def parse_coords(s: str) -> List[Tuple[float, float]]:
    """Parse 'x1,y1;x2,y2;...' -> [(x1,y1), (x2,y2), ...]"""
    pts = []
    for pair in s.split(";"):
        x, y = pair.strip().split(",")
        pts.append((float(x), float(y)))
    return pts

def load_or_create_zone(conn, zone_id: Optional[int], zone_name: Optional[str], zone_coords: Optional[str]):
    """Return (zone_id, polygon_np). If no zone args, returns (None, empty polygon)."""
    if zone_id is not None:
        row = conn.execute("SELECT id, coordinates FROM zones WHERE id=?", (zone_id,)).fetchone()
        if not row:
            raise ValueError(f"Zone id {zone_id} not found.")
        coords = json.loads(row["coordinates"])
        return row["id"], np.array(coords, dtype=np.float32)

    if zone_coords:
        if not zone_name:
            zone_name = "Zone-" + str(int(time.time()))
        coords = parse_coords(zone_coords)
        zid = insert_zone(conn, name=zone_name, coordinates=coords)
        return zid, np.array(coords, dtype=np.float32)

    # No zone: counting disabled; detection only
    return None, np.array([], dtype=np.float32)

def set_ffmpeg_low_latency_options(source: str):
    """
    Configure OpenCV's FFmpeg capture options to be more forgiving/low-latency.
    Works for RTSP and often for HLS.
    """
    if isinstance(source, str) and (
        source.startswith("rtsp://") or source.startswith("rtsps://")
        or source.startswith("http://") or source.startswith("https://")
        or source.endswith(".m3u8")
    ):
        base_opts = "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|reorder_queue_size;0|buffer_size;102400"
        # Preserve any user-provided additions
        existing = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = base_opts + ("|" + existing if existing else "")

def open_capture(source: str) -> cv2.VideoCapture:
    set_ffmpeg_low_latency_options(source)
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    # Reduce internal buffering (may be ignored by some builds, but harmless)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def reopen_capture(cap: Optional[cv2.VideoCapture], source: str, backoff_sec: float) -> cv2.VideoCapture:
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    time.sleep(backoff_sec)
    return open_capture(source)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Live CCTV people counting (RTSP/HLS).")
    ap.add_argument("--db", required=False, default="people_count.db", help="Path to SQLite DB")
    ap.add_argument("--source", required=True,
                    help="CCTV URL (rtsp://... or http(s)://...m3u8). Example: rtsp://user:pass@ip:554/Streaming/Channels/101")
    ap.add_argument("--model", default=os.environ.get("YOLO_MODEL", "yolov8s.pt"), help="Ultralytics model name/path")
    ap.add_argument("--zone-id", type=int, default=None, help="Use existing zone by id")
    ap.add_argument("--zone-name", type=str, default=None, help="Zone name (when creating)")
    ap.add_argument("--zone-coords", type=str, default=None,
                    help='Polygon coords "x1,y1;x2,y2;...". Example: "400,250;600,250;600,400;400,400"')
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO image size")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--max-fps", type=float, default=12.0, help="Max processing FPS (skip frames to keep up)")
    ap.add_argument("--save", type=str, default=None, help="Optionally record annotated output to MP4")
    ap.add_argument("--no-gui", action="store_true", help="Disable imshow window (headless)")
    args = ap.parse_args()

    # Initialize DB and zone
    conn = init_db(args.db)
    zone_id, polygon = load_or_create_zone(conn, args.zone_id, args.zone_name, args.zone_coords)
    have_polygon = polygon.size > 0
    zone = sv.PolygonZone(polygon=polygon) if have_polygon else None
    zone_annot = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color(255,0,0), thickness=3, text_thickness=1, text_scale=0.6) if have_polygon else None

    # Model / tracker
    model = YOLO(args.model)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color(0,255,0))

    # State for counting
    prev_inside_ids: set[int] = set()
    last_det_id_by_tid: Dict[int, int] = {}
    entries_total = 0
    exits_total = 0

    # Open live source
    source = args.source
    cap = open_capture(source)

    if not cap.isOpened():
        # Try a quick reconnect once
        cap = reopen_capture(cap, source, backoff_sec=1.0)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open live source: {source}")

    # Optional recording
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Use a fallback size/fps if stream does not report them
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fps = cap.get(cv2.CAP_PROP_FPS) or args.max_fps
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    print("▶️  LIVE mode. Press 'q' to stop (if GUI enabled).")
    next_allowed_time = 0.0
    backoff = 1.0
    max_backoff = 8.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            sys.stderr.write("[live] frame read failed; reconnecting...\n")
            cap = reopen_capture(cap, source, backoff_sec=backoff)
            backoff = min(max_backoff, backoff * 2.0)
            continue
        backoff = 1.0  # reset when ok

        # Throttle processing FPS to keep up with real-time
        now = time.time()
        if now < next_allowed_time:
            # We still want to consume the buffer so we stay near-live; just skip processing.
            continue
        next_allowed_time = now + (1.0 / max(args.max_fps, 0.1))

        # DETECT → TRACK
        results = model(frame, imgsz=args.imgsz, conf=args.conf)[0]
        dets = sv.Detections.from_ultralytics(results)
        dets = dets[dets.class_id == 0]  # COCO 'person'
        dets = tracker.update_with_detections(dets)

        if have_polygon:
            zone.trigger(detections=dets)

        # ZONE CHECK + DB writes
        current_inside_ids: set[int] = set()
        ts = int(time.time())

        # Iterate detections
        for xyxy, track_id, conf, cls in zip(dets.xyxy, dets.tracker_id, dets.confidence, dets.class_id):
            if track_id is None:
                continue

            x1, y1, x2, y2 = map(float, xyxy)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            inside = False
            if have_polygon:
                inside = cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0
            if inside:
                current_inside_ids.add(int(track_id))

            det_id = insert_detection(
                conn,
                zone_id if (inside and zone_id is not None) else None,
                ts,
                int(track_id),
                (x1, y1, x2, y2),
                float(conf),
                int(cls) if cls is not None else 0
            )
            last_det_id_by_tid[int(track_id)] = det_id

        # Edge-triggered entry/exit
        if have_polygon and zone_id is not None:
            # Entries
            for tid in (current_inside_ids - prev_inside_ids):
                det_for_tid = last_det_id_by_tid.get(tid)
                if det_for_tid:
                    insert_event(conn, det_for_tid, "entry", ts)
                    entries_total += 1
            # Exits
            for tid in (prev_inside_ids - current_inside_ids):
                det_for_tid = last_det_id_by_tid.get(tid)
                if det_for_tid:
                    insert_event(conn, det_for_tid, "exit", ts)
                    exits_total += 1

        prev_inside_ids = current_inside_ids

        # Annotate
        frame = box_annotator.annotate(scene=frame, detections=dets)
        if have_polygon and zone_annot:
            frame = zone_annot.annotate(scene=frame)
        cv2.putText(frame, f"LIVE", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(frame, f"In zone: {len(current_inside_ids)}", (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Entries: {entries_total}", (20,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Exits: {exits_total}", (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if writer:
            writer.write(frame)

        if not args.no_gui:
            cv2.imshow("people_count (LIVE)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("✅ Live processing ended.")

if __name__ == "__main__":
    main()
