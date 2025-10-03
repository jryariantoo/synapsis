# People Counter in Polygon Zone (YOLOv8 + Supervision)

This is a standalone Python script converted from a notebook workflow so you can run and modify it locally.

## What it does

- Detects **people** using YOLOv8 (COCO weights).
- Tracks them with ByteTrack (via `supervision`) for stable IDs.
- Lets you define a **polygon zone** (interactively or via CLI).
- Shows a live **count of people in the zone** and an optional **entries** counter.
- Works with a video file or your webcam, and can save the annotated output.

## 1) Install dependencies

> Use a fresh virtual environment (recommended).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

## 2) Run it

### Webcam with interactive polygon

```bash
python people_counter_zone.py --source 0
```

- On the first frame, **click** to add polygon points.
- Press **ENTER** to finish, **BACKSPACE** to undo last point, **ESC** to cancel.

### Video file with polygon via CLI

```bash
python people_counter_zone.py --source path/to/video.mp4 --polygon "200,100 600,100 600,400 200,400"
```

### Save output

```bash
python people_counter_zone.py --source path/to/video.mp4 --save out.mp4
```

### Useful options

- `--model yolov8n.pt` (default) or try `yolov8s.pt` for better accuracy.
- `--device 0` to use GPU 0 (CUDA); use `--device cpu` to force CPU.
- `--conf 0.4` and `--iou 0.5` adjust detection thresholds.
- `--show_ids` to draw track IDs.
- `--show_traces` to draw motion traces.
- `--entries_counter` to keep a cumulative count of outside -> inside entries.

## 3) Notes & Tips

- First run will auto-download the YOLO weights (internet required once).
- If your video is rotated or huge, consider pre-processing for speed.
- To change the "inside" test, edit `inside_polygon()`.
- If you want a **permanent zone** for multiple videos, pass `--polygon` instead of drawing each time.

## Troubleshooting

- **No module named ultralytics/supervision**: `pip install -r requirements.txt`
- **Black window / no video**: Check `--source` path/index.
- **Slow FPS**: Use a smaller model (`yolov8n.pt`), lower `--conf`, or enable GPU with `--device 0`.

usage: people_counter.py [-h] --source SOURCE [--db DB] [--zone-id ZONE_ID]
[--zone-name ZONE_NAME] [--zone-coords ZONE_COORDS]
[--imgsz IMGSZ] [--conf CONF] [--max-fps MAX_FPS]
[--model MODEL] [--save SAVE] [--no-gui]

People counter (CCTV live / HLS) with saving.

options:
-h, --help show this help message and exit
--db DB SQLite DB path (default: people_count.db)
--source SOURCE RTSP/HLS URL (e.g., ...playlist.m3u8)
--zone-id ZONE_ID Use existing zone id from DB
--zone-name ZONE_NAME Name of zone when creating via --zone-coords
--zone-coords ZONE_COORDS
Polygon coords "x1,y1;x2,y2;..."
--imgsz IMGSZ YOLO image size (default: 640)
--conf CONF YOLO confidence threshold (default: 0.25)
--max-fps MAX_FPS Max processing FPS (default: 12.0)
--model MODEL YOLO model (default: yolov8s.pt)
--save SAVE Output MP4 path (if not given, auto path is used)
--no-gui Disable live preview window (headless mode)
