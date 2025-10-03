# 1. Clone & set up environment
`python -m venv .venv`
`source .venv/bin/activate`

`pip install -r requirements.txt`

# 2. Start the FastAPI backend
`uvicorn api:app --host 0.0.0.0 --port 8000 --reload`

Open docs at: http://localhost:8000/docs
Endpoints used by the dashboard:
- GET /api/stats/ (historical)
- GET /api/stats/live (recent window)
- POST /api/config/area (insert polygon)

# 3. Start the Streamlit dashboard
`streamlit run dashboard.py`

Dashboard at: http://localhost:8501

Tabs:
- Home: choose MP4 or HLS URL, live counters
- Stats: historical & live stats
- Update Coordinate: click-to-place vertices on a frozen frame, then save polygon to backend

# 4. Run the people counter (worker)
`python people_counter.py --help`

## Expected --help output:
usage: people_counter.py [-h] --source SOURCE
                         [--zone-id ZONE_ID | --zone-coords "x1,y1;..."]
                         [--zone-name ZONE_NAME]
                         [--imgsz IMGSZ] [--conf CONF]
                         [--max-fps MAX_FPS] [--model MODEL]
                         [--save SAVE] [--no-gui]

Options:
  -h, --help            Show this help and exit
  --source SOURCE       Video path or HLS URL (.m3u8)
  --zone-id ZONE_ID     Use polygon by zone_id from DB
  --zone-coords STR     Inline polygon: "x1,y1;x2,y2;..."
  --zone-name STR       Name when inserting a new polygon
  --imgsz IMGSZ         YOLO input size (default: 640)
  --conf CONF           Confidence threshold (default: 0.25)
  --max-fps MAX_FPS     Cap processing FPS (default: 20)
  --model MODEL         YOLO weights (default: yolov8s.pt)
  --save SAVE           Output annotated MP4 path
  --no-gui              Disable OpenCV window (headless)

## Run with a local MP4

`python people_counter.py \
  --source "datas/samples/sample3.mp4" \
  --zone-id 1 \
  --imgsz 640 --conf 0.25 --max-fps 20 \
  --save ./datas/results/sample3_annotated.mp4`

## Run with live CCTV (HLS)

`python people_counter.py \
  --source "https://cctvjss.jogjakota.go.id/malioboro/NolKm_Utara.stream/playlist.m3u8" \
  --zone-id 2 \
  --imgsz 640 --conf 0.25 --max-fps 20 \
  --save ./datas/results/malioboro_live.mp4`  

## Use inline polygon (no prior zone)

`python people_counter.py \
  --source "datas/samples/sample2.mp4" \
  --zone-coords "860,465;1060,465;1060,615;860,615" \
  --zone-name "CenterBox-1920x1080" \
  --imgsz 640 --conf 0.25 --max-fps 20 \
  --save ./datas/results/sample2.mp4`  


# Challenge 1 — Database Design
    ## A. ERD
        ZONES ||--o{ DETECTIONS : has
        DETECTIONS ||--o{ ENTRY_EXIT_EVENTS : triggers

        ZONES {
        int id PK
        text name
        text coordinates   
        timestamp created_at
        }

        DETECTIONS {
        int id PK
        int zone_id FK -> ZONES.id
        timestamp timestamp
        int track_id
        real bbox_x1
        real bbox_y1
        real bbox_x2
        real bbox_y2
        real confidence
        int class_id
        }
        
        ENTRY_EXIT_EVENTS {
        int id PK
        int detection_id FK -> DETECTIONS.id
        text event_type 
        timestamp event_time
        }

    ## B. Why this design?

        zones stores editable polygons.

        detections keeps raw per-frame detections with tracker IDs.

        entry_exit_events stores only boundary crossings (compact, fast to aggregate).

# Challenge 2 — Dataset / Video Sources

    ## For dataset collection, I used a combination of:
        A. 3 CCTV live streams from Jogja:
            - Malioboro_10_Kepatihan.stream
            - Malioboro_30_Pasar_Beringharjo.stream
            - NolKm_Utara.stream (in .m3u8 format)

        B. Additional videos from YouTube, stored locally in:
            datas/samples/
            ├── sample1.mp4
            ├── sample2.mp4
            ├── sample3.mp4
            └── test_1.mp4

    This provides a combination of live video feeds (bonus) and static videos for offline testing.

# Challenge 3 — Detection & Tracking Approach

    ## Simple processing pipeline:
        1. Input video (V): can be a static MP4 or live CCTV in HLS (.m3u8).
        2. YOLOv8 Detector (D): detects “person” objects and outputs bounding boxes.
        3. ByteTrack Tracker (T): tracks objects across frames, assigning a unique track_id to each person.
        4. Polygon Check (Z): checks whether the bounding box center lies inside the polygon (zone).
        5. Database (DB):
        - If a person newly enters or exits the polygon → save an entry/exit event.
        - If the person is only detected but does not cross → save the detection only.    

# Checklist Task 
1. Database Design — Done
SQLite schema with zones, detections, and entry_exit_events. Works for storing polygons, raw detections, and entry/exit events.

2. Dataset Collection — Done
Mixed sources: 3 live CCTV Jogja streams (HLS .m3u8) + several local MP4 samples under datas/samples/.

3. Object Detection & Tracking — Done (baseline; needs tuning)
Using YOLOv8s for demo (no fine-tuning) + ByteTrack via supervision.
Note: zero training and small model → good for demo but can be improved (misses/ID switches in crowded scenes).

4. Counting & Polygon Area — Done
Center-in-polygon logic + set-difference of track IDs for entry/exit.
Accuracy is dependent on detector & tracker stability.

5. Forecasting — Not Implemented
Prioritized improving detection/tracking first; forecasting would be more useful after the base counts are stable.

6. API Integration (Backend + Frontend) — Done
FastAPI exposes historical/live stats and zone insertion; dashboard calls the API.

7. Dashboard — Partially Done
Streamlit app with Home/Stats/Update Coordinate. 
Stats tab is solid; Home and Coordinate editor "work" but still have bugs (e.g., some streams don’t snapshot in all browsers).

8. Deployment — Not Using Docker
For this submission, focus is on local run. Docker can be added later; instructions below cover local setup end-to-end.
