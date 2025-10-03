import numpy as np
import supervision as sv
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# load model
model = YOLO("yolov8n.pt")

# polygon zone (adjust coordinates to your frame size)
# sample3 coordinates
polygon = np.array([
    [400, 250],   # top-left
    [600, 250],   # top-right
    [600, 400],   # bottom-right
    [400, 400]    # bottom-left
])

# video path
video_path = "./data/sample3.mp4"
video_info = sv.VideoInfo.from_video_path(video_path)
zone = sv.PolygonZone(polygon=polygon)

# annotators
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.WHITE,
    thickness=6,
    text_thickness=2,
    text_scale=1.2
)

# take one frame from video
generator = sv.get_video_frames_generator(video_path)
frame = next(generator)  # just 1st frame

# run detection
results = model(frame, imgsz=640)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[detections.class_id == 0]  # only persons
zone.trigger(detections=detections)

# annotate
labels = [f"{model.names[class_id]} {conf:0.2f}" for _, _, conf, class_id, _, _ in detections]
frame = box_annotator.annotate(scene=frame, detections=detections)
frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
frame = zone_annotator.annotate(scene=frame)

# display using matplotlib
plt.imshow(frame[..., ::-1])  # BGR→RGB
plt.axis("off")
cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
