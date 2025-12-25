import cv2
from ultralytics import YOLO
import os

def analyze_video(video_path):
    os.makedirs("outputs", exist_ok=True)

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        "outputs/annotated_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = 0
    total_weight_proxy = 0
    bird_counts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # YOLO tracking
        results = model.track(frame, persist=True, conf=0.3, imgsz=640)
        boxes = results[0].boxes

        count = 0

        if boxes.id is not None:
            count = len(boxes.id)

            for box, track_id in zip(boxes.xyxy, boxes.id):
                x1, y1, x2, y2 = map(int, box)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw ID
                cv2.putText(
                    frame,
                    f"ID {int(track_id)}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Weight proxy (area)
                area = (x2 - x1) * (y2 - y1)
                total_weight_proxy += area

        bird_counts.append(count)
        out.write(frame)

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()

    avg_weight_proxy = total_weight_proxy / frame_count if frame_count else 0

    return {
        "frames_processed": frame_count,
        "max_bird_count": max(bird_counts),
        "avg_bird_count": sum(bird_counts) / len(bird_counts),
        "avg_weight_proxy": avg_weight_proxy
    }
