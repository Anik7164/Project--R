
from pathlib import Path
import csv

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / "data" / "videos" / "test1.mp4"
MODEL_PATH = Path(r"F:\AI_Models\Ultralytics\yolo26n.pt")

OUTPUT_DIR = ROOT / "outputs" / "tracking"
OUTPUT_VIDEO = OUTPUT_DIR / "test1_tracked.mp4"
OUTPUT_CSV = OUTPUT_DIR / "test1_tracks.csv"


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output: {OUTPUT_VIDEO}")

    frame_number = 0
    rows_written = 0
    seen_ids = set()

    try:
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([
                "frame", "timestamp_seconds", "track_id",
                "class_name", "confidence",
                "x1", "y1", "x2", "y2"
            ])

            cap = cv2.VideoCapture(str(VIDEO_PATH))
            try:
                while True:
                    success, frame = cap.read()
                    if not success:
                        break

                    results = model.track(
                        source=frame,
                        tracker="bytetrack.yaml",
                        persist=True,
                        classes=[0],  # COCO person class
                        conf=0.35,
                        imgsz=640,
                        device=device,
                        verbose=False,
                    )

                    result = results[0]
                    writer.write(result.plot())

                    boxes = result.boxes
                    if boxes is not None and boxes.id is not None:
                        coordinates = boxes.xyxy.cpu().tolist()
                        ids = boxes.id.int().cpu().tolist()
                        confidences = boxes.conf.cpu().tolist()
                        class_ids = boxes.cls.int().cpu().tolist()

                        timestamp = frame_number / fps

                        for bbox, track_id, confidence, class_id in zip(
                            coordinates, ids, confidences, class_ids
                        ):
                            seen_ids.add(track_id)
                            csv_writer.writerow([
                                frame_number,
                                round(timestamp, 3),
                                track_id,
                                model.names[class_id],
                                round(confidence, 4),
                                *[round(value, 2) for value in bbox],
                            ])
                            rows_written += 1

                    frame_number += 1

                    if frame_number % 30 == 0:
                        print(
                            f"Processed {frame_number}/{total_frames} "
                            f"frames | unique track IDs: {len(seen_ids)}"
                        )
            finally:
                cap.release()
    finally:
        writer.release()

    print("\nTracking finished!")
    print("Frames processed:", frame_number)
    print("Unique track IDs observed:", len(seen_ids))
    print("Detection records:", rows_written)
    print("Tracked video:", OUTPUT_VIDEO)
    print("Tracking CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
