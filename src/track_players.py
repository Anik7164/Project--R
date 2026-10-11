
from pathlib import Path
import csv

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = ROOT / "data" / "videos" / "test1.mp4"
MODEL_PATH = Path(r"F:\AI_Models\Ultralytics\yolo26n.pt")
TRACKER_CONFIG = ROOT / "botsort_custom.yaml"

OUTPUT_DIR = ROOT / "outputs" / "tracking"
OUTPUT_VIDEO = OUTPUT_DIR / "test1_botsort_buffer90.mp4"
OUTPUT_CSV = OUTPUT_DIR / "test1_botsort_buffer90_tracks.csv"

# Detection settings
CONFIDENCE = 0.35
IMAGE_SIZE = 960

# Conservative box validation: avoid rejecting distant players.
MIN_BOX_WIDTH = 8
MIN_BOX_HEIGHT = 18
MAX_BOX_WIDTH_RATIO = 0.45
MAX_BOX_HEIGHT_RATIO = 0.98
MAX_BOX_AREA_RATIO = 0.18

# Optional diagnostic: rejected boxes are orange.
SHOW_REJECTED = False


def validate_box(box, frame_width, frame_height):
    """Validate box geometry without changing tracker identities."""
    x1, y1, x2, y2 = map(float, box)

    if x2 <= x1 or y2 <= y1:
        return False, "invalid coordinates"

    if (
        x2 <= 0 or y2 <= 0
        or x1 >= frame_width or y1 >= frame_height
    ):
        return False, "outside frame"

    # Clip to the frame before checking dimensions.
    x1 = max(0.0, min(x1, float(frame_width)))
    y1 = max(0.0, min(y1, float(frame_height)))
    x2 = max(0.0, min(x2, float(frame_width)))
    y2 = max(0.0, min(y2, float(frame_height)))

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width < MIN_BOX_WIDTH or box_height < MIN_BOX_HEIGHT:
        return False, "too small"

    if box_width > frame_width * MAX_BOX_WIDTH_RATIO:
        return False, "too wide"

    if box_height > frame_height * MAX_BOX_HEIGHT_RATIO:
        return False, "too tall"

    area_ratio = (
        box_width * box_height
        / (frame_width * frame_height)
    )

    if area_ratio > MAX_BOX_AREA_RATIO:
        return False, "too large"

    aspect_ratio = box_height / max(box_width, 1.0)

    if not 0.45 <= aspect_ratio <= 7.0:
        return False, "implausible aspect ratio"

    return True, "accepted"


def main():
    for path in (VIDEO_PATH, MODEL_PATH, TRACKER_CONFIG):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Tracker:", TRACKER_CONFIG)

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {OUTPUT_VIDEO}")

    frame_number = 0
    rows_written = 0
    rejected = 0
    seen_ids = set()
    rejection_reasons = {}

    try:
        with OUTPUT_CSV.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([
                "frame",
                "timestamp_seconds",
                "track_id",
                "class_name",
                "confidence",
                "x1", "y1", "x2", "y2",
            ])

            while True:
                success, frame = cap.read()
                if not success:
                    break

                # IMPORTANT: use the custom BoT-SORT config.
                # persist=True preserves tracker state between frames.
                results = model.track(
                    source=frame,
                    tracker=str(TRACKER_CONFIG),
                    persist=True,
                    classes=[0],
                    conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    device=device,
                    verbose=False,
                )

                result = results[0]
                annotated = frame.copy()
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
                        accepted, reason = validate_box(
                            bbox, frame_width, frame_height
                        )

                        x1, y1, x2, y2 = map(int, bbox)

                        if not accepted:
                            rejected += 1
                            rejection_reasons[reason] = (
                                rejection_reasons.get(reason, 0) + 1
                            )

                            if SHOW_REJECTED:
                                x1 = max(0, min(x1, frame_width - 1))
                                y1 = max(0, min(y1, frame_height - 1))
                                x2 = max(0, min(x2, frame_width - 1))
                                y2 = max(0, min(y2, frame_height - 1))

                                cv2.rectangle(
                                    annotated, (x1, y1), (x2, y2),
                                    (0, 165, 255), 2,
                                )
                            continue

                        # Clamp accepted coordinates to the image.
                        x1 = max(0, min(x1, frame_width - 1))
                        y1 = max(0, min(y1, frame_height - 1))
                        x2 = max(0, min(x2, frame_width - 1))
                        y2 = max(0, min(y2, frame_height - 1))

                        if x2 <= x1 or y2 <= y1:
                            rejected += 1
                            continue

                        track_id = int(track_id)
                        confidence = float(confidence)
                        class_id = int(class_id)
                        seen_ids.add(track_id)

                        csv_writer.writerow([
                            frame_number,
                            round(timestamp, 3),
                            track_id,
                            model.names[class_id],
                            round(confidence, 4),
                            x1, y1, x2, y2,
                        ])
                        rows_written += 1

                        cv2.rectangle(
                            annotated, (x1, y1), (x2, y2),
                            (0, 255, 0), 2,
                        )
                        cv2.putText(
                            annotated,
                            f"ID {track_id} {confidence:.2f}",
                            (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2,
                        )

                writer.write(annotated)
                frame_number += 1

                if frame_number % 30 == 0:
                    print(
                        f"Processed {frame_number}/{total_frames} "
                        f"| unique IDs: {len(seen_ids)} "
                        f"| records: {rows_written} "
                        f"| rejected: {rejected}"
                    )

    finally:
        cap.release()
        writer.release()

    print("\nBoT-SORT tracking finished.")
    print("Frames processed:", frame_number)
    print("Unique IDs observed:", len(seen_ids))
    print("Detection records:", rows_written)
    print("Rejected boxes:", rejected)
    print("Rejection reasons:", rejection_reasons)
    print("Video:", OUTPUT_VIDEO)
    print("CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
