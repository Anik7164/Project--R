
"""
Track players with BoT-SORT and give them stable player IDs.

BoT-SORT track IDs break whenever a player is lost for a while: occlusion,
leaving the view during a camera pan, or a broadcast cut. The identity
layer in identity.py re-links those broken tracks so a player keeps the
same ID while the camera moves.

Outputs (in outputs/tracking/):
    <name>_tracks.csv  one row per detection, in frame order.
        track_id    stable player ID (use this one)
        tracker_id  raw BoT-SORT ID, kept for comparison
        scene       shot number; it increases at every detected cut
    <name>_tracks_mot.txt  the same tracks in MOTChallenge format, for
        evaluate_tracking.py and SoccerNet-style evaluation.
    <name>_tracked.mp4  annotated video.

Usage:
    python src/track_players.py --video data/videos/test1.mp4
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from detector_classes import default_model, people_classes
from identity import StablePlayerTracker
from reid import load_default as load_reid

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VIDEO = ROOT / "data" / "videos" / "test1.mp4"
# Override with --model or the YOLO_MODEL environment variable; a trained
# football model in models/ (train_detector.py) is used when there is one.
DEFAULT_MODEL = default_model()
DEFAULT_TRACKER = ROOT / "botsort_custom.yaml"
OUTPUT_DIR = ROOT / "outputs" / "tracking"

# Detection settings
CONFIDENCE = 0.35
IMAGE_SIZE = 960
# NMS IoU. Lower than the Ultralytics default (0.7) so two boxes on one
# player are merged more often. Ignored by NMS-free models like YOLO26.
NMS_IOU = 0.5

# Conservative box validation: avoid rejecting distant players.
MIN_BOX_WIDTH = 8
MIN_BOX_HEIGHT = 18
MAX_BOX_WIDTH_RATIO = 0.45
MAX_BOX_HEIGHT_RATIO = 0.98
MAX_BOX_AREA_RATIO = 0.18

# Optional diagnostic: rejected boxes are orange.
SHOW_REJECTED = False

CSV_HEADER = [
    "frame",
    "timestamp_seconds",
    "track_id",
    "tracker_id",
    "class_name",
    "confidence",
    "x1", "y1", "x2", "y2",
    "scene",
]


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


def colour_for(player_id):
    """A distinct, stable BGR colour per player ID."""
    hue = (player_id * 47) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


class OrderedTrackWriter:
    """Writes rows in frame order while pending IDs wait to be resolved.

    A new BoT-SORT ID is pending for a few frames before it gets a player
    ID. Rows are held back until every ID in their frame is resolved.
    """

    def __init__(self, csv_writer, mot_file, delay):
        self.csv_writer = csv_writer
        self.mot_file = mot_file
        self.delay = delay
        self.frames = deque()
        self.pending_rows = defaultdict(list)
        self.rows_written = 0
        self.player_ids = set()

    def add_frame(self, frame_number, rows):
        """rows: dicts with a 'track_id' that may be None (pending)."""
        for row in rows:
            if row["track_id"] is None:
                self.pending_rows[row["tracker_id"]].append(row)
        self.frames.append((frame_number, rows))
        while self.frames and self.frames[0][0] < frame_number - self.delay:
            self._write(*self.frames.popleft())

    def resolve(self, resolved):
        for tracker_id, player_id in resolved.items():
            for row in self.pending_rows.pop(tracker_id, []):
                # None means the ID was dropped as a false detection.
                row["track_id"] = player_id if player_id is not None else -1

    def close(self):
        while self.frames:
            self._write(*self.frames.popleft())

    def _write(self, frame_number, rows):
        for row in rows:
            if row["track_id"] is None or row["track_id"] < 0:
                continue
            self.csv_writer.writerow([row[key] for key in CSV_HEADER])
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            self.mot_file.write(
                f"{frame_number + 1},{row['track_id']},{x1},{y1},"
                f"{x2 - x1},{y2 - y1},{row['confidence']},-1,-1,-1\n"
            )
            self.rows_written += 1
            self.player_ids.add(row["track_id"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--no-video", action="store_true",
        help="skip writing the annotated video (faster)",
    )
    parser.add_argument(
        "--no-reid", action="store_true",
        help="match players on colour only (no deep appearance model)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for path in (args.video, args.model, args.tracker):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.video.stem
    output_video = args.output_dir / f"{name}_tracked.mp4"
    output_csv = args.output_dir / f"{name}_tracks.csv"
    output_mot = args.output_dir / f"{name}_tracks_mot.txt"

    device = 0 if torch.cuda.is_available() else "cpu"

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Tracker:", args.tracker)

    model = YOLO(str(args.model))
    person_classes = people_classes(model)
    print("Detector:", args.model, "| people classes:",
          [model.names[c] for c in person_classes])

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {output_video}")

    reid = None if args.no_reid else load_reid("cuda:0" if device == 0 else "cpu")
    print("ReID:", "on" if reid else "off")
    stable = StablePlayerTracker(reid=reid)
    delay = stable.identity.config.pending_timeout + 2

    frame_number = 0
    rejected = 0
    tracker_ids_seen = set()
    rejection_reasons = {}

    try:
        with output_csv.open("w", newline="", encoding="utf-8") as file, \
                output_mot.open("w", encoding="utf-8") as mot_file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(CSV_HEADER)
            track_writer = OrderedTrackWriter(csv_writer, mot_file, delay)

            while True:
                success, frame = cap.read()
                if not success:
                    break

                # IMPORTANT: use the custom BoT-SORT config.
                # persist=True preserves tracker state between frames.
                results = model.track(
                    source=frame,
                    tracker=str(args.tracker),
                    persist=True,
                    classes=person_classes,
                    conf=CONFIDENCE,
                    iou=NMS_IOU,
                    imgsz=IMAGE_SIZE,
                    device=device,
                    verbose=False,
                )

                result = results[0]
                annotated = frame.copy() if writer else None
                boxes = result.boxes
                timestamp = frame_number / fps

                detections = []
                if boxes is not None and boxes.id is not None:
                    for bbox, tracker_id, confidence, class_id in zip(
                        boxes.xyxy.cpu().tolist(),
                        boxes.id.int().cpu().tolist(),
                        boxes.conf.cpu().tolist(),
                        boxes.cls.int().cpu().tolist(),
                    ):
                        accepted, reason = validate_box(
                            bbox, frame_width, frame_height
                        )
                        if not accepted:
                            rejected += 1
                            rejection_reasons[reason] = (
                                rejection_reasons.get(reason, 0) + 1
                            )
                            if SHOW_REJECTED and annotated is not None:
                                x1, y1, x2, y2 = map(int, bbox)
                                cv2.rectangle(
                                    annotated, (x1, y1), (x2, y2),
                                    (0, 165, 255), 2,
                                )
                            continue

                        # Clamp accepted coordinates to the image.
                        x1, y1, x2, y2 = map(int, bbox)
                        x1 = max(0, min(x1, frame_width - 1))
                        y1 = max(0, min(y1, frame_height - 1))
                        x2 = max(0, min(x2, frame_width - 1))
                        y2 = max(0, min(y2, frame_height - 1))
                        if x2 <= x1 or y2 <= y1:
                            rejected += 1
                            continue

                        detections.append((
                            int(tracker_id), (x1, y1, x2, y2),
                            float(confidence), int(class_id),
                        ))

                keep, assignments, resolved = stable.process(
                    frame_number,
                    frame,
                    [d[0] for d in detections],
                    [d[1] for d in detections],
                    [d[2] for d in detections],
                )

                rows = []
                for index, player_id in zip(keep, assignments):
                    tracker_id, (x1, y1, x2, y2), confidence, class_id = (
                        detections[index]
                    )
                    tracker_ids_seen.add(tracker_id)
                    rows.append({
                        "frame": frame_number,
                        "timestamp_seconds": round(timestamp, 3),
                        "track_id": player_id,
                        "tracker_id": tracker_id,
                        "class_name": model.names[class_id],
                        "confidence": round(confidence, 4),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "scene": stable.identity.scene,
                    })

                    if annotated is not None:
                        if player_id is None:
                            colour, label = (160, 160, 160), "?"
                        else:
                            colour, label = colour_for(player_id), f"P{player_id}"
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                        cv2.putText(
                            annotated, label, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2,
                        )

                track_writer.resolve(resolved)
                track_writer.add_frame(frame_number, rows)

                if annotated is not None:
                    if stable.last_cut:
                        cv2.putText(
                            annotated, "SCENE CUT", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                        )
                    writer.write(annotated)
                frame_number += 1

                if frame_number % 30 == 0:
                    print(
                        f"Processed {frame_number}/{total_frames} "
                        f"| BoT-SORT IDs: {len(tracker_ids_seen)} "
                        f"| player IDs: {stable.identity.next_player_id - 1} "
                        f"| scenes: {stable.identity.scene} "
                        f"| rejected: {rejected}"
                    )

            track_writer.resolve(stable.flush())
            track_writer.close()

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print("\nTracking finished.")
    print("Frames processed:", frame_number)
    print("BoT-SORT IDs observed:", len(tracker_ids_seen))
    print("Stable player IDs written:", len(track_writer.player_ids))
    print("Scenes (shots):", stable.identity.scene)
    print("Detection records:", track_writer.rows_written)
    print("Rejected boxes:", rejected)
    print("Rejection reasons:", rejection_reasons)
    if writer is not None:
        print("Video:", output_video)
    print("CSV:", output_csv)
    print("MOT:", output_mot)


if __name__ == "__main__":
    main()
