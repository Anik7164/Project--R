
from pathlib import Path
import csv

import cv2

ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / "data" / "videos" / "test1.mp4"
TRACKS_CSV = ROOT / "outputs" / "tracking" / "test1_tracks.csv"
OUTPUT_DIR = ROOT / "outputs" / "player_crops"

# Save one crop every 5 frames to avoid excessive duplicates.
FRAME_INTERVAL = 5

# Skip very small detections, which are unlikely to contain
# readable jersey numbers.
MIN_WIDTH = 70
MIN_HEIGHT = 100


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not TRACKS_CSV.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {TRACKS_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    saved = 0
    skipped = 0
    current_frame = -1
    frame = None

    try:
        with TRACKS_CSV.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                frame_number = int(row["frame"])

                if frame_number % FRAME_INTERVAL != 0:
                    continue

                # Read each video frame only once, even when
                # multiple players appear in the same frame.
                if frame_number != current_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ok, frame = cap.read()
                    current_frame = frame_number

                    if not ok:
                        skipped += 1
                        continue

                height, width = frame.shape[:2]

                x1 = int(float(row["x1"]))
                y1 = int(float(row["y1"]))
                x2 = int(float(row["x2"]))
                y2 = int(float(row["y2"]))

                box_w = x2 - x1
                box_h = y2 - y1

                if box_w < MIN_WIDTH or box_h < MIN_HEIGHT:
                    skipped += 1
                    continue

                # Add a small margin around the player.
                pad_x = int(box_w * 0.08)
                pad_y = int(box_h * 0.05)

                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(width, x2 + pad_x)
                y2 = min(height, y2 + pad_y)

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    skipped += 1
                    continue

                track_id = int(row["track_id"])
                filename = (
                    f"frame_{frame_number:06d}"
                    f"_track_{track_id:04d}.jpg"
                )

                if cv2.imwrite(str(OUTPUT_DIR / filename), crop):
                    saved += 1

    finally:
        cap.release()

    print("\nCrop extraction finished.")
    print("Crops saved:", saved)
    print("Detections skipped:", skipped)
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
