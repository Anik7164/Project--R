
import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import csv
import math
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = ROOT / "data" / "videos" / "test1.mp4"

# Written by track_players.py; track_id is the stable player ID.
TRACKS_CSV = ROOT / "outputs" / "tracking" / "test1_tracks.csv"

OUTPUT_DIR = ROOT / "outputs" / "jersey_crops"
REVIEW_CSV = OUTPUT_DIR / "jersey_crop_review.csv"

# Keep several candidates per track; jersey_reading.py needs at least
# two crops to agree before it trusts a name or number.
MAX_CROPS_PER_TRACK = 6

# Avoid selecting nearly identical neighboring frames.
MIN_FRAME_GAP = 15

# Basic full-person detection filters.
MIN_PERSON_WIDTH = 25
MIN_PERSON_HEIGHT = 55
MAX_FRAME_AREA_RATIO = 0.25

# Upper-body region within the person bounding box.
TORSO_TOP = 0.10
TORSO_BOTTOM = 0.63
TORSO_LEFT = 0.05
TORSO_RIGHT = 0.95


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def calculate_sharpness(image):
    """Higher Laplacian variance often indicates a sharper image."""
    if image.size == 0:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_tracks():
    if not TRACKS_CSV.exists():
        raise FileNotFoundError(
            f"Tracking CSV not found:\n{TRACKS_CSV}\n"
            "Check the filename and location."
        )

    with TRACKS_CSV.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    rows = read_tracks()

    if not rows:
        raise RuntimeError("The tracking CSV contains no rows.")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_area = frame_width * frame_height

    print(f"Video: {frame_width}x{frame_height}")
    print(f"Frames: {total_frames}")
    print(f"Tracking rows: {len(rows)}")
    print("Preparing torso crops...")

    # Sort by frame so nearby video frames are read in order.
    rows.sort(key=lambda r: int(float(r["frame"])))

    # Store candidate crops before selecting the best frames.
    candidates_by_track = {}
    cached_frame_number = -1
    cached_frame = None

    for index, row in enumerate(rows, start=1):
        try:
            frame_number = int(float(row["frame"]))
            track_id = int(float(row["track_id"]))
            confidence = float(row["confidence"])

            x1 = int(float(row["x1"]))
            y1 = int(float(row["y1"]))
            x2 = int(float(row["x2"]))
            y2 = int(float(row["y2"]))

        except (KeyError, TypeError, ValueError):
            continue

        if not (0 <= frame_number < total_frames):
            continue

        # Clip person box to the frame.
        x1 = clamp(x1, 0, frame_width - 1)
        x2 = clamp(x2, 0, frame_width)
        y1 = clamp(y1, 0, frame_height - 1)
        y2 = clamp(y2, 0, frame_height)

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width < MIN_PERSON_WIDTH:
            continue

        if box_height < MIN_PERSON_HEIGHT:
            continue

        area_ratio = (
            box_width * box_height / frame_area
        )

        if area_ratio > MAX_FRAME_AREA_RATIO:
            continue

        # Read each frame only once where possible.
        if frame_number != cached_frame_number:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = cap.read()

            if not success:
                continue

            cached_frame = frame
            cached_frame_number = frame_number

        frame = cached_frame

        # Extract the upper body rather than the entire player.
        torso_x1 = int(x1 + box_width * TORSO_LEFT)
        torso_x2 = int(x1 + box_width * TORSO_RIGHT)
        torso_y1 = int(y1 + box_height * TORSO_TOP)
        torso_y2 = int(y1 + box_height * TORSO_BOTTOM)

        torso_x1 = clamp(torso_x1, 0, frame_width - 1)
        torso_x2 = clamp(torso_x2, 0, frame_width)
        torso_y1 = clamp(torso_y1, 0, frame_height - 1)
        torso_y2 = clamp(torso_y2, 0, frame_height)

        crop = frame[
            torso_y1:torso_y2,
            torso_x1:torso_x2
        ]

        if crop.size == 0:
            continue

        crop_height, crop_width = crop.shape[:2]

        if crop_width < 15 or crop_height < 15:
            continue

        sharpness = calculate_sharpness(crop)

        # Normalize sharpness so it cannot dominate the score.
        sharpness_score = (
            math.log1p(max(0.0, sharpness))
            / math.log1p(1000.0)
        )
        sharpness_score = min(1.0, sharpness_score)

        # Favor larger crops, but cap the benefit of large boxes.
        size_score = min(
            1.0,
            (crop_width * crop_height) / (100.0 * 100.0)
        )

        confidence_score = clamp(confidence, 0.0, 1.0)

        # This ranks general crop usefulness, not actual OCR accuracy.
        quality_score = (
            0.50 * sharpness_score
            + 0.30 * confidence_score
            + 0.20 * size_score
        )

        candidates_by_track.setdefault(
            track_id, []
        ).append({
            "frame": frame_number,
            "confidence": confidence,
            "sharpness": sharpness,
            "area_ratio": area_ratio,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "quality_score": quality_score,
            "crop": crop.copy(),
        })

        if index % 5000 == 0:
            print(f"Processed {index}/{len(rows)} rows")

    cap.release()

    review_rows = []
    saved_count = 0

    for track_id, candidates in candidates_by_track.items():
        # Best score first.
        candidates.sort(
            key=lambda item: item["quality_score"],
            reverse=True
        )

        selected = []

        for candidate in candidates:
            # Avoid nearly identical frames for the same track.
            if any(
                abs(candidate["frame"] - chosen["frame"])
                < MIN_FRAME_GAP
                for chosen in selected
            ):
                continue

            selected.append(candidate)

            if len(selected) >= MAX_CROPS_PER_TRACK:
                break

        # Save a separate folder for each track ID.
        track_dir = OUTPUT_DIR / f"track_{track_id:04d}"
        track_dir.mkdir(parents=True, exist_ok=True)

        for rank, item in enumerate(selected, start=1):
            filename = (
                f"rank_{rank}_frame_{item['frame']:06d}.jpg"
            )
            output_path = track_dir / filename

            cv2.imwrite(
                str(output_path),
                item["crop"],
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            review_rows.append({
                "track_id": track_id,
                "rank": rank,
                "frame": item["frame"],
                "confidence": round(item["confidence"], 4),
                "sharpness": round(item["sharpness"], 2),
                "quality_score": round(
                    item["quality_score"], 4
                ),
                "crop_width": item["crop_width"],
                "crop_height": item["crop_height"],
                "filename": str(output_path.relative_to(ROOT)),
            })

            saved_count += 1

    with REVIEW_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = [
            "track_id",
            "rank",
            "frame",
            "confidence",
            "sharpness",
            "quality_score",
            "crop_width",
            "crop_height",
            "filename",
        ]

        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(review_rows)

    print("\nFinished.")
    print(f"Tracks with candidate crops: {len(candidates_by_track)}")
    print(f"Crops saved: {saved_count}")
    print(f"Review CSV: {REVIEW_CSV}")
    print(f"Crop folders: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
