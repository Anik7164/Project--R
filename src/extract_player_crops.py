
import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
from pathlib import Path
from collections import defaultdict
import csv

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = ROOT / "data" / "videos" / "test1.mp4"
# Written by track_players.py; track_id is the stable player ID.
TRACKS_CSV = ROOT / "outputs" / "tracking" / "test1_tracks.csv"
OUTPUT_DIR = ROOT / "outputs" / "player_crops"
REVIEW_CSV = OUTPUT_DIR / "crop_review.csv"

# Sample frames from the tracking CSV.
FRAME_INTERVAL = 5

# Person detection filters.
MIN_WIDTH = 45
MIN_HEIGHT = 70
MIN_TRACK_CONFIDENCE = 0.35

# Select only the best N crops per track.
MAX_CROPS_PER_TRACK = 4

# Crop padding.
PAD_X_RATIO = 0.04
PAD_Y_RATIO = 0.03

# Quality thresholds.
MIN_SHARPNESS = 8.0
MIN_PERSON_FRAME_AREA_RATIO = 0.0008
MAX_ASPECT_RATIO = 1.8

CLEAN_OLD_CROPS = True


def calculate_sharpness(image):
    """Estimate focus using the Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def overlay_likelihood(frame, x1, y1, x2, y2):
    """
    Heuristic for broadcast graphics.

    Broadcast graphics often have large, high-contrast regions
    near the bottom of the frame. This is only a weak warning
    signal, not a reliable overlay detector.
    """
    frame_h, frame_w = frame.shape[:2]
    box_h = y2 - y1
    box_w = x2 - x1

    if box_h <= 0 or box_w <= 0:
        return 1.0

    # Focus on the bottom portion of the frame.
    if y2 < frame_h * 0.68:
        return 0.0

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 1.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    edge_ratio = float(np.mean(edges > 0))

    # A high edge density alone does not prove a graphic exists.
    # Return a mild penalty rather than automatically rejecting it.
    return min(1.0, max(0.0, (edge_ratio - 0.12) / 0.30))


def score_candidate(frame, row, width, height):
    """Return candidate crop and quality information, or None."""
    confidence = float(row["confidence"])

    if confidence < MIN_TRACK_CONFIDENCE:
        return None

    x1 = int(float(row["x1"]))
    y1 = int(float(row["y1"]))
    x2 = int(float(row["x2"]))
    y2 = int(float(row["y2"]))

    if x2 <= x1 or y2 <= y1:
        return None

    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return None

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w < MIN_WIDTH or box_h < MIN_HEIGHT:
        return None

    aspect_ratio = box_h / max(box_w, 1)

    # Reject extremely wide detections; retain a generous range
    # because camera angles and partially visible players vary.
    if aspect_ratio < 0.45 or aspect_ratio > MAX_ASPECT_RATIO:
        return None

    area_ratio = (box_w * box_h) / float(width * height)

    if area_ratio < MIN_PERSON_FRAME_AREA_RATIO:
        return None

    pad_x = int(box_w * PAD_X_RATIO)
    pad_y = int(box_h * PAD_Y_RATIO)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x)
    crop_y2 = min(height, y2 + pad_y)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    if crop.size == 0:
        return None

    sharpness = calculate_sharpness(crop)

    if sharpness < MIN_SHARPNESS:
        return None

    # Normalize factors to comparable ranges.
    size_score = min(1.0, area_ratio / 0.04)
    confidence_score = min(1.0, max(0.0, confidence))
    sharpness_score = min(1.0, sharpness / 150.0)

    overlay_score = overlay_likelihood(
        frame, crop_x1, crop_y1, crop_x2, crop_y2
    )

    # Prioritize large, confident, sharp detections.
    # Overlay penalty is deliberately modest because this
    # heuristic can confuse field markings with graphics.
    quality_score = (
        0.35 * size_score
        + 0.30 * confidence_score
        + 0.35 * sharpness_score
        - 0.10 * overlay_score
    )

    return {
        "crop": crop,
        "confidence": confidence,
        "width": box_w,
        "height": box_h,
        "area_ratio": area_ratio,
        "sharpness": sharpness,
        "overlay_score": overlay_score,
        "quality_score": quality_score,
    }


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not TRACKS_CSV.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {TRACKS_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAN_OLD_CROPS:
        for old_file in OUTPUT_DIR.glob("*.jpg"):
            old_file.unlink()

        if REVIEW_CSV.exists():
            REVIEW_CSV.unlink()

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates = defaultdict(list)
    skipped = 0
    current_frame = -1
    frame = None

    try:
        with TRACKS_CSV.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)

            required = {
                "frame", "track_id", "confidence",
                "x1", "y1", "x2", "y2",
            }

            if not reader.fieldnames or not required.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"CSV is missing required columns: {required}"
                )

            for row in reader:
                frame_number = int(row["frame"])

                if frame_number % FRAME_INTERVAL != 0:
                    continue

                if not 0 <= frame_number < frame_count:
                    skipped += 1
                    continue

                if frame_number != current_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ok, frame = cap.read()
                    current_frame = frame_number

                    if not ok:
                        skipped += 1
                        frame = None
                        continue

                if frame is None:
                    skipped += 1
                    continue

                height, width = frame.shape[:2]

                candidate = score_candidate(
                    frame, row, width, height
                )

                if candidate is None:
                    skipped += 1
                    continue

                track_id = int(row["track_id"])
                candidate["frame"] = frame_number
                candidate["track_id"] = track_id

                candidates[track_id].append(candidate)

                if frame_number % 30 == 0:
                    print(
                        f"Evaluated frame {frame_number}/"
                        f"{frame_count} | "
                        f"tracks collected: {len(candidates)}"
                    )

    finally:
        cap.release()

    saved = 0
    review_rows = []

    with REVIEW_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        columns = [
            "track_id", "frame", "quality_score",
            "confidence", "sharpness", "crop_width",
            "crop_height", "area_ratio", "overlay_score",
            "filename",
        ]

        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for track_id, items in sorted(candidates.items()):
            # Select the best crops for each track.
            # Enforce a minimum temporal separation so the
            # review set does not contain near-identical frames.
            items.sort(
                key=lambda item: item["quality_score"],
                reverse=True,
            )

            selected = []

            for item in items:
                if any(
                    abs(item["frame"] - chosen["frame"]) < 15
                    for chosen in selected
                ):
                    continue

                selected.append(item)

                if len(selected) >= MAX_CROPS_PER_TRACK:
                    break

            for rank, item in enumerate(selected, start=1):
                frame_number = item["frame"]

                filename = (
                    f"track_{track_id:04d}"
                    f"_rank_{rank}"
                    f"_frame_{frame_number:06d}.jpg"
                )

                output_path = OUTPUT_DIR / filename

                if not cv2.imwrite(
                    str(output_path), item["crop"]
                ):
                    print(f"Could not save: {output_path}")
                    continue

                saved += 1

                review_row = {
                    "track_id": track_id,
                    "frame": frame_number,
                    "quality_score": round(
                        item["quality_score"], 4
                    ),
                    "confidence": round(item["confidence"], 4),
                    "sharpness": round(item["sharpness"], 2),
                    "crop_width": item["width"],
                    "crop_height": item["height"],
                    "area_ratio": round(item["area_ratio"], 6),
                    "overlay_score": round(
                        item["overlay_score"], 4
                    ),
                    "filename": filename,
                }

                writer.writerow(review_row)
                review_rows.append(review_row)

    print("\nBest-crop selection finished.")
    print("Tracks with candidates:", len(candidates))
    print("Crops saved:", saved)
    print("Candidates rejected:", skipped)
    print("Crop directory:", OUTPUT_DIR)
    print("Review CSV:", REVIEW_CSV)


if __name__ == "__main__":
    main()
