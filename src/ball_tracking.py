"""
Find the ball in every frame.

YOLO's COCO "sports ball" class finds the ball in many frames but also
fires on heads, socks and pitch markings, and misses it when it is small
or blurred. This script keeps the candidate that continues the ball's
path (corrected for camera motion), and fills short gaps by linear
interpolation.

Output: outputs/tracking/<name>_ball.csv
    frame, timestamp_seconds, x, y, confidence, source, scene, sx, sy
    source is "detected" or "interpolated"; frames without a ball are
    left out. (sx, sy) is the position in the pixels of the first frame
    of the shot, i.e. with camera motion removed, so a ball waiting for a
    free kick stays still in (sx, sy) while the camera pans.

Usage:
    python src/ball_tracking.py --video data/videos/test1.mp4
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from camera_motion import CameraMotionEstimator, warp_points

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = ROOT / "data" / "videos" / "test1.mp4"
DEFAULT_MODEL = ROOT / "yolo26n.pt"
OUTPUT_DIR = ROOT / "outputs" / "tracking"

BALL_CLASS = 32
CONFIDENCE = 0.15
# The ball is tiny; a larger input size finds it far more often.
IMAGE_SIZE = 1280

# Largest believable ball movement per frame, as a fraction of the frame
# width (a hard shot crosses about a third of a wide shot per second).
MAX_STEP_RATIO = 0.06
# Without a recent position, accept only fairly confident detections.
REACQUIRE_CONFIDENCE = 0.35
# Gaps up to this many frames are interpolated.
MAX_GAP = 12
# Implausibly large "balls" are heads or other objects.
MAX_BALL_SIZE_RATIO = 0.04


def compose(a, b):
    """2x3 affine for applying a, then b."""
    a3 = np.vstack([a, [0, 0, 1]])
    b3 = np.vstack([b, [0, 0, 1]])
    return (b3 @ a3)[:2]


def invert(warp):
    return cv2.invertAffineTransform(warp)


class BallSelector:
    """Picks one ball candidate per frame by continuity with the last one."""

    def __init__(self, frame_width):
        self.max_step = MAX_STEP_RATIO * frame_width
        self.max_size = MAX_BALL_SIZE_RATIO * frame_width
        self.last = None        # (x, y) in current-frame pixels
        self.velocity = np.zeros(2)
        self.missing = 0

    def update(self, candidates, warp=None, scene_cut=False):
        """candidates: [(x, y, w, h, conf)]. Returns (x, y, conf) or None."""
        if scene_cut:
            self.last, self.velocity, self.missing = None, np.zeros(2), 0
        elif self.last is not None and warp is not None:
            self.last = warp_points([self.last], warp)[0]

        candidates = [c for c in candidates if max(c[2], c[3]) <= self.max_size]
        best, best_score = None, None

        for x, y, _, _, conf in candidates:
            if self.last is None:
                if conf < REACQUIRE_CONFIDENCE:
                    continue
                score = -conf
            else:
                expected = self.last + self.velocity * (self.missing + 1)
                distance = np.hypot(x - expected[0], y - expected[1])
                if distance > self.max_step * (self.missing + 1):
                    continue
                score = distance / self.max_step - conf
            if best_score is None or score < best_score:
                best, best_score = (x, y, conf), score

        if best is None:
            self.missing += 1
            if self.missing > MAX_GAP:
                self.last, self.velocity = None, np.zeros(2)
            return None

        position = np.array(best[:2])
        if self.last is not None:
            step = (position - self.last) / (self.missing + 1)
            self.velocity = 0.5 * self.velocity + 0.5 * step
        self.last = position
        self.missing = 0
        return best


def interpolate_gaps(points, max_gap=MAX_GAP, scenes=None):
    """points: {frame: (x, y, conf)} -> {frame: (x, y, conf, source)}.

    Gaps of at most max_gap frames between two detections in the same
    shot are filled linearly (confidence 0, source "interpolated").
    """
    frames = sorted(points)
    filled = {f: (*points[f], "detected") for f in frames}
    for start, end in zip(frames, frames[1:]):
        gap = end - start
        if scenes is not None and scenes[start] != scenes[end]:
            continue
        if 1 < gap <= max_gap + 1:
            a, b = np.array(points[start][:2]), np.array(points[end][:2])
            for f in range(start + 1, end):
                t = (f - start) / gap
                x, y = a + t * (b - a)
                filled[f] = (float(x), float(y), 0.0, "interpolated")
    return dict(sorted(filled.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    for path in (args.video, args.model):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / f"{args.video.stem}_ball.csv"

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    selector = BallSelector(width)
    camera = CameraMotionEstimator()
    points = {}
    # Shot number and shot-start -> current frame warp, per frame.
    scene_of, to_frame = {}, {}
    scene, cumulative = 0, np.eye(3)[:2]
    frame_number = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            result = model.predict(
                frame, classes=[BALL_CLASS], conf=CONFIDENCE,
                imgsz=IMAGE_SIZE, device=device, verbose=False,
            )[0]
            candidates = [
                (*box, conf)
                for box, conf in zip(
                    result.boxes.xywh.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                )
            ]
            warp, cut = camera.update(frame)
            if cut:
                scene, cumulative = scene + 1, np.eye(3)[:2]
            else:
                cumulative = compose(cumulative, warp)
            scene_of[frame_number], to_frame[frame_number] = scene, cumulative
            ball = selector.update(candidates, warp, cut)
            if ball is not None:
                points[frame_number] = ball
            frame_number += 1
            if frame_number % 100 == 0:
                print(f"Processed {frame_number} frames | ball found in {len(points)}")
    finally:
        cap.release()

    filled = interpolate_gaps(points, scenes=scene_of)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", "timestamp_seconds", "x", "y", "confidence", "source",
            "scene", "sx", "sy",
        ])
        for frame, (x, y, conf, source) in filled.items():
            sx, sy = warp_points([[x, y]], invert(to_frame[frame]))[0]
            writer.writerow([
                frame, round(frame / fps, 3), round(x, 1), round(y, 1),
                round(conf, 4), source, scene_of[frame],
                round(float(sx), 1), round(float(sy), 1),
            ])

    print(f"Ball in {len(filled)}/{frame_number} frames "
          f"({len(points)} detected). CSV: {output_csv}")


if __name__ == "__main__":
    main()
