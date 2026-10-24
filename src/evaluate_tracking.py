"""
Measure how fragmented player IDs are, with or without ground truth.

Without ground truth it reports ID statistics for the stable player IDs
(track_id) and the raw BoT-SORT IDs (tracker_id) from the same run, so the
effect of the identity layer is visible on any clip:

    python src/evaluate_tracking.py outputs/tracking/test1_tracks.csv

With MOTChallenge-format ground truth (e.g. a SoccerNet tracking
sequence's gt/gt.txt) it also reports IDF1, MOTA and ID switches. This
needs `pip install motmetrics`:

    python src/evaluate_tracking.py outputs/tracking/seq_tracks.csv --gt gt.txt
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_tracks(csv_path, id_column):
    """Return {frame: [(id, x1, y1, x2, y2), ...]}."""
    frames = defaultdict(list)
    with Path(csv_path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            frames[int(row["frame"])].append((
                int(row[id_column]),
                float(row["x1"]), float(row["y1"]),
                float(row["x2"]), float(row["y2"]),
            ))
    return frames


def id_statistics(frames, fps):
    lengths = defaultdict(int)
    concurrent = []
    for rows in frames.values():
        concurrent.append(len(rows))
        for row in rows:
            lengths[row[0]] += 1

    values = np.array(list(lengths.values())) if lengths else np.zeros(1)
    return {
        "unique_ids": len(lengths),
        "max_people_in_one_frame": max(concurrent, default=0),
        "mean_frames_per_id": round(float(values.mean()), 1),
        "median_frames_per_id": float(np.median(values)),
        "ids_shorter_than_1s": int((values < fps).sum()),
    }


def load_mot(path):
    """Read a MOTChallenge file into {frame (0-based): [(id, x, y, w, h)]}."""
    frames = defaultdict(list)
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame, track_id = int(float(parts[0])) - 1, int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            frames[frame].append((track_id, x, y, w, h))
    return frames


def iou_distance(gt_boxes, hyp_boxes, max_distance=0.5):
    """1 - IoU between (x, y, w, h) boxes; NaN where too far to match.

    Written out here because motmetrics' own helper fails on NumPy 2.
    """
    distances = np.full((len(gt_boxes), len(hyp_boxes)), np.nan)
    for i, (gx, gy, gw, gh) in enumerate(gt_boxes):
        for j, (hx, hy, hw, hh) in enumerate(hyp_boxes):
            iw = min(gx + gw, hx + hw) - max(gx, hx)
            ih = min(gy + gh, hy + hh) - max(gy, hy)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            distance = 1 - inter / (gw * gh + hw * hh - inter)
            if distance <= max_distance:
                distances[i, j] = distance
    return distances


def clear_mot_metrics(tracks, gt_frames):
    try:
        import motmetrics as mm
    except ImportError as error:
        raise SystemExit("Install motmetrics for --gt: pip install motmetrics") from error

    accumulator = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(set(gt_frames) | set(tracks)):
        gt = gt_frames.get(frame, [])
        hyp = [
            (tid, x1, y1, x2 - x1, y2 - y1)
            for tid, x1, y1, x2, y2 in tracks.get(frame, [])
        ]
        distances = iou_distance([g[1:] for g in gt], [h[1:] for h in hyp])
        accumulator.update([g[0] for g in gt], [h[0] for h in hyp], distances)

    host = mm.metrics.create()
    summary = host.compute(
        accumulator,
        metrics=["idf1", "mota", "num_switches", "num_fragmentations",
                 "precision", "recall"],
    )
    return {k: round(float(v), 4) for k, v in summary.iloc[0].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tracks_csv", type=Path)
    parser.add_argument("--gt", type=Path, help="MOTChallenge ground truth")
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    for label, column in (
        ("Stable player IDs (track_id)", "track_id"),
        ("Raw BoT-SORT IDs (tracker_id)", "tracker_id"),
    ):
        tracks = load_tracks(args.tracks_csv, column)
        print(f"\n{label}")
        for key, value in id_statistics(tracks, args.fps).items():
            print(f"  {key}: {value}")
        if args.gt:
            for key, value in clear_mot_metrics(tracks, load_mot(args.gt)).items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
