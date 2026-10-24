"""
Assign every tracked player to a team (or goalkeeper / referee).

Reads the tracks written by track_players.py, looks at the players' kit
colours and writes one row per stable player ID:

    outputs/tracking/<name>_teams.csv
        track_id, team, role, confidence, shirt_colour, shirt_colour_bn

Usage:
    python src/identify_teams.py --video data/videos/test1.mp4
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

from appearance import BENGALI_COLOUR_NAMES, extract_descriptor, shirt_colour_name
from identity import overlaps_other
from team_identification import TeamClassifier, TeamVotes

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = ROOT / "data" / "videos" / "test1.mp4"
TRACKING_DIR = ROOT / "outputs" / "tracking"

# Look at every Nth frame; kit colours do not change quickly.
FRAME_STEP = 5
# Descriptors kept per player, so memory stays bounded on full matches.
MAX_SAMPLES_PER_PLAYER = 40
OVERLAP_LIMIT = 0.15


def read_track_rows(tracks_csv):
    """Yield (frame, [(track_id, box), ...]) in frame order."""
    current, rows = None, []
    with tracks_csv.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            frame = int(row["frame"])
            if frame != current and rows:
                yield current, rows
                rows = []
            current = frame
            rows.append((
                int(row["track_id"]),
                tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")),
            ))
    if rows:
        yield current, rows


def collect_samples(video_path, tracks_csv):
    """Return per-frame samples: [(player_ids, boxes, descriptors), ...]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    samples = []
    per_player = defaultdict(int)
    frame_index, frame = -1, None

    try:
        for target, rows in read_track_rows(tracks_csv):
            if target % FRAME_STEP:
                continue
            while frame_index < target:
                success, frame = cap.read()
                if not success:
                    return samples
                frame_index += 1

            boxes = [box for _, box in rows]
            player_ids, kept_boxes, descriptors = [], [], []
            for index, (player_id, box) in enumerate(rows):
                descriptor = None
                if (
                    per_player[player_id] < MAX_SAMPLES_PER_PLAYER
                    and not overlaps_other(index, boxes, OVERLAP_LIMIT)
                ):
                    descriptor = extract_descriptor(frame, box)
                if descriptor is not None:
                    per_player[player_id] += 1
                player_ids.append(player_id)
                kept_boxes.append(box)
                descriptors.append(descriptor)
            samples.append((player_ids, kept_boxes, descriptors))
    finally:
        cap.release()
    return samples


def identify_teams(samples):
    """Fit team colours and decide each player's team and role."""
    classifier = TeamClassifier().fit(
        [d for _, _, descriptors in samples for d in descriptors]
    )
    votes = TeamVotes()
    player_descriptors = defaultdict(list)

    for player_ids, boxes, descriptors in samples:
        labels = [classifier.predict(d)[0] for d in descriptors]
        votes.add_frame(player_ids, boxes, labels)
        for player_id, descriptor in zip(player_ids, descriptors):
            if descriptor is not None:
                player_descriptors[player_id].append(descriptor)

    decisions = votes.decide()
    for player_id, decision in decisions.items():
        own = player_descriptors[player_id]
        colour = shirt_colour_name(sum(own) / len(own)) if own else None
        decision["shirt_colour"] = colour
        decision["shirt_colour_bn"] = BENGALI_COLOUR_NAMES.get(colour)
    return classifier, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--tracks", type=Path, help="tracks CSV")
    parser.add_argument("--output", type=Path, help="teams CSV")
    args = parser.parse_args()

    name = args.video.stem
    tracks_csv = args.tracks or TRACKING_DIR / f"{name}_tracks.csv"
    output_csv = args.output or tracks_csv.with_name(f"{name}_teams.csv")

    for path in (args.video, tracks_csv):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    samples = collect_samples(args.video, tracks_csv)
    classifier, decisions = identify_teams(samples)

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "track_id", "team", "role", "confidence",
            "shirt_colour", "shirt_colour_bn",
        ])
        for player_id in sorted(decisions):
            d = decisions[player_id]
            writer.writerow([
                player_id, d["team"] or "", d["role"], d["confidence"],
                d["shirt_colour"] or "", d["shirt_colour_bn"] or "",
            ])

    roles = defaultdict(int)
    for d in decisions.values():
        roles[(d["team"], d["role"])] += 1
    print("Team colours:", classifier.colours)
    for (team, role), count in sorted(roles.items(), key=str):
        print(f"  {team or '-'} {role}: {count} IDs")
    print("Teams CSV:", output_csv)


if __name__ == "__main__":
    main()
