"""
Assign every tracked player to a team (or goalkeeper / referee).

Reads the tracks written by track_players.py, looks at the players' kit
colours and writes one row per stable player ID:

    outputs/tracking/<name>_teams.csv
        track_id, team, role, confidence, shirt_colour, shirt_colour_bn,
        team_colour, team_colour_bn

shirt_colour is read from this ID's own crops and can be off (shadows,
skin, a bad crop); team_colour is the colour of the whole team, used by
the commentary. For goalkeepers and referees both are their own colour.

Usage:
    python src/identify_teams.py --video data/videos/test1.mp4
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import csv
from collections import Counter, defaultdict
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
# With a football detector (train_detector.py) the detected class decides
# a player's role when this share of their boxes agree.
ROLE_SHARE = 0.6
DETECTOR_ROLES = {"player", "goalkeeper", "referee"}
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


def detector_roles(tracks_csv):
    """{track_id: role} from the class_name column, when the tracks come
    from a football detector; {} for a COCO "person" detector."""
    counts = defaultdict(Counter)
    with tracks_csv.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("class_name") in DETECTOR_ROLES:
                counts[int(row["track_id"])][row["class_name"]] += 1
    roles = {}
    for track_id, votes in counts.items():
        role, count = votes.most_common(1)[0]
        if count >= ROLE_SHARE * sum(votes.values()):
            roles[track_id] = role
    return roles


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


def identify_teams(samples, roles=None):
    """Fit team colours and decide each player's team and role (roles:
    detector_roles(), optional)."""
    roles = roles or {}
    # Goalkeepers and referees the detector found are left out of the
    # team colours.
    classifier = TeamClassifier().fit([
        d for player_ids, _, descriptors in samples
        for player_id, d in zip(player_ids, descriptors)
        if roles.get(player_id, "player") == "player"
    ])
    votes = TeamVotes()
    player_descriptors = defaultdict(list)

    for player_ids, boxes, descriptors in samples:
        labels = [classifier.predict(d)[0] for d in descriptors]
        votes.add_frame(player_ids, boxes, labels, descriptors)
        for player_id, descriptor in zip(player_ids, descriptors):
            if descriptor is not None:
                player_descriptors[player_id].append(descriptor)

    decisions = votes.decide(roles)
    for player_id, decision in decisions.items():
        own = player_descriptors[player_id]
        colour = shirt_colour_name(sum(own) / len(own)) if own else None
        decision["shirt_colour"] = colour
        decision["shirt_colour_bn"] = BENGALI_COLOUR_NAMES.get(colour)
        if decision["role"] == "player" and decision["team"]:
            team_colour = classifier.colours.get(decision["team"])
        else:
            team_colour = colour
        decision["team_colour"] = team_colour
        decision["team_colour_bn"] = BENGALI_COLOUR_NAMES.get(team_colour)
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
    roles = detector_roles(tracks_csv)
    if roles:
        print(f"Roles from the football detector for {len(roles)} IDs")
    classifier, decisions = identify_teams(samples, roles)

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "track_id", "team", "role", "confidence",
            "shirt_colour", "shirt_colour_bn", "team_colour", "team_colour_bn",
        ])
        for player_id in sorted(decisions):
            d = decisions[player_id]
            writer.writerow([
                player_id, d["team"] or "", d["role"], d["confidence"],
                d["shirt_colour"] or "", d["shirt_colour_bn"] or "",
                d["team_colour"] or "", d["team_colour_bn"] or "",
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
