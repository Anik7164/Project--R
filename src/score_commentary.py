"""
How much of the commentary is right: the events (match_events.py) against
the list of what really happened (mark_events.py, or a hand-written
<name>_truth.txt).

A line is right if an event of a type it may stand for happened within
WINDOW seconds of it, e.g. "the referee stops play" is right for a
whistle, a foul or a free kick; "the crowd is up, orange attack" for an
attack, a shot or a goal. If both name a team, the team must match too
(reported separately as well). Passes, runs and "who keeps the ball"
lines are too many to write down, so they are counted as not checked.

    python src/score_commentary.py --video data/videos/match5.mp4

Prints precision (share of checked lines that are right), and per real
event type how many were mentioned (recall); writes
outputs/tracking/<name>_score.json.
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = ROOT / "outputs" / "tracking"

WINDOW = 6.0
# Which real events each commentary event may stand for, and how far
# before (-) / after (+) the line the real event may be.
STANDS_FOR = {
    "goal": ({"goal"}, (-8.0, 8.0)),
    "big_chance": ({"attack", "shot", "goal", "corner"}, (-6.0, 12.0)),
    "missed_shot": ({"shot"}, (-6.0, 6.0)),
    "save": ({"shot"}, (-6.0, 6.0)),
    "whistle_stop": ({"whistle", "foul", "freekick", "card", "goal", "corner",
                      "goal_kick"}, (-6.0, 6.0)),
    "foul": ({"foul", "freekick"}, (-6.0, 30.0)),
    "freekick": ({"freekick", "foul"}, (-30.0, 6.0)),
    "corner_kick": ({"corner"}, (-20.0, 6.0)),
    "throw_in": ({"throw_in"}, (-10.0, 6.0)),
    "goal_kick": ({"goal_kick"}, (-15.0, 6.0)),
    "yellow_card": ({"card"}, (-30.0, 10.0)),
    "red_card": ({"card"}, (-30.0, 10.0)),
    "tackle_won": ({"tackle"}, (-3.0, 3.0)),
    "interception": ({"tackle"}, (-3.0, 3.0)),
}
NOT_CHECKED = {"match_intro", "pass", "long_pass", "run_with_ball", "possession_spell"}
TRUTH_TYPES = {"goal", "shot", "attack", "foul", "freekick", "corner", "throw_in",
               "goal_kick", "tackle", "card", "whistle"}


def parse_time(text):
    minutes, _, seconds = text.partition(":")
    return float(minutes) * 60 + float(seconds) if seconds else float(minutes)


def load_truth(name):
    json_path = TRACKING_DIR / f"{name}_truth.json"
    text_path = TRACKING_DIR / f"{name}_truth.txt"
    truth = []
    if json_path.exists():
        truth += json.loads(json_path.read_text(encoding="utf-8"))
    if text_path.exists():
        for number, line in enumerate(text_path.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split("#")[0].split()
            if not parts:
                continue
            if len(parts) < 2 or not re.fullmatch(r"\d+(:\d+(\.\d+)?)?", parts[0]):
                raise SystemExit(f"{text_path}:{number}: write it as '7:49 goal orange'")
            kind = parts[1].lower().replace("-", "_")
            kind = {"freekick": "freekick", "free_kick": "freekick", "throwin": "throw_in",
                    "goalkick": "goal_kick", "chance": "attack"}.get(kind, kind)
            if kind not in TRUTH_TYPES:
                raise SystemExit(f"{text_path}:{number}: unknown type '{parts[1]}'; "
                                 f"use one of {sorted(TRUTH_TYPES)}")
            truth.append({"time": parse_time(parts[0]), "type": kind,
                          "team": " ".join(parts[2:]) or None})
    if not truth:
        raise SystemExit(f"No list of real events: make {json_path.name} with "
                         f"mark_events.py, or write {text_path}")
    return sorted(truth, key=lambda e: e["time"])


def team_colours(name):
    path = TRACKING_DIR / f"{name}_teams.csv"
    colours = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                colours[int(row["track_id"])] = row.get("team_colour") or None
    return colours


def event_team(event, colours):
    if event.get("team_colour"):
        return event["team_colour"]
    for player in event.get("players", []):
        colour = colours.get(player.get("track_id"))
        if colour:
            return colour
    return None


def same_team(a, b):
    a, b = (a or "").lower().strip(), (b or "").lower().strip()
    return not a or not b or a == b or a in b or b in a


def score(events, truth, colours):
    lines, used = [], set()
    for event in events:
        kind = event.get("template_type", event["type"])
        kind = "goal" if kind == "goal_celebration" else kind
        if kind in NOT_CHECKED or kind not in STANDS_FOR:
            lines.append((event, "not checked", None))
            continue
        allowed, (before, after) = STANDS_FOR[kind]
        team = event_team(event, colours)
        candidates = [(abs(t["time"] - event["time"]), i) for i, t in enumerate(truth)
                      if t["type"] in allowed
                      and before <= t["time"] - event["time"] <= after]
        if not candidates:
            lines.append((event, "wrong", None))
            continue
        _, best = min(candidates)
        right_team = same_team(team, truth[best].get("team"))
        used.add(best)
        lines.append((event, "right" if right_team else "wrong team", truth[best]))
    return lines, used


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    name = args.video.stem

    events = json.loads((TRACKING_DIR / f"{name}_events.json").read_text(encoding="utf-8"))
    truth = load_truth(name)
    colours = team_colours(name)
    lines, used = score(events, truth, colours)

    print("time     line             verdict      real event")
    for event, verdict, real in lines:
        kind = event.get("template_type", event["type"])
        real_text = f"{real['time'] // 60:.0f}:{real['time'] % 60:04.1f} {real['type']}" \
            f" {real.get('team') or ''}" if real else ""
        print(f"{event['time'] // 60:.0f}:{event['time'] % 60:04.1f}  {kind:<16} "
              f"{verdict:<12} {real_text}")

    checked = [v for _, v, _ in lines if v != "not checked"]
    right = checked.count("right")
    event_only = right + checked.count("wrong team")
    summary = {
        "lines": len(lines),
        "checked": len(checked),
        "right": right,
        "precision": round(right / len(checked), 3) if checked else None,
        "precision_ignoring_team": round(event_only / len(checked), 3) if checked else None,
        "not_checked": len(lines) - len(checked),
        "recall": {},
    }
    for kind in sorted({t["type"] for t in truth}):
        indices = [i for i, t in enumerate(truth) if t["type"] == kind]
        summary["recall"][kind] = f"{sum(i in used for i in indices)}/{len(indices)}"

    print(f"\nChecked lines: {len(checked)} of {len(lines)} "
          f"({summary['not_checked']} passes/runs/possession not checked)")
    if checked:
        print(f"Right: {right}/{len(checked)} = {100 * summary['precision']:.0f}%   "
              f"(event right, team aside: {100 * summary['precision_ignoring_team']:.0f}%)")
    print("Real events mentioned:", ", ".join(f"{k} {v}" for k, v in summary["recall"].items()))
    output = TRACKING_DIR / f"{name}_score.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Score:", output)


if __name__ == "__main__":
    main()
