"""
How the commentary refers to a player, and filling the Bengali templates.

Fallback chain, best first:
    1. the name on the shirt (or from the roster),
    2. the shirt number,
    3. the shirt colour (the team's colour, so every player of a team is
       called the same, e.g. "লাল জার্সির খেলোয়াড়").

Templates in dataset.json use {minute}, {jersey} and {player}. A template
is only used if every slot it needs can be filled: with a name and a
number any template fits; otherwise only templates without {jersey} are
used and {player} becomes "7 নম্বর জার্সির খেলোয়াড়" or
"লাল জার্সির খেলোয়াড়".

Usage (after match_events.py and jersey_reading.py):
    python src/player_identity.py --video data/videos/test1.mp4
"""

import argparse
import csv
import json
import random
from pathlib import Path

from appearance import BENGALI_COLOUR_NAMES

ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = ROOT / "outputs" / "tracking"
DATASET = ROOT / "dataset.json"

BENGALI_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

# Which player an event's commentary is about.
MAIN_ROLE = {
    "goal": "scorer",
    "foul": "fouler",
    "yellow_card": "booked",
    "red_card": "booked",
    "freekick": "taker",
    "corner_kick": "taker",
    "goal_kick": "taker",
    "penalty": "taker",
}


def bengali_number(value):
    return str(value).translate(BENGALI_DIGITS)


def describe(identity, team_info=None):
    """Return the best available way to name a player.

    identity: {"name", "jersey_number"} (either may be empty).
    team_info: {"team_colour", "shirt_colour"} from identify_teams.py.
    """
    identity = identity or {}
    team_info = team_info or {}
    name = identity.get("name") or None
    number = identity.get("jersey_number") or None
    colour = (
        team_info.get("team_colour") or team_info.get("shirt_colour") or None
    )

    if name:
        level = "name"
        label = name
    elif number:
        level = "number"
        label = f"{bengali_number(number)} নম্বর জার্সির খেলোয়াড়"
    elif colour:
        level = "colour"
        label = f"{BENGALI_COLOUR_NAMES.get(colour, colour)} জার্সির খেলোয়াড়"
    else:
        level = "unknown"
        label = "একজন খেলোয়াড়"
    return {
        "level": level,
        "player": label,
        "jersey": bengali_number(number) if number else None,
    }


def fill_template(templates, minute, who, rng=random):
    """Pick a template whose slots can all be filled, and fill it."""
    usable = [
        t for t in templates
        if "{jersey}" not in t or (who["jersey"] and who["level"] == "name")
    ]
    if not usable:
        return None
    template = rng.choice(usable)
    return template.format(
        minute=bengali_number(minute),
        jersey=who["jersey"] or "",
        player=who["player"],
    )


def video_time(seconds):
    """Seconds into the video as m:ss.s, e.g. 82.24 -> "1:22.2"."""
    minutes, rest = divmod(float(seconds), 60)
    return f"{int(minutes)}:{rest:04.1f}"


def load_by_track(path):
    rows = {}
    if Path(path).exists():
        with Path(path).open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rows[int(row["track_id"])] = row
    return rows


def commentary(events, identities, teams, templates_by_type, rng=random):
    """[(event, Bengali line or None)] for a list of match events."""
    lines = []
    for event in events:
        templates = templates_by_type.get(event["type"], [])
        role = MAIN_ROLE.get(event["type"])
        main = next(
            (p for p in event["players"] if p["role"] == role),
            event["players"][0] if event["players"] else None,
        )
        track_id = main["track_id"] if main else None
        team_info = teams.get(track_id)
        if team_info is None and event.get("team"):
            # Nobody identified, but the team is known (e.g. from the
            # scoreboard): "a player in red".
            team_info = next(
                (t for t in teams.values() if t.get("team") == event["team"]
                 and t.get("role", "player") == "player"),
                None,
            )
        who = describe(identities.get(track_id), team_info)
        lines.append((event, fill_template(templates, event["minute"], who, rng)))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    name = args.video.stem
    events = json.loads(
        (TRACKING_DIR / f"{name}_events.json").read_text(encoding="utf-8")
    )
    identities = load_by_track(TRACKING_DIR / f"{name}_identities.csv")
    teams = load_by_track(TRACKING_DIR / f"{name}_teams.csv")
    templates = {
        item["event_type"]: item["templates"]
        for item in json.loads(DATASET.read_text(encoding="utf-8"))
    }

    rng = random.Random(args.seed)
    print("video time  match min  event        player  commentary")
    for event, line in commentary(events, identities, teams, templates, rng):
        role = MAIN_ROLE.get(event["type"])
        main = next(
            (p for p in event["players"] if p["role"] == role), None
        )
        player = f"P{main['track_id']}" if main else "-"
        print(
            f"{video_time(event['time']):>10}  {event['minute']:>8}'  "
            f"{event['type']:<12} {player:<7} {line or '-'}"
        )


if __name__ == "__main__":
    main()
