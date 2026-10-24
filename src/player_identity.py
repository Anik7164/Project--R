"""
How the commentary refers to a player, and filling the Bengali templates.

Fallback chain, best first:
    1. the name on the shirt (or from the roster),
    2. the shirt number,
    3. the shirt colour.

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
    team_info: {"shirt_colour"} from identify_teams.py.
    """
    identity = identity or {}
    team_info = team_info or {}
    name = identity.get("name") or None
    number = identity.get("jersey_number") or None
    colour = team_info.get("shirt_colour") or None

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
        who = describe(identities.get(track_id), teams.get(track_id))
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
    for event, line in commentary(events, identities, teams, templates, rng):
        print(f"{event['minute']:>3}' {event['type']:<12} {line or '-'}")


if __name__ == "__main__":
    main()
