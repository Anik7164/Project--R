"""
Set a team's kit colour by hand in <name>_teams.csv, without running
identify_teams.py again (when the automatic name is wrong).

    python src/fix_team_colour.py --video data/videos/match5.mp4 team_b=white
"""

import argparse
import csv
from pathlib import Path

from appearance import BENGALI_COLOUR_NAMES

TRACKING_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tracking"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("names", nargs="+", metavar="TEAM=COLOUR")
    args = parser.parse_args()
    path = TRACKING_DIR / f"{args.video.stem}_teams.csv"
    names = dict(item.split("=", 1) for item in args.names)
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    changed = 0
    for row in rows:
        if row["role"] == "player" and row["team"] in names:
            row["team_colour"] = names[row["team"]]
            row["team_colour_bn"] = BENGALI_COLOUR_NAMES.get(names[row["team"]], names[row["team"]])
            changed += 1
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"{changed} players renamed in {path}")


if __name__ == "__main__":
    main()
