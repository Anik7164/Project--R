"""
Read the name and number on each player's shirt.

Takes the torso crops saved by extract_jersey_crops.py (one folder per
stable player ID), runs OCR on each crop and lets the crops vote. A name
or number is only accepted when several crops agree, because one blurry
crop easily turns a 6 into an 8.

An optional roster makes the result far more reliable:
    {"team_a": {"7": "Ronaldo", "10": "Messi"}, "team_b": {...}}
A read number is turned into the roster name, and a read name is
snapped to the closest roster name.

Output: outputs/tracking/<name>_identities.csv
    track_id, name, name_source, jersey_number, number_votes, name_votes

OCR uses EasyOCR (pip install easyocr; it downloads its model on first
use). Any object with readtext(image, allowlist=...) returning
[(box, text, confidence)] can be used instead.

Usage:
    python src/extract_jersey_crops.py
    python src/jersey_reading.py --video data/videos/test1.mp4 [--roster roster.json]
"""

import argparse
import csv
import difflib
import json
import re
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CROPS = ROOT / "outputs" / "jersey_crops"
TRACKING_DIR = ROOT / "outputs" / "tracking"

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.-' "

MIN_OCR_CONFIDENCE = 0.3
MIN_NAME_LENGTH = 3
# A value needs at least this many crops and this share of the votes.
MIN_VOTES = 2
MIN_SHARE = 0.5
# Upscale small crops so OCR has enough pixels to work with.
MIN_OCR_HEIGHT = 160
ROSTER_NAME_MATCH = 0.7


def prepare(crop):
    """Upscale and boost contrast of a torso crop for OCR."""
    height = crop.shape[0]
    if height < MIN_OCR_HEIGHT:
        scale = MIN_OCR_HEIGHT / float(height)
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def parse_number(results):
    """Best shirt number (1-99) in OCR results, as (text, confidence)."""
    best = None
    for _, text, confidence in results:
        digits = re.sub(r"\D", "", text)
        if not 1 <= len(digits) <= 2 or confidence < MIN_OCR_CONFIDENCE:
            continue
        if not 1 <= int(digits) <= 99:
            continue
        digits = str(int(digits))
        if best is None or confidence > best[1]:
            best = (digits, confidence)
    return best


def parse_name(results):
    """Best surname-like word in OCR results, as (text, confidence)."""
    best = None
    for _, text, confidence in results:
        name = re.sub(r"[^A-Z.\-' ]", "", text.upper()).strip(" .-'")
        if len(name.replace(" ", "")) < MIN_NAME_LENGTH:
            continue
        if confidence < MIN_OCR_CONFIDENCE:
            continue
        if best is None or confidence > best[1]:
            best = (name, confidence)
    return best


def vote(readings, min_votes=MIN_VOTES, min_share=MIN_SHARE):
    """readings: [(value, confidence)] -> (value, votes) or (None, 0)."""
    readings = [r for r in readings if r is not None]
    if not readings:
        return None, 0
    weights = Counter()
    counts = Counter()
    for value, confidence in readings:
        weights[value] += confidence
        counts[value] += 1
    value, weight = weights.most_common(1)[0]
    if counts[value] < min_votes or weight / sum(weights.values()) < min_share:
        return None, 0
    return value, counts[value]


def read_player(reader, crops):
    """OCR every crop of one player; return the voted name and number."""
    numbers, names = [], []
    for crop in crops:
        image = prepare(crop)
        numbers.append(parse_number(reader.readtext(image, allowlist=DIGITS)))
        names.append(parse_name(reader.readtext(image, allowlist=LETTERS)))
    number, number_votes = vote(numbers)
    name, name_votes = vote(names)
    return {
        "jersey_number": number,
        "number_votes": number_votes,
        "name": name,
        "name_votes": name_votes,
    }


def apply_roster(identity, team, roster):
    """Fill or correct the name from the team's roster, if there is one."""
    names_by_number = roster.get(team or "", {})
    if not names_by_number:
        identity["name_source"] = "jersey" if identity["name"] else ""
        return identity

    number = identity["jersey_number"]
    if number in names_by_number:
        identity["name"] = names_by_number[number]
        identity["name_source"] = "roster"
        return identity

    if identity["name"]:
        upper = {n.upper(): n for n in names_by_number.values()}
        match = difflib.get_close_matches(
            identity["name"], list(upper), n=1, cutoff=ROSTER_NAME_MATCH
        )
        if match:
            identity["name"] = upper[match[0]]
            identity["name_source"] = "roster"
            # The roster also tells us the number.
            for number, name in names_by_number.items():
                if name == identity["name"]:
                    identity["jersey_number"] = number
            return identity
    identity["name_source"] = "jersey" if identity["name"] else ""
    return identity


def load_crops(crops_dir):
    """{track_id: [crop images]} from track_XXXX folders."""
    crops = {}
    for folder in sorted(Path(crops_dir).glob("track_*")):
        try:
            track_id = int(folder.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        images = [cv2.imread(str(p)) for p in sorted(folder.glob("*.jpg"))]
        crops[track_id] = [im for im in images if im is not None]
    return crops


def load_teams(teams_csv):
    teams = {}
    if teams_csv.exists():
        with teams_csv.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                teams[int(row["track_id"])] = row["team"] or None
    return teams


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True,
                        help="its name locates the tracking outputs")
    parser.add_argument("--crops", type=Path, default=DEFAULT_CROPS)
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=args.gpu)

    name = args.video.stem
    teams = load_teams(TRACKING_DIR / f"{name}_teams.csv")
    roster = (
        json.loads(args.roster.read_text(encoding="utf-8"))
        if args.roster else {}
    )

    output = TRACKING_DIR / f"{name}_identities.csv"
    rows = []
    for track_id, crops in load_crops(args.crops).items():
        identity = read_player(reader, crops)
        identity = apply_roster(identity, teams.get(track_id), roster)
        rows.append((track_id, identity))
        print(f"P{track_id}: name={identity['name']} number={identity['jersey_number']}")

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "track_id", "name", "name_source", "jersey_number",
            "number_votes", "name_votes",
        ])
        for track_id, identity in rows:
            writer.writerow([
                track_id, identity["name"] or "", identity["name_source"],
                identity["jersey_number"] or "", identity["number_votes"],
                identity["name_votes"],
            ])
    print("Identities:", output)


if __name__ == "__main__":
    main()
