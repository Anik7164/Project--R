"""
Continuous commentary: fill the silences between the real event lines.

The event lines (goal, whistle, crowd, tackle, ...) are left as they are.
Where nothing is said for more than MAX_SILENCE seconds, a short line is
added every EVERY seconds about what is going on at that moment: which
team has the ball (from the ball and player tracks), the ball changing
hands, or, when the ball is not seen, a neutral line about the game.

Run it after player_identity.py and before speak_commentary.py:

    python src/player_identity.py --video data/videos/match4.mp4
    python src/fill_commentary.py --video data/videos/match4.mp4
    python src/speak_commentary.py --video data/videos/match4.mp4 --voice edge

It rewrites <name>_commentary.json. The event-only version is kept as
<name>_commentary_events.json and read again on a re-run, until
player_identity.py writes a new commentary file.
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import csv
import json
import random
from pathlib import Path

import cv2

from match_events import TRACKING_DIR, find_touches, load_match
from play_events import reliable

EVERY = 5.0
MAX_SILENCE = 8.0
LOOK_BACK = 5.0          # seconds of play a filler line describes
KEEP_SHARE = 0.65        # share of the ball time for "team X has the ball"
SPEECH_GAP = 3.0         # leave room after a line before the next one

KEEPING = [
    "বল এখন {team} জার্সির দলের দখলে।",
    "{team} জার্সির দলের পায়ে বল।",
    "পাস দিয়ে খেলা গড়ছে {team} জার্সির দল।",
    "ধীরে ধীরে সামনে এগোচ্ছে {team} জার্সির দল।",
    "{team} জার্সির দল বল ঘুরিয়ে সুযোগ খুঁজছে।",
    "বল নিজেদের পায়ে রেখেছে {team} জার্সির দল।",
]
CHANGE = [
    "বলের দখল বদলাল, এবার {team} জার্সির দলের পালা।",
    "বল কেড়ে নিল {team} জার্সির দল!",
    "এবার বল {team} জার্সির দলের কাছে।",
]
NEUTRAL = [
    "চলছে বলের লড়াই।",
    "দুই দলই সুযোগ খুঁজছে।",
    "কেউ কাউকে এক ইঞ্চি জায়গা ছাড়ছে না।",
    "ম্যাচ জমে উঠেছে, দর্শকরা তাকিয়ে আছেন মাঠের দিকে।",
    "দুই দলের লড়াই চলছে সমানে সমানে।",
    "খেলা চলছে, গোলের অপেক্ষায় দর্শকরা।",
]


def bengali_colours(teams_csv):
    colours = {}
    if teams_csv.exists():
        with teams_csv.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                if row["team"] and row["role"] == "player" and row.get("team_colour_bn"):
                    colours.setdefault(row["team"], row["team_colour_bn"])
    return colours


def ball_team(touches, fps, at):
    """(team that had the ball in the LOOK_BACK seconds before at, team
    before that) or (None, None)."""
    start, end = (at - LOOK_BACK) * fps, at * fps
    time_by_team, order = {}, []
    for t in touches:
        overlap = min(t.end, end) - max(t.start, start) + 1
        if overlap > 0 and t.team:
            time_by_team[t.team] = time_by_team.get(t.team, 0) + overlap
            order.append(t.team)
    if not time_by_team:
        return None, None
    team = max(time_by_team, key=time_by_team.get)
    if time_by_team[team] < KEEP_SHARE * sum(time_by_team.values()):
        return None, None
    earlier = [t.team for t in touches if t.end < start and t.team]
    return team, (earlier[-1] if earlier else None)


def pick(options, last_text, rng):
    choices = [o for o in options if o != last_text] or options
    return rng.choice(choices)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--every", type=float, default=EVERY,
                        help="seconds between filler lines (default 5)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="no filler before this second (intro)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    name = args.video.stem
    rng = random.Random(args.seed)

    events_only = TRACKING_DIR / f"{name}_commentary_events.json"
    output = TRACKING_DIR / f"{name}_commentary.json"
    # A commentary file with no filler lines in it is a fresh one from
    # player_identity.py: it replaces the event-only copy kept from before.
    if output.exists() and (not events_only.exists() or not any(
            l.get("type") == "filler" for l in json.loads(output.read_text(encoding="utf-8")))):
        events_only.write_text(output.read_text(encoding="utf-8"), encoding="utf-8")
    lines = [l for l in json.loads(events_only.read_text(encoding="utf-8"))]

    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    duration = (capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    capture.release()

    match = load_match(TRACKING_DIR / f"{name}_tracks.csv",
                       TRACKING_DIR / f"{name}_teams.csv",
                       TRACKING_DIR / f"{name}_ball.csv", fps)
    touches = sorted(reliable(match, find_touches(match)), key=lambda t: t.start)
    colours = bengali_colours(TRACKING_DIR / f"{name}_teams.csv")

    spoken = sorted(l["time"] for l in lines)
    marks = [max(args.start, spoken[0] if spoken else 0.0)] + spoken + [duration]
    fillers, last_text, last_team = [], None, None
    for before, after in zip(marks, marks[1:]):
        if after - before <= MAX_SILENCE:
            continue
        at = before + max(args.every, SPEECH_GAP)
        while at < after - SPEECH_GAP:
            if at < args.start:
                at += args.every
                continue
            team, _ = ball_team(touches, fps, at)
            if team and team in colours:
                options = CHANGE if last_team and last_team != team else KEEPING
                last_team = team
                text = pick(options, last_text, rng).format(team=colours[team])
            else:
                text = pick(NEUTRAL, last_text, rng)
            fillers.append({"time": round(at, 2), "minute": int(at // 60) + 1,
                            "type": "filler", "text": text})
            last_text = text
            at += args.every

    merged = sorted(lines + fillers, key=lambda l: l["time"])
    output.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(lines)} event lines + {len(fillers)} filler lines -> {output}")
    print(f"Next: python src/speak_commentary.py --video {args.video} --voice edge")


if __name__ == "__main__":
    main()
