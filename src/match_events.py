"""
Match events and the players involved in them.

Inputs (from the earlier scripts):
    <name>_tracks.csv   stable player tracks    (track_players.py)
    <name>_teams.csv    team and role per ID    (identify_teams.py)
    <name>_ball.csv     ball position per frame (ball_tracking.py)

What happened is decided in one of two ways:
  * from an action-spotting model's output (SoccerNet "results_spotting"
    JSON, --spotting): goals, fouls, cards, corners, free kicks and goal
    kicks with timestamps; this module then works out who was involved;
  * goals always come from the broadcast scoreboard (scoreboard.py) when
    one is found: the score graphic changing is far more reliable than
    anything seen on the pitch;
  * from rules on the tracks alone: a card is a yellow or red rectangle
    held up above the referee's head (card_detection.py); a set piece is a ball that stood still
    and is then played; the player who plays it is the taker; a set piece
    taken by a goalkeeper is a goal kick; a free kick implies a foul by the
    opponent nearest to the fouled player just before play stopped.

Who was involved is decided from ball possession:
  * goal: the last player to touch the ball (scorer) and the previous
    teammate to touch it (assist);
  * foul: the player on the ball (fouled) and the nearest opponent (fouler);
  * card: the player the referee is facing up to when showing it, else the
    fouler of the latest foul;
  * corner / free kick / goal kick: the first player to touch the ball
    after it stood still.

Output: <name>_events.json, a list of events:
    {"frame", "time", "minute", "type", "players": [{"role", "track_id",
     "team"}], "confidence", "source"}
type uses the event names of dataset.json (goal, foul, yellow_card, ...).

Usage:
    python src/match_events.py --video data/videos/test1.mp4 \\
        [--spotting results_spotting.json] [--start-minute 0]
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from card_detection import detect_card
from scoreboard import find_score_changes, parse_score

ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = ROOT / "outputs" / "tracking"

# The ball is "on" a player when it is within this many box heights of
# the player's feet.
POSSESSION_DISTANCE = 0.6
# A touch needs this many frames in a row (filters players running past).
MIN_TOUCH_FRAMES = 3

# A set piece: the ball stays within this many pixels (camera motion
# removed) for at least this many seconds, then moves away.
STILL_RADIUS = 6.0
STILL_SECONDS = 1.5

# Search windows, in seconds.
GOAL_LOOKBACK = 15.0
ASSIST_LOOKBACK = 10.0
FOUL_WINDOW = 3.0
FOUL_LOOKBACK_FROM_RESTART = 20.0
CARD_FOUL_LOOKBACK = 90.0
SET_PIECE_WINDOW = 6.0

# Card showing: look at every Nth frame; a card needs this many hits
# with gaps of at most CARD_GAP seconds between them.
CARD_FRAME_STEP = 2
CARD_MIN_HITS = 3
CARD_GAP = 1.0

# SoccerNet-v2 action-spotting labels -> dataset.json event types.
SOCCERNET_LABELS = {
    "Goal": "goal",
    "Foul": "foul",
    "Yellow card": "yellow_card",
    "Red card": "red_card",
    "Yellow->red card": "red_card",
    "Corner": "corner_kick",
    "Direct free-kick": "freekick",
    "Indirect free-kick": "freekick",
    "Clearance": "goal_kick",  # SoccerNet calls goal kicks "Clearance"
    "Penalty": "penalty",
}


@dataclass
class Touch:
    start: int
    end: int
    track_id: int
    team: str | None


@dataclass
class MatchData:
    fps: float
    tracks: dict      # frame -> [(track_id, (x1, y1, x2, y2))]
    teams: dict       # track_id -> {"team", "role"}
    ball: dict        # frame -> (x, y, sx, sy, scene)
    team_colours: dict = None  # team -> shirt colour name

    def team(self, track_id):
        return self.teams.get(track_id, {}).get("team")

    def role(self, track_id):
        return self.teams.get(track_id, {}).get("role", "player")

    def players(self, frame):
        """Non-referee boxes in a frame."""
        return [
            (tid, box) for tid, box in self.tracks.get(frame, [])
            if self.role(tid) != "referee"
        ]

    def referees(self, frame):
        return [
            (tid, box) for tid, box in self.tracks.get(frame, [])
            if self.role(tid) == "referee"
        ]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_match(tracks_csv, teams_csv, ball_csv, fps):
    tracks = defaultdict(list)
    with Path(tracks_csv).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            tracks[int(row["frame"])].append((
                int(row["track_id"]),
                tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")),
            ))

    teams = {}
    team_colours = {}
    if teams_csv and Path(teams_csv).exists():
        with Path(teams_csv).open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                teams[int(row["track_id"])] = {
                    "team": row["team"] or None,
                    "role": row["role"],
                }
                if row["team"] and row["role"] == "player" and row.get(
                    "team_colour"
                ):
                    team_colours[row["team"]] = row["team_colour"]

    ball = {}
    with Path(ball_csv).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            ball[int(row["frame"])] = (
                float(row["x"]), float(row["y"]),
                float(row["sx"]), float(row["sy"]), int(row["scene"]),
            )
    return MatchData(fps=fps, tracks=dict(tracks), teams=teams, ball=ball,
                     team_colours=team_colours)


def load_spotting(path, fps, half=1):
    """Read SoccerNet results_spotting JSON -> [(frame, type, confidence)].

    Only predictions for the given half are used; "position" is in ms.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    events = []
    for item in data.get("predictions", []):
        event_type = SOCCERNET_LABELS.get(item.get("label"))
        if event_type is None:
            continue
        if int(item.get("half", half)) != half:
            continue
        frame = int(round(float(item["position"]) / 1000.0 * fps))
        events.append((frame, event_type, float(item.get("confidence", 1.0))))
    return sorted(events)


# ----------------------------------------------------------------------
# Possession
# ----------------------------------------------------------------------

def _feet(box):
    return np.array([(box[0] + box[2]) / 2.0, box[3]])


def _height(box):
    return max(1.0, box[3] - box[1])


def ball_holder(match, frame):
    """The player whose feet are nearest the ball, if close enough."""
    if frame not in match.ball:
        return None
    ball = np.array(match.ball[frame][:2])
    best, best_distance = None, None
    for tid, box in match.players(frame):
        distance = np.linalg.norm(_feet(box) - ball) / _height(box)
        if distance <= POSSESSION_DISTANCE and (
            best_distance is None or distance < best_distance
        ):
            best, best_distance = tid, distance
    return best


def find_touches(match):
    """Runs of frames in which one player has the ball (gaps of up to two
    frames allowed)."""
    touches = []
    run = None  # [track_id, start, last]

    def close():
        if run and run[2] - run[1] + 1 >= MIN_TOUCH_FRAMES:
            touches.append(Touch(run[1], run[2], run[0], match.team(run[0])))

    for frame in sorted(match.ball):
        holder = ball_holder(match, frame)
        if holder is None:
            continue
        if run and run[0] == holder and frame - run[2] <= 2:
            run[2] = frame
        else:
            close()
            run = [holder, frame, frame]
    close()
    return touches


def touches_between(touches, first, last):
    return [t for t in touches if t.end >= first and t.start <= last]


def nearest_opponent(match, frame, track_id):
    """Nearest player of the other team to track_id in a frame."""
    team = match.team(track_id)
    boxes = dict(match.players(frame))
    if track_id not in boxes or team is None:
        return None
    centre = _feet(boxes[track_id])
    best, best_distance = None, None
    for tid, box in boxes.items():
        other = match.team(tid)
        if tid == track_id or other is None or other == team:
            continue
        distance = np.linalg.norm(_feet(box) - centre)
        if best_distance is None or distance < best_distance:
            best, best_distance = tid, distance
    return best


def nearest_to_referee(match, frame):
    referees = match.referees(frame)
    if not referees:
        return None
    _, ref_box = referees[0]
    centre = _feet(ref_box)
    best, best_distance = None, None
    for tid, box in match.players(frame):
        distance = np.linalg.norm(_feet(box) - centre) / _height(ref_box)
        if distance <= 3.0 and (best_distance is None or distance < best_distance):
            best, best_distance = tid, distance
    return best


# ----------------------------------------------------------------------
# Set pieces from the ball alone
# ----------------------------------------------------------------------

def find_restarts(match):
    """Frames where a ball that stood still starts moving."""
    frames = sorted(match.ball)
    min_still = int(STILL_SECONDS * match.fps)
    restarts = []
    anchor, anchor_frame, still_frames, prev = None, None, 0, None

    for frame in frames:
        _, _, sx, sy, scene = match.ball[frame]
        point = np.array([sx, sy])
        broken = prev is None or frame - prev > 2 or match.ball[prev][4] != scene
        if broken or anchor is None:
            anchor, anchor_frame, still_frames = point, frame, 1
        elif np.linalg.norm(point - anchor) <= STILL_RADIUS:
            still_frames += 1
        else:
            if still_frames >= min_still:
                restarts.append((anchor_frame, frame))
            anchor, anchor_frame, still_frames = point, frame, 1
        prev = frame
    return restarts


# ----------------------------------------------------------------------
# Attribution
# ----------------------------------------------------------------------

def _player(match, role, track_id):
    return {"role": role, "track_id": track_id, "team": match.team(track_id)}


def attribute(match, touches, event_type, frame, fouls=()):
    """Players involved in an event at `frame`."""
    fps = match.fps

    if event_type == "goal":
        before = [t for t in touches_between(
            touches, frame - int(GOAL_LOOKBACK * fps), frame)]
        if not before:
            return []
        scorer = before[-1]
        players = [_player(match, "scorer", scorer.track_id)]
        for touch in reversed(before[:-1]):
            if scorer.start - touch.end > ASSIST_LOOKBACK * fps:
                break
            if touch.track_id != scorer.track_id and touch.team == scorer.team:
                players.append(_player(match, "assist", touch.track_id))
                break
        return players

    if event_type == "foul":
        around = touches_between(touches, frame - int(FOUL_WINDOW * fps), frame)
        if not around:
            return []
        fouled = around[-1]
        players = [_player(match, "fouled", fouled.track_id)]
        fouler = nearest_opponent(match, fouled.end, fouled.track_id)
        if fouler is not None:
            players.insert(0, _player(match, "fouler", fouler))
        return players

    if event_type in ("yellow_card", "red_card"):
        recipient = nearest_to_referee(match, frame)
        if recipient is None:
            recent = [
                f for f in fouls
                if 0 <= frame - f["frame"] <= CARD_FOUL_LOOKBACK * fps
            ]
            for foul in reversed(recent):
                fouler = [p for p in foul["players"] if p["role"] == "fouler"]
                if fouler:
                    recipient = fouler[0]["track_id"]
                    break
        return [] if recipient is None else [_player(match, "booked", recipient)]

    if event_type in ("corner_kick", "freekick", "goal_kick", "penalty"):
        # The taker often stands over the ball before kicking it, so take
        # the first touch still going on as the ball is played.
        window = int(SET_PIECE_WINDOW * fps)
        around = touches_between(touches, frame - 2, frame + window)
        if not around:
            return []
        taker = min(around, key=lambda t: t.end)
        return [_player(match, "taker", taker.track_id)]

    return []


def _event(match, frame, event_type, players, confidence, source, start_minute):
    seconds = frame / match.fps
    return {
        "frame": frame,
        "time": round(seconds, 2),
        "minute": int(start_minute + seconds // 60) + 1,
        "type": event_type,
        "players": players,
        "confidence": round(confidence, 3),
        "source": source,
    }


def events_from_spotting(match, spotted, start_minute=0):
    touches = find_touches(match)
    events, fouls = [], []
    for frame, event_type, confidence in spotted:
        players = attribute(match, touches, event_type, frame, fouls)
        event = _event(match, frame, event_type, players, confidence,
                       "spotting", start_minute)
        events.append(event)
        if event_type == "foul":
            fouls.append(event)
    return events


def find_cards(match, video_path):
    """[(first frame, "yellow_card" | "red_card")] seen in the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    hits = {"yellow": [], "red": []}
    frame_number = -1
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_number += 1
            if frame_number % CARD_FRAME_STEP:
                continue
            referees = match.referees(frame_number)
            if not referees:
                continue
            others = [box for _, box in match.players(frame_number)]
            for _, box in referees:
                colour = detect_card(frame, box, others)
                if colour:
                    hits[colour].append(frame_number)
    finally:
        cap.release()

    cards = []
    gap = CARD_GAP * match.fps
    for colour, frames in hits.items():
        group = []
        for frame in frames + [None]:
            if group and (frame is None or frame - group[-1] > gap):
                if len(group) >= CARD_MIN_HITS:
                    cards.append((group[0], f"{colour}_card"))
                group = []
            if frame is not None:
                group.append(frame)
    return sorted(cards)


def events_from_rules(match, start_minute=0, video_path=None):
    """Cards seen in the video, set pieces from a still ball, and the foul
    that caused each free kick."""
    touches = find_touches(match)
    events = []
    for still_from, moved_at in find_restarts(match):
        taker_players = attribute(match, touches, "freekick", moved_at)
        if not taker_players:
            continue
        taker = taker_players[0]["track_id"]
        if match.role(taker) == "goalkeeper":
            events.append(_event(match, moved_at, "goal_kick",
                                 taker_players, 0.5, "rules", start_minute))
            continue

        events.append(_event(match, moved_at, "freekick",
                             taker_players, 0.4, "rules", start_minute))

        # The foul: the last touch before play stopped, by the free-kick
        # team, and the nearest opponent at that moment.
        team = match.team(taker)
        earlier = touches_between(
            touches,
            still_from - int(FOUL_LOOKBACK_FROM_RESTART * match.fps),
            still_from,
        )
        earlier = [t for t in earlier if t.team == team and t.start < still_from]
        if earlier:
            fouled = earlier[-1]
            foul_frame = min(fouled.end, still_from)
            fouler = nearest_opponent(match, foul_frame, fouled.track_id)
            if fouler is not None:
                events.append(_event(
                    match, foul_frame, "foul",
                    [_player(match, "fouler", fouler),
                     _player(match, "fouled", fouled.track_id)],
                    0.3, "rules", start_minute,
                ))

    if video_path is not None:
        fouls = sorted(
            (e for e in events if e["type"] == "foul"), key=lambda e: e["frame"]
        )
        for frame, card in find_cards(match, video_path):
            earlier = [f for f in fouls if f["frame"] <= frame]
            players = attribute(match, touches, card, frame, earlier)
            events.append(_event(match, frame, card, players, 0.5,
                                 "rules", start_minute))
    return sorted(events, key=lambda e: e["frame"])


def nearest_to_ball(match, frame, team):
    """Player of `team` nearest the ball at its last sighting within
    GOAL_LOOKBACK before `frame`, however far away."""
    first = frame - int(GOAL_LOOKBACK * match.fps)
    for f in range(frame, first, -1):
        if f not in match.ball:
            continue
        ball = np.array(match.ball[f][:2])
        mates = [(tid, box) for tid, box in match.players(f)
                 if match.team(tid) == team]
        if mates:
            return min(
                mates,
                key=lambda m: np.linalg.norm(_feet(m[1]) - ball) / _height(m[1]),
            )[0]
    return None


def scoring_teams(match, changes):
    """{"home"/"away": team} from the colours of the board's two halves
    and the teams' shirt colours; one match is enough for both."""
    colours = match.team_colours or {}
    side_colour = {c["side"]: c.get("side_colour") for c in changes}
    mapping = {}
    for side, colour in side_colour.items():
        found = [t for t, c in colours.items() if colour and c == colour]
        if len(found) == 1:
            mapping[side] = found[0]
    teams = sorted(colours)
    if len(mapping) == 1 and len(teams) == 2:
        side, team = next(iter(mapping.items()))
        other_side = "away" if side == "home" else "home"
        mapping[other_side] = teams[1] if team == teams[0] else teams[0]
    if len(set(mapping.values())) < len(mapping):
        return {}
    return mapping


def goals_from_scoreboard(match, video_path, start_score=(0, 0),
                          start_minute=0):
    """One goal event per change of the score graphic.

    The scorer is the last player on the ball before it (see attribute),
    from the scoring team when the board's colours tell which team that
    is.
    """
    touches = find_touches(match)
    changes = find_score_changes(video_path, start_score)
    side_team = scoring_teams(match, changes)
    events = []
    for change in changes:
        frame = change["frame"]
        team = side_team.get(change["side"])
        own = [t for t in touches if t.team == team] if team else touches
        players = attribute(match, own, "goal", frame)
        if not players and team:
            nearest = nearest_to_ball(match, frame, team)
            if nearest is not None:
                players = [_player(match, "scorer", nearest)]
        event = _event(match, frame, "goal", players, 0.9, "scoreboard",
                       start_minute)
        event["team"] = team
        event["side"] = change["side"]
        event["score"] = change["score"]
        events.append(event)
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True,
                        help="the match video; its name also locates the CSVs")
    parser.add_argument("--spotting", type=Path,
                        help="SoccerNet results_spotting JSON")
    parser.add_argument("--half", type=int, default=1)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--start-minute", type=int, default=0,
                        help="match minute at the start of the video")
    parser.add_argument("--start-score", default="0-0",
                        help="score shown at the start of the video")
    parser.add_argument("--no-scoreboard", action="store_true",
                        help="do not read goals from the score graphic")
    args = parser.parse_args()

    name = args.video.stem
    match = load_match(
        TRACKING_DIR / f"{name}_tracks.csv",
        TRACKING_DIR / f"{name}_teams.csv",
        TRACKING_DIR / f"{name}_ball.csv",
        args.fps,
    )
    if args.spotting:
        spotted = load_spotting(args.spotting, args.fps, args.half)
        events = events_from_spotting(match, spotted, args.start_minute)
    else:
        events = events_from_rules(match, args.start_minute, args.video)
    if not args.no_scoreboard:
        goals = goals_from_scoreboard(
            match, args.video, parse_score(args.start_score),
            args.start_minute,
        )
        if goals:
            events = [e for e in events if e["type"] != "goal"] + goals
            events.sort(key=lambda e: e["frame"])

    output = TRACKING_DIR / f"{name}_events.json"
    output.write_text(json.dumps(events, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    for event in events:
        who = ", ".join(f"{p['role']} P{p['track_id']}" for p in event["players"])
        score = f" ({event['score']})" if "score" in event else ""
        print(f"{event['minute']:>3}'  {event['type']:<12} {who}{score}")
    print("Events:", output)


if __name__ == "__main__":
    main()
