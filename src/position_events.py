"""
Attacks from where the players and the ball are, not from the picture.

A model that looks at the whole picture (local_events.py, train_events.py)
learns what one ground looks like and means nothing on the next one. A
few numbers from the tracks mean the same thing from any camera, and
they did carry over between the two matches with a truth list:

    players in view        fewer when the camera follows an attack
    where they stand       left or right of the picture's middle
    how big they are       the camera zooms in near a goal
    ball off the middle    the ball is near the side of the picture
    ball height            how far up the picture the ball is
    ball speed             a long ball, a cross, a shot

Each is averaged over 6 seconds and measured against the match's own
average (so the camera place and zoom of that match drop out). WEIGHTS
says how much each one counts; attacks() returns the clearest moments.

How good (2026-11-12): weights learnt on one match and used on the
other, the clearest 10% of each match: match4 11 of 15 lines right
(picking moments at random gives 64%), match5 8 of 13 (random 40%).
Only two matches, and these six numbers were picked because they pointed
the same way in both, so the real figure on a new match will be lower.
In the whole pipeline (match_events.py --positions, weights from the
other match) it added 5 lines per match where nothing else was said:
match4 3 right, match5 2 right, with 3 more real shots mentioned. So it
finds more but half of what it adds is wrong, and the share of right
lines went down a little (67% to 65%, 71% to 67%). It is therefore off
unless --positions is given, until a better ball detector or more
labelled matches make it worth it.
Shots alone were tried the same way and were not good enough to say
(5 of 22 right): the ball is seen too rarely.

    python src/position_events.py --fit data/videos/match4.mp4 data/videos/match5.mp4
prints new WEIGHTS from matches with a truth list (and the leave-one-out
check) when more labelled matches exist.
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
from pathlib import Path

import numpy as np

STEP = 0.5                       # seconds
SMOOTH_SECONDS = 6.0
NAMES = ("players", "players_x", "player_height", "ball_off_middle", "ball_y", "ball_speed")
# Learnt on match4 + match5 (fit() below); for numbers measured against
# the match's own mean and spread.
WEIGHTS = {"players": -0.333, "players_x": -0.167, "player_height": -0.064,
           "ball_off_middle": 0.243, "ball_y": -0.05, "ball_speed": 0.162}
SHARE = 0.10                     # the clearest share of play can become a line
GAP_SECONDS = 12.0
BALL_GAP_STEPS = 6               # the ball's place is carried over gaps this short


def numbers(match, width, height):
    """(steps, 6) array, one row per STEP, columns as NAMES; NaN where
    nothing is known. match: match_events.MatchData."""
    import pandas as pd

    steps = 1 + int(max(list(match.tracks) + list(match.ball) + [0]) / match.fps / STEP)
    rows = {}
    for frame, boxes in match.tracks.items():
        step = int(frame / match.fps / STEP)
        if step in rows and rows[step][0] <= frame:
            continue
        rows[step] = (frame, boxes)          # the first frame of each step
    out = np.full((steps, len(NAMES)), np.nan)
    out[:, 0] = 0.0
    for step, (_, boxes) in rows.items():
        out[step, 0] = len(boxes)
        out[step, 1] = np.mean([(b[0] + b[2]) / 2 for _, b in boxes]) / width
        out[step, 2] = np.median([b[3] - b[1] for _, b in boxes]) / height
    seen = match.ball_detected if match.ball_detected is not None else set(match.ball)
    ball = np.full((steps, 2), np.nan)
    sums = {}
    for frame in seen:
        x, y = match.ball[frame][:2]
        sums.setdefault(int(frame / match.fps / STEP), []).append((x / width, y / height))
    for step, places in sums.items():
        ball[step] = np.mean(places, axis=0)
    filled = pd.DataFrame(ball).interpolate(limit=BALL_GAP_STEPS).to_numpy()
    out[:, 3] = np.abs(filled[:, 0] - 0.5)
    out[:, 4] = ball[:, 1]
    out[1:, 5] = np.hypot(*np.diff(filled, axis=0).T) / STEP
    return out


def scores(values, play, weights=None):
    """One number per step: how much it looks like an attack. values:
    numbers(); play: True for the steps that are play."""
    import pandas as pd

    weights = weights or WEIGHTS
    table = pd.DataFrame(values, columns=NAMES)
    table = table.fillna(table[play].median()).fillna(0.0)
    width = int(round(SMOOTH_SECONDS / STEP))
    smooth = table.rolling(width, center=True, min_periods=1).mean()
    spread = smooth[play].std().replace(0.0, 1.0).fillna(1.0)
    z = (smooth - smooth[play].mean()) / spread
    return sum(weights[name] * z[name] for name in NAMES).to_numpy()


def attacks(match, width, height, quiet=None, weights=None):
    """[(seconds, confidence)]: the clearest attack moments of the match.
    quiet(seconds) -> True where there is no play."""
    import pandas as pd

    values = numbers(match, width, height)
    play = np.array([not (quiet and quiet(i * STEP)) for i in range(len(values))], dtype=bool)
    if play.sum() < 120 or not match.ball:      # under a minute of play, or no ball at all
        return []
    score = scores(values, play, weights)
    smooth = pd.Series(score).rolling(5, center=True, min_periods=1).mean().to_numpy()
    limit = np.quantile(smooth[play], 1 - SHARE)
    taken = []
    for i in np.argsort(-smooth):
        if smooth[i] < limit:
            break
        if play[i] and all(abs(i - j) * STEP >= GAP_SECONDS for j in taken):
            taken.append(int(i))
    top = smooth[play].max() or 1.0
    return sorted((i * STEP, float(min(1.0, max(0.0, smooth[i] / top)))) for i in taken)


# ----------------------------------------------------------------------
# Learning the weights from matches with a truth list
# ----------------------------------------------------------------------
ATTACK_TYPES = {"attack", "shot", "goal", "corner"}
LABEL_SECONDS = 4.0


def _labelled(video, kickoff=0.0, pauses=()):
    import cv2
    import pandas as pd

    from match_events import TRACKING_DIR, _silent, load_match
    from score_commentary import load_truth

    name = Path(video).stem
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    size = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    capture.release()
    match = load_match(TRACKING_DIR / f"{name}_tracks.csv", TRACKING_DIR / f"{name}_teams.csv",
                       TRACKING_DIR / f"{name}_ball.csv", fps)
    values = numbers(match, *size)
    truth = load_truth(name)
    times = np.arange(len(values)) * STEP
    play = np.array([not _silent(t, kickoff, pauses) for t in times])
    table = pd.DataFrame(values, columns=NAMES)
    table = table.fillna(table[play].median()).fillna(0.0)
    smooth = table.rolling(int(round(SMOOTH_SECONDS / STEP)), center=True, min_periods=1).mean()
    z = ((smooth - smooth[play].mean()) / smooth[play].std().replace(0.0, 1.0)).to_numpy()
    yes = np.zeros(len(values), dtype=bool)
    for event in truth:
        if event["type"] in ATTACK_TYPES:
            yes |= np.abs(times - event["time"]) <= LABEL_SECONDS
    return {"name": name, "z": z[play], "yes": yes[play], "match": match, "size": size,
            "truth": truth, "kickoff": kickoff, "pauses": pauses}


def fit(matches):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(
        np.vstack([m["z"] for m in matches]), np.concatenate([m["yes"] for m in matches]))
    return {name: round(float(w), 3) for name, w in zip(NAMES, model.coef_[0])}


def main():
    from match_events import _clock, _clock_range, _silent

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fit", type=Path, nargs="+", required=True, metavar="VIDEO",
                        help="matches with tracks, ball and a truth list")
    parser.add_argument("--kickoff", type=_clock, nargs="*", default=[],
                        help="kick-off time of each video, in the same order")
    parser.add_argument("--pause", type=_clock_range, nargs="*", default=[],
                        help="one stretch that is not play per video (0-0 for none)")
    args = parser.parse_args()
    matches = []
    for i, video in enumerate(args.fit):
        kickoff = args.kickoff[i] if i < len(args.kickoff) else 0.0
        pauses = [args.pause[i]] if i < len(args.pause) and args.pause[i][1] > 0 else []
        matches.append(_labelled(video, kickoff, pauses))
    if len(matches) >= 2:
        print("Weights from the other matches, used on a match they have not seen:")
        for held in matches:
            weights = fit([m for m in matches if m is not held])
            found = attacks(held["match"], *held["size"],
                            lambda s: _silent(s, held["kickoff"], held["pauses"]), weights)
            real = [e["time"] for e in held["truth"] if e["type"] in ATTACK_TYPES]
            right = sum(any(-6 <= t - s <= 12 for t in real) for s, _ in found)
            print(f"  {held['name']}: {right}/{len(found)} lines right   (weights {weights})")
    print("WEIGHTS =", fit(matches))


if __name__ == "__main__":
    main()
