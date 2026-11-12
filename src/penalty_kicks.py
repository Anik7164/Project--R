"""
Penalty kicks, found from one camera with no trained model.

Before a penalty everybody on the pitch stands and waits: the taker over
the ball, the keeper on the line, the rest outside the box. Nothing else
in a match looks like that for so long. So:

1. player_speed(): how fast the tracked players move, second by second,
   in body heights per second (so it does not matter how far the camera
   is). Median over the players in view.
2. still_stretches(): stretches where that stays under STILL_SPEED for
   STILL_SECONDS or longer. The kick is at the end of the stretch.
3. find(): three or more such kicks, each within SHOOTOUT_GAP_SECONDS of
   the one before, are a penalty shootout. Inside a shootout a strong
   crowd roar far from every kick found is one more kick (the camera was
   cut or the players were not tracked while they waited). A kick on its
   own counts as a penalty in play only if the crowd roars as it is
   taken.

Checked on the one shootout there is footage of (match5, 8 kicks): all 8
found, none in the 11 minutes of open play before it, none in match4.
The numbers below were chosen on that shootout, so they need a second
match to confirm. A penalty in open play has not been seen at all yet;
that part follows the same idea but is untested. Whether the kick went in
is not decided here (the crowd roars for a save too).
"""

from math import hypot

import numpy as np

STILL_SPEED = 0.25              # body heights per second
STILL_SECONDS = 12.0            # players wait at least this long before a penalty
STILL_BRIDGE_SECONDS = 2.0      # a shorter break does not end a stretch
SHOOTOUT_KICKS = 3
SHOOTOUT_GAP_SECONDS = 180.0
SHOOTOUT_AFTER_SECONDS = 60.0   # celebrations after the last kick are not play
ROAR_DB = 5.0
ROAR_NEAR_KICK_SECONDS = 4.0    # a penalty in play: the roar starts this close to it
ROAR_OWN_KICK_SECONDS = 30.0    # in a shootout: a roar this far from every kick is a kick
AFTER_QUIET_SECONDS = 5.0       # players also stand still before a kick-off


def player_speed(match):
    """Median speed of the tracked players for every second of the video,
    in body heights per second; NaN where nobody is tracked from one
    second to the next."""
    first = {}      # (track, second) -> (seconds, x, feet y, height) when first seen in it
    for frame in sorted(match.tracks):
        second = int(frame / match.fps)
        for track_id, (x1, y1, x2, y2) in match.tracks[frame]:
            first.setdefault((track_id, second),
                             (frame / match.fps, (x1 + x2) / 2, y2, y2 - y1))
    if not first:
        return np.zeros(0)
    moves = [[] for _ in range(1 + max(second for _, second in first))]
    for (track_id, second), (at, x, y, height) in first.items():
        later = first.get((track_id, second + 1))
        if later and height > 0 and later[0] > at:
            moves[second].append(hypot(later[1] - x, later[2] - y) / height / (later[0] - at))
    return np.array([np.median(m) if m else np.nan for m in moves])


def still_stretches(speed, limit=STILL_SPEED, shortest=STILL_SECONDS,
                    bridge=STILL_BRIDGE_SECONDS):
    """[(start, end)] in seconds: the players hardly move from start to end."""
    stretches = []
    for second, still in enumerate(np.nan_to_num(speed, nan=np.inf) < limit):
        if not still:
            continue
        if stretches and second - stretches[-1][1] <= bridge:
            stretches[-1][1] = second + 1
        else:
            stretches.append([second, second + 1])
    return [(float(a), float(b)) for a, b in stretches if b - a >= shortest]


def _strong_roars(cues):
    return [c["time"] for c in cues
            if c["type"] == "crowd_roar" and c["strength"] >= ROAR_DB]


def find(match, cues=(), quiet=None):
    """(kicks, shootouts). kicks: [(seconds, "shootout_kick" or
    "penalty_kick")]; shootouts: [(start, end)] in seconds, with every
    kick of that shootout inside. quiet(seconds) -> True where there is
    no play (before kick-off, half-time): nothing is looked for there."""
    quiet = quiet or (lambda seconds: False)
    stretches = [(start, end) for start, end in still_stretches(player_speed(match))
                 if not quiet(end) and not quiet(end - AFTER_QUIET_SECONDS)]
    roars = [t for t in _strong_roars(cues) if not quiet(t)]

    runs = []       # stretches whose kicks follow each other closely
    for stretch in stretches:
        if runs and stretch[1] - runs[-1][-1][1] <= SHOOTOUT_GAP_SECONDS:
            runs[-1].append(stretch)
        else:
            runs.append([stretch])

    kicks, shootouts = [], []
    for run in runs:
        if len(run) < SHOOTOUT_KICKS:
            kicks += [(end, "penalty_kick") for _, end in run
                      if any(abs(t - end) <= ROAR_NEAR_KICK_SECONDS for t in roars)]
            continue
        times = [end for _, end in run]
        for t in roars:
            if run[0][0] <= t <= times[-1] + ROAR_OWN_KICK_SECONDS and all(
                    abs(t - other) >= ROAR_OWN_KICK_SECONDS for other in times):
                times.append(t)
        kicks += [(t, "shootout_kick") for t in times]
        shootouts.append((run[0][0], max(times) + SHOOTOUT_AFTER_SECONDS))
    return sorted(kicks), shootouts
