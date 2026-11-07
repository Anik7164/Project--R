"""
Open-play moments for commentary: passes, long passes, tackles,
interceptions and runs with the ball, plus a match introduction that
names the two teams by their kit colours.

They come from ball possession (match_events.find_touches): the player
whose feet are nearest the ball, frame by frame.

  pass            touch by A, then by a teammate B within PASS_SECONDS,
                  the ball travelling at least PASS_MIN_HEIGHTS
  long_pass       a pass over LONG_PASS_HEIGHTS (about 20 m)
  tackle_won      the ball changes team with the two players right next to
                  each other (within TACKLE_HEIGHTS) and almost no gap
  interception    the ball changes team after a short flight
  run_with_ball   one player keeps the ball for RUN_SECONDS and carries it
                  RUN_HEIGHTS or more

Distances are in player heights (box height, about 1.8 m), so they work
at any zoom. The ball position is the camera-motion-corrected one (sx,
sy) and is only compared inside one camera shot.

A broadcast commentator does not name every pass, so the moments are
thinned out: at most one every MIN_GAP_SECONDS, none close to a bigger
event (goal, card, foul, ...), the most interesting kind first.
"""

import numpy as np

PASS_SECONDS = 3.0
PASS_MIN_HEIGHTS = 3.0
LONG_PASS_HEIGHTS = 12.0
TACKLE_GAP_SECONDS = 0.4
TACKLE_HEIGHTS = 1.5
INTERCEPTION_SECONDS = 2.5
WIN_KEEP_SECONDS = 0.6
REVERSAL_SECONDS = 1.5
RUN_SECONDS = 2.0
RUN_HEIGHTS = 6.0

# A touch counts here only if the player keeps the ball this long and the
# ball was really seen (not interpolated) in this share of its frames;
# shorter or guessed "touches" are mostly the ball rolling past someone.
MIN_TOUCH_SECONDS = 0.3
MIN_SEEN_SHARE = 0.5

MIN_GAP_SECONDS = 6.0
NEAR_BIG_EVENT_SECONDS = 5.0
# Most interesting first.
PRIORITY = ["tackle_won", "run_with_ball", "interception", "long_pass", "pass"]


def _box(match, frame, track_id):
    return dict(match.tracks.get(frame, [])).get(track_id)


def _height(box):
    return max(1.0, box[3] - box[1])


def _ball(match, frame):
    """(stabilised position, scene) at or just around a frame."""
    for f in (frame, frame - 1, frame + 1, frame - 2, frame + 2):
        if f in match.ball:
            x, y, sx, sy, scene = match.ball[f]
            return np.array([sx, sy]), scene
    return None, None


def _scale(match, frame, *track_ids):
    """Mean box height of these players around a frame."""
    heights = [_height(b) for t in track_ids
               for b in [_box(match, frame, t)] if b is not None]
    return float(np.mean(heights)) if heights else None


def _player(match, role, track_id):
    return {"role": role, "track_id": track_id, "team": match.team(track_id)}


def reliable(match, touches):
    seen = match.ball_detected
    kept = []
    for touch in touches:
        frames = range(touch.start, touch.end + 1)
        if len(frames) < MIN_TOUCH_SECONDS * match.fps:
            continue
        if seen is not None and sum(f in seen for f in frames) < MIN_SEEN_SHARE * len(frames):
            continue
        kept.append(touch)
    return kept


def candidates(match, touches):
    """[(frame, type, players)] for every moment found, before thinning."""
    fps = match.fps
    touches = reliable(match, touches)
    found = []
    for touch in touches:
        # A run: one player keeps the ball and carries it far.
        if (touch.end - touch.start) / fps >= RUN_SECONDS:
            start, scene_a = _ball(match, touch.start)
            end, scene_b = _ball(match, touch.end)
            scale = _scale(match, touch.start, touch.track_id)
            if (start is not None and end is not None and scene_a == scene_b
                    and scale and np.linalg.norm(end - start) / scale >= RUN_HEIGHTS):
                found.append((touch.start, "run_with_ball",
                              [_player(match, "runner", touch.track_id)]))

    for i, (a, b) in enumerate(zip(touches, touches[1:])):
        if a.track_id == b.track_id or a.team is None or b.team is None:
            continue
        gap = (b.start - a.end) / fps
        if gap < 0 or gap > PASS_SECONDS:
            continue
        start, scene_a = _ball(match, a.end)
        end, scene_b = _ball(match, b.start)
        if start is None or end is None or scene_a != scene_b:
            continue
        scale = _scale(match, a.end, a.track_id) or _scale(match, b.start, b.track_id)
        if not scale:
            continue
        travel = np.linalg.norm(end - start) / scale

        if a.team == b.team:
            if travel < PASS_MIN_HEIGHTS:
                continue  # one player split into two IDs, or a scramble
            kind = "long_pass" if travel >= LONG_PASS_HEIGHTS else "pass"
            found.append((a.end, kind, [_player(match, "passer", a.track_id),
                                        _player(match, "receiver", b.track_id)]))
            continue

        # The ball changed team. In a duel it goes back and forth; only a
        # change that lasts (the winner keeps it, no quick reversal) counts.
        if (b.end - b.start) / fps < WIN_KEEP_SECONDS:
            continue
        earlier = touches[i - 1] if i >= 1 else None
        if (earlier is not None and earlier.team == b.team
                and (a.start - earlier.end) / fps <= REVERSAL_SECONDS):
            continue  # winning the ball straight back in a duel
        later = touches[i + 2] if i + 2 < len(touches) else None
        if (later is not None and later.team == a.team
                and (later.start - b.end) / fps <= REVERSAL_SECONDS):
            continue
        box_a, box_b = _box(match, b.start, a.track_id), _box(match, b.start, b.track_id)
        if box_a is not None and box_b is not None:
            feet_a = np.array([(box_a[0] + box_a[2]) / 2, box_a[3]])
            feet_b = np.array([(box_b[0] + box_b[2]) / 2, box_b[3]])
            close = np.linalg.norm(feet_a - feet_b) / scale <= TACKLE_HEIGHTS
        else:
            close = False
        if gap <= TACKLE_GAP_SECONDS and close:
            found.append((b.start, "tackle_won",
                          [_player(match, "tackler", b.track_id),
                           _player(match, "opponent", a.track_id)]))
        elif gap <= INTERCEPTION_SECONDS and travel >= PASS_MIN_HEIGHTS:
            found.append((b.start, "interception",
                          [_player(match, "interceptor", b.track_id),
                           _player(match, "passer", a.track_id)]))
    return sorted(found, key=lambda c: c[0])


def thin_out(found, fps, busy_frames=()):
    """Keep the most interesting moments, at least MIN_GAP_SECONDS apart and
    away from bigger events (busy_frames)."""
    gap = MIN_GAP_SECONDS * fps
    near = NEAR_BIG_EVENT_SECONDS * fps
    kept = []
    for frame, kind, players in sorted(found, key=lambda c: (PRIORITY.index(c[1]), c[0])):
        if any(abs(frame - f) <= near for f in busy_frames):
            continue
        if any(abs(frame - k[0]) < gap for k in kept):
            continue
        kept.append((frame, kind, players))
    return sorted(kept, key=lambda c: c[0])


def match_intro(match):
    """The two teams by kit colour, for a line at the start; None if the
    colours are not known."""
    colours = match.team_colours or {}
    if len(colours) < 2:
        return None
    team_a, team_b = sorted(colours)[:2]
    return {"team": team_a, "opponent": team_b,
            "team_colour": colours[team_a], "opponent_colour": colours[team_b]}


# Team-level lines for quiet spells: who has the ball. They need neither a
# clear ball in every frame nor a named player, so they stay right on
# weak footage.
SPELL_SECONDS = 10.0
SPELL_SHARE = 0.7          # of the ball time in the spell
SPELL_MIN_BALL_SECONDS = 3.0
SPELL_GAP_SECONDS = 15.0   # from any other line, and between spells
SPELL_SAME_TEAM_SECONDS = 60.0  # "orange keep the ball" not again within a minute


def possession_spells(match, touches, busy_frames=()):
    """[(frame, team)]: spells where one team clearly keeps the ball."""
    fps = match.fps
    touches = [t for t in reliable(match, touches) if t.team is not None]
    if not touches:
        return []
    size, gap = int(SPELL_SECONDS * fps), SPELL_GAP_SECONDS * fps
    spoken = sorted(busy_frames)
    found = []
    for end in range(touches[0].start + size, touches[-1].end + 1, size // 2):
        start = end - size
        time_by_team = {}
        for touch in touches:
            overlap = min(touch.end, end) - max(touch.start, start) + 1
            if overlap > 0:
                time_by_team[touch.team] = time_by_team.get(touch.team, 0) + overlap
        total = sum(time_by_team.values())
        if total < SPELL_MIN_BALL_SECONDS * fps:
            continue
        team = max(time_by_team, key=time_by_team.get)
        if time_by_team[team] < SPELL_SHARE * total:
            continue
        if any(abs(end - f) < gap for f in spoken):
            continue
        if any(t == team and end - f < SPELL_SAME_TEAM_SECONDS * fps for f, t in found):
            continue
        found.append((end, team))
        spoken.append(end)
    return found
