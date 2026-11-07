"""
Commentary moments from the sound (audio_cues.py), and using the sound to
check what the event model claims.

On a local match the picture is weak but the sound is not:

  * a spotted foul, free kick or offside is believed only if the
    referee's whistle was heard around it; a spotted shot or corner only
    if the crowd reacted, unless the model is very sure (SURE_CONFIDENCE).
    Throw-ins and goal kicks make no sound, so they need that certainty;
  * a crowd roar with no other event near it becomes "big_chance": the
    crowd saw something dangerous (the lines say "what a moment" and name
    the attacking team, not "shot" or "goal", so they stay true);
  * a whistle with no other event near it becomes "whistle_stop".

Speaking less but rightly is the point: on footage like this a wrong
"free kick!" is worse than a plain "the referee stops play".
"""

# Event model claims that need the sound to back them up.
NEEDS_WHISTLE = {"foul": (-3.0, 6.0), "offside": (-3.0, 6.0),
                 "freekick": (-40.0, 3.0)}     # (seconds before, after) the event
NEEDS_ROAR = {"missed_shot": (-6.0, 6.0), "corner_kick": (-20.0, 6.0)}
SILENT = {"throw_in", "goal_kick"}
SURE_CONFIDENCE = 0.6

# A cue this close to another event belongs to it.
NEAR_EVENT_SECONDS = 8.0
MIN_ROAR_DB = 3.5
WHISTLE_GAP_SECONDS = 15.0
POSSESSION_LOOKBACK = 4.0   # whose attack: who had the ball just before the roar


def _cues(cues, kind):
    return [c["time"] for c in cues if c["type"] == kind]


def _heard(times, at, window):
    before, after = window
    return any(at + before <= t <= at + after for t in times)


def check_with_sound(events, cues):
    """Drop event-model claims the sound does not back up."""
    events = merge_repeats(events)
    if not cues:
        return events
    whistles, roars = _cues(cues, "whistle"), _cues(cues, "crowd_roar")
    kept = []
    for event in events:
        kind, at = event["type"], event["time"]
        if event.get("source") != "spotting" or event["confidence"] >= SURE_CONFIDENCE:
            kept.append(event)
        elif kind in NEEDS_WHISTLE:
            if _heard(whistles, at, NEEDS_WHISTLE[kind]):
                kept.append(event)
        elif kind in NEEDS_ROAR:
            if _heard(roars, at, NEEDS_ROAR[kind]):
                kept.append(event)
        elif kind not in SILENT:
            kept.append(event)
    return kept


SAME_EVENT_SECONDS = 10.0


def merge_repeats(events):
    """One event model call per moment: the same type again within
    SAME_EVENT_SECONDS (e.g. two "free kick" spots 6 s apart) is one event,
    the surer one."""
    kept = []
    for event in sorted(events, key=lambda e: e["time"]):
        last = next((k for k in reversed(kept) if k["type"] == event["type"]), None)
        if (last is not None and event.get("source") == "spotting"
                and last.get("source") == "spotting"
                and event["time"] - last["time"] <= SAME_EVENT_SECONDS):
            if event["confidence"] > last["confidence"]:
                kept[kept.index(last)] = event
            continue
        kept.append(event)
    return kept


def attacking_team(match, touches, frame):
    """The team with the most ball time in the seconds before a frame."""
    first = frame - int(POSSESSION_LOOKBACK * match.fps)
    time_by_team = {}
    for touch in touches:
        if touch.team is None or touch.end < first or touch.start > frame:
            continue
        overlap = min(touch.end, frame) - max(touch.start, first) + 1
        time_by_team[touch.team] = time_by_team.get(touch.team, 0) + overlap
    if not time_by_team:
        return None
    return max(time_by_team, key=time_by_team.get)


def events_from_sound(match, events, cues, touches, make_event):
    """big_chance and whistle_stop moments where nothing else is said.
    make_event(frame, type, players, confidence, source) -> event."""
    taken = [e["time"] for e in events if e["type"] != "match_intro"]
    found = []
    last_whistle = None
    for cue in sorted(cues, key=lambda c: c["time"]):
        at = cue["time"]
        if any(abs(at - t) <= NEAR_EVENT_SECONDS for t in taken):
            continue
        frame = int(at * match.fps)
        if cue["type"] == "crowd_roar" and cue["strength"] >= MIN_ROAR_DB:
            event = make_event(frame, "big_chance", [], min(1.0, cue["strength"] / 8), "crowd")
            team = attacking_team(match, touches, frame)
            if team:
                event["team"] = team
                event["team_colour"] = (match.team_colours or {}).get(team)
        elif cue["type"] == "whistle":
            if last_whistle is not None and at - last_whistle < WHISTLE_GAP_SECONDS:
                continue
            last_whistle = at
            event = make_event(frame, "whistle_stop", [], 0.8, "whistle")
        else:
            continue
        found.append(event)
        taken.append(at)
    return found
