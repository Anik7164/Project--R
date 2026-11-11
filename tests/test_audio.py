import numpy as np

import audio_cues as ac
from audio_events import check_with_sound, merge_repeats
from match_events import MatchData, Touch
from play_events import possession_spells

RATE = ac.SAMPLE_RATE


def _noise(seconds, level=0.01, seed=0):
    return np.random.default_rng(seed).normal(0, level, int(seconds * RATE)).astype(np.float32)


def _tone(seconds, start_hz, end_hz=None, level=0.2):
    t = np.arange(int(seconds * RATE)) / RATE
    end_hz = start_hz if end_hz is None else end_hz
    phase = 2 * np.pi * (start_hz * t + (end_hz - start_hz) * t ** 2 / (2 * seconds))
    return (level * np.sin(phase)).astype(np.float32)


def test_flat_whistle_found_sliding_voice_not():
    audio = _noise(12)
    for at in (1, 4, 7):  # three blows of the same whistle
        audio[at * RATE:at * RATE + RATE // 2] += _tone(0.5, 4000)
    audio[10 * RATE:10 * RATE + RATE // 2] += _tone(0.5, 3000, 4200)  # a slide
    whistles = ac.find_whistles(audio)
    assert [round(w["time"]) for w in whistles] == [1, 4, 7]
    assert all(abs(w["pitch"] - 4000) < 50 for w in whistles)


def test_crowd_roar_against_the_recent_level():
    levels = np.full(200, -20.0)
    levels[100:106] = -15.0  # 3 s, 5 dB louder
    levels[150] = -10.0      # one loud bang
    [roar] = ac.find_roars(levels)
    assert roar["time"] == 50.0 and roar["strength"] == 5.0


def _ev(t, kind, conf=0.5, source="spotting"):
    return {"time": t, "type": kind, "confidence": conf, "source": source, "frame": int(t * 30)}


def test_sound_backs_up_or_drops_model_events():
    cues = [{"time": 100.0, "type": "whistle"}, {"time": 200.0, "type": "crowd_roar"}]
    events = [_ev(99, "foul"), _ev(300, "foul"), _ev(201, "missed_shot"),
              _ev(50, "throw_in"), _ev(60, "throw_in", 0.8), _ev(400, "goal", 1.0, "scoreboard"),
              _ev(102, "yellow_card"), _ev(500, "red_card", 0.9)]
    kept = [(e["time"], e["type"]) for e in check_with_sound(events, cues)]
    assert kept == [(60, "throw_in"), (99, "foul"), (201, "missed_shot"), (400, "goal"),
                    (500, "red_card")]


def test_repeated_spots_merge():
    merged = merge_repeats([_ev(302, "freekick", 0.62), _ev(308, "freekick", 0.70)])
    assert len(merged) == 1 and merged[0]["confidence"] == 0.70


def test_possession_spell_when_one_team_keeps_the_ball():
    fps = 30.0
    frames = range(0, 600)
    tracks = {f: [(1, (90, 240, 110, 300)), (2, (190, 240, 210, 300))] for f in frames}
    match = MatchData(fps=fps, tracks=tracks,
                      teams={1: {"team": "team_a"}, 2: {"team": "team_a"}},
                      ball={f: (100, 300, 100, 300, 1) for f in frames},
                      team_colours={"team_a": "orange"}, ball_detected=set(frames))
    touches = [Touch(0, 149, 1, "team_a"), Touch(160, 349, 2, "team_a")]
    [(frame, team)] = possession_spells(match, touches)
    assert team == "team_a"
    assert possession_spells(match, touches, busy_frames=[frame]) == []


def test_goal_from_long_roar_and_a_huddle():
    from audio_events import goals_from_sound

    fps = 10.0
    tracks = {}
    for f in range(0, 600):
        # team_a: 5 players close together after 30 s; team_b spread out
        a = [(i, (100 + i * (15 if f > 300 else 120), 200, 110 + i * (15 if f > 300 else 120), 260))
             for i in range(5)]
        b = [(10 + i, (100 + i * 150, 400, 110 + i * 150, 460)) for i in range(5)]
        tracks[f] = a + b
    match = MatchData(fps=fps, tracks=tracks,
                      teams={**{i: {"team": "team_a", "role": "player"} for i in range(5)},
                             **{10 + i: {"team": "team_b", "role": "player"} for i in range(5)}},
                      ball={}, team_colours={"team_a": "orange", "team_b": "white"})
    make = lambda f, t, p, c, s: {"frame": f, "type": t, "players": p, "confidence": c,
                                  "source": s, "time": f / fps}
    roar = {"type": "crowd_roar", "time": 25.0, "duration": 4.0, "strength": 5.0,
            "longest": 3.0, "longest_at": 26.0}
    [goal] = goals_from_sound(match, [roar], make)
    assert goal["team"] == "team_a" and goal["time"] == 26.0
    early = dict(roar, time=5.0, longest_at=5.0)   # no huddle after it
    assert goals_from_sound(match, [early], make) == []
    short = dict(roar, longest=1.5)                # too short for a goal
    assert goals_from_sound(match, [short], make) == []


def test_silent_before_kickoff_and_in_pauses():
    from match_events import _clock, _clock_range, _silent
    assert _clock("1:22") == 82 and _clock("90") == 90
    pause = [_clock_range("7:13-7:25")]
    assert _silent(50, 82) and not _silent(90, 82)
    assert _silent(440, 82, pause) and not _silent(450, 82, pause)
