"""The live engine and the web app, without models, a camera or a GPU."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "webapp"))

import audio_cues as ac  # noqa: E402
import live_engine  # noqa: E402
from live_engine import FrameSource, LiveMatch, SoundTrack  # noqa: E402

RATE = ac.SAMPLE_RATE


def _tone(seconds, hz, level=0.2):
    t = np.arange(int(seconds * RATE)) / RATE
    return (level * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _feed(track, audio):
    for first in range(0, len(audio), track.CHUNK):
        track._add(audio[first:first + track.CHUNK])


def test_sound_arriving_in_pieces_gives_the_same_whistles_and_roar():
    audio = np.random.default_rng(0).normal(0, 0.01, 60 * RATE).astype(np.float32)
    for at in (5, 20.8, 33, 47):       # 20.8 s: across the join of two 3 s pieces
        first = int(at * RATE)
        audio[first:first + RATE // 2] += _tone(0.5, 4000)
    audio[40 * RATE:44 * RATE] *= 2.5  # the crowd, 8 dB up for 4 s
    track = SoundTrack([], lambda: 0.0, live=True)
    _feed(track, audio)

    cues = track.cues(60.0)
    whole = ac.find_whistles(audio)
    assert [round(c["time"]) for c in cues if c["type"] == "whistle"] == \
        [round(c["time"]) for c in whole] == [5, 21, 33, 47]
    [roar] = [c for c in cues if c["type"] == "crowd_roar"]
    assert 39.5 <= roar["time"] <= 40.5 and roar["strength"] > 6
    # Only what has been heard by now.
    assert [round(c["time"]) for c in track.cues(30.0)] == [5, 21]


def test_sound_of_a_file_started_part_way_keeps_match_time():
    audio = np.random.default_rng(1).normal(0, 0.01, 10 * RATE).astype(np.float32)
    audio[4 * RATE:4 * RATE + RATE // 2] += _tone(0.5, 3600)
    track = SoundTrack([], lambda: 0.0, live=False, start=100.0)
    _feed(track, audio)
    assert [round(c["time"]) for c in track.cues(200.0)] == [104]


def _match(**settings):
    said = []
    match = LiveMatch({"kind": "camera", "index": 0}, {"voice": "off", **settings},
                      said.append, lambda data: None)
    match.frames_done = 1
    return match, said


def _lines(said):
    return [m["line"] for m in said if m["type"] == "line"]


def test_silence_gets_a_filler_line_but_not_before_kickoff_or_in_a_break():
    match, said = _match()
    match.kickoff = 30.0
    match.clock = 20.0
    match._tick()
    assert not _lines(said)                 # before kick-off
    match.clock = 45.0
    match._tick()
    [line] = _lines(said)
    assert line["type"] == "filler" and line["minute"] == 1
    match.clock = 50.0
    match._tick()
    assert len(_lines(said)) == 1           # said 5 s ago: not yet again
    match.operator("half_time")
    match.clock = 80.0
    match._tick()
    assert [l["type"] for l in _lines(said)] == ["filler", "half_time"]
    match.operator("second_half")
    match.clock = 95.0
    match._tick()
    assert _lines(said)[-1]["type"] == "filler" and _lines(said)[-1]["minute"] == 46


def test_operator_goal_counts_the_score_and_names_the_team():
    match, said = _match()
    match.team_colours = {"team_a": "orange", "team_b": "white"}
    match.clock = 600.0
    match.operator("goal", "team_b")
    match.operator("goal", "team_b")
    match.operator("yellow_card", "team_a")
    goal_1, goal_2, card = _lines(said)
    assert match.score == {"team_a": 0, "team_b": 2}
    assert "সাদা" in goal_1["text"] and "০-১" in goal_1["text"] and "০-২" in goal_2["text"]
    assert "কমলা" in card["text"] and "-কে" not in card["text"]
    match.operator("undo_goal", "team_b")
    assert match.score["team_b"] == 1
    with pytest.raises(ValueError):
        match.operator("goal")              # which team?


def test_a_roar_becomes_one_line_once_it_is_settled():
    match, said = _match(filler=False)

    class Sound:
        samples, error = 1, None

        def cues(self, until):
            return [c for c in [{"time": 100.0, "type": "crowd_roar", "strength": 5.0,
                                 "duration": 2.0}] if c["time"] <= until]

    match.sound = Sound()
    for now in (101.0, 105.0):              # too early: something bigger may follow
        match.clock = now
        match._tick()
    assert not _lines(said)
    for now in (110.0, 112.0, 114.0):       # later looks find it again: said once
        match.clock = now
        match._tick()
    [line] = _lines(said)
    assert line["type"] == "big_chance" and line["time"] == 100.0 and line["delay"] == 10.0


def test_a_still_ball_is_no_free_kick_without_the_whistle(monkeypatch):
    match, said = _match(filler=False)
    kick = {"frame": 1200, "time": 100.0, "minute": 2, "type": "freekick", "players": [],
            "confidence": 0.4, "source": "rules"}
    monkeypatch.setattr(live_engine.rules, "events_from_rules", lambda *a, **k: [dict(kick)])

    class Sound:
        samples, error = 1, None
        heard = []

        def cues(self, until):
            return list(self.heard)

    match.sound = Sound()
    match.clock = 104.0
    match._tick()
    assert not _lines(said)
    Sound.heard = [{"time": 95.0, "type": "whistle", "strength": 20.0, "duration": 0.5,
                    "pitch": 4000}]
    match.clock = 106.0
    match._tick()
    assert [l["type"] for l in _lines(said)] == ["freekick"]


def test_team_names_stay_put_when_the_colours_are_fitted_again():
    from test_teams import sim_samples

    samples = sim_samples(frames=900)
    match, _ = _match()
    match.samples.extend(samples[:60])
    match._refit_teams(10.0)
    first = {tid: d["team"] for tid, d in match.teams.items() if d["role"] == "player"}
    assert set(first.values()) == {"team_a", "team_b"}
    assert set(match.team_colours) == {"team_a", "team_b"}
    for more in range(60, len(samples), 30):   # k-means may name the groups the other way round
        match.samples.extend(samples[more:more + 30])
        match.last_fit = None
        match._refit_teams(100.0)
        now = {tid: d["team"] for tid, d in match.teams.items() if d["role"] == "player"}
        assert sum(now[tid] == team for tid, team in first.items()) >= 0.9 * len(first)


def test_a_video_is_handed_out_at_the_analysed_rate(tmp_path):
    import cv2

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    for i in range(40):                     # 2 s at 20 frames a second
        writer.write(np.full((48, 64, 3), i * 5, dtype=np.uint8))
    writer.release()
    source = FrameSource({"kind": "file", "path": path}, process_fps=5.0, realtime=False)
    source.start()
    times = []
    while True:
        item = source.get(timeout=1.0)
        if item is None:
            if source.ended:
                break
            continue
        times.append(round(item[0], 2))
    assert times == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]


def test_web_app_answers_and_guards_the_operator(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import server

    monkeypatch.setattr(server, "VIDEO_DIR", tmp_path)
    (tmp_path / "match9.mp4").write_bytes(b"x")
    with TestClient(server.app) as client:
        state = client.get("/api/state").json()
        assert state["operator"] and [v["name"] for v in state["videos"]] == ["match9.mp4"]
        assert client.get("/").status_code == 200
        assert client.post("/api/start", json={"source": {"kind": "file", "name": "../x.mp4"}}
                           ).status_code == 400
        assert client.post("/api/start", json={"source": {"kind": "url", "url": "file:///etc"}}
                           ).status_code == 400
        assert client.post("/api/operator", json={"action": "kickoff"}).status_code == 409
        # An upload never writes over a video that is already there.
        sent = client.post("/api/upload", files={"file": ("match9.mp4", b"new", "video/mp4")})
        assert sent.json()["name"] == "match9 (2).mp4"
        assert (tmp_path / "match9.mp4").read_bytes() == b"x"

        monkeypatch.setattr(server.hub, "key", "secret")
        assert not client.get("/api/state").json()["operator"]
        assert client.post("/api/stop").status_code == 403
        assert client.post("/api/stop", headers={"X-Key": "secret"}).status_code == 200


def _standing(match, first, last, fps=12.0):
    """Six players standing still from `first` to `last` seconds, running otherwise."""
    for frame in range(int(first * fps), int(last * fps)):
        match.tracks[frame] = [(i, (100 * i, 200, 100 * i + 40, 300)) for i in range(1, 7)]


def _running(match, first, last, fps=12.0):
    for frame in range(int(first * fps), int(last * fps)):
        shift = int(40 * (frame / fps))
        match.tracks[frame] = [(i, (100 * i + shift % 900, 200, 100 * i + 40 + shift % 900, 300))
                               for i in range(1, 7)]


def test_shootout_kick_is_said_after_the_wait_ends_not_during_it():
    match, said = _match(filler=False)
    match.clock = 5.0
    match.operator("shootout")
    _running(match, 0, 10)
    _standing(match, 10, 40)                # the taker and keeper wait: still going on
    match.clock = 40.0
    match._tick()
    assert [l["type"] for l in _lines(said)] == ["shootout_start"]
    _running(match, 40, 50)                 # the kick at 0:40, then everybody moves
    match.clock = 50.0
    match._tick()
    match.clock = 52.0
    match._tick()                           # the same kick, seen again
    kinds = [l["type"] for l in _lines(said)]
    assert kinds == ["shootout_start", "shootout_kick"]
    assert 38.0 <= _lines(said)[-1]["time"] <= 42.0
    assert match.phase() == "shootout"
