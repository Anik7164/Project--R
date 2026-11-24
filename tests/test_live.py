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
    # A match is in the picture (six people), and it is not a wall-clock camera.
    match.watch.live = False
    for _ in range(3):
        match.watch.frame(0.0, None, 6)
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


def test_cameras_are_set_up_first_and_kickoff_starts_the_commentary():
    MAX_SILENCE = live_engine.filler.MAX_SILENCE

    match, said = _match(wait_kickoff=True)
    match.state = "running"
    _running(match, 0, 10)
    _standing(match, 10, 40)                # something happens while the cameras are set up
    _running(match, 40, 52)
    for clock in (40.0, 50.0, 52.0):
        match.clock = clock
        match._tick()
    assert not _lines(said)
    stats = match.stats()
    assert stats["waiting_kickoff"] and stats["hold"] == live_engine.HOLD["kickoff"]
    assert match.phase() == "before"
    match.operator("kickoff")               # at 0:52
    match.clock = 56.0
    match._tick()
    assert [l["type"] for l in _lines(said)] == ["kickoff"]   # nothing from before it
    assert not match.stats()["waiting_kickoff"] and match.minute(56.0) == 1
    match.clock = 52.0 + MAX_SILENCE + 1
    match._tick()
    assert _lines(said)[-1]["type"] == "filler"


def test_cameras_change_while_the_match_runs():
    import threading
    import time

    import cv2

    ok, jpeg = cv2.imencode(".jpg", np.full((90, 160, 3), 120, dtype=np.uint8))
    feeds = {name: live_engine.PushFeed(name) for name in "ABC"}
    spec = {name: {"kind": "phone", "feed": feed, "id": name, "label": f"phone {name}"}
            for name, feed in feeds.items()}
    match = live_engine.LiveMatch(spec["A"], {}, lambda item: None, lambda item: None,
                                  others=[spec["B"]])

    def phones():
        while not match.stopping.is_set():
            for feed in feeds.values():
                feed.push(jpeg.tobytes())
            time.sleep(0.05)

    threading.Thread(target=phones, daemon=True).start()
    match.source = live_engine.FrameSource(spec["A"], 12.0, True)
    match.source.start()
    match._start_other(2, spec["B"])
    try:
        numbers = lambda: [c["number"] for c in match.stats()["cameras"]]   # noqa: E731
        match.put_on_air(2)
        match.set_cameras(spec["A"], [])                # 2 cameras -> 1: the split view goes
        assert numbers() == [1] and match.others == {} and match.on_air == 1
        match.set_cameras(spec["A"], [spec["B"], spec["C"]])   # -> 3
        assert numbers() == [1, 2, 3] and set(match.others) == {2, 3}
        two = match.others[2]
        match.set_cameras(spec["A"], [spec["B"]])       # camera 2 unchanged: not opened again
        assert match.others[2] is two and numbers() == [1, 2]
        zero = match.source.zero
        match.set_cameras(spec["C"], [spec["B"]])       # another camera 1, the same clock
        assert match.spec is spec["C"] and match.source.spec["feed"] is feeds["C"]
        assert match.source.zero == zero
        dead = {"kind": "phone", "feed": live_engine.PushFeed("D"), "id": "D", "label": "phone D"}
        with pytest.raises(ValueError):
            match.set_cameras(dead, [spec["B"]], wait=0.5)   # sends nothing: camera 1 stays
        assert match.spec is spec["C"]
    finally:
        match.stopping.set()
        match.source.stopping.set()


# ----------------------------------------------------------------------
# Nothing is said unless a match is in the picture
# ----------------------------------------------------------------------

def _noise(seed, size=(90, 160)):
    return np.random.default_rng(seed).integers(0, 255, (*size, 3), dtype=np.uint8)


def test_picture_watch_tells_why_there_is_no_match_to_talk_about():
    from live_engine import PictureWatch

    watch = PictureWatch(need=2, live=True)
    assert watch.state(wall=0.0) == "none"                 # no frame yet

    for i in range(30):                                    # 3 s of black: a covered lens
        watch.frame(i / 10, np.zeros((90, 160, 3), np.uint8), 4, wall=i / 10)
    assert watch.state(wall=3.0) == "blank"

    watch = PictureWatch(need=2, live=True)
    still = _noise(0)
    for i in range(70):                                    # 7 s of the very same picture
        watch.frame(i / 10, still, 4, wall=i / 10)
    assert watch.state(wall=7.0) == "frozen"

    watch = PictureWatch(need=2, live=True)
    for i in range(300):                                   # a desk: one person, no ball
        watch.frame(i / 10, _noise(i), 1, wall=i / 10)
    assert watch.state(wall=30.0) == "empty"
    for i in range(300, 310):                              # a lamp the ball model takes for a ball
        watch.frame(i / 10, _noise(i), [0.9], wall=i / 10, ball=i % 2 == 0)
    assert watch.state(wall=31.0) == "empty"
    for i in range(310, 400):                              # people sitting close to a webcam
        watch.frame(i / 10, _noise(i), [0.8, 0.7, 0.6, 0.3], wall=i / 10, ball=i % 3 == 0)
    assert watch.state(wall=40.0) == "empty"
    for i in range(400, 480):                              # the camera turns to the pitch
        watch.frame(i / 10, _noise(i), [0.2, 0.15, 0.1, 0.12, 0.5], wall=i / 10)
    assert watch.state(wall=48.0) == "ok"
    assert watch.state(wall=58.0) == "none"                # the camera went quiet

    watch = PictureWatch(need=2, live=False)               # a video: no wall clock
    for i in range(3):
        watch.frame(i / 10, _noise(i), 2)
    assert watch.state(wall=1e9) == "ok"
    assert PictureWatch(need=0, live=False).state() == "none"


def _no_match_in_view(match, first, last):
    """The camera shows a wall from `first` to `last` seconds."""
    for i in range(int(first * 5), int(last * 5)):
        match.watch.frame(i / 5, _noise(i), 0)


def test_no_line_at_all_while_the_camera_shows_no_match():
    match, said = _match()
    _no_match_in_view(match, 0, 60)
    for now in (30.0, 45.0, 60.0):          # silences that would get a filler line
        match.clock = now
        match._tick()
    assert not _lines(said) and match.phase() == "hold"
    stats = [m for m in said if m["type"] == "stats"][-1]
    assert stats["picture"] == "empty"

    # A roar heard while nothing was in the picture is not said later either.
    class Sound:
        samples, error = 1, None

        def cues(self, until):
            return [c for c in [{"time": 50.0, "type": "crowd_roar", "strength": 5.0,
                                 "duration": 2.0}] if c["time"] <= until]

    match.sound = Sound()
    for i in range(40):                      # the match comes into view at 1:01 (8 s of it)
        match.watch.frame(61.0 + i / 5, _noise(1000 + i), 6)
    match.clock = 69.0
    match._tick()
    assert match.picture == "ok" and not _lines(said)
    match.clock = 82.0
    match._tick()                            # play goes on: the filler is back
    assert [l["type"] for l in _lines(said)] == ["filler"]


def test_before_the_first_picture_nothing_is_said_and_the_buttons_refuse():
    said = []
    match = LiveMatch({"kind": "camera", "index": 0}, {"voice": "off"}, said.append,
                      lambda data: None)
    match.clock = 30.0
    match._tick()                            # the camera has not sent a picture
    assert not _lines(said) and match.picture == "none"
    with pytest.raises(ValueError, match="No picture"):
        match.operator("kickoff")
    match.operator("pause")                  # says nothing: allowed
    assert not _lines(said)


def test_stop_drops_the_lines_that_were_still_waiting():
    match, said = _match()

    class Voice:
        dropped = False

        def backlog(self):
            return 0

        def say(self, line):
            pass

        def drop(self):
            self.dropped = True

    match.speaker = Voice()
    match.stop()
    match.clock = 45.0
    match._tick()                            # a look that was already on its way
    assert not _lines(said) and match.speaker.dropped
    assert match.stats()["cut"]


def test_a_phone_sends_pictures_in_and_they_come_out_as_a_camera():
    import time

    import cv2

    from live_engine import PushFeed

    feed = PushFeed("pitch side")
    source = FrameSource({"kind": "phone", "feed": feed}, process_fps=50.0)
    assert source.live and source.describe() == "phone camera pitch side"
    source.start()
    got = []
    for i in range(4):
        ok, jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), 40 + 50 * i, dtype=np.uint8))
        feed.push(jpeg.tobytes())
        item = source.get(timeout=2.0)
        assert item is not None
        time.sleep(0.05)                    # the camera's next picture, not the same instant
        got.append(int(item[1].mean()))
    source.stopping.set()
    assert [round(value, -1) for value in got] == [40, 90, 140, 190]
    assert source.last_wall is not None


def test_two_cameras_are_chosen_by_hand_and_a_phone_can_be_one(tmp_path, monkeypatch):
    import cv2
    from fastapi.testclient import TestClient

    import server

    monkeypatch.setattr(server, "VIDEO_DIR", tmp_path)
    monkeypatch.setattr(server, "camera_names", lambda: {0: "Laptop camera", 1: "USB camera"})
    started = []

    class FakeMatch:
        state, id, spec, other_specs = "running", "x", None, []

        def __init__(self, spec, settings, publish_json, publish_bytes, second=None, others=None):
            self.spec, self.other_specs = spec, list(others or [])
            started.append((spec, (self.other_specs or [None])[0]))

        def start(self):
            pass

        def stop(self):
            self.state = "ended"

        def hello(self):
            return {"type": "hello", "stats": {}, "lines": [], "notes": []}

    monkeypatch.setattr(live_engine, "LiveMatch", FakeMatch)
    with TestClient(server.app) as client:
        monkeypatch.setattr(server.hub, "match", None)
        labels = [c["label"] for c in client.get("/api/cameras").json()["cameras"]]
        assert labels == ["Camera 0: Laptop camera", "Camera 1: USB camera"]

        # The same device cannot be camera 1 and camera 2.
        same = client.post("/api/start", json={"source": {"kind": "camera", "index": 1},
                                               "second": {"kind": "camera", "index": 1}})
        assert same.status_code == 400 and "same device" in same.json()["detail"]
        assert client.post("/api/start", json={"source": {"kind": "camera", "index": 0},
                                               "second": {"kind": "file", "name": "x.mp4"}}
                           ).status_code == 400

        # Camera 1 is what the operator says it is: here the USB camera, not number 0.
        assert client.post("/api/start", json={"source": {"kind": "camera", "index": 1},
                                               "second": {"kind": "camera", "index": 0}}
                           ).status_code == 200
        spec, second = started[-1]
        assert (spec["index"], second["index"]) == (1, 0)
        assert spec["label"] == "camera 1 (USB camera)"
        # ... and the running match's cameras are not opened a second time for a look.
        assert client.post("/api/camera/preview", json={"kind": "camera", "index": 0}
                           ).status_code == 409
        client.post("/api/stop")

        # A phone that sends its camera shows in the lists and can be either camera.
        with client.websocket_connect("/ws/camera?name=Pitch%20side") as phone:
            told = phone.receive_json()
            assert told["name"] == "Pitch side"
            ok, jpeg = cv2.imencode(".jpg", np.full((90, 160, 3), 120, dtype=np.uint8))
            phone.send_bytes(jpeg.tobytes())
            cameras = client.get("/api/cameras").json()["cameras"]
            assert cameras[-1] == {"id": f"phone:{told['id']}", "label": "Phone camera: Pitch side"}
            look = client.post("/api/camera/preview", json={"kind": "phone", "id": told["id"]})
            assert look.status_code == 200 and look.headers["content-type"] == "image/jpeg"
            assert client.post("/api/start", json={
                "source": {"kind": "phone", "id": told["id"]},
                "second": {"kind": "camera", "index": 0}}).status_code == 200
            assert started[-1][0]["feed"].name == "Pitch side"
            client.post("/api/stop")
        gone = client.post("/api/start", json={"source": {"kind": "phone", "id": told["id"]}})
        assert gone.status_code == 400 and "not connected" in gone.json()["detail"]

        # Only the operator may add a camera.
        monkeypatch.setattr(server.hub, "key", "secret")
        assert client.get("/api/cameras").status_code == 403
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect("/ws/camera?name=x") as phone:
                phone.receive_json()
        assert refused.value.code == 4403
        # The iPhone app's pages come from the app, so the server must let them in.
        allowed = client.get("/api/state", headers={"Origin": "capacitor://localhost"})
        assert allowed.headers["access-control-allow-origin"] == "capacitor://localhost"
        assert "access-control-allow-origin" not in client.get(
            "/api/state", headers={"Origin": "https://example.com"}).headers
        assert client.get("/camera.html").status_code == 200


def test_viewers_need_no_password_and_a_chosen_iphone_starts_filming(tmp_path, monkeypatch):
    import threading

    import cv2
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    import server

    monkeypatch.setattr(server, "VIDEO_DIR", tmp_path)
    monkeypatch.setattr(server, "camera_names", lambda: {0: "Laptop camera"})
    monkeypatch.setattr(server, "save_operator_key", lambda key: None)
    monkeypatch.setattr(server, "wrong_passwords", {})
    started = []

    class FakeMatch:
        state, id, spec, other_specs = "running", "x", None, []

        def __init__(self, spec, settings, publish_json, publish_bytes, second=None, others=None):
            self.spec, self.other_specs = spec, list(others or [])
            started.append(spec)

        def start(self):
            pass

        def stop(self):
            self.state = "ended"

        def hello(self):
            return {"type": "hello", "stats": {}, "lines": [], "notes": []}

    def until(socket_, kind):
        while True:
            message = socket_.receive_json()
            if message.get("type") == kind:
                return message

    monkeypatch.setattr(live_engine, "LiveMatch", FakeMatch)
    ok, jpeg = cv2.imencode(".jpg", np.full((90, 160, 3), 120, dtype=np.uint8))
    op = {"X-Key": "secret"}
    with TestClient(server.app) as client:
        monkeypatch.setattr(server.hub, "match", None)
        monkeypatch.setattr(server.hub, "key", "secret")
        monkeypatch.setattr(server.hub, "apps", {})
        monkeypatch.setattr(server.hub, "camera_tokens", {})
        monkeypatch.setattr(server.hub, "app_feeds", {})
        # Watching needs nothing; the page knows a password makes it the operator.
        state = client.get("/api/state").json()
        assert not state["operator"] and state["password"]
        assert client.get("/api/state", headers=op).json()["operator"]
        # The pairing link only finds the PC: no password in it.
        assert "k=" not in client.get("/connect/info").json()["link"]

        # A watching iPhone with the app open is a camera the operator can choose...
        with client.websocket_connect("/ws?app=abc&name=Goal%20side") as iphone:
            cameras = client.get("/api/cameras", headers=op).json()["cameras"]
            assert cameras[-1] == {"id": "phone:app-abc", "wake": True,
                                   "label": "iPhone: Goal side (app, starts by itself)"}
            # ... and choosing it asks the app to film, with a token for this phone only.
            assert client.post("/api/camera/wake", json={"id": "app-abc"}).status_code == 403
            assert client.post("/api/camera/wake", json={"id": "app-abc"}, headers=op).json() == \
                {"ok": True, "sending": False}
            token = until(iphone, "camera_request")["token"]
            for wrong in (f"/ws/camera?app=abc&token=nope", f"/ws/camera?app=xyz&token={token}"):
                with pytest.raises(WebSocketDisconnect) as refused:
                    with client.websocket_connect(wrong) as camera:
                        camera.receive_json()
                assert refused.value.code == 4403
            with client.websocket_connect(f"/ws/camera?app=abc&token={token}&name=Goal%20side") as camera:
                assert camera.receive_json() == {"id": "app-abc", "name": "Goal side"}
                camera.send_bytes(jpeg.tobytes())
                cameras = client.get("/api/cameras", headers=op).json()["cameras"]
                assert [c["id"] for c in cameras].count("phone:app-abc") == 1
                assert cameras[-1] == {"id": "phone:app-abc", "label": "Phone camera: Goal side"}
            # It stopped filming: it is still there, to be asked again.
            assert client.get("/api/cameras", headers=op).json()["cameras"][-1]["wake"]

            # Start with it as camera 1: the PC asks it and waits for its picture.
            answer = {}

            def start():
                answer["start"] = client.post("/api/start", headers=op, json={
                    "source": {"kind": "phone", "id": "app-abc"}})

            starting = threading.Thread(target=start)
            starting.start()
            token = until(iphone, "camera_request")["token"]
            with client.websocket_connect(f"/ws/camera?app=abc&token={token}&name=Goal%20side") as camera:
                camera.receive_json()
                camera.send_bytes(jpeg.tobytes())
                starting.join(10)
                assert answer["start"].status_code == 200
                assert started[-1]["feed"].name == "Goal side"
                # The same feed each time it films: a match using it goes on when it comes back.
                assert started[-1]["feed"] is server.hub.app_feeds["abc"]
                client.post("/api/stop", headers=op)
        # The app closed: gone from the list, and it cannot be started.
        assert all(c["id"] != "phone:app-abc"
                   for c in client.get("/api/cameras", headers=op).json()["cameras"])
        gone = client.post("/api/start", headers=op, json={"source": {"kind": "phone", "id": "app-abc"}})
        assert gone.status_code == 400 and "not connected" in gone.json()["detail"]

        # The password is changed on the PC itself only.
        assert client.post("/api/password", headers=op, json={"password": "goal2026"}).status_code == 403
        monkeypatch.setattr(server, "LOOPBACK", server.LOOPBACK | {"testclient"})
        assert client.post("/api/password", json={"password": "abc"}).status_code == 400
        assert client.post("/api/password", json={"password": "goal2026"}).json() == {"ok": True}
        monkeypatch.setattr(server, "LOOPBACK", server.LOOPBACK - {"testclient"})
        assert not client.get("/api/state", headers=op).json()["operator"]
        assert client.get("/api/state", headers={"X-Key": "goal2026"}).json()["operator"]

        # Guessing: after 10 wrong passwords even the right one waits a minute.
        for _ in range(10):
            assert not client.get("/api/state", headers={"X-Key": "guess"}).json()["operator"]
        assert not client.get("/api/state", headers={"X-Key": "goal2026"}).json()["operator"]
        server.wrong_passwords.clear()
        assert client.get("/api/state", headers={"X-Key": "goal2026"}).json()["operator"]


def test_an_iphone_the_pc_no_longer_uses_is_told_to_stop(monkeypatch):
    import threading

    import cv2
    from fastapi.testclient import TestClient

    import server

    monkeypatch.setattr(server, "camera_names", lambda: {0: "Laptop camera"})

    class FakeMatch:
        state, id = "running", "x"

        def __init__(self, spec, settings, publish_json, publish_bytes, second=None, others=None):
            self.spec, self.other_specs = spec, list(others or [])
            self.stopping = threading.Event()

        def start(self):
            pass

        def stop(self):
            self.stopping.set()
            self.state = "ended"

        def set_cameras(self, main, others):
            self.spec, self.other_specs = main or self.spec, list(others)

        def hello(self):
            return {"type": "hello", "stats": {}, "lines": [], "notes": []}

    def until(socket_, kind):
        while True:
            message = socket_.receive_json()
            if message.get("type") == kind:
                return message

    monkeypatch.setattr(live_engine, "LiveMatch", FakeMatch)
    ok, jpeg = cv2.imencode(".jpg", np.full((90, 160, 3), 120, dtype=np.uint8))
    with TestClient(server.app) as client:
        for name, value in (("match", None), ("key", None), ("apps", {}), ("camera_tokens", {}),
                            ("app_feeds", {}), ("taken", set()), ("selected", set())):
            monkeypatch.setattr(server.hub, name, value)
        with client.websocket_connect("/ws?app=abc&name=Goal%20side") as iphone:
            client.post("/api/camera/wake", json={"id": "app-abc"})
            token = until(iphone, "camera_request")["token"]
            with client.websocket_connect(f"/ws/camera?app=abc&token={token}&name=Goal%20side") as camera:
                camera.receive_json()
                camera.send_bytes(jpeg.tobytes())
                # Still chosen: it keeps filming. Another device chosen: it is told to stop.
                client.post("/api/camera/selection", json={"ids": ["camera:0", "phone:app-abc"]})
                assert server.hub.taken == {"abc"}
                client.post("/api/camera/selection", json={"ids": ["camera:0"]})
                assert until(iphone, "camera_release") == {"type": "camera_release"}
                assert server.hub.taken == set()

                # In a match as camera 2 it is kept, even when not chosen on a page...
                client.post("/api/camera/selection", json={"ids": ["camera:0", "phone:app-abc"]})
                assert client.post("/api/start", json={
                    "source": {"kind": "camera", "index": 0},
                    "others": [{"kind": "phone", "id": "app-abc"}]}).status_code == 200
                client.post("/api/camera/selection", json={"ids": ["camera:0"]})
                assert server.hub.taken == {"abc"}
                # ... until the match goes back to one camera.
                assert client.post("/api/match/cameras", json={
                    "source": {"kind": "camera", "index": 0}, "others": []}).json() == {"ok": True}
                assert server.hub.match.other_specs == []
                until(iphone, "camera_release")

                # Stop lets every phone of the match go.
                client.post("/api/match/cameras", json={
                    "source": {"kind": "camera", "index": 0},
                    "others": [{"kind": "phone", "id": "app-abc"}]})
                assert server.hub.taken == {"abc"}
                client.post("/api/stop")
                until(iphone, "camera_release")
                assert server.hub.taken == set()
        # No match: cameras cannot be changed in one.
        assert client.post("/api/match/cameras", json={
            "source": {"kind": "camera", "index": 0}}).status_code == 409


def test_three_cameras_side_by_side_and_viewers_see_the_one_on_air(tmp_path, monkeypatch):
    import asyncio
    import struct

    from fastapi.testclient import TestClient

    import server

    monkeypatch.setattr(server, "VIDEO_DIR", tmp_path)
    monkeypatch.setattr(server, "camera_names", lambda: {})
    made = []

    class FakeMatch:
        state, id, on_air = "running", "x", 1

        def __init__(self, spec, settings, publish_json, publish_bytes, second=None, others=None):
            self.spec, self.other_specs = spec, list(others or [])
            made.append(self)

        def start(self):
            pass

        def stop(self):
            self.state = "ended"

        def hello(self):
            return {"type": "hello", "stats": {}, "lines": [], "notes": []}

        def put_on_air(self, number):
            if not 1 <= int(number) <= 1 + len(self.other_specs):
                raise ValueError(f"There is no camera {number} in this match.")
            self.on_air = int(number)

    monkeypatch.setattr(live_engine, "LiveMatch", FakeMatch)
    with TestClient(server.app) as client:
        monkeypatch.setattr(server.hub, "match", None)
        cameras = [{"kind": "camera", "index": i} for i in range(4)]
        # At most 3 cameras, and no device twice.
        assert client.post("/api/start", json={"source": cameras[0], "others": cameras[1:]}
                           ).status_code == 400
        twice = client.post("/api/start", json={"source": cameras[0],
                                                "others": [cameras[1], cameras[0]]})
        assert twice.status_code == 400 and "camera 1 and camera 3" in twice.json()["detail"].lower()
        assert client.post("/api/start", json={"source": cameras[0], "others": cameras[1:3]}
                           ).status_code == 200
        match = made[-1]
        assert [spec["index"] for spec in match.other_specs] == [1, 2]
        assert client.post("/api/camera/preview", json=cameras[2]).status_code == 409

        # The operator chooses the camera on air.
        assert client.post("/api/operator", json={"action": "on_air", "camera": 4}).status_code == 400
        assert client.post("/api/operator", json={"action": "on_air", "camera": 3}).json()["on_air"] == 3
        assert match.on_air == 3

    # The operator gets all cameras; a viewer and the broadcast page only
    # the one on air, as the main picture (kind 1).
    hub = server.Hub()
    hub.match = match
    hub.key = "secret"
    operator, viewer = asyncio.Queue(), asyncio.Queue()
    hub.clients = {operator: False, viewer: True}
    pictures = {number: bytes([live_engine.PICTURE_KIND[number]]) + struct.pack("<d", 1.0) + b"jpg"
                for number in (1, 2, 3)}
    for number in (1, 2, 3):
        hub._deliver(pictures[number])
    hub._deliver(b"\x02" + struct.pack("<d", 1.0) + b"sound")
    sent_to = lambda queue: [queue.get_nowait() for _ in range(queue.qsize())]
    assert sent_to(operator) == [pictures[1], pictures[2], pictures[3],
                                 b"\x02" + struct.pack("<d", 1.0) + b"sound"]
    assert sent_to(viewer) == [b"\x01" + pictures[3][1:], b"\x02" + struct.pack("<d", 1.0) + b"sound"]


def test_a_camera_that_stops_goes_off_air(tmp_path, monkeypatch):
    match = live_engine.LiveMatch({"kind": "camera", "index": 0}, {}, lambda item: None,
                                  lambda item: None,
                                  others=[{"kind": "camera", "index": 1, "label": "camera 1"},
                                          {"kind": "camera", "index": 2, "label": "camera 2"}])
    assert [c["number"] for c in match.stats()["cameras"]] == [1, 2, 3]
    match.put_on_air(2)
    assert match.stats()["on_air"] == 2
    with pytest.raises(ValueError):
        match.put_on_air(4)
    with pytest.raises(ValueError):
        live_engine.LiveMatch({"kind": "camera", "index": 0}, {}, print, print,
                              others=[{"kind": "camera", "index": i} for i in (1, 2, 3)])

    class DeadCamera:
        error, ended, stopping, live, rate = "unplugged", True, None, True, 0.0

        def __init__(self, *args, **kwargs):
            import threading
            self.stopping = threading.Event()

        def describe(self):
            return "camera 1"

        def start(self):
            pass

        def get(self):
            return None

    monkeypatch.setattr(live_engine, "FrameSource", DeadCamera)
    match._other_loop(2, match.other_specs[0])
    stats = match.stats()
    assert stats["on_air"] == 1 and stats["cameras"][1]["error"] == "unplugged"
    with pytest.raises(ValueError):
        match.put_on_air(2)


def test_a_phone_camera_that_closes_goes_off_air_and_comes_back(monkeypatch):
    import threading
    import time

    import cv2

    monkeypatch.setattr(live_engine, "QUIET_SECONDS", 0.5)
    feed = live_engine.PushFeed("Goal side")
    match = live_engine.LiveMatch({"kind": "camera", "index": 0}, {}, lambda item: None,
                                  lambda item: None,
                                  others=[{"kind": "phone", "feed": feed, "id": "app-abc",
                                           "label": "phone camera Goal side"}])
    notes = []
    monkeypatch.setattr(match, "note", notes.append)
    ok, jpeg = cv2.imencode(".jpg", np.full((90, 160, 3), 120, dtype=np.uint8))
    pushing = threading.Event()
    pushing.set()

    def phone():
        while not match.stopping.is_set():
            if pushing.is_set():
                feed.push(jpeg.tobytes())
            time.sleep(0.05)

    threading.Thread(target=phone, daemon=True).start()
    loop = threading.Thread(target=match._other_loop, args=(2, match.other_specs[0]), daemon=True)
    loop.start()
    time.sleep(0.6)
    match.put_on_air(2)
    pushing.clear()
    feed.close()                       # the iPhone closed its camera
    deadline = time.monotonic() + 5
    while not notes and time.monotonic() < deadline:
        time.sleep(0.05)
    assert notes and "sends no pictures" in notes[0] and match.on_air == 1
    feed.reopen()                      # ... and films again: the same feed
    pushing.set()
    deadline = time.monotonic() + 5
    while len(notes) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "is back" in notes[-1]
    match.put_on_air(2)
    match.stopping.set()
    loop.join(3)


def test_no_picture_reaches_a_page_after_stop():
    import asyncio

    import server

    async def check():
        hub = server.Hub()
        client = server.Outbox()
        hub.clients[client] = False
        hub._deliver(b"\x01" + b"\0" * 8 + b"jpeg")
        assert client.qsize() == 1
        hub._deliver({"type": "stats", "state": "ended", "cut": True})
        assert not client.pictures and client.qsize() == 1     # the stats only
        assert (await client.get())["cut"]

    asyncio.run(check())


def test_https_certificate_is_made_once_for_this_pc(tmp_path):
    pytest.importorskip("cryptography")
    import ssl

    import server

    cert, key = server.certificate("192.168.0.5", tmp_path)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)       # a pair the server can really use
    first = cert.read_bytes()
    assert server.certificate("192.168.0.5", tmp_path)[0].read_bytes() == first
    assert server.certificate("192.168.0.9", tmp_path)[0].read_bytes() != first


def test_the_picture_is_shown_at_its_own_pace_not_the_analysis_pace(tmp_path):
    import cv2

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    for i in range(20):                     # 1 s at 20 frames a second
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    shown = []
    source = FrameSource({"kind": "file", "path": path}, process_fps=4.0, realtime=True,
                         on_picture=lambda at, frame: shown.append(round(at, 2)),
                         display_fps=20.0)
    source.start()
    analysed = []
    while True:
        item = source.get(timeout=1.0)
        if item is None:
            if source.ended:
                break
            continue
        analysed.append(round(item[0], 2))
    assert analysed == [0.0, 0.25, 0.5, 0.75]
    assert len(shown) == 20 and shown[:3] == [0.0, 0.05, 0.1]


def test_the_camera_picture_is_sent_while_the_models_are_still_loading(monkeypatch):
    import threading

    from live_engine import PushFeed

    import cv2

    feed = PushFeed("pitch side")
    pictures, loading = [], threading.Event()
    match = LiveMatch({"kind": "phone", "feed": feed}, {"voice": "off"},
                      lambda message: None, pictures.append)
    monkeypatch.setattr(match, "_load_models", lambda: loading.wait(10))
    match.start()
    ok, jpeg = cv2.imencode(".jpg", np.full((90, 160, 3), 120, dtype=np.uint8))
    for _ in range(20):
        feed.push(jpeg.tobytes())
        if pictures:
            break
        threading.Event().wait(0.1)
    assert pictures and pictures[0][:1] == b"\x01"     # on its way before any analysis
    assert match.state == "starting" and match.frames_done == 0
    assert "Loading the models" in match.message or "Opening" in match.message
    match.stop()
    loading.set()


def test_a_slow_browser_gets_the_newest_picture_not_a_backlog():
    import asyncio
    import struct

    import server

    async def run():
        box = server.Outbox()
        for i in range(10):                          # ten pictures of camera 1 while busy
            box.put_nowait(b"\x01" + struct.pack("<d", i) + b"jpg")
        box.put_nowait(b"\x03" + struct.pack("<d", 9) + b"jpg")
        box.put_nowait({"type": "line"})
        box.put_nowait(b"\x02" + struct.pack("<d", 9) + b"sound")
        return [await box.get() for _ in range(4)]

    sent = asyncio.run(run())
    pictures = [item for item in sent if isinstance(item, bytes) and item[0] in (1, 3)]
    assert len(pictures) == 2 and struct.unpack("<d", pictures[0][1:9])[0] == 9.0
    assert {"type": "line"} in sent and any(isinstance(i, bytes) and i[0] == 2 for i in sent)
