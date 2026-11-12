"""
Live commentary engine: one camera (or stream, or a video played as if
live) in, Bengali commentary lines and speech out, while the match runs.

The offline scripts in src/ read a whole video. Here the same parts run
one frame at a time and the same rules are asked again every TICK seconds
on the last WINDOW seconds of play:

  every frame   people detection + BoT-SORT + stable IDs (track_players.py),
                ball (ball_tracking.py), kit colour samples (identify_teams.py)
  sound         referee whistle and crowd roar (audio_cues.py), from the
                source's own sound or a microphone, through ffmpeg
  every TICK    teams refitted; set pieces, passes, tackles, possession,
                crowd moments, goals from roar + huddle, penalties
                (match_events.py and friends) on the last WINDOW seconds;
                what is new and settled becomes a line (player_identity.py),
                silences get a filler line (fill_commentary.py)
  speech        each line is spoken (speak_commentary.py voices) and sent on

An event is only said once it is settled: a goal needs the celebration
that follows it, a whistle needs a few seconds to see nothing bigger
happened. That wait (SETTLE) is why the browser shows the picture a few
seconds late: the line then lands on the moment it is about.

Not live yet: cards (they need a scan of every frame around the referee
and were unreliable from one camera anyway), the scoreboard, jersey
numbers, the SoccerNet event model. The operator buttons cover what the
camera cannot see (kick-off, half-time, a goal, a card).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import project_cache  # noqa: E402,F401  keeps downloads/caches in <project>/.cache

import asyncio  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import random  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict, deque  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import audio_cues  # noqa: E402
import audio_events  # noqa: E402
import fill_commentary as filler  # noqa: E402
import match_events as rules  # noqa: E402
import penalty_kicks  # noqa: E402
import player_identity  # noqa: E402
from appearance import BENGALI_COLOUR_NAMES, extract_descriptor  # noqa: E402

TRACKER = ROOT / "botsort_custom.yaml"
LIVE_DIR = ROOT / "outputs" / "live"

TICK = 2.0               # seconds between two looks at the match
WINDOW = 240.0           # seconds of play the rules look at
WINDOW_EDGE = 15.0       # events this close to the window's start are cut off ones
TEAM_REFIT_SECONDS = 30.0
TEAM_SAMPLE_PER_SECOND = 5.0
MAX_TEAM_SAMPLES = 3000  # frames of kit samples kept (about 10 min)
STREAM_WIDTH = 960       # picture sent to the browsers
STREAM_QUALITY = 70

# Seconds after an event before it can be said: what its rule still needs
# to see (see the module text).
SETTLE = {
    "goal": 26.0, "big_chance": 9.0, "whistle_stop": 9.0,
    "penalty_kick": 6.0, "shootout_kick": 3.0, "shootout_start": 0.0,
    "freekick": 2.0, "goal_kick": 2.0, "foul": 1.0,
    "possession_spell": 1.0,
}
DEFAULT_SETTLE = 3.0
# ... and how stale it may be when found (a foul is only known at the
# free kick; a goal after the huddle).
MAX_LATE = {"goal": 60.0, "foul": 40.0, "shootout_start": 240.0, "shootout_kick": 25.0}
DEFAULT_MAX_LATE = 12.0
# The same event found again by a later look.
SAME_EVENT = {"goal": 60.0, "possession_spell": 15.0, "shootout_start": 600.0,
              "big_chance": 12.0, "whistle_stop": 12.0}
DEFAULT_SAME_EVENT = 6.0
# Lines that only describe play give way to anything else.
PLAY_TYPES = {"pass", "long_pass", "run_with_ball", "tackle_won", "interception",
              "possession_spell", "filler"}
PLAY_GAP = 5.0           # between two play lines
SPEECH_BACKLOG = 2       # play lines are not spoken when this many wait


def clock_text(seconds):
    minutes, rest = divmod(int(max(0, seconds)), 60)
    return f"{minutes}:{rest:02d}"


# ----------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------

class FrameSource(threading.Thread):
    """Reads the camera, stream or file and hands out (seconds, frame).

    A camera or stream is live: the newest frame wins and old ones are
    dropped. A file is played at its own speed the same way (realtime),
    or every sampled frame is handed out however long the work takes
    (realtime False: for testing on a slow machine)."""

    def __init__(self, spec, process_fps, realtime=True):
        super().__init__(daemon=True)
        self.spec, self.process_fps, self.realtime = spec, process_fps, realtime
        self.live = spec["kind"] in ("camera", "url")
        self.drop = self.live or realtime
        self.cond = threading.Condition()
        self.slot = None
        self.queue = queue.Queue(maxsize=2)
        self.stopping = threading.Event()
        self.ended = False
        self.error = None
        self.dropped = 0
        self.duration = None

    def _open(self):
        kind = self.spec["kind"]
        if kind == "camera":
            index = int(self.spec.get("index", 0))
            capture = (cv2.VideoCapture(index, cv2.CAP_DSHOW)
                       if sys.platform == "win32" else cv2.VideoCapture(index))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        elif kind == "url":
            capture = cv2.VideoCapture(self.spec["url"])
        else:
            capture = cv2.VideoCapture(str(self.spec["path"]))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open the video source: {self.describe()}")
        return capture

    def describe(self):
        kind = self.spec["kind"]
        if kind == "camera":
            return f"camera {self.spec.get('index', 0)}"
        return self.spec["url"] if kind == "url" else Path(self.spec["path"]).name

    def _offer(self, item):
        if not self.drop:
            while not self.stopping.is_set():
                try:
                    self.queue.put(item, timeout=0.2)
                    return
                except queue.Full:
                    pass
            return
        with self.cond:
            if self.slot is not None:
                self.dropped += 1
            self.slot = item
            self.cond.notify()

    def get(self, timeout=0.5):
        if not self.drop:
            try:
                return self.queue.get(timeout=timeout)
            except queue.Empty:
                return None
        with self.cond:
            if self.slot is None:
                self.cond.wait(timeout)
            item, self.slot = self.slot, None
            return item

    def run(self):
        capture = None
        try:
            capture = self._open()
            step = 1.0 / self.process_fps
            start, next_at = time.monotonic(), 0.0
            if self.live:
                misses = 0
                while not self.stopping.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        misses += 1
                        if misses > 50:
                            raise RuntimeError("The camera or stream stopped sending pictures.")
                        time.sleep(0.1)
                        continue
                    misses = 0
                    at = time.monotonic() - start
                    if at < next_at:
                        continue
                    next_at = max(next_at + step, at)
                    self._offer((at, frame))
            else:
                fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                if not 1.0 <= fps <= 240.0:
                    fps = 25.0
                frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                self.duration = frames / fps if frames else None
                start_at = float(self.spec.get("start", 0.0) or 0.0)
                index = 0
                if start_at > 0:
                    index = int(start_at * fps)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    next_at = index / fps
                while not self.stopping.is_set():
                    at = index / fps
                    index += 1
                    if at + 1e-6 < next_at:
                        if not capture.grab():
                            break
                        continue
                    ok, frame = capture.read()
                    if not ok:
                        break
                    next_at += step
                    if self.realtime:
                        wait = start + (at - start_at) - time.monotonic()
                        if wait > 0 and self.stopping.wait(wait):
                            break
                    self._offer((at, frame))
        except Exception as error:  # shown in the browser
            self.error = str(error)
        finally:
            if capture is not None:
                capture.release()
            self.ended = True
            with self.cond:
                self.cond.notify()


# ----------------------------------------------------------------------
# Sound
# ----------------------------------------------------------------------

class SoundTrack(threading.Thread):
    """The match sound through ffmpeg, as 16 kHz mono: whistles and crowd
    loudness are worked out as it arrives, and it is passed on to the
    browsers."""

    CHUNK = audio_cues.SAMPLE_RATE // 10          # 0.1 s
    TONE_BLOCK = 3 * audio_cues.SAMPLE_RATE       # whistles are looked for in 3 s pieces
    TONE_CARRY = audio_cues.SAMPLE_RATE           # ... overlapping by 1 s

    def __init__(self, input_args, clock, live, start=0.0, on_chunk=None):
        super().__init__(daemon=True)
        self.input_args, self.clock, self.live = input_args, clock, live
        self.start_at = start
        self.on_chunk = on_chunk
        self.lock = threading.Lock()
        self.say_lock = threading.RLock()   # one look or operator button at a time
        self.stopping = threading.Event()
        self.samples = 0
        self.levels = []             # dB per LOUDNESS_STEP
        self.tones = []              # whistle-like tones, every pitch
        self.error = None
        self._loud = np.zeros(0, dtype=np.float32)
        self._tone = np.zeros(0, dtype=np.float32)
        self._tone_offset = 0        # sample index of _tone[0]
        self.process = None

    @property
    def seconds(self):
        return self.start_at + self.samples / audio_cues.SAMPLE_RATE

    def run(self):
        ffmpeg = audio_cues.ffmpeg_path()
        if ffmpeg is None:
            self.error = "ffmpeg not found (pip install imageio-ffmpeg): no sound cues."
            return
        seek = ["-ss", str(self.start_at)] if self.start_at > 0 else []
        command = [ffmpeg, "-loglevel", "error", "-nostdin", *seek, *self.input_args,
                   "-vn", "-ac", "1", "-ar", str(audio_cues.SAMPLE_RATE),
                   "-f", "s16le", "pipe:1"]
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
        except OSError as error:
            self.error = f"ffmpeg could not start: {error}"
            return
        size = self.CHUNK * 2
        try:
            while not self.stopping.is_set():
                data = self.process.stdout.read(size)
                if not data:
                    break
                data = data[:len(data) // 2 * 2]
                at = self.seconds
                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self._add(chunk)
                if self.on_chunk is not None:
                    self.on_chunk(at, data)
                # A file's sound must not run ahead of its picture.
                while (not self.live and not self.stopping.is_set()
                       and self.seconds > self.clock() + 1.0):
                    time.sleep(0.05)
        finally:
            if self.samples == 0 and self.process.poll() not in (None, 0):
                message = self.process.stderr.read().decode("utf-8", "replace").strip()
                self.error = "No sound from the source" + (f": {message[-200:]}" if message else ".")
            self.close()

    def close(self):
        self.stopping.set()
        if self.process is not None and self.process.poll() is None:
            self.process.kill()

    def _add(self, chunk):
        rate = audio_cues.SAMPLE_RATE
        base = self.samples
        self.samples += len(chunk)

        self._loud = np.concatenate([self._loud, chunk])
        block = int(audio_cues.LOUDNESS_STEP * rate)
        if len(self._loud) >= block:
            count = len(self._loud) // block
            levels = audio_cues.loudness(self._loud[:count * block])
            self._loud = self._loud[count * block:]
            with self.lock:
                self.levels.extend(float(v) for v in levels)

        if len(self._tone) == 0:
            self._tone_offset = base
        self._tone = np.concatenate([self._tone, chunk])
        if len(self._tone) >= self.TONE_BLOCK:
            found = audio_cues.find_tones(self._tone, rate)
            offset = self.start_at + self._tone_offset / rate
            limit = (len(self._tone) - self.TONE_CARRY) / rate
            with self.lock:
                for cue in found:
                    if cue["time"] >= limit:
                        continue          # the next piece hears all of it
                    cue = dict(cue, time=round(cue["time"] + offset, 2))
                    last = self.tones[-1] if self.tones else None
                    if last and cue["time"] - (last["time"] + last["duration"]) <= \
                            audio_cues.WHISTLE_MERGE_SECONDS:
                        last["duration"] = round(cue["time"] + cue["duration"] - last["time"], 2)
                        last["strength"] = max(last["strength"], cue["strength"])
                    else:
                        self.tones.append(cue)
            keep = self.TONE_CARRY
            self._tone_offset += len(self._tone) - keep
            self._tone = self._tone[-keep:]

    def cues(self, until):
        """Whistles (the referee's pitch so far) and crowd roars up to now."""
        with self.lock:
            tones = [dict(t) for t in self.tones]
            levels = np.array(self.levels, dtype=np.float64)
        whistles = audio_cues.referee_whistle(tones)
        history = int(600 / audio_cues.LOUDNESS_STEP)
        first = max(0, len(levels) - history)
        roars = audio_cues.find_roars(levels[first:])
        shift = self.start_at + first * audio_cues.LOUDNESS_STEP
        for roar in roars:
            roar["time"] = round(roar["time"] + shift, 2)
            if "longest_at" in roar:
                roar["longest_at"] = round(roar["longest_at"] + shift, 2)
        return sorted((c for c in whistles + roars if c["time"] <= until),
                      key=lambda c: c["time"])


class _CameraRecorder:
    """Keeps the camera motion the stable-ID layer worked out, for the ball."""

    def __init__(self, inner):
        self.inner = inner
        self.last = (None, True)

    def update(self, frame, boxes=()):
        self.last = self.inner.update(frame, boxes)
        return self.last

    def __getattr__(self, name):
        return getattr(self.inner, name)


# ----------------------------------------------------------------------
# Speech
# ----------------------------------------------------------------------

class Speaker(threading.Thread):
    """Speaks lines one after another and saves each as a file the
    browsers fetch. voice: "edge" (online, natural), "mms" (offline) or
    "off"."""

    def __init__(self, voice, speaker, folder, on_ready, on_note):
        super().__init__(daemon=True)
        self.voice_name, self.speaker_name = voice, speaker
        self.folder, self.on_ready, self.on_note = folder, on_ready, on_note
        self.queue = queue.Queue()
        self.voice = None
        self.stopping = threading.Event()

    def backlog(self):
        return self.queue.qsize()

    def say(self, line):
        if self.voice_name != "off":
            self.queue.put(line)

    def _load(self):
        import speak_commentary as speech

        self.speech = speech
        if self.voice_name == "edge":
            try:
                import edge_tts  # noqa: F401
                return "edge"
            except ImportError:
                self.on_note("The edge voice needs: pip install edge-tts. Using the offline voice.")
        try:
            self.mms = speech.MmsVoice()
            return "mms"
        except Exception as error:
            self.on_note(f"No voice available ({type(error).__name__}: {error}). "
                         "Lines are shown without speech.")
            return None

    def _edge(self, text):
        import edge_tts

        voice = self.speech.EDGE_VOICES.get(self.speaker_name, self.speaker_name)

        async def synthesise():
            chunks = []
            async for chunk in edge_tts.Communicate(text, voice, rate="+10%").stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        for attempt in range(2):
            try:
                data = asyncio.run(synthesise())
                if data:
                    return data
            except Exception:
                time.sleep(0.5)
        return None

    def run(self):
        kind = None
        while not self.stopping.is_set():
            try:
                line = self.queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if self.voice is None:
                kind = self._load()
                self.voice = kind or False
            if not kind:
                continue
            started = time.monotonic()
            try:
                self.folder.mkdir(parents=True, exist_ok=True)
                name = None
                if kind == "edge":
                    data = self._edge(line["text"])
                    if data:
                        name = f"{line['id']}.mp3"
                        (self.folder / name).write_bytes(data)
                    else:
                        if not hasattr(self, "mms"):
                            self.on_note("The online voice did not answer; using the offline voice "
                                         "for the lines it misses.")
                            try:
                                self.mms = self.speech.MmsVoice()
                            except Exception:
                                self.mms = None
                        if self.mms is None:
                            continue
                if name is None:
                    name = f"{line['id']}.wav"
                    self.speech.write_wav(self.folder / name, self.mms(line["text"]))
                self.on_ready(line, name, time.monotonic() - started)
            except Exception as error:
                self.on_note(f"Speech failed for one line ({type(error).__name__}: {error}).")


# ----------------------------------------------------------------------
# The match
# ----------------------------------------------------------------------

DEFAULTS = {
    "process_fps": 12.0,     # frames per second looked at
    "realtime": True,        # a file is played at its own speed (frames dropped if slow)
    "voice": "edge",         # edge | mms | off
    "speaker": "male",
    "people_model": "auto",  # auto | football | coco
    "image_size": 960,
    "ball_image_size": 1280,
    "reid": True,
    "penalties": True,
    "filler": True,
    "boxes": True,           # draw players and ball on the picture sent out
    "mic": "",               # ffmpeg sound input for a camera, e.g. audio=Microphone (USB)
    "start": 0.0,            # a file: start this many seconds in
}


class LiveMatch:
    def __init__(self, spec, settings, publish_json, publish_bytes):
        self.spec = spec
        self.settings = {**DEFAULTS, **{k: v for k, v in (settings or {}).items()
                                        if k in DEFAULTS and v is not None}}
        self.publish_json, self.publish_bytes = publish_json, publish_bytes
        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.folder = LIVE_DIR / self.id
        self.fps = float(self.settings["process_fps"])
        self.rng = random.Random()

        self.lock = threading.Lock()
        self.say_lock = threading.RLock()   # one look or operator button at a time
        self.stopping = threading.Event()
        self.state = "starting"       # starting | running | ended | failed
        self.message = "Starting"
        self.clock = 0.0              # seconds of match seen so far
        self.started_wall = time.time()

        # What has been seen (frame index = seconds * fps).
        self.tracks = {}
        self.ball = {}
        self.ball_detected = set()
        self.role_votes = defaultdict(Counter)
        self.samples = deque()
        self.sample_count = Counter()
        self.teams = {}               # track id -> decision of identify_teams
        self.team_colours = {}        # team -> colour name
        self.colour_overrides = {}
        self.classifier = None
        self.last_fit = None
        self.descriptors_at_fit = 0

        # Match clock: set by the operator; without it everything is play.
        self.kickoff = None
        self.second_half = None
        self.pauses = []              # [start, end or None]
        self.full_time = None
        self.shootouts = []
        self.manual_shootout = None
        self.score = {"team_a": 0, "team_b": 0}

        self.lines = []
        self.events = []
        self.recent_texts = deque(maxlen=8)
        self.intro_said = False
        self.last_filler_team = None
        self.cue_counts = {"whistle": 0, "crowd_roar": 0}
        self.notes = []
        self.source = None
        self.sound = None
        self.speaker = None
        self.frames_done = 0
        self.rate = 0.0
        self.templates = {item["event_type"]: item["templates"] for item in json.loads(
            player_identity.DATASET.read_text(encoding="utf-8"))}

    # -- running -------------------------------------------------------

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.stopping.set()

    def note(self, text):
        self.notes.append(text)
        print("[live]", text)
        self.publish_json({"type": "note", "text": text})

    def _status(self, message=None, state=None):
        if state:
            self.state = state
        if message:
            self.message = message
        self.publish_json({"type": "status", "state": self.state, "message": self.message})

    def _run(self):
        analyser = None
        try:
            self._status("Loading the models (the first start takes a while)")
            self._load_models()
            self.source = FrameSource({**self.spec, "start": self.settings["start"]},
                                      self.fps, bool(self.settings["realtime"]))
            self.source.start()
            self._start_sound()
            self.speaker = Speaker(self.settings["voice"], self.settings["speaker"],
                                   self.folder / "audio", self._spoken, self.note)
            self.speaker.start()
            analyser = threading.Thread(target=self._analyse_loop, daemon=True)
            analyser.start()
            self._status(f"Live: {self.source.describe()}", "running")

            last_stats, done_at = time.monotonic(), deque(maxlen=40)
            while not self.stopping.is_set():
                item = self.source.get()
                if item is None:
                    if self.source.ended:
                        break
                    continue
                self._process(*item)
                done_at.append(time.monotonic())
                if len(done_at) > 1:
                    self.rate = (len(done_at) - 1) / max(1e-6, done_at[-1] - done_at[0])
                if time.monotonic() - last_stats >= 1.0:
                    last_stats = time.monotonic()
                    self.publish_json(self.stats())
            if self.source.error:
                raise RuntimeError(self.source.error)
            self._flush_tracks()
            self._tick(final=True)
            self._status("The match feed ended", "ended")
        except Exception as error:
            traceback.print_exc()
            self._status(f"Stopped: {error}", "failed")
        finally:
            self.stopping.set()
            if self.source is not None:
                self.source.stopping.set()
            if self.sound is not None:
                self.sound.close()
            if self.speaker is not None:
                # Let the last lines be spoken before the voice goes.
                deadline = time.monotonic() + 20
                while self.speaker.backlog() and time.monotonic() < deadline:
                    time.sleep(0.2)
                self.speaker.stopping.set()
            self.save()
            self.publish_json(self.stats())

    def _load_models(self):
        import torch
        from ultralytics import YOLO

        import ball_tracking
        import track_players
        from detector_classes import ball_class, default_model, is_football_model, people_classes
        from identity import StablePlayerTracker, overlaps_other
        from reid import load_default as load_reid

        self.tp, self.bt, self.overlaps_other = track_players, ball_tracking, overlaps_other
        self.device = 0 if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.note("No GPU found: running on the CPU, far slower than the match.")
        path = default_model()
        coco_path = ROOT / "yolo26n.pt"
        want = self.settings["people_model"]
        main = YOLO(str(path))
        self.ball_model = YOLO(str(path))
        self.ball_id = ball_class(self.ball_model)
        self.coco = None
        self.probe = None
        if want == "coco" and coco_path.exists():
            main = YOLO(str(coco_path))
        elif want == "auto" and is_football_model(main) and coco_path.exists():
            # Like detector_classes.check_people_model, on the first frames.
            self.coco = YOLO(str(coco_path))
            self.probe = {"frames": 0, "ours": 0, "theirs": 0}
        self.people_model = main
        self.people_classes = people_classes(main)
        self.people_classes_fn = people_classes
        reid = None
        if self.settings["reid"]:
            reid = load_reid("cuda:0" if self.device == 0 else "cpu")
        self.stable = StablePlayerTracker(reid=reid)
        self.stable.camera = _CameraRecorder(self.stable.camera)
        self.delay = self.stable.identity.config.pending_timeout + 2
        self.held = deque()
        self.pending_rows = defaultdict(list)
        self.selector = None
        self.scene, self.cumulative = 0, np.eye(3)[:2]
        self.to_frame = {}
        self.last_ball = None
        self.last_frame = -1
        self.sample_step = max(1, int(round(self.fps / TEAM_SAMPLE_PER_SECOND)))

    def _start_sound(self):
        kind = self.spec["kind"]
        if kind == "camera":
            mic = (self.settings["mic"] or "").strip()
            if not mic:
                self.note("No microphone chosen: whistles and the crowd are not heard, "
                          "so only what the picture shows is said.")
                return
            if sys.platform == "win32":
                args = ["-f", "dshow", "-i", mic if mic.startswith("audio=") else f"audio={mic}"]
            elif sys.platform == "darwin":
                args = ["-f", "avfoundation", "-i", mic]
            else:
                args = ["-f", "pulse", "-i", mic]
        else:
            args = ["-i", self.spec["url"] if kind == "url" else str(self.spec["path"])]
        self.sound = SoundTrack(args, lambda: self.clock, self.source.live,
                                0.0 if self.source.live else float(self.settings["start"]),
                                self._sound_chunk)
        self.sound.start()

    def _sound_chunk(self, at, data):
        self.publish_bytes(b"\x02" + struct.pack("<d", at) + data)

    # -- every frame ---------------------------------------------------

    def _probe_people_model(self, frame):
        """The football model finds nobody on close phone footage; the COCO
        model does. Decided on the first frames."""
        size = int(self.settings["image_size"])
        for model, key in ((self.people_model, "ours"), (self.coco, "theirs")):
            found = model.predict(frame, imgsz=size, conf=0.35, device=self.device,
                                  classes=self.people_classes_fn(model), verbose=False)[0]
            self.probe[key] += len(found.boxes)
        self.probe["frames"] += 1
        if self.probe["frames"] < 8:
            return
        if self.probe["ours"] < 0.4 * self.probe["theirs"]:
            self.people_model = self.coco
            self.people_classes = self.people_classes_fn(self.coco)
            self.note("The football detector finds too few players on this camera: "
                      "using the general model for people (the ball still comes from the football model).")
        self.probe = self.coco = None

    def _process(self, at, frame):
        index = max(self.last_frame + 1, int(round(at * self.fps)))
        self.last_frame = index
        height, width = frame.shape[:2]
        if self.probe is not None:
            self._probe_people_model(frame)

        result = self.people_model.track(
            source=frame, tracker=str(TRACKER), persist=True,
            classes=self.people_classes, conf=self.tp.CONFIDENCE, iou=self.tp.NMS_IOU,
            imgsz=int(self.settings["image_size"]), device=self.device, verbose=False)[0]
        detections = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for bbox, tracker_id, confidence, class_id in zip(
                    boxes.xyxy.cpu().tolist(), boxes.id.int().cpu().tolist(),
                    boxes.conf.cpu().tolist(), boxes.cls.int().cpu().tolist()):
                if not self.tp.validate_box(bbox, width, height)[0]:
                    continue
                x1, y1, x2, y2 = (int(v) for v in bbox)
                x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
                y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append((int(tracker_id), (x1, y1, x2, y2), float(confidence),
                                   self.people_model.names[int(class_id)]))

        keep, assignments, resolved = self.stable.process(
            index, frame, [d[0] for d in detections], [d[1] for d in detections],
            [d[2] for d in detections])
        warp, cut = self.stable.camera.last

        sample = self.frames_done % self.sample_step == 0
        kept_boxes = [detections[i][1] for i in keep]
        rows = []
        for position, (i, player_id) in enumerate(zip(keep, assignments)):
            tracker_id, box, _, class_name = detections[i]
            descriptor = None
            if sample and not self.overlaps_other(position, kept_boxes, 0.15):
                descriptor = extract_descriptor(frame, box)
            rows.append({"track_id": player_id, "tracker_id": tracker_id, "box": box,
                         "class": class_name, "descriptor": descriptor})
        for tracker_id, player_id in resolved.items():
            for row in self.pending_rows.pop(tracker_id, []):
                row["track_id"] = player_id if player_id is not None else -1
        for row in rows:
            if row["track_id"] is None:
                self.pending_rows[row["tracker_id"]].append(row)
        self.held.append((index, rows, sample, height))
        while self.held and self.held[0][0] < index - self.delay:
            self._commit(*self.held.popleft())

        ball = self._find_ball(index, frame, width, warp, cut)
        self.frames_done += 1
        self.clock = at
        self._send_picture(at, frame, rows, ball)

    def _commit(self, index, rows, sample, height):
        from identify_teams import DETECTOR_ROLES, MAX_SAMPLES_PER_PLAYER, ROLE_MAX_HEIGHT

        good = [r for r in rows if r["track_id"] is not None and r["track_id"] >= 0]
        with self.lock:
            self.tracks[index] = [(r["track_id"], r["box"]) for r in good]
            for row in good:
                box = row["box"]
                if row["class"] in DETECTOR_ROLES and box[3] - box[1] <= ROLE_MAX_HEIGHT * height:
                    self.role_votes[row["track_id"]][row["class"]] += 1
            if sample and good:
                ids, boxes, descriptors = [], [], []
                for row in good:
                    descriptor = row["descriptor"]
                    if descriptor is not None and \
                            self.sample_count[row["track_id"]] >= MAX_SAMPLES_PER_PLAYER:
                        descriptor = None
                    if descriptor is not None:
                        self.sample_count[row["track_id"]] += 1
                    ids.append(row["track_id"])
                    boxes.append(row["box"])
                    descriptors.append(descriptor)
                self.samples.append((ids, boxes, descriptors))
                if len(self.samples) > MAX_TEAM_SAMPLES:
                    old_ids, _, old_descriptors = self.samples.popleft()
                    for player_id, descriptor in zip(old_ids, old_descriptors):
                        if descriptor is not None:
                            self.sample_count[player_id] -= 1

    def _flush_tracks(self):
        for tracker_id, player_id in self.stable.flush().items():
            for row in self.pending_rows.pop(tracker_id, []):
                row["track_id"] = player_id if player_id is not None else -1
        while self.held:
            self._commit(*self.held.popleft())

    def _find_ball(self, index, frame, width, warp, cut):
        bt = self.bt
        result = self.ball_model.predict(
            frame, classes=[self.ball_id], conf=bt.CONFIDENCE,
            imgsz=int(self.settings["ball_image_size"]), device=self.device, verbose=False)[0]
        candidates = [(*box, conf) for box, conf in zip(result.boxes.xywh.cpu().tolist(),
                                                        result.boxes.conf.cpu().tolist())]
        if self.selector is None:
            self.selector = bt.BallSelector(width)
        if cut or warp is None:
            self.scene, self.cumulative = self.scene + 1, np.eye(3)[:2]
        else:
            self.cumulative = bt.compose(self.cumulative, warp)
        self.to_frame[index] = self.cumulative
        for old in [f for f in self.to_frame if f < index - 4 * bt.MAX_GAP]:
            del self.to_frame[old]
        ball = self.selector.update(candidates, warp, cut)
        if ball is None:
            return None
        x, y, _ = ball

        def still(px, py, frame_index):
            sx, sy = bt.warp_points([[px, py]], bt.invert(self.to_frame[frame_index]))[0]
            return float(sx), float(sy)

        with self.lock:
            last = self.last_ball
            if last and last[3] == self.scene and 1 < index - last[0] <= bt.MAX_GAP + 1:
                gap = index - last[0]
                for between in sorted(f for f in self.to_frame if last[0] < f < index):
                    share = (between - last[0]) / gap
                    bx, by = last[1] + share * (x - last[1]), last[2] + share * (y - last[2])
                    self.ball[between] = (bx, by, *still(bx, by, between), self.scene)
            self.ball[index] = (float(x), float(y), *still(x, y, index), self.scene)
            self.ball_detected.add(index)
        self.last_ball = (index, float(x), float(y), self.scene)
        return float(x), float(y)

    COLOUR_BGR = {
        "red": (40, 40, 230), "orange": (0, 140, 255), "yellow": (0, 220, 255),
        "green": (60, 170, 60), "sky blue": (235, 190, 110), "blue": (220, 90, 30),
        "purple": (170, 60, 150), "pink": (180, 120, 255), "white": (245, 245, 245),
        "grey": (150, 150, 150), "black": (30, 30, 30),
    }

    def _send_picture(self, at, frame, rows, ball):
        scale = min(1.0, STREAM_WIDTH / frame.shape[1])
        picture = frame if scale == 1.0 else cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if self.settings["boxes"]:
            if scale == 1.0:
                picture = picture.copy()
            for row in rows:
                info = self.teams.get(row["track_id"]) or {}
                colour = self.COLOUR_BGR.get(info.get("team_colour"), (200, 200, 200))
                x1, y1, x2, y2 = (int(v * scale) for v in row["box"])
                cv2.rectangle(picture, (x1, y1), (x2, y2), (0, 0, 0), 3)
                cv2.rectangle(picture, (x1, y1), (x2, y2), colour, 1)
                if info.get("role") == "referee":
                    cv2.putText(picture, "REF", (x1, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if ball is not None:
                centre = (int(ball[0] * scale), int(ball[1] * scale))
                cv2.circle(picture, centre, 9, (0, 0, 0), 3)
                cv2.circle(picture, centre, 9, (0, 255, 255), 1)
        ok, jpeg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        if ok:
            self.publish_bytes(b"\x01" + struct.pack("<d", at) + jpeg.tobytes())

    # -- match clock ---------------------------------------------------

    def silent(self, seconds):
        """Nothing is said before kick-off, in a break or after full time."""
        if self.kickoff is not None and seconds < self.kickoff:
            return True
        if self.full_time is not None and seconds >= self.full_time:
            return True
        return any(start <= seconds and (end is None or seconds <= end)
                   for start, end in self.pauses)

    def in_shootout(self, seconds):
        return any(start <= seconds <= end for start, end in self.shootouts)

    def minute(self, seconds):
        if self.second_half is not None and seconds >= self.second_half:
            return 45 + int((seconds - self.second_half) // 60) + 1
        start = self.kickoff if self.kickoff is not None else float(self.settings["start"])
        return int(max(0.0, seconds - start) // 60) + 1

    def phase(self):
        now = self.clock
        if self.full_time is not None and now >= self.full_time:
            return "full_time"
        if self.kickoff is not None and now < self.kickoff:
            return "before"
        if any(start <= now and (end is None or now <= end) for start, end in self.pauses):
            return "break"
        if self.in_shootout(now):
            return "shootout"
        return "play"

    # -- teams ---------------------------------------------------------

    def _refit_teams(self, now):
        from identify_teams import MIN_ROLE_VOTES, ROLE_SHARE, identify_teams
        from team_identification import MIN_SAMPLES, TEAM_LABELS

        with self.lock:
            samples = list(self.samples)
            votes = {tid: Counter(v) for tid, v in self.role_votes.items()}
        count = sum(1 for _, _, ds in samples for d in ds if d is not None)
        if count < MIN_SAMPLES:
            return
        if self.last_fit is not None and (now - self.last_fit < TEAM_REFIT_SECONDS
                                          or count == self.descriptors_at_fit):
            return
        roles = {}
        for track_id, counts in votes.items():
            role, most = counts.most_common(1)[0]
            if most >= MIN_ROLE_VOTES and most >= ROLE_SHARE * sum(counts.values()):
                roles[track_id] = role
        try:
            classifier, decisions = identify_teams(samples, roles)
        except ValueError:
            return
        # k-means names its two groups in any order: keep "team_a" the
        # team it was at the last fit.
        if self.classifier is not None:
            old, new = self.classifier.centres, classifier.centres
            same = np.linalg.norm(old[0] - new[0]) + np.linalg.norm(old[1] - new[1])
            swapped = np.linalg.norm(old[0] - new[1]) + np.linalg.norm(old[1] - new[0])
            if swapped < same:
                a, b = TEAM_LABELS
                flip = {a: b, b: a}
                classifier.centres = new[::-1].copy()
                classifier.colours = {flip[t]: c for t, c in classifier.colours.items()}
                for decision in decisions.values():
                    if decision["team"] in flip:
                        decision["team"] = flip[decision["team"]]
        colours = {**classifier.colours, **self.colour_overrides}
        for decision in decisions.values():
            if decision["role"] == "player" and decision["team"] in colours:
                decision["team_colour"] = colours[decision["team"]]
                decision["team_colour_bn"] = BENGALI_COLOUR_NAMES.get(colours[decision["team"]])
        with self.lock:
            self.classifier = classifier
            self.teams.update(decisions)
            self.team_colours = {t: c for t, c in colours.items() if c}
        self.last_fit, self.descriptors_at_fit = now, count

    def set_colour(self, team, colour):
        if colour not in BENGALI_COLOUR_NAMES or team not in ("team_a", "team_b"):
            raise ValueError("unknown team or colour")
        with self.lock:
            self.colour_overrides[team] = colour
            self.team_colours[team] = colour
            for decision in self.teams.values():
                if decision.get("team") == team and decision.get("role") == "player":
                    decision["team_colour"] = colour
                    decision["team_colour_bn"] = BENGALI_COLOUR_NAMES[colour]

    # -- every TICK ----------------------------------------------------

    def _analyse_loop(self):
        while not self.stopping.wait(TICK):
            try:
                self._tick()
            except Exception as error:
                traceback.print_exc()
                self.note(f"One look at the match failed ({type(error).__name__}: {error}).")

    def _snapshot(self, now):
        first = int(max(0.0, now - WINDOW) * self.fps)
        with self.lock:
            # Old frames are dropped so a full match stays small in memory.
            for store in (self.tracks, self.ball):
                for frame in [f for f in store if f < first]:
                    del store[frame]
            self.ball_detected = {f for f in self.ball_detected if f >= first}
            match = rules.MatchData(
                fps=self.fps, tracks=dict(self.tracks),
                teams={tid: dict(d) for tid, d in self.teams.items()},
                ball=dict(self.ball), team_colours=dict(self.team_colours),
                ball_detected=set(self.ball_detected))
        return match, first / self.fps

    def _tick(self, final=False):
        with self.say_lock:
            self._look(final)

    def _look(self, final):
        now = self.clock
        if self.frames_done == 0:
            return
        self._refit_teams(now)
        match, window_start = self._snapshot(now)
        cues = []
        if self.sound is not None:
            if self.sound.error and self.sound.error not in self.notes:
                self.note(self.sound.error)
            cues = self.sound.cues(now)
            for kind in self.cue_counts:
                self.cue_counts[kind] = sum(c["type"] == kind for c in cues)
            cues = [c for c in cues if c["time"] >= window_start]

        def make(frame, kind, players, confidence, source):
            return rules._event(match, frame, kind, players, confidence, source, 0)

        events = rules.events_from_rules(match, 0, None)
        if self.sound is not None and self.sound.samples:
            # A still ball is a free kick only if the referee blew: a wrong
            # "free kick!" is worse than saying nothing.
            whistles = [c["time"] for c in cues if c["type"] == "whistle"]
            events = [e for e in events if e["type"] not in audio_events.NEEDS_WHISTLE or any(
                e["time"] + audio_events.NEEDS_WHISTLE[e["type"]][0] <= w
                <= e["time"] + audio_events.NEEDS_WHISTLE[e["type"]][1] for w in whistles)]
        penalties = []
        if self.settings["penalties"]:
            # A wait that reaches the newest tracks is still going on: the
            # kick has not been taken yet.
            speed = penalty_kicks.player_speed(match)
            known_until = len(speed) - 1 - penalty_kicks.STILL_BRIDGE_SECONDS - 1

            def no_kick(seconds):
                return self.silent(seconds) or seconds > known_until

            kicks, shootouts = penalty_kicks.find(match, cues, no_kick)
            if self.manual_shootout is not None:
                # The operator said it is a shootout: every long wait ends in a kick.
                kicks = [(at, "shootout_kick") for at, _ in kicks]
                kicks += [(end, "shootout_kick")
                          for _, end in penalty_kicks.still_stretches(speed)
                          if self.manual_shootout <= end <= known_until
                          and all(abs(end - at) > 5 for at, _ in kicks)]
            penalties = [make(int(s * self.fps), kind, [], 0.7, "stillness") for s, kind in kicks]
            penalties += [make(int(start * self.fps), "shootout_start", [], 0.7, "stillness")
                          for start, _ in shootouts]
            for start, end in shootouts:
                known = next((s for s in self.shootouts if s[0] <= end and start <= s[1]), None)
                if known:
                    known[0], known[1] = min(known[0], start), max(known[1], end)
                else:
                    self.shootouts.append([start, end])

        def quiet(seconds):
            return self.silent(seconds) or self.in_shootout(seconds)

        if cues:
            goals = [g for g in audio_events.goals_from_sound(match, cues, make)
                     if not quiet(g["time"])]
            if goals:
                after = rules.AFTER_GOAL_SECONDS
                events = [e for e in events if not any(
                    0 <= e["time"] - g["time"] <= after for g in goals)] + goals
        events = [e for e in rules.add_play_events(match, events, 0)
                  if e["type"] != "match_intro"]
        if cues:
            # (the penalties count as said, so a kick's roar is not a second line)
            events += audio_events.events_from_sound(
                match, events + penalties, cues, rules.find_touches(match), make)
        events = rules.add_possession_spells(match, sorted(events, key=lambda e: e["frame"]), 0)
        events = [e for e in events if not quiet(e["time"])] + penalties

        if not self.intro_said and len(match.team_colours) >= 2 and not self.silent(now):
            intro = make(int(now * self.fps), "match_intro", [], 1.0, "teams")
            teams = sorted(match.team_colours)[:2]
            intro.update(team=teams[0], opponent=teams[1],
                         team_colour=match.team_colours[teams[0]],
                         opponent_colour=match.team_colours[teams[1]])
            self.intro_said = True
            self._say(intro, match, now)

        for event in sorted(events, key=lambda e: e["time"]):
            kind, at = event["type"], event["time"]
            settle = 0.0 if final else SETTLE.get(kind, DEFAULT_SETTLE)
            if at > now - settle:
                continue
            if now - at > SETTLE.get(kind, DEFAULT_SETTLE) + MAX_LATE.get(kind, DEFAULT_MAX_LATE):
                continue
            if window_start > 0 and at < window_start + WINDOW_EDGE and kind != "shootout_start":
                continue
            same = SAME_EVENT.get(kind, DEFAULT_SAME_EVENT)
            if any(e["type"] == kind and abs(e["time"] - at) <= same for e in self.events):
                continue
            if kind in PLAY_TYPES:
                if any(abs(l["time"] - at) < PLAY_GAP for l in self.lines):
                    continue
                if self.lines and now - self.lines[-1]["said_at"] < PLAY_GAP:
                    continue
            if kind == "goal":
                self._count_goal(event)
            self._say(event, match, now)

        if self.settings["filler"] and not final:
            self._fill_silence(match, now)

    def _count_goal(self, event):
        team = event.get("team")
        if team in self.score:
            self.score[team] += 1
            event["score"] = f"{self.score['team_a']}-{self.score['team_b']}"
            event["side"] = "home" if team == "team_a" else "away"

    def _fill_silence(self, match, now):
        """fill_commentary.py, live: a short line when nothing was said
        for a while."""
        if self.silent(now) or (self.kickoff is None and not self.lines and now < 5):
            return
        last = self.lines[-1]["said_at"] if self.lines else float(self.settings["start"])
        shootout = self.in_shootout(now)
        if not shootout and any(now > end for _, end in self.shootouts):
            return                      # after the shootout: the match is over
        wait = filler.SHOOTOUT_EVERY if shootout else filler.MAX_SILENCE
        if now - last < wait:
            return
        last_text = self.lines[-1]["text"] if self.lines else None
        team = None
        if shootout:
            text = filler.pick(filler.SHOOTOUT, last_text, self.rng)
        else:
            touches = sorted(play_reliable(match), key=lambda t: t.start)
            team = filler.ball_team(touches, self.fps, now)[0]
            colour = BENGALI_COLOUR_NAMES.get((match.team_colours or {}).get(team))
            if team and colour:
                options = (filler.CHANGE if self.last_filler_team
                           and self.last_filler_team != team else filler.KEEPING)
                self.last_filler_team = team
                text = filler.pick(options, last_text, self.rng).format(team=colour)
            else:
                text = filler.pick(filler.NEUTRAL, last_text, self.rng)
        event = {"frame": int(now * self.fps), "time": round(now - 1.0, 2), "type": "filler",
                 "players": [], "confidence": 0.5, "source": "filler"}
        if team:
            event["team"] = team
        self._say(event, match, now, text)

    # -- lines ---------------------------------------------------------

    def _say(self, event, match, now, text=None):
        event["minute"] = self.minute(event["time"])
        if text is None:
            for _ in range(4):
                text = player_identity.commentary(
                    [event], {}, match.teams, self.templates, self.rng)[0][1]
                if text is None or text not in self.recent_texts:
                    break
        self.events.append(event)
        if not text:
            return
        self.recent_texts.append(text)
        line = {
            "id": len(self.lines) + 1,
            "time": event["time"],            # the moment in the match feed
            "said_at": round(now, 2),         # when the system knew
            "delay": round(now - event["time"], 1),
            "minute": event["minute"],
            "type": event["type"],
            "source": event.get("source"),
            "team": event.get("team"),
            "text": text,
            "audio": None,
            "score": dict(self.score),
        }
        self.lines.append(line)
        self.publish_json({"type": "line", "line": line})
        low = event["type"] in PLAY_TYPES
        if self.speaker is not None and not (low and self.speaker.backlog() >= SPEECH_BACKLOG):
            self.speaker.say(line)

    def _spoken(self, line, name, took):
        line["audio"] = f"/live/{self.id}/audio/{name}"
        line["voice_seconds"] = round(took, 2)
        self.publish_json({"type": "audio", "id": line["id"], "url": line["audio"],
                           "time": line["time"]})

    # -- operator ------------------------------------------------------

    def operator(self, action, team=None):
        """What the person at the camera tells the system: the match clock
        and the things one camera cannot see."""
        with self.say_lock:
            self._operator(action, team)
        self.publish_json(self.stats())

    def _operator(self, action, team):
        now = self.clock
        with self.lock:
            teams = {tid: dict(d) for tid, d in self.teams.items()}
            colours = dict(self.team_colours)
        if colours.get(team):
            # "a player in orange", also when nobody of that team is tracked yet.
            teams[-1] = {"team": team, "role": "player", "team_colour": colours[team]}
        match = rules.MatchData(fps=self.fps, tracks={}, teams=teams, ball={},
                                team_colours=colours, ball_detected=set())
        event = {"frame": int(now * self.fps), "time": round(now, 2), "players": [],
                 "confidence": 1.0, "source": "operator"}
        text = None
        if action == "kickoff":
            self.kickoff, self.intro_said = now, True
            self.pauses = [p for p in self.pauses if p[1] is not None and p[1] < now]
            event["type"] = "kickoff"
            names = [f"{BENGALI_COLOUR_NAMES.get(c, c)} জার্সির দল" for c in
                     (colours.get("team_a"), colours.get("team_b")) if c]
            text = (f"খেলা শুরু! মুখোমুখি {names[0]} ও {names[1]}।" if len(names) == 2
                    else "রেফারির বাঁশি, খেলা শুরু!")
        elif action == "half_time":
            self.pauses.append([now, None])
            event["type"] = "half_time"
            text = "রেফারির বাঁশি, শেষ হলো প্রথমার্ধের খেলা। " + self._score_text()
        elif action == "second_half":
            for pause in self.pauses:
                if pause[1] is None:
                    pause[1] = now
            self.second_half = now
            event["type"] = "second_half"
            text = "শুরু হলো দ্বিতীয়ার্ধের খেলা!"
        elif action == "pause":
            self.pauses.append([now, None])
            event["type"] = "pause"
        elif action == "resume":
            for pause in self.pauses:
                if pause[1] is None:
                    pause[1] = now
            event["type"] = "resume"
        elif action == "shootout":
            self.manual_shootout = now
            self.shootouts.append([now, float("inf")])
            event["type"] = "shootout_start"
        elif action == "full_time":
            self.full_time = now
            event["type"] = "full_time"
            text = "শেষ বাঁশি! খেলা শেষ। " + self._score_text()
        elif action in ("goal", "yellow_card", "red_card", "corner_kick", "freekick", "penalty"):
            if team not in ("team_a", "team_b"):
                raise ValueError("this needs a team")
            event.update(type=action, team=team, team_colour=colours.get(team))
            if action == "goal":
                event["template_type"] = "goal_celebration"
                self._count_goal(event)
            elif action == "penalty":
                event["template_type"] = "penalty_kick"
        elif action == "undo_goal":
            if team in self.score and self.score[team] > 0:
                self.score[team] -= 1
            event["type"] = "undo_goal"
        else:
            raise ValueError(f"unknown action {action!r}")
        self._say(event, match, now, text)

    def _score_text(self):
        return f"স্কোর {player_identity.bengali_number(self.score['team_a'])}-" \
               f"{player_identity.bengali_number(self.score['team_b'])}।"

    # -- out -----------------------------------------------------------

    def stats(self):
        now = self.clock
        with self.lock:
            recent = [f for f in self.tracks if f >= (now - 3) * self.fps]
            people = (sum(len(self.tracks[f]) for f in recent) / len(recent)) if recent else 0
            span = int(10 * self.fps)
            first = int(now * self.fps) - span
            ball_share = sum(1 for f in self.ball_detected if f >= first) / max(1, span)
            teams = {}
            for team in ("team_a", "team_b"):
                colour = self.team_colours.get(team)
                teams[team] = {"colour": colour,
                               "colour_bn": BENGALI_COLOUR_NAMES.get(colour),
                               "players": sum(1 for d in self.teams.values()
                                              if d.get("team") == team)}
        return {
            "type": "stats", "session": self.id, "state": self.state, "message": self.message,
            "clock": round(now, 1), "minute": self.minute(now), "phase": self.phase(),
            "rate": round(self.rate, 1), "target_rate": self.fps,
            "dropped": self.source.dropped if self.source else 0,
            "duration": self.source.duration if self.source else None,
            "players": round(people, 1), "ball_share": round(min(1.0, ball_share), 2),
            "whistles": self.cue_counts["whistle"], "roars": self.cue_counts["crowd_roar"],
            "sound": bool(self.sound is not None and self.sound.samples),
            "teams": teams, "score": dict(self.score), "lines": len(self.lines),
            "speech_waiting": self.speaker.backlog() if self.speaker else 0,
            "source": self.source.describe() if self.source else "",
            "live_source": bool(self.source and self.source.live),
            "realtime": bool(self.settings["realtime"]),
        }

    def hello(self):
        """Everything a browser that joins late needs."""
        return {"type": "hello", "stats": self.stats(), "lines": self.lines[-60:],
                "notes": self.notes[-10:], "settings": self.settings}

    def save(self):
        """outputs/live/<start time>/: the lines (as <name>_commentary.json
        of the offline scripts, plus when each was known) and the events."""
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "commentary.json").write_text(
                json.dumps(self.lines, ensure_ascii=False, indent=2), encoding="utf-8")
            (self.folder / "events.json").write_text(
                json.dumps(self.events, ensure_ascii=False, indent=2), encoding="utf-8")
            delays = [l["delay"] for l in self.lines if l["source"] not in ("operator", "filler")]
            summary = {
                "session": self.id, "source": self.source.describe() if self.source else None,
                "settings": self.settings, "seconds": round(self.clock, 1),
                "frames": self.frames_done,
                "dropped": self.source.dropped if self.source else 0,
                "lines": len(self.lines), "score": self.score,
                "median_delay_seconds": float(np.median(delays)) if delays else None,
                "notes": self.notes,
            }
            (self.folder / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as error:
            print("[live] could not save the session:", error)


def play_reliable(match):
    import play_events

    return play_events.reliable(match, rules.find_touches(match))
