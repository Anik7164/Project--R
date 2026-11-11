"""
What the sound says: the referee's whistle and the crowd.

On a local match filmed from the stands the ball is often a few pixels
wide, but the sound is clear and costs almost nothing to analyse:

  whistle     a referee's whistle is a flat, pure tone around 2.8-4.5 kHz
              (with harmonics). A peak in that band that stays at the same
              pitch for WHISTLE_SECONDS is a whistle; voices and horns
              slide in pitch, claps and bangs are not tones, so neither
              passes. A whistle means play stopped: foul, offside, ball out
              for a free kick, half-time, or a goal being given.
  crowd_roar  the crowd gets louder than it has been over the last
              BASELINE_SECONDS by ROAR_DB for ROAR_SECONDS: a goal, a big
              chance, a great save, a hard tackle.

Both work one step at a time from the past only, so they also run live.

Output: outputs/tracking/<name>_audio.json
    {"version", "cues": [{"time", "type": "whistle"|"crowd_roar", "strength",
                          "duration", ...}]}
strength: whistle tone prominence in dB, or the roar's rise in dB.

Usage:
    python src/audio_cues.py --video data/videos/match1.mp4
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = ROOT / "outputs" / "tracking"

SAMPLE_RATE = 16000
FRAME = 1024              # 64 ms analysis window (16 Hz pitch steps)
HOP = 320                 # 20 ms step

WHISTLE_BAND = (2800.0, 4500.0)
WHISTLE_PROMINENCE_DB = 18.0   # tone above the band's median level
WHISTLE_PITCH_HZ = 40.0        # largest pitch change between steps
WHISTLE_FLAT_HZ = 150.0        # largest pitch range over the whole blow
WHISTLE_SECONDS = 0.25
WHISTLE_GAP = 0.06             # a step or so may drop out
WHISTLE_MERGE_SECONDS = 1.0
# The referee blows the same whistle all match: keep the tones near the
# pitch heard most (voices and horns are spread over the band).
WHISTLE_PITCH_MATCH = 120.0
MIN_REFEREE_BLOWS = 3    # pips of one blow count once

LOUDNESS_STEP = 0.5            # seconds
BASELINE_SECONDS = 30.0
ROAR_DB = 3.0
ROAR_SECONDS = 1.5
ROAR_MERGE_SECONDS = 12.0


def ffmpeg_path():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def load_audio(video):
    """Mono float32 samples at SAMPLE_RATE (None if the video has no sound)."""
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise SystemExit("ffmpeg is needed: pip install imageio-ffmpeg")
    result = subprocess.run(
        [ffmpeg, "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        return None
    return np.frombuffer(result.stdout, dtype=np.float32)


def _tones(samples, rate):
    """Per HOP step: (is a tone, pitch in Hz, prominence in dB)."""
    window = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1 / rate)
    band = (freqs >= WHISTLE_BAND[0]) & (freqs <= WHISTLE_BAND[1])
    count = max(0, (len(samples) - FRAME) // HOP)
    frames = np.lib.stride_tricks.sliding_window_view(samples, FRAME)[::HOP][:count]
    out = []
    for first in range(0, count, 2048):  # in blocks, to keep memory small
        spectra = 20 * np.log10(np.abs(np.fft.rfft(frames[first:first + 2048] * window,
                                                   axis=1))[:, band] + 1e-9)
        peaks = spectra.argmax(axis=1)
        prominence = spectra[np.arange(len(peaks)), peaks] - np.median(spectra, axis=1)
        out += list(zip(prominence >= WHISTLE_PROMINENCE_DB, freqs[band][peaks], prominence))
    return out


def find_whistles(samples, rate=SAMPLE_RATE):
    step = HOP / rate
    max_gap = int(round(WHISTLE_GAP / step))
    runs, run, gap = [], None, 0
    for i, (tone, pitch, prominence) in enumerate(_tones(samples, rate)):
        if tone and run is not None and abs(pitch - run["pitches"][-1]) <= WHISTLE_PITCH_HZ:
            run["pitches"].append(pitch)
            run["end"], run["strength"], gap = i, max(run["strength"], prominence), 0
            continue
        if run is not None and not tone and gap < max_gap:
            gap += 1
            continue
        if run is not None:
            runs.append(run)
        run = {"start": i, "end": i, "strength": prominence, "pitches": [pitch]} if tone else None
        gap = 0
    if run is not None:
        runs.append(run)

    cues = []
    for w in runs:
        start, duration = w["start"] * step, (w["end"] - w["start"] + 1) * step
        # A whistle holds its pitch; a voice or a horn slides.
        if duration < WHISTLE_SECONDS or np.ptp(w["pitches"]) > WHISTLE_FLAT_HZ:
            continue
        if cues and start - (cues[-1]["time"] + cues[-1]["duration"]) <= WHISTLE_MERGE_SECONDS:
            last = cues[-1]
            last["duration"] = round(start + duration - last["time"], 2)
            last["strength"] = round(max(last["strength"], float(w["strength"])), 1)
            continue
        cues.append({"time": round(start, 2), "type": "whistle",
                     "strength": round(float(w["strength"]), 1),
                     "duration": round(duration, 2),
                     "pitch": round(float(np.median(w["pitches"])))})
    return referee_whistle(cues)


def referee_whistle(cues):
    """Keep the tones at the referee's whistle pitch: the pitch with the
    most blowing time. Live, call it on the cues heard so far."""
    if len(cues) < MIN_REFEREE_BLOWS:
        return cues
    pitches = np.array([c["pitch"] for c in cues], dtype=float)
    weights = np.array([c["duration"] for c in cues])
    support = [weights[np.abs(pitches - p) <= WHISTLE_PITCH_MATCH].sum() for p in pitches]
    referee = pitches[int(np.argmax(support))]
    near = [c for c in cues if abs(c["pitch"] - referee) <= WHISTLE_PITCH_MATCH]
    return near if len(near) >= MIN_REFEREE_BLOWS else cues


def loudness(samples, rate=SAMPLE_RATE):
    size = int(LOUDNESS_STEP * rate)
    count = len(samples) // size
    blocks = samples[:count * size].reshape(count, size)
    return 20 * np.log10(np.sqrt(np.mean(blocks ** 2, axis=1)) + 1e-9)


def find_roars(levels):
    """levels: dB per LOUDNESS_STEP. Rise over the median of the past
    BASELINE_SECONDS (past only, so it works live)."""
    history = int(BASELINE_SECONDS / LOUDNESS_STEP)
    need = int(round(ROAR_SECONDS / LOUDNESS_STEP))
    cues, run = [], []
    for i in range(len(levels)):
        if i < history // 2:
            continue  # not enough past yet to know what "normal" is
        baseline = float(np.median(levels[max(0, i - history):i]))
        rise = levels[i] - baseline
        if rise >= ROAR_DB:
            run.append((i, rise))
            continue
        if len(run) >= need:
            cues.append(run)
        run = []
    if len(run) >= need:
        cues.append(run)

    merged = []
    for run in cues:
        start = run[0][0] * LOUDNESS_STEP
        duration = len(run) * LOUDNESS_STEP
        rise = max(r for _, r in run)
        if merged and start - merged[-1]["time"] <= ROAR_MERGE_SECONDS:
            last = merged[-1]
            last["duration"] = round(start + duration - last["time"], 2)
            last["strength"] = round(float(max(last["strength"], rise)), 1)
            if duration > last.get("longest", 0):
                last["longest"], last["longest_at"] = duration, round(start, 2)
            continue
        merged.append({"time": round(start, 2), "type": "crowd_roar",
                       "strength": round(float(rise), 1), "duration": duration,
                       "longest": duration, "longest_at": round(start, 2)})
    return merged


def audio_cues(video):
    samples = load_audio(video)
    if samples is None or len(samples) < SAMPLE_RATE:
        return []
    cues = find_whistles(samples) + find_roars(loudness(samples))
    return sorted(cues, key=lambda c: c["time"])


# Bump when the cues change, so an old <name>_audio.json is made again.
CUES_VERSION = 2


def save_cues(path, cues):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": CUES_VERSION, "cues": cues}, indent=2),
                    encoding="utf-8")


def load_cues(video):
    """Cached <name>_audio.json, made again if missing or from an older
    version of this script."""
    path = TRACKING_DIR / f"{Path(video).stem}_audio.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get("version") == CUES_VERSION:
            return cached["cues"]
    cues = audio_cues(video)
    save_cues(path, cues)
    return cues


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    cues = audio_cues(args.video)
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    output = TRACKING_DIR / f"{args.video.stem}_audio.json"
    save_cues(output, cues)
    for cue in cues:
        t = cue["time"]
        print(f"{int(t // 60)}:{t % 60:04.1f}  {cue['type']:<10}  "
              f"{cue['strength']:>5.1f} dB  {cue['duration']:.1f} s")
    print(f"{len(cues)} cues:", output)


if __name__ == "__main__":
    main()
