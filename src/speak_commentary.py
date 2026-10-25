"""
Speak the Bengali commentary and put it on the video.

Reads outputs/tracking/<name>_commentary.json (from player_identity.py),
turns every line into Bengali speech and places it at the moment of its
event. A line that would start while the previous one is still being
spoken waits for it (live commentators do not talk over themselves).

Voices:
  mms   Meta MMS-TTS Bengali (facebook/mms-tts-ben, VITS). Runs offline
        on this PC, ~150 MB, downloaded once into .cache/huggingface.
        Licence CC-BY-NC 4.0: fine for research, not for a product.
        Needs: pip install transformers
  edge  Microsoft Edge neural voices (bn-BD-PradeepNeural, male, or
        bn-BD-NabanitaNeural, female). Much more natural, but online.
        Needs: pip install edge-tts

Outputs (outputs/tracking/):
  <name>_commentary.wav           the commentary track, as long as the video
  <name>_commentary.mp4           the video with the commentary over the
                                  original sound (quieter) - needs ffmpeg,
                                  found on PATH or from pip install imageio-ffmpeg
  <name>_commentary_lines/*.wav   one file per line

Usage:
    python src/speak_commentary.py --video data/videos/test1.mp4
    python src/speak_commentary.py --video data/videos/test1.mp4 --voice edge
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import asyncio
import json
import shutil
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

ROOT = project_cache.ROOT
TRACKING_DIR = ROOT / "outputs" / "tracking"
SAMPLE_RATE = 24000
MMS_MODEL = "facebook/mms-tts-ben"
EDGE_VOICES = {"male": "bn-BD-PradeepNeural", "female": "bn-BD-NabanitaNeural"}
# Short pause between two lines that follow each other.
GAP_SECONDS = 0.3
# Original match sound under the commentary.
BACKGROUND_VOLUME = 0.35


# ----------------------------------------------------------------------
# Voices: each turns a line into float32 mono audio at SAMPLE_RATE
# ----------------------------------------------------------------------

def resample(audio, rate):
    if rate == SAMPLE_RATE or len(audio) == 0:
        return audio.astype(np.float32)
    duration = len(audio) / rate
    target = np.linspace(0, duration, int(round(duration * SAMPLE_RATE)),
                         endpoint=False)
    return np.interp(target, np.arange(len(audio)) / rate, audio).astype(np.float32)


class MmsVoice:
    def __init__(self, device=None):
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MMS_MODEL)
        self.model = VitsModel.from_pretrained(MMS_MODEL).to(self.device).eval()

    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            audio = self.model(**inputs).waveform[0].cpu().numpy()
        return resample(audio, self.model.config.sampling_rate)


class EdgeVoice:
    def __init__(self, voice="male", rate="+10%"):
        import edge_tts  # noqa: F401  (fail early if missing)

        self.voice = EDGE_VOICES.get(voice, voice)
        self.rate = rate  # commentators speak a little faster

    def __call__(self, text):
        import edge_tts

        async def synthesise():
            chunks = []
            async for chunk in edge_tts.Communicate(
                    text, self.voice, rate=self.rate).stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        return decode_audio(asyncio.run(synthesise()))


def ffmpeg_path():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def decode_audio(data):
    """MP3/any bytes -> float32 mono at SAMPLE_RATE, through ffmpeg."""
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is needed for the edge voice: "
                           "pip install imageio-ffmpeg")
    result = subprocess.run(
        [ffmpeg, "-loglevel", "error", "-i", "pipe:0", "-f", "f32le",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        input=data, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


# ----------------------------------------------------------------------
# Timeline
# ----------------------------------------------------------------------

def place_lines(clips, duration):
    """clips: [(start seconds, audio)] -> (track, [actual start seconds]).

    A line starts at its event, or right after the previous line ends.
    """
    starts, end = [], 0.0
    for start, audio in sorted(clips, key=lambda c: c[0]):
        start = max(start, end + (GAP_SECONDS if end else 0.0))
        starts.append(start)
        end = start + len(audio) / SAMPLE_RATE
    length = max(duration, end)
    track = np.zeros(int(np.ceil(length * SAMPLE_RATE)) + 1, np.float32)
    for start, (_, audio) in zip(starts, sorted(clips, key=lambda c: c[0])):
        first = int(start * SAMPLE_RATE)
        track[first:first + len(audio)] += audio
    peak = float(np.abs(track).max()) if len(track) else 0.0
    if peak > 0.98:
        track *= 0.98 / peak
    return track, starts


def write_wav(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(pcm.tobytes())


def mux(video, commentary_wav, output, background=BACKGROUND_VOLUME):
    """Video + commentary (+ the original sound, quieter, if it has any)."""
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        print("No ffmpeg (pip install imageio-ffmpeg): only the WAV was written.")
        return None
    with_sound = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(video),
        "-i", str(commentary_wav), "-filter_complex",
        f"[0:a]volume={background}[bg];[bg][1:a]amix=inputs=2:duration=first:normalize=0[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
        "-shortest", str(output)]
    commentary_only = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(video),
        "-i", str(commentary_wav), "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(output)]
    if subprocess.run(with_sound).returncode != 0:
        # The video has no sound track.
        subprocess.run(commentary_only, check=True)
    return output


def video_duration(video):
    capture = cv2.VideoCapture(str(video))
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    capture.release()
    return frames / fps if frames > 0 else 0.0


def speak(video, lines, voice, output_dir=TRACKING_DIR):
    name = Path(video).stem
    clips = []
    for index, line in enumerate(lines, 1):
        audio = voice(line["text"])
        write_wav(output_dir / f"{name}_commentary_lines" / f"{index:03d}.wav", audio)
        clips.append((float(line["time"]), audio))
    track, starts = place_lines(clips, video_duration(video))
    wav = output_dir / f"{name}_commentary.wav"
    write_wav(wav, track)
    for line, start in zip(sorted(lines, key=lambda l: l["time"]), starts):
        delay = start - float(line["time"])
        note = f" (+{delay:.1f} s)" if delay > 0.05 else ""
        print(f"{int(start // 60)}:{start % 60:04.1f}{note}  {line['text']}")
    print("Commentary audio:", wav)
    video_out = mux(video, wav, output_dir / f"{name}_commentary.mp4")
    if video_out:
        print("Video with commentary:", video_out)
    return wav, video_out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--voice", choices=["mms", "edge"], default="mms")
    parser.add_argument("--speaker", default="male",
                        help="edge: male, female or a voice name")
    args = parser.parse_args()

    commentary = TRACKING_DIR / f"{args.video.stem}_commentary.json"
    if not commentary.exists():
        raise SystemExit(f"No {commentary}. Run player_identity.py first.")
    lines = json.loads(commentary.read_text(encoding="utf-8"))
    voice = MmsVoice() if args.voice == "mms" else EdgeVoice(args.speaker)
    speak(args.video, lines, voice)


if __name__ == "__main__":
    main()
