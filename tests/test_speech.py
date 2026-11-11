"""Commentary audio timeline and video muxing, with a fake voice."""

import json
import wave

import numpy as np
import pytest

import speak_commentary as sc


def fake_voice(text):
    seconds = 0.1 * len(text.split())
    t = np.arange(int(seconds * sc.SAMPLE_RATE)) / sc.SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_lines_wait_for_each_other():
    one = np.zeros(sc.SAMPLE_RATE * 2, np.float32)  # 2 s
    track, starts = sc.place_lines([(5.0, one), (1.0, one), (2.0, one)], 10.0)
    gap = sc.GAP_SECONDS
    assert starts == pytest.approx([1.0, 3.0 + gap, 5.0 + 2 * gap])
    assert len(track) >= 10.0 * sc.SAMPLE_RATE


def test_speak_writes_wav_and_video(tmp_path):
    import cv2
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for _ in range(75):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    lines = [{"time": 0.5, "text": "গোল! দারুণ গোল"}, {"time": 1.0, "text": "কর্নার"}]
    wav, out = sc.speak(video, lines, fake_voice, output_dir=tmp_path)
    with wave.open(str(wav)) as file:
        assert file.getframerate() == sc.SAMPLE_RATE
        assert file.getnframes() >= 3 * sc.SAMPLE_RATE
    if sc.ffmpeg_path():
        assert out.exists() and out.stat().st_size > 0
    assert (tmp_path / "clip_commentary_lines" / "002.wav").exists()
