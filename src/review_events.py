"""
Check the commentary by eye: plays the video around every commentary
line, with the line, its verdict from score_commentary.py and the real
event it was matched to written on the picture.

    python src/review_events.py --video data/videos/match5.mp4

Keys:
    n or space  next line          p   previous line
    r           play this one again
    q or Esc    quit

Each line plays from BEFORE seconds before it to AFTER seconds after.
--only wrong (or right, "wrong team", "not checked") shows just those.
Without a <name>_truth list it still plays every line, with no verdict.
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from score_commentary import TRACKING_DIR, event_team, load_truth, score, team_colours

BEFORE, AFTER = 4.0, 8.0
MAX_WIDTH = 1280
VERDICT_COLOUR = {"right": (80, 200, 80), "wrong": (60, 60, 230),
                  "wrong team": (40, 170, 240), "not checked": (180, 180, 180),
                  None: (220, 220, 220)}


def clock(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--only", choices=["right", "wrong", "wrong team", "not checked"])
    args = parser.parse_args()
    name = args.video.stem

    events = json.loads((TRACKING_DIR / f"{name}_events.json").read_text(encoding="utf-8"))
    colours = team_colours(name)
    try:
        lines, _ = score(events, load_truth(name), colours)
    except SystemExit:
        lines = [(e, None, None) for e in events]
    if args.only:
        lines = [line for line in lines if line[1] == args.only]
    if not lines:
        raise SystemExit("No lines to show.")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    window = "Review commentary  (n next, p previous, r replay, q quit)"
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    except cv2.error as error:
        raise SystemExit("OpenCV cannot open a window here (a 'headless' copy). Fix:\n"
                         "  pip uninstall -y opencv-python-headless\n"
                         "  pip install --force-reinstall --no-deps opencv-python==5.0.0.93"
                         ) from error

    index = 0
    while 0 <= index < len(lines):
        event, verdict, real = lines[index]
        kind = event.get("template_type", event["type"])
        team = event_team(event, colours)
        top = f"{index + 1}/{len(lines)}   {clock(event['time'])}  {kind}" + (f"  ({team})" if team else "")
        bottom = (verdict or "no truth list") + (
            f"   real: {clock(real['time'])} {real['type']} {real.get('team') or ''}" if real else "")
        start = max(0.0, event["time"] - BEFORE)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        action = None
        while action is None:
            ok, frame = capture.read()
            now = capture.get(cv2.CAP_PROP_POS_FRAMES) / fps
            if not ok or now > event["time"] + AFTER:
                key = cv2.waitKey(0) & 0xFF   # hold the last picture until a key
            else:
                scale = min(1.0, MAX_WIDTH / frame.shape[1])
                view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
                bar = np.zeros((60, view.shape[1], 3), np.uint8)
                cv2.putText(bar, top, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                cv2.putText(bar, bottom, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            VERDICT_COLOUR.get(verdict, (220, 220, 220)), 1)
                marker = "  << NOW" if abs(now - event["time"]) < 0.5 else ""
                cv2.putText(view, clock(now) + marker, (10, view.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(window, np.vstack([bar, view]))
                key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key in (ord("q"), 27):
                action = "quit"
            elif key in (ord("n"), ord(" ")):
                action = "next"
            elif key == ord("p"):
                action = "previous"
            elif key == ord("r"):
                action = "replay"
            elif cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                action = "quit"
        if action == "quit":
            break
        index += {"next": 1, "previous": -1, "replay": 0}[action]
        index = max(0, index)
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
