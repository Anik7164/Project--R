"""
Write down what really happens in a match, while watching it: one key
per event. The list is the answer sheet for score_commentary.py.

    python src/mark_events.py --video data/videos/match5.mp4

Keys (the time saved is where the video is when you press):
    g goal        s shot (wide, over, saved)   a attack / big chance
    f foul        k free kick                  c corner
    t throw-in    o goal kick                  x tackle (ball won)
    y card        w whistle / play stops (other reasons)
    p penalty kick (in play or in a shootout)
    1 / 2         right after an event key: which team (1 = first kit,
                  2 = second kit, as you name them at the start)
    space         pause / play          j / l   back / forward 5 s
    [ / ]         slower / faster       u       undo the last event
    q or Esc      save and quit (run again to continue: it keeps the list)

The video has no sound here; if whistles matter, mark them by "w" while
watching with sound in a normal player, or write the list by hand
instead: outputs/tracking/<name>_truth.txt, one event per line,
    7:49 goal orange
    5:03 foul
(type from the keys above: goal shot attack foul freekick corner
throw_in goal_kick tackle card whistle penalty).

Output: outputs/tracking/<name>_truth.json
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = ROOT / "outputs" / "tracking"

KEYS = {"g": "goal", "s": "shot", "a": "attack", "f": "foul", "k": "freekick",
        "c": "corner", "t": "throw_in", "o": "goal_kick", "x": "tackle",
        "y": "card", "w": "whistle", "p": "penalty"}
MAX_WIDTH = 1280


def clock(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--teams", nargs=2, metavar=("KIT1", "KIT2"),
                        help="kit colours for keys 1 and 2, e.g. orange white")
    args = parser.parse_args()

    output = TRACKING_DIR / f"{args.video.stem}_truth.json"
    events = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    teams = args.teams or [input("Kit colour of team 1 (e.g. orange): ").strip() or "team1",
                           input("Kit colour of team 2 (e.g. white): ").strip() or "team2"]

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    window = "Mark events (see keys in the title bar of the console)"
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    except cv2.error as error:
        raise SystemExit(
            "This OpenCV cannot open windows (a 'headless' copy, often pulled in by "
            "easyocr). Fix:\n  pip uninstall -y opencv-python-headless\n"
            "  pip install --force-reinstall --no-deps opencv-python==5.0.0.93\n"
            "Or write the list by hand in outputs/tracking/<name>_truth.txt.") from error
    print(__doc__.split("Keys")[1].split("The video")[0])

    speed, paused, frame_number = 1.0, False, 0
    frame = None
    while True:
        if not paused or frame is None:
            ok, frame = capture.read()
            if not ok:
                paused = True
                frame_number = total - 1
            else:
                frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        seconds = frame_number / fps
        view = frame.copy()
        scale = min(1.0, MAX_WIDTH / view.shape[1])
        if scale < 1:
            view = cv2.resize(view, None, fx=scale, fy=scale)
        last = events[-1] if events else None
        bar = np.zeros((34, view.shape[1], 3), np.uint8)
        text = (f"{clock(seconds)}  x{speed:g}{'  PAUSED' if paused else ''}   "
                f"{len(events)} events"
                + (f"   last: {clock(last['time'])} {last['type']} {last.get('team') or ''}"
                   if last else ""))
        cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(window, np.vstack([bar, view]))

        key = cv2.waitKey(max(1, int(1000 / (fps * speed))) if not paused else 30) & 0xFF
        if key == 255:
            continue
        char = chr(key).lower() if key < 128 else ""
        if key in (27,) or char == "q":
            break
        if char == " ":
            paused = not paused
        elif char in ("j", "l"):
            target = frame_number + (5 if char == "l" else -5) * fps
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, int(target))))
            frame = None
        elif char == "[":
            speed = max(0.25, speed / 2)
        elif char == "]":
            speed = min(8.0, speed * 2)
        elif char == "u" and events:
            removed = events.pop()
            print("undone:", clock(removed["time"]), removed["type"])
        elif char in ("1", "2") and events:
            events[-1]["team"] = teams[int(char) - 1]
            print(f"  team: {events[-1]['team']}")
        elif char in KEYS:
            events.append({"time": round(seconds, 1), "type": KEYS[char]})
            print(f"{clock(seconds)}  {KEYS[char]}")

    capture.release()
    cv2.destroyAllWindows()
    events.sort(key=lambda e: e["time"])
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"{len(events)} events saved: {output}")
    print(f"Next: python src/score_commentary.py --video {args.video}")


if __name__ == "__main__":
    main()
