"""
Mark the ball by clicking: the fastest way to teach the detector the
balls it misses.

Only frames where the current model does NOT find the ball confidently
are shown (the ones it has to learn). For each frame:

    left click      the ball is here (a box of the right size is drawn;
                    clicking on a yellow model guess uses its box)
    N or right click  no ball visible in this frame
    S               skip (unsure; nothing saved)
    U               undo the last frame
    Q or Esc        stop (everything so far is saved; run again to go on)

A magnifier next to the mouse shows the area 4x bigger, as the ball is
often only 8-10 pixels wide.

Saved like label-frames, in data/local_det (images + labels, with the
model's confident player boxes added), so train_detector.py prepare-local
and train use them. Frames you clicked are also used for validation, so
the ball mAP50 after training is measured on real labels.

    python src/click_balls.py data/videos/match1.mp4 data/videos/match2.mp4 --frames 300
    python src/train_detector.py finetune
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from detector_classes import default_model
from train_detector import (AUTO_PEOPLE_CONFIDENCE, CLASSES, HAND_LIST, IMAGE_SIZE,
                            LOCAL, _auto_stems, _hand_stems, _yolo_row, AUTO_LIST)

# A frame is "hard" if the model's best ball is below this confidence.
HARD_CONFIDENCE = 0.4
# Ball diameter (22 cm) relative to a player's height (1.8 m).
BALL_PER_PLAYER = 0.12
MIN_BALL = 6
SNAP_PIXELS = 10
MAX_WINDOW_WIDTH = 1400
ZOOM, ZOOM_HALF = 4, 40
BAR = 30


def hard_frames(videos, model, device, per_video, every=1.0):
    """[(video, frame, ball guesses, people rows)] where the model is unsure."""
    names = model.names
    index = {i: CLASSES.index(n) for i, n in names.items() if n in CLASSES}
    if len(index) != len(CLASSES):
        raise SystemExit("Needs the football model (player/goalkeeper/referee/ball); give --model.")
    ball = CLASSES.index("ball")
    done = _hand_stems()
    chosen = []
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise SystemExit(f"Cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(every * fps))
        found = []
        print(f"Finding hard frames in {Path(video).name} ...")
        for number in range(step // 2, total, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, number)
            ok, frame = capture.read()
            if not ok:
                continue
            if f"{Path(video).stem}_{number:06d}" in done:
                continue
            height, width = frame.shape[:2]
            result = model.predict(frame, imgsz=IMAGE_SIZE, conf=0.1, device=device,
                                   verbose=False)[0]
            balls, people, heights = [], [], []
            for box, cls, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                       result.boxes.conf.tolist()):
                if index[int(cls)] == ball:
                    balls.append((box, score))
                elif score >= AUTO_PEOPLE_CONFIDENCE:
                    people.append((index[int(cls)], box))
            if not people:
                continue  # no pitch in view (crowd, flags, close-up)
            if max((s for _, s in balls), default=0.0) >= HARD_CONFIDENCE:
                continue  # the model already finds this ball
            found.append((video, number, balls, people))
        capture.release()
        if len(found) > per_video:  # spread over the whole video
            found = [found[int(i * len(found) / per_video)] for i in range(per_video)]
        print(f"  {len(found)} hard frames")
        chosen += found
    return chosen


def ball_box(x, y, balls, people):
    """Box for a click: a model guess under the click, else a ball-sized
    box from the height of the nearest player."""
    for (x1, y1, x2, y2), _ in balls:
        if x1 - SNAP_PIXELS <= x <= x2 + SNAP_PIXELS and y1 - SNAP_PIXELS <= y <= y2 + SNAP_PIXELS:
            return x1, y1, x2, y2
    nearest = min(people, key=lambda p: np.hypot((p[1][0] + p[1][2]) / 2 - x, p[1][3] - y))
    size = max(MIN_BALL, BALL_PER_PLAYER * (nearest[1][3] - nearest[1][1]))
    return x - size / 2, y - size / 2, x + size / 2, y + size / 2


class Window:
    def __init__(self, name):
        self.name, self.mouse, self.click = name, (0, 0), None
        try:
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        except cv2.error as error:
            raise SystemExit("OpenCV cannot open a window here (a 'headless' copy, often "
                             "pulled in by easyocr). Fix:\n  pip uninstall -y "
                             "opencv-python-headless\n  pip install --force-reinstall "
                             "--no-deps opencv-python==5.0.0.93") from error
        cv2.setMouseCallback(name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        y -= BAR  # the status bar sits above the picture
        self.mouse = (x, max(0, y))
        if y < 0:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = ("ball", x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.click = ("none", x, y)

    def show(self, frame, scale, guesses, status):
        view = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame.copy()
        for (x1, y1, x2, y2), score in guesses:
            cv2.rectangle(view, (int(x1 * scale) - 3, int(y1 * scale) - 3),
                          (int(x2 * scale) + 3, int(y2 * scale) + 3), (0, 255, 255), 1)
        mx, my = self.mouse
        fx, fy = int(mx / scale), int(my / scale)
        h, w = frame.shape[:2]
        x1, y1 = max(0, fx - ZOOM_HALF), max(0, fy - ZOOM_HALF)
        patch = frame[y1:min(h, fy + ZOOM_HALF), x1:min(w, fx + ZOOM_HALF)]
        if patch.size:
            patch = cv2.resize(patch, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
            ph, pw = patch.shape[:2]
            cv2.drawMarker(patch, ((fx - x1) * ZOOM, (fy - y1) * ZOOM), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 1)
            vh, vw = view.shape[:2]
            # Put the magnifier in the corner away from the mouse.
            ox = 10 if mx > vw // 2 else vw - pw - 10
            oy = 10 if my > vh // 2 else vh - ph - 10
            if 0 <= ox and ox + pw <= vw and 0 <= oy and oy + ph <= vh:
                view[oy:oy + ph, ox:ox + pw] = patch
                cv2.rectangle(view, (ox, oy), (ox + pw, oy + ph), (255, 255, 255), 1)
        bar = np.zeros((BAR, view.shape[1], 3), np.uint8)
        cv2.putText(bar, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(self.name, np.vstack([bar, view]))


def save(video, number, frame, ball, people):
    height, width = frame.shape[:2]
    name = f"{Path(video).stem}_{number:06d}"
    rows = [_yolo_row(CLASSES.index("ball"), *ball, width, height)] if ball else []
    rows += [_yolo_row(cls, *box, width, height) for cls, box in people]
    (LOCAL / "images").mkdir(parents=True, exist_ok=True)
    (LOCAL / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(LOCAL / "images" / f"{name}.jpg"), frame)
    (LOCAL / "labels" / f"{name}.txt").write_text("\n".join(rows) + "\n" if rows else "")
    return name


def write_lists(hand):
    HAND_LIST.parent.mkdir(parents=True, exist_ok=True)
    HAND_LIST.write_text("\n".join(sorted(hand)) + "\n")
    auto = _auto_stems()
    if auto & hand:  # clicked by hand now: auto-label must never overwrite it
        AUTO_LIST.write_text("\n".join(sorted(auto - hand)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--frames", type=int, default=300, help="frames to show in all")
    parser.add_argument("--model", type=Path, default=None,
                        help="default: the newest models/football_*.pt")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model or default_model()))
    todo = hard_frames(args.videos, model, device, max(1, args.frames // len(args.videos)))
    if not todo:
        print("No hard frames left to label.")
        return
    (LOCAL / "classes.txt").parent.mkdir(parents=True, exist_ok=True)
    (LOCAL / "classes.txt").write_text("\n".join(CLASSES) + "\n")

    hand = _hand_stems()
    window = Window("Click the ball  (N = no ball, S = skip, U = undo, Q = quit)")
    captures = {}
    history = []  # (position, name)
    position, balls_marked, empty = 0, 0, 0
    while position < len(todo):
        video, number, guesses, people = todo[position]
        capture = captures.setdefault(video, cv2.VideoCapture(str(video)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, frame = capture.read()
        if not ok:
            position += 1
            continue
        scale = min(1.0, MAX_WINDOW_WIDTH / frame.shape[1])
        window.click = None
        action = None
        while action is None:
            window.show(frame, scale, guesses,
                        f"{position + 1}/{len(todo)}  {Path(video).name} frame {number}   "
                        f"balls {balls_marked}, no ball {empty}   "
                        "click ball | N none | S skip | U undo | Q quit")
            key = cv2.waitKey(20) & 0xFF
            if window.click is not None:
                kind, x, y = window.click
                action = ("ball", x / scale, y / scale) if kind == "ball" else ("none",)
            elif key in (ord("n"), ord("N")):
                action = ("none",)
            elif key in (ord("s"), ord("S")):
                action = ("skip",)
            elif key in (ord("u"), ord("U")):
                action = ("undo",)
            elif key in (ord("q"), ord("Q"), 27):
                action = ("quit",)
            elif cv2.getWindowProperty(window.name, cv2.WND_PROP_VISIBLE) < 1:
                action = ("quit",)
        if action[0] == "quit":
            break
        if action[0] == "undo":
            if history:
                position, name = history.pop()
                if name:
                    hand.discard(name)
                    label = (LOCAL / "labels" / f"{name}.txt").read_text()
                    if any(line.startswith("3 ") for line in label.splitlines()):
                        balls_marked -= 1
                    else:
                        empty -= 1
                    (LOCAL / "labels" / f"{name}.txt").unlink()
            continue
        name = None
        if action[0] == "ball":
            name = save(video, number, frame, ball_box(action[1], action[2], guesses, people), people)
            balls_marked += 1
        elif action[0] == "none":
            name = save(video, number, frame, None, people)
            empty += 1
        if name:
            hand.add(name)
            write_lists(hand)
        history.append((position, name))
        position += 1

    for capture in captures.values():
        capture.release()
    cv2.destroyAllWindows()
    write_lists(hand)
    print(f"\nSaved {balls_marked} balls and {empty} frames without a ball "
          f"({len(hand)} hand-labelled frames in all) in {LOCAL}")
    print("Next:  python src/train_detector.py finetune")


if __name__ == "__main__":
    main()
