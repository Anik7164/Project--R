"""
Goals from the broadcast scoreboard.

The score graphic in the corner of a broadcast is the most reliable goal
signal there is: it changes from 0-0 to 1-0 a few seconds after the goal
and stays that way. This module

1. finds the scoreboard: the part of the top of the frame that looks the
   same in most frames (it stays while the picture behind it changes);
2. finds the two score digits in it: the tall, narrow, bright shapes
   nearest its middle;
3. watches each digit; when one changes shape and stays changed for a
   couple of seconds (while the rest of the scoreboard is unchanged, so
   it is not a replay wipe or a graphic covering it), that side scored.

Digits are not read, only their changes are counted, starting from
--start-score (default 0-0). The left digit is the home team. The main
colour of each half of the board (badge, colour stripe) is reported too,
so a side can be matched to a team's shirt colour.

Usage:
    python src/scoreboard.py --video data/videos/test1.mp4
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
from pathlib import Path

import cv2
import numpy as np

from appearance import HUE_NAMES

# Frames sampled to find the scoreboard.
LOCATE_STEP = 25
# Only the top of the frame is searched.
TOP_FRACTION = 0.25
# A pixel is part of the scoreboard if it is this close to its median
# colour in at least STATIC_SHARE of the sampled frames.
STATIC_TOLERANCE = 20
STATIC_SHARE = 0.6
MIN_BOARD_AREA = 0.002  # of the frame

# Score digits: bright, tall (share of board height) and narrow.
DIGIT_BRIGHTNESS = 170
DIGIT_MIN_HEIGHT = 0.3
DIGIT_MAX_ASPECT = 0.85

# Watching the score.
WATCH_STEP = 5
# The board counts as visible if it differs this little (mean absolute
# difference, digits left out) from its usual look.
BOARD_MAX_DIFFERENCE = 25.0
# A digit changed if its shape overlaps the previous one less than this.
SAME_DIGIT_IOU = 0.6
# ... for this long without going back.
CHANGE_SECONDS = 2.0
SIGNATURE_SIZE = (16, 24)


def locate_scoreboard(video_path):
    """Return (median board image, (x, y, w, h)) or (None, None)."""
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = []
    for index in range(0, max(total, 1), LOCATE_STEP):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            break
        top = frame[: int(frame.shape[0] * TOP_FRACTION)]
        samples.append(top.astype(np.int16))
    capture.release()
    if len(samples) < 5:
        return None, None

    stack = np.array(samples)
    median = np.median(stack, axis=0)
    close = (np.abs(stack - median).max(axis=-1) < STATIC_TOLERANCE).mean(axis=0)
    mask = (close >= STATIC_SHARE).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None, None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[best]
    frame_area = stack.shape[1] * stack.shape[2] / TOP_FRACTION
    if area < MIN_BOARD_AREA * frame_area:
        return None, None
    board = median[y:y + h, x:x + w].astype(np.uint8)
    return board, (int(x), int(y), int(w), int(h))


def find_digits(board):
    """Boxes (x, y, w, h) of the two score digits in a board image, left
    first, or None."""
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    bright = (gray >= DIGIT_BRIGHTNESS).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(bright)
    height, width = gray.shape
    candidates = []
    for x, y, w, h, area in stats[1:count]:
        if h < DIGIT_MIN_HEIGHT * height or w > DIGIT_MAX_ASPECT * h:
            continue
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue  # part of the board edge or a badge background
        candidates.append((int(x), int(y), int(w), int(h)))
    if len(candidates) < 2:
        return None
    centre = width / 2.0
    left = [c for c in candidates if c[0] + c[2] / 2.0 < centre]
    right = [c for c in candidates if c[0] + c[2] / 2.0 >= centre]
    if not left or not right:
        return None
    # The score sits around the middle of the board.
    left = max(left, key=lambda c: c[0])
    right = min(right, key=lambda c: c[0])
    return [_pad(left, height, width), _pad(right, height, width)]


def side_colours(board, digits):
    """Main colour name of the left and right half of the board, from its
    strongly coloured pixels (badges, colour stripes), digits left out."""
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    keep = (hsv[..., 1] >= 80) & (hsv[..., 2] >= 50)
    for x, y, w, h in digits:
        keep[y:y + h, x:x + w] = False
    middle = board.shape[1] // 2
    names = []
    for part, mask in ((hsv[:, :middle], keep[:, :middle]),
                       (hsv[:, middle:], keep[:, middle:])):
        hues = part[..., 0][mask]
        if len(hues) == 0:
            names.append(None)
            continue
        counts = {}
        for hue in hues:
            name = next(n for upper, n in HUE_NAMES if hue < upper)
            counts[name] = counts.get(name, 0) + 1
        names.append(max(counts, key=counts.get))
    return names


def _pad(box, height, width):
    # Room for a wider digit than the one in the median image ("1" -> "2").
    x, y, w, h = box
    grow = max(2, int(0.35 * h)) - w // 2 if w < 0.5 * h else 2
    x0, x1 = max(0, x - grow), min(width, x + w + grow)
    y0, y1 = max(0, y - 2), min(height, y + h + 2)
    return (x0, y0, x1 - x0, y1 - y0)


def _signature(board, box):
    x, y, w, h = box
    gray = cv2.cvtColor(board[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    shape = cv2.resize(gray, SIGNATURE_SIZE, interpolation=cv2.INTER_AREA)
    return shape >= DIGIT_BRIGHTNESS


def _iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def _board_visible(board, usual, digits):
    difference = np.abs(board.astype(np.int16) - usual.astype(np.int16))
    keep = np.ones(board.shape[:2], dtype=bool)
    for x, y, w, h in digits:
        keep[y:y + h, x:x + w] = False
    return float(difference[keep].mean()) <= BOARD_MAX_DIFFERENCE


def find_score_changes(video_path, start_score=(0, 0)):
    """Return [{"frame", "time", "side", "side_colour", "score"}] for each
    score change.

    side is "home" (left digit) or "away" (right digit), side_colour the
    main colour of that half of the board; score is the score after the
    change.
    """
    usual, box = locate_scoreboard(video_path)
    if usual is None:
        return []
    digits = find_digits(usual)
    if digits is None:
        return []
    colours = side_colours(usual, digits)
    x, y, w, h = box

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    needed = max(1, int(round(CHANGE_SECONDS * fps / WATCH_STEP)))
    reference = [None, None]
    candidate = [None, None]  # (first frame, signature, hits)
    score = list(start_score)
    changes = []
    index = -1
    while True:
        ok = capture.grab()
        if not ok:
            break
        index += 1
        if index % WATCH_STEP:
            continue
        ok, frame = capture.retrieve()
        if not ok:
            break
        board = frame[y:y + h, x:x + w]
        if board.shape[:2] != usual.shape[:2]:
            break
        if not _board_visible(board, usual, digits):
            candidate = [None, None]
            continue
        for side in (0, 1):
            signature = _signature(board, digits[side])
            if reference[side] is None:
                reference[side] = signature
                continue
            if _iou(signature, reference[side]) >= SAME_DIGIT_IOU:
                candidate[side] = None
                continue
            if candidate[side] is None or _iou(
                signature, candidate[side][1]
            ) < SAME_DIGIT_IOU:
                candidate[side] = (index, signature, 1)
                continue
            first, _, hits = candidate[side]
            candidate[side] = (first, signature, hits + 1)
            if hits + 1 >= needed:
                reference[side] = signature
                candidate[side] = None
                score[side] += 1
                changes.append({
                    "frame": first,
                    "time": round(first / fps, 2),
                    "side": ("home", "away")[side],
                    "side_colour": colours[side],
                    "score": f"{score[0]}-{score[1]}",
                })
    capture.release()
    return changes


def parse_score(text):
    home, away = text.split("-")
    return int(home), int(away)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-score", default="0-0",
                        help="score at the start of the video, e.g. 1-0")
    args = parser.parse_args()

    board, box = locate_scoreboard(args.video)
    print("Scoreboard:", box or "not found")
    digits = find_digits(board) if board is not None else None
    if digits:
        home, away = side_colours(board, digits)
        print(f"Home side colour: {home}, away side colour: {away}")
    for change in find_score_changes(args.video, parse_score(args.start_score)):
        minutes, seconds = divmod(change["time"], 60)
        print(f"{int(minutes)}:{seconds:04.1f}  goal for {change['side']}"
              f"  -> {change['score']}")


if __name__ == "__main__":
    main()
