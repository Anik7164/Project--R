"""
Spot a yellow or red card held up by the referee.

A shown card is a small, bright, solid yellow or red rectangle above the
referee's head. Pixels inside other people's boxes are ignored, so a
yellow shirt behind the referee is not mistaken for a card.
"""

import cv2
import numpy as np

# Search region around the referee box, in box widths/heights.
SIDE_REACH = 0.7
ABOVE_REACH = 0.45
BELOW_TOP = 0.10

MIN_SATURATION = 120
YELLOW_HUE = (18, 35)
YELLOW_MIN_VALUE = 150
RED_HUES = ((0, 8), (170, 180))
RED_MIN_VALUE = 100

# Card area relative to the referee's box height squared.
MIN_AREA = 0.002
MAX_AREA = 0.05
MIN_FILL = 0.55
ASPECT_RANGE = (0.35, 2.8)


def _search_region(frame_shape, box):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    frame_h, frame_w = frame_shape[:2]
    left = int(max(0, x1 - SIDE_REACH * width))
    right = int(min(frame_w, x2 + SIDE_REACH * width))
    top = int(max(0, y1 - ABOVE_REACH * height))
    bottom = int(min(frame_h, y1 + BELOW_TOP * height))
    return left, top, right, bottom


def _has_card(mask, height):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    area_min = MIN_AREA * height * height
    area_max = MAX_AREA * height * height
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if not area_min <= area <= area_max:
            continue
        if area / float(w * h) < MIN_FILL:
            continue
        if not ASPECT_RANGE[0] <= h / float(w) <= ASPECT_RANGE[1]:
            continue
        return True
    return False


def detect_card(frame, referee_box, other_boxes=()):
    """Return "yellow", "red" or None for one referee box in one frame."""
    left, top, right, bottom = _search_region(frame.shape, referee_box)
    if right <= left or bottom <= top:
        return None

    hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    ignore = np.zeros(hue.shape, dtype=bool)
    for x1, y1, x2, y2 in other_boxes:
        ix1, iy1 = int(max(x1, left)) - left, int(max(y1, top)) - top
        ix2, iy2 = int(min(x2, right)) - left, int(min(y2, bottom)) - top
        if ix2 > ix1 and iy2 > iy1:
            ignore[iy1:iy2, ix1:ix2] = True

    saturated = (sat >= MIN_SATURATION) & ~ignore
    yellow = (
        saturated & (hue >= YELLOW_HUE[0]) & (hue <= YELLOW_HUE[1])
        & (val >= YELLOW_MIN_VALUE)
    )
    red = saturated & (val >= RED_MIN_VALUE) & (
        ((hue >= RED_HUES[0][0]) & (hue <= RED_HUES[0][1]))
        | ((hue >= RED_HUES[1][0]) & (hue <= RED_HUES[1][1]))
    )

    height = referee_box[3] - referee_box[1]
    if _has_card(yellow.astype(np.uint8), height):
        return "yellow"
    if _has_card(red.astype(np.uint8), height):
        return "red"
    return None
