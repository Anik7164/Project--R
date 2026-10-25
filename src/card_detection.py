"""
Spot a yellow or red card held up by the referee.

A shown card is a small, bright, solid yellow or red rectangle above the
referee's head. Two ways to find it:

* detect_card: next to a known referee box (wide shots, where the
  referee is tracked). Pixels inside other people's boxes are ignored,
  so a yellow shirt behind the referee is not mistaken for a card.
* find_shown_cards: anywhere in the frame, for close-ups where the
  referee is not tracked. A very bright, saturated, solid, compact blob
  standing alone (not part of an advert or a shirt) is a candidate; it
  counts only when a person detector finds someone right below it (a
  raised arm), and only if that keeps happening for about half a second.
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


# ----------------------------------------------------------------------
# Cards anywhere in the frame (close-ups)
# ----------------------------------------------------------------------

# Cards are lit and printed in pure colours: brighter and more saturated
# than shirts in most light.
SHOWN_MIN_SATURATION = 170
SHOWN_MIN_VALUE = 220
SHOWN_RED_HUES = ((0, 12), (172, 180))
SHOWN_YELLOW_HUE = (20, 34)
# Blob area as a share of the frame.
SHOWN_MIN_AREA = 0.0006
SHOWN_MAX_AREA = 0.006
SHOWN_MIN_SOLIDITY = 0.85
SHOWN_ELONGATION = (1.2, 3.0)
# The card is held up: in the upper part of the frame.
SHOWN_MAX_Y = 0.6
# Share of equally bright, saturated pixels allowed around the blob
# (adverts and graphics are surrounded by more of the same).
SHOWN_MAX_BUSY = 0.05
# A person whose box top is at most this many box heights below the card
# centre and who is at least this much taller than the card holds it.
HOLDER_BELOW = 1.0
HOLDER_ABOVE = 0.2
HOLDER_SIDE = 0.5
HOLDER_MIN_SCALE = 1 / 0.35
HOLDER_MAX_SCALE = 1 / 0.05
# Confirmed sightings needed, at most SHOWN_GAP seconds apart.
SHOWN_MIN_HITS = 5
SHOWN_GAP = 0.5


def card_candidates(frame):
    """[(colour, (cx, cy), area)] of card-like blobs in one frame."""
    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (sat >= SHOWN_MIN_SATURATION) & (val >= SHOWN_MIN_VALUE)
    red = bright & (
        ((hue >= SHOWN_RED_HUES[0][0]) & (hue <= SHOWN_RED_HUES[0][1]))
        | ((hue >= SHOWN_RED_HUES[1][0]) & (hue <= SHOWN_RED_HUES[1][1]))
    )
    yellow = bright & (hue >= SHOWN_YELLOW_HUE[0]) & (hue <= SHOWN_YELLOW_HUE[1])

    found = []
    kernel = np.ones((3, 3), np.uint8)
    for colour, mask in (("red", red), ("yellow", yellow)):
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if not SHOWN_MIN_AREA * frame_area <= area <= SHOWN_MAX_AREA * frame_area:
                continue
            if x == 0 or x + w >= frame_w or y + h >= frame_h:
                continue
            blob = (labels[y:y + h, x:x + w] == label).astype(np.uint8)
            contour = cv2.findContours(
                blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0][0]
            hull = cv2.contourArea(cv2.convexHull(contour))
            (cx, cy), (rw, rh), _ = cv2.minAreaRect(contour)
            if hull == 0 or min(rw, rh) == 0:
                continue
            cx, cy = x + cx, y + cy
            if cy > SHOWN_MAX_Y * frame_h:
                continue
            if area / hull < SHOWN_MIN_SOLIDITY:
                continue
            if not SHOWN_ELONGATION[0] <= max(rw, rh) / min(rw, rh) <= SHOWN_ELONGATION[1]:
                continue
            pad = max(w, h)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(frame_w, x + w + pad), min(frame_h, y + h + pad)
            ring = bright[y0:y1, x0:x1].copy()
            ring[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
            ring_area = (y1 - y0) * (x1 - x0) - w * h
            if ring.sum() / max(1, ring_area) >= SHOWN_MAX_BUSY:
                continue
            found.append((colour, (float(cx), float(cy)), int(area)))
    return found


def held_up(candidate, person_boxes):
    """True if someone in person_boxes stands right below the card."""
    _, (cx, cy), area = candidate
    side = np.sqrt(area)
    for x1, y1, x2, y2 in person_boxes:
        width, height = x2 - x1, y2 - y1
        if not HOLDER_MIN_SCALE * side <= height <= HOLDER_MAX_SCALE * side:
            continue
        if not x1 - HOLDER_SIDE * width <= cx <= x2 + HOLDER_SIDE * width:
            continue
        if y1 - HOLDER_BELOW * height <= cy <= y1 + HOLDER_ABOVE * height:
            return True
    return False


def find_shown_cards(video_path, detect_people, fps=None):
    """[(first frame, "yellow_card" | "red_card")] held up anywhere.

    detect_people(frame) -> list of (x1, y1, x2, y2) person boxes; it is
    only called on frames with a card-like blob, so it can be a full
    detector.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = fps or capture.get(cv2.CAP_PROP_FPS) or 25.0
    hits = {"yellow": [], "red": []}
    index = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            candidates = card_candidates(frame)
            if not candidates:
                continue
            people = detect_people(frame)
            for colour in {c[0] for c in candidates if held_up(c, people)}:
                hits[colour].append(index)
    finally:
        capture.release()

    cards = []
    gap = SHOWN_GAP * fps
    for colour, frames in hits.items():
        group = []
        for frame in frames + [None]:
            if group and (frame is None or frame - group[-1] > gap):
                if len(group) >= SHOWN_MIN_HITS:
                    cards.append((group[0], f"{colour}_card"))
                group = []
            if frame is not None:
                group.append(frame)
    return sorted(cards)


def yolo_people(model_path):
    """A detect_people function using an Ultralytics YOLO model."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    def detect(frame):
        result = model.predict(frame, classes=[0], conf=0.25, verbose=False)[0]
        return [tuple(map(float, b)) for b in result.boxes.xyxy.cpu().numpy()]

    return detect
