"""
Lightweight appearance descriptors for football players.

A descriptor is a colour histogram of the shirt and shorts regions of a
player box, with grass pixels removed. It is cheap enough to run on every
detection in real time on a CPU, and it separates teams, goalkeepers and
referees well. It cannot tell two players of the same team apart; the
identity layer relies on motion for that, and later on jersey numbers.
"""

import cv2
import numpy as np

# Vertical bands of the person box (fractions of box height).
SHIRT_BAND = (0.15, 0.50)
SHORTS_BAND = (0.50, 0.72)
# Ignore the box edges, which mostly contain background.
SIDE_MARGIN = 0.20

# Pixels below this saturation are described by brightness, not hue,
# so white, black and grey kits still get a useful descriptor.
MIN_SATURATION = 50
HUE_BINS = 16
SAT_BINS = 4
VALUE_BINS = 8

# OpenCV hue range (0-180) that counts as grass.
GRASS_HUE = (32, 90)
GRASS_MIN_SATURATION = 40

SHIRT_WEIGHT = 0.7

DESCRIPTOR_SIZE = 2 * (HUE_BINS * SAT_BINS + VALUE_BINS)


def grass_mask(hsv):
    """Return True for pixels that look like pitch grass."""
    hue, sat = hsv[..., 0], hsv[..., 1]
    return (
        (hue >= GRASS_HUE[0]) & (hue <= GRASS_HUE[1])
        & (sat >= GRASS_MIN_SATURATION)
    )


def _band(frame, box, band):
    x1, y1, x2, y2 = box
    height = y2 - y1
    width = x2 - x1
    top = int(y1 + band[0] * height)
    bottom = int(y1 + band[1] * height)
    left = int(x1 + SIDE_MARGIN * width)
    right = int(x2 - SIDE_MARGIN * width)

    frame_h, frame_w = frame.shape[:2]
    top, bottom = max(0, top), min(frame_h, bottom)
    left, right = max(0, left), min(frame_w, right)

    if bottom <= top or right <= left:
        return None
    return frame[top:bottom, left:right]


def _region_histogram(region):
    """Hue/saturation histogram for coloured pixels plus a value histogram
    for unsaturated ones, normalised to sum to one."""
    size = HUE_BINS * SAT_BINS + VALUE_BINS
    if region is None or region.size == 0:
        return np.zeros(size, dtype=np.float32)

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hsv = hsv[~grass_mask(hsv)]
    if len(hsv) == 0:
        return np.zeros(size, dtype=np.float32)

    coloured = hsv[hsv[:, 1] >= MIN_SATURATION]
    plain = hsv[hsv[:, 1] < MIN_SATURATION]

    hue_sat, _, _ = np.histogram2d(
        coloured[:, 0], coloured[:, 1],
        bins=(HUE_BINS, SAT_BINS),
        range=((0, 180), (MIN_SATURATION, 256)),
    )
    value, _ = np.histogram(plain[:, 2], bins=VALUE_BINS, range=(0, 256))

    hist = np.concatenate([hue_sat.ravel(), value]).astype(np.float32)
    return hist / hist.sum()


def extract_descriptor(frame, box):
    """Return the appearance descriptor of one player box (x1, y1, x2, y2).

    Returns None when the box is too small or almost all grass.
    """
    box = [float(v) for v in box]
    if box[2] - box[0] < 4 or box[3] - box[1] < 10:
        return None

    shirt = _region_histogram(_band(frame, box, SHIRT_BAND))
    shorts = _region_histogram(_band(frame, box, SHORTS_BAND))

    if shirt.sum() == 0 and shorts.sum() == 0:
        return None

    descriptor = np.concatenate([
        SHIRT_WEIGHT * shirt,
        (1.0 - SHIRT_WEIGHT) * shorts,
    ])
    return descriptor / descriptor.sum()


def descriptor_distance(a, b):
    """Hellinger distance between two descriptors, in [0, 1]."""
    if a is None or b is None:
        return 1.0
    overlap = float(np.sum(np.sqrt(a * b)))
    return float(np.sqrt(max(0.0, 1.0 - overlap)))


def blend(old, new, momentum=0.9):
    """Exponential moving average of descriptors, renormalised."""
    if old is None:
        return new
    if new is None:
        return old
    mixed = momentum * old + (1.0 - momentum) * new
    return mixed / mixed.sum()
