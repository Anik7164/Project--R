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
# Hue bins start at this hue, so that one bin boundary falls between red
# (hue 0-3, 175-180) and orange: phone footage puts orange kits at hue 5-11
# (match1: median 6), which a bin starting at 0 lumped with red.
HUE_OFFSET = 4
SAT_BINS = 4
VALUE_BINS = 8

# OpenCV hue range (0-180) that counts as grass.
GRASS_HUE = (32, 90)
GRASS_MIN_SATURATION = 40

SHIRT_WEIGHT = 0.7
DARK_VALUE = 80

DESCRIPTOR_SIZE = 2 * (HUE_BINS * SAT_BINS + VALUE_BINS)


def grass_mask(hsv):
    """Return True for pixels that look like pitch grass."""
    hue, sat = hsv[..., 0], hsv[..., 1]
    return (
        (hue >= GRASS_HUE[0]) & (hue <= GRASS_HUE[1])
        & (sat >= GRASS_MIN_SATURATION)
    )


# Share of grass pixels around the feet below which a person is not on the
# pitch (spectators, staff in the stands, faces in close-ups, logos).
MIN_GRASS_AT_FEET = 0.25


# The pitch is not always green: sun-dried grass and colour-graded phone
# video put it at hue 20-30 (match5), bare earth patches lower still. So
# the colour of the ground is also learnt from each frame: the hues of the
# bottom part of the picture (where the pitch is when the camera looks at
# play), as the narrowest hue window holding PITCH_SHARE of its coloured
# pixels, at most PITCH_MAX_WIDTH wide.
PITCH_ROWS = 0.4
PITCH_SHARE = 0.8
PITCH_MAX_WIDTH = 24


def pitch_hues(frame):
    """(low, high) hue of this frame's ground, or None if it has too few
    coloured pixels to tell (night, grey artificial turf)."""
    height = frame.shape[0]
    ground = cv2.cvtColor(frame[int(height * (1 - PITCH_ROWS)):, ::4], cv2.COLOR_BGR2HSV)
    hue = ground[..., 0][(ground[..., 1] >= GRASS_MIN_SATURATION) & (ground[..., 2] >= 40)]
    if hue.size < 500:
        return None
    counts = np.bincount(hue.ravel(), minlength=180).astype(float)
    need = PITCH_SHARE * counts.sum()
    cumulative = np.concatenate([[0.0], np.cumsum(counts)])
    for width in range(1, PITCH_MAX_WIDTH + 1):
        sums = cumulative[width:] - cumulative[:-width]
        start = int(np.argmax(sums))
        if sums[start] >= need:
            return start, start + width - 1
    return None


def on_pitch(frame, box, ground=None):
    """True if the person in box is standing on the pitch: on green grass,
    or on the ground colour of this frame (ground = pitch_hues(frame))."""
    x1, y1, x2, y2 = [float(v) for v in box]
    width, height = x2 - x1, y2 - y1
    frame_h, frame_w = frame.shape[:2]
    top = int(max(0, y2 - 0.1 * height))
    bottom = int(min(frame_h, y2 + 0.1 * height))
    left = int(max(0, x1 - 0.25 * width))
    right = int(min(frame_w, x2 + 0.25 * width))
    if bottom <= top or right <= left:
        return False
    region = frame[top:bottom, left:right]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pitch = grass_mask(hsv)
    if ground is not None:
        hue = hsv[..., 0]
        pitch |= ((hue >= ground[0]) & (hue <= ground[1])
                  & (hsv[..., 1] >= GRASS_MIN_SATURATION))
    return float(np.mean(pitch)) >= MIN_GRASS_AT_FEET


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

    # Very dark pixels have no reliable hue (phone video gives black kits
    # a strong colour cast), so they count as plain dark.
    is_coloured = (hsv[:, 1] >= MIN_SATURATION) & (hsv[:, 2] >= DARK_VALUE)
    coloured = hsv[is_coloured]
    plain = hsv[~is_coloured]

    hue = (coloured[:, 0].astype(np.float32) - HUE_OFFSET) % 180
    hue_sat, _, _ = np.histogram2d(
        hue, coloured[:, 1],
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


# Colour names for commentary, keyed by OpenCV hue (0-180) upper bound.
HUE_NAMES = [
    (5, "red"), (20, "orange"), (33, "yellow"), (85, "green"),
    (98, "sky blue"), (130, "blue"), (150, "purple"), (165, "pink"),
    (181, "red"),
]

# Pale kits (white, very light sky blue) in shade come out around value
# 160-170; real grey kits are darker.
WHITE_VALUE = 150

BENGALI_COLOUR_NAMES = {
    "red": "লাল",
    "orange": "কমলা",
    "yellow": "হলুদ",
    "green": "সবুজ",
    "sky blue": "আকাশি",
    "blue": "নীল",
    "purple": "বেগুনি",
    "pink": "গোলাপি",
    "white": "সাদা",
    "grey": "ধূসর",
    "black": "কালো",
}


def team_colour_names(first, second):
    """Names of two teams' kits from their mean descriptors. What both
    teams' crops share (skin, the pitch showing between arm and body, the
    colour cast of the video) is taken out first, so each team is named by
    what sets it apart: on warm phone video both teams otherwise come out
    "orange"."""
    if first is None or second is None:
        return shirt_colour_name(first), shirt_colour_name(second)
    common = np.minimum(first, second)
    names = []
    for own in (first, second):
        rest = own - common
        names.append(shirt_colour_name(rest / rest.sum()) if rest.sum() > 0.05
                     else shirt_colour_name(own))
    return tuple(names)


def shirt_colour_name(descriptor):
    """English name of the dominant shirt colour in a descriptor."""
    if descriptor is None:
        return None
    coloured_size = HUE_BINS * SAT_BINS
    shirt = descriptor[: coloured_size + VALUE_BINS]
    hue_sat = shirt[:coloured_size].reshape(HUE_BINS, SAT_BINS)
    value = shirt[coloured_size:]

    if hue_sat.sum() >= value.sum():
        per_bin = hue_sat.sum(axis=1)
        hue_bin = int(np.argmax(per_bin))
        # Weighted circular mean of the peak bin and its neighbours: finer
        # than one bin (11 hue steps), which matters between red and orange.
        bins = [(hue_bin + d) % HUE_BINS for d in (-1, 0, 1)]
        angles = np.array([(b + 0.5) * 2 * np.pi / HUE_BINS for b in bins])
        weights = per_bin[bins]
        angle = np.arctan2(weights @ np.sin(angles), weights @ np.cos(angles))
        hue = ((angle % (2 * np.pi)) * 180.0 / (2 * np.pi) + HUE_OFFSET) % 180
        for upper, name in HUE_NAMES:
            if hue < upper:
                return name

    mean_value = float(
        np.dot(value, (np.arange(VALUE_BINS) + 0.5) * 256.0 / VALUE_BINS)
        / max(value.sum(), 1e-9)
    )
    if mean_value < 70:
        return "black"
    if mean_value < WHITE_VALUE:
        return "grey"
    return "white"
