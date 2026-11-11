"""
Label the ball in your own videos automatically, with no hand work.

The detector already finds the ball in about half the frames of a local
match. The other half are what it needs to learn, and nobody wants to
draw thousands of boxes. A ball moves smoothly, so the frames around a
miss tell where it was:

  1. every frame of short windows of the video goes through the current
     detector at a LOW confidence (0.05), so a faint ball is still there
     among the candidates;
  2. candidates are linked across frames (camera motion removed) into
     tracks; a track counts as the ball only if it lasts, moves like a
     ball, is confidently seen at least once and is on the grass (white
     caps in the crowd move with the camera and fool the detector);
  3. labels come from those tracks:
       - a faint detection on the track becomes a sure label (the model
         was unsure; now it learns it was right),
       - a frame the model missed inside a short gap (1-2 frames) of a
         track going straight on gets a box placed between the
         neighbours (the model learns the ball it did not see),
       - candidates off the track (heads, socks, line marks) get no box,
         so the model learns they are not the ball;
  4. people boxes are the model's own confident ones (players are
     already found well: mAP50 0.97).

This is self-training (pseudo-labelling): it does not reach what hand
labels give, but it costs nothing and its labels are mostly right
because the track checks every one of them. Frames where the ball
cannot be followed get no labels at all, rather than wrong ones.

Used through train_detector.py:
    python src/train_detector.py auto-label "data/videos/match.mp4"
    python src/train_detector.py auto-finetune "data/videos/match.mp4"
"""

import numpy as np

LOW_CONFIDENCE = 0.05
# Linking: largest ball movement per frame, as a share of frame width.
MAX_STEP_RATIO = 0.06
MAX_LINK_GAP = 6          # frames a track may go unseen and still link
MAX_FILL_GAP = 2          # missed frames filled in between two detections
# ... only where the ball goes straight on: its step before and after the
# gap differ by at most this share of the larger step (plus 2 px).
MAX_TURN = 0.5
MIN_DETECTIONS = 6        # a track is the ball only with this many ...
MIN_PEAK_CONFIDENCE = 0.3  # ... and at least one detection this sure
STILL_MIN_CONFIDENCE = 0.4  # a track that never moves must be this sure
MAX_BALL_SIZE_RATIO = 0.04
# A ball box is about square; a tall or wide one is a leg, a sock or a line.
MAX_ASPECT = 1.8
# A track is dropped if fewer than this share of its points pass `keep`.
MIN_KEEP_SHARE = 0.5


class _Track:
    def __init__(self, frame, point):
        self.points = {frame: point}   # frame -> (x, y, w, h, conf)
        self.velocity = np.zeros(2)

    @property
    def last_frame(self):
        return max(self.points)

    def predict(self, frame):
        last = self.points[self.last_frame]
        return np.array(last[:2]) + self.velocity * (frame - self.last_frame)

    def add(self, frame, point):
        last = self.points[self.last_frame]
        step = (np.array(point[:2]) - np.array(last[:2])) / (frame - self.last_frame)
        self.velocity = step if len(self.points) == 1 else 0.5 * self.velocity + 0.5 * step
        self.points[frame] = point

    def score(self):
        return sum(p[4] for p in self.points.values())


def link(candidates, frame_width):
    """candidates: {frame: [(x, y, w, h, conf)]} in shot-stabilised pixels
    (camera motion removed), for consecutive frames of one shot.
    Returns the tracks (lists are linked greedily, surest first)."""
    max_step = MAX_STEP_RATIO * frame_width
    max_size = MAX_BALL_SIZE_RATIO * frame_width
    tracks, open_tracks = [], []
    for frame in sorted(candidates):
        open_tracks = [t for t in open_tracks if frame - t.last_frame <= MAX_LINK_GAP]
        taken = set()
        for point in sorted(candidates[frame], key=lambda p: -p[4]):
            w, h = max(point[2], 1e-6), max(point[3], 1e-6)
            if max(w, h) > max_size or max(w / h, h / w) > MAX_ASPECT:
                continue
            best, best_distance = None, None
            for track in open_tracks:
                if id(track) in taken:
                    continue
                gap = frame - track.last_frame
                distance = np.linalg.norm(track.predict(frame) - point[:2])
                if distance <= max_step * gap and (best is None or distance < best_distance):
                    best, best_distance = track, distance
            if best is None:
                best = _Track(frame, point)
                tracks.append(best)
                open_tracks.append(best)
            else:
                best.add(frame, point)
            taken.add(id(best))
    return tracks


def is_ball(track):
    points = list(track.points.values())
    if len(points) < MIN_DETECTIONS:
        return False
    confidences = [p[4] for p in points]
    if max(confidences) < MIN_PEAK_CONFIDENCE:
        return False
    xy = np.array([p[:2] for p in points])
    size = float(np.median([max(p[2], p[3]) for p in points]))
    moved = np.linalg.norm(xy - np.median(xy, axis=0), axis=1).max()
    if moved < max(size, 2.0) and np.mean(confidences) < STILL_MIN_CONFIDENCE:
        return False  # a still, faint "ball": a line mark or a spot
    return True


def _straight(track, a, b):
    """Does the ball go straight on across the gap a..b?"""
    frames = sorted(track.points)
    i, j = frames.index(a), frames.index(b)
    if i == 0 or j == len(frames) - 1:
        return False
    p = {f: np.array(track.points[f][:2]) for f in (frames[i - 1], a, b, frames[j + 1])}
    before = (p[a] - p[frames[i - 1]]) / (a - frames[i - 1])
    across = (p[b] - p[a]) / (b - a)
    after = (p[frames[j + 1]] - p[b]) / (frames[j + 1] - b)
    limit = MAX_TURN * max(np.linalg.norm(before), np.linalg.norm(after)) + 2.0
    return (np.linalg.norm(before - across) <= limit
            and np.linalg.norm(after - across) <= limit)


def ball_labels(candidates, frame_width, keep=None):
    """{frame: (x, y, w, h, source)} for the frames where the ball can be
    followed; source is "detected", "faint" (a low-confidence detection
    confirmed by the track) or "filled" (a missed frame inside the
    track). Coordinates are the same (stabilised) pixels as the input.

    keep(frame, (x, y, w, h)) -> bool, if given, vets each label (e.g. is
    it on the grass); a track most of whose points fail is not the ball.
    """
    tracks = sorted((t for t in link(candidates, frame_width) if is_ball(t)),
                    key=lambda t: -t.score())
    labels = {}
    for track in tracks:  # surest track first; one ball per frame
        frames = sorted(track.points)
        own = {}
        for f in frames:
            x, y, w, h, conf = track.points[f]
            own[f] = (x, y, w, h, "detected" if conf >= MIN_PEAK_CONFIDENCE else "faint")
        if keep is not None:
            passed = {f for f in own if keep(f, own[f][:4])}
            if len(passed) < MIN_KEEP_SHARE * len(own):
                continue  # e.g. a white cap in the crowd moving with the camera
            own = {f: own[f] for f in passed}
        for a, b in zip(frames, frames[1:]):
            if 1 < b - a <= MAX_FILL_GAP + 1 and a in own and b in own and _straight(track, a, b):
                pa, pb = np.array(track.points[a][:4]), np.array(track.points[b][:4])
                for f in range(a + 1, b):
                    x, y, w, h = pa + (f - a) / (b - a) * (pb - pa)
                    box = (float(x), float(y), float(w), float(h))
                    if keep is None or keep(f, box):
                        own[f] = (*box, "filled")
        if any(f in labels for f in own):
            continue  # overlaps a surer track: there is only one ball
        labels.update(own)
    return labels


def pick_frames(labels, gap, limit):
    """Frames to save, at least `gap` apart; the ones the model found hard
    (faint, then filled) first, as they teach the most."""
    order = {"faint": 0, "filled": 1, "detected": 2}
    chosen = []
    for frame in sorted(labels, key=lambda f: (order[labels[f][4]], f)):
        if len(chosen) >= limit:
            break
        if all(abs(frame - c) >= gap for c in chosen):
            chosen.append(frame)
    return sorted(chosen)
