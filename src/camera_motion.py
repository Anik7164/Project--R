"""
Camera motion and scene-cut estimation for broadcast football video.

Each frame is compared with the previous one. Background corners (with
player boxes masked out) are tracked with sparse optical flow, and a
similarity transform (pan, zoom, small rotation) is fitted with RANSAC.
The transform maps pixel positions in the previous frame to the current
frame, so positions of players who left the view can be carried along
while the camera moves.

A scene cut (replay, close-up, switch to another camera) is reported when
the colour content changes abruptly or the background can no longer be
followed. Positions from before a cut are not comparable with positions
after it.
"""

import cv2
import numpy as np

# Work on a downscaled copy for speed.
PROCESS_WIDTH = 640

MAX_CORNERS = 400
CORNER_QUALITY = 0.01
CORNER_MIN_DISTANCE = 8

# Broadcast graphics (scoreboard, clock) are static and would pull the
# estimate towards "no motion", so ignore the top band of the frame.
IGNORE_TOP_RATIO = 0.10
BOX_MASK_PADDING = 0.15

MIN_INLIERS = 12
MIN_TRACKED_RATIO = 0.25
CUT_HIST_CORRELATION = 0.55

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _histogram(frame_small):
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


class CameraMotionEstimator:
    def __init__(self, process_width=PROCESS_WIDTH):
        self.process_width = process_width
        self.prev_gray = None
        self.prev_hist = None
        self.scale = 1.0

    def reset(self):
        self.prev_gray = None
        self.prev_hist = None

    def _background_mask(self, shape, boxes):
        height, width = shape
        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[: int(height * IGNORE_TOP_RATIO), :] = 0

        for x1, y1, x2, y2 in boxes:
            pad_x = (x2 - x1) * BOX_MASK_PADDING
            pad_y = (y2 - y1) * BOX_MASK_PADDING
            left = int(max(0, (x1 - pad_x) * self.scale))
            top = int(max(0, (y1 - pad_y) * self.scale))
            right = int(min(width, (x2 + pad_x) * self.scale))
            bottom = int(min(height, (y2 + pad_y) * self.scale))
            mask[top:bottom, left:right] = 0
        return mask

    def update(self, frame, boxes=()):
        """Return (warp, is_cut) for this frame.

        warp is a 2x3 affine matrix in full-resolution pixels mapping the
        previous frame to this one. On a cut (or the first frame) the warp
        is the identity and is meaningless.
        """
        frame_h, frame_w = frame.shape[:2]
        self.scale = min(1.0, self.process_width / float(frame_w))
        small = cv2.resize(
            frame,
            (int(frame_w * self.scale), int(frame_h * self.scale)),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = _histogram(small)

        prev_gray, prev_hist = self.prev_gray, self.prev_hist
        self.prev_gray, self.prev_hist = gray, hist

        if prev_gray is None:
            return IDENTITY.copy(), True

        correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
        if correlation < CUT_HIST_CORRELATION:
            return IDENTITY.copy(), True

        mask = self._background_mask(gray.shape, boxes)
        corners = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=CORNER_QUALITY,
            minDistance=CORNER_MIN_DISTANCE,
            mask=mask,
        )

        # A flat, textureless view (e.g. tight shot of grass) gives no
        # corners. That is not a cut; assume a static camera.
        if corners is None or len(corners) < MIN_INLIERS:
            return IDENTITY.copy(), False

        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, corners, None
        )
        status = status.reshape(-1).astype(bool)
        if status.mean() < MIN_TRACKED_RATIO:
            return IDENTITY.copy(), True

        src = corners.reshape(-1, 2)[status]
        dst = moved.reshape(-1, 2)[status]

        warp, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0
        )
        if warp is None or inliers is None or inliers.sum() < MIN_INLIERS:
            return IDENTITY.copy(), False

        # Back to full-resolution pixels: only the translation scales.
        warp = warp.astype(np.float64)
        warp[:, 2] /= self.scale
        return warp, False


def warp_points(points, warp):
    """Apply a 2x3 affine warp to an (N, 2) array of points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return points @ warp[:, :2].T + warp[:, 2]


def warp_box(box, warp):
    """Warp an (x1, y1, x2, y2) box, keeping it axis aligned."""
    x1, y1, x2, y2 = box
    corners = warp_points([[x1, y1], [x2, y2]], warp)
    return np.array([
        min(corners[:, 0]), min(corners[:, 1]),
        max(corners[:, 0]), max(corners[:, 1]),
    ])
