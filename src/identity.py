"""
Stable player identities on top of BoT-SORT.

BoT-SORT gives a new track ID whenever it loses a player for longer than
its buffer: after occlusion, after the player leaves the view while the
camera pans, and after every broadcast cut. In a match clip that turns
22 players into hundreds or thousands of IDs.

IdentityManager keeps its own player IDs. Every BoT-SORT track ID that it
has not seen before is held as "pending" for a few frames, then either
re-linked to a player who is currently missing or given a new player ID:

* Positions of missing players are carried along with the estimated
  camera motion, so a player who left the view during a pan is expected
  where they left, in current-frame pixels.
* Inside one shot, a candidate must be near that expected position
  (the allowed distance grows with the time missing) and look similar.
* Across a cut positions mean nothing, so only appearance is used, with a
  stricter threshold and a ratio test: if two missing players look alike
  (e.g. teammates) no link is made, because a wrong link is worse than a
  new ID.

It also drops boxes that duplicate a bigger box in the same frame (a
partial-body second detection of the same player).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from appearance import blend, descriptor_distance, extract_descriptor
from camera_motion import CameraMotionEstimator, warp_box


@dataclass
class IdentityConfig:
    # Frames a new BoT-SORT ID is observed before it gets a player ID.
    confirm_frames: int = 3
    # A pending ID that is not confirmed in this many frames is resolved
    # anyway if seen at least min_pending_hits times, otherwise dropped.
    pending_timeout: int = 12
    min_pending_hits: int = 2

    # Forget missing players after this many frames (~10 s at 30 fps).
    max_missing_frames: int = 300

    # Motion gate, in units of the player's box height.
    gate_base: float = 1.0
    gate_growth_per_frame: float = 0.04
    gate_max: float = 6.0
    # Frames of constant velocity extrapolation for missing players.
    max_extrapolation_frames: int = 15
    # Missing longer than this (~2 s at 30 fps), a player could be
    # anywhere, so only a distinctive appearance can re-link them.
    motion_horizon: int = 60

    # Appearance gates (Hellinger distance, 0 = identical).
    appearance_gate: float = 0.45
    appearance_gate_after_cut: float = 0.30
    # After a cut, or after motion_horizon, the best candidate must be
    # clearly better than the runner-up: best <= ratio * second_best and second_best - best >=
    # margin. Teammates in the same kit never pass this.
    cut_ratio: float = 0.75
    cut_margin: float = 0.10

    # A player entering at the image border can only be someone who was
    # last seen (or is now expected) near or beyond the border, not
    # someone who disappeared in the middle of the view (occlusion).
    border_margin: float = 0.5

    # Appearance score used when a player had no clean view.
    unknown_appearance: float = 0.3

    appearance_weight: float = 0.5
    max_cost: float = 0.85

    # A bound BoT-SORT ID whose appearance stops matching its player for
    # this many frames in a row is treated as an ID switch and re-linked.
    switch_distance: float = 0.6
    switch_patience: int = 5

    # Boxes covered by another box beyond this fraction get no
    # appearance update.
    overlap_limit: float = 0.15

    # Duplicate suppression inside a frame.
    duplicate_containment: float = 0.9
    duplicate_area_ratio: float = 0.6

    feature_momentum: float = 0.9


@dataclass
class Player:
    player_id: int
    box: np.ndarray
    feature: np.ndarray | None
    last_frame: int
    scene: int
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hits: int = 1
    mismatches: int = 0


@dataclass
class Pending:
    first_frame: int
    first_box: np.ndarray
    box: np.ndarray
    last_frame: int
    scene: int
    features: list = field(default_factory=list)
    hits: int = 0

    def mean_feature(self):
        feats = [f for f in self.features if f is not None]
        if not feats:
            return None
        mean = np.mean(feats, axis=0)
        return mean / mean.sum()


def _center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def _height(box):
    return max(1.0, float(box[3] - box[1]))


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def find_duplicates(boxes, confidences, containment=0.9, area_ratio=0.6):
    """Indices of boxes that are mostly inside a larger, more confident box.

    A player is sometimes detected twice: once fully and once as a partial
    body box inside the first. Players standing behind each other have
    boxes of similar size, so only clearly smaller boxes are dropped.
    """
    drop = set()
    for i, small in enumerate(boxes):
        small_area = _area(small)
        if small_area <= 0:
            continue
        for j, big in enumerate(boxes):
            if i == j or j in drop:
                continue
            big_area = _area(big)
            if small_area > area_ratio * big_area:
                continue
            if confidences[i] > confidences[j]:
                continue
            ix1, iy1 = max(small[0], big[0]), max(small[1], big[1])
            ix2, iy2 = min(small[2], big[2]), min(small[3], big[3])
            inter = _area((ix1, iy1, ix2, iy2))
            if inter / small_area >= containment:
                drop.add(i)
                break
    return drop


def overlaps_other(index, boxes, limit):
    """True if more than `limit` of boxes[index] is covered by another box."""
    box = boxes[index]
    area = _area(box)
    if area <= 0:
        return True
    for other_index, other in enumerate(boxes):
        if other_index == index:
            continue
        ix1, iy1 = max(box[0], other[0]), max(box[1], other[1])
        ix2, iy2 = min(box[2], other[2]), min(box[3], other[3])
        if _area((ix1, iy1, ix2, iy2)) / area > limit:
            return True
    return False


class IdentityManager:
    def __init__(self, config=None):
        self.config = config or IdentityConfig()
        self.players = {}
        self.pending = {}
        self.bindings = {}  # BoT-SORT track ID -> player ID
        self.next_player_id = 1
        self.scene = 0
        self.frame = -1
        self.frame_size = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_index, tracker_ids, boxes, features,
               warp=None, scene_cut=False, frame_size=None):
        """Process one frame.

        Args:
            frame_index: frame number.
            tracker_ids: BoT-SORT track IDs of this frame's detections.
            boxes: matching (x1, y1, x2, y2) boxes.
            features: matching appearance descriptors (or None).
            warp: 2x3 camera motion from the previous frame, if known.
            scene_cut: True if this frame starts a new shot.
            frame_size: (width, height) of the frame, enables the border
                rule for players entering the view.

        Returns:
            (assignments, resolved)
            assignments: player ID per detection, or None while pending.
            resolved: {tracker_id: player_id} for pending IDs resolved in
                this frame, so earlier rows of that ID can be filled in.
                A value of None means the ID was dropped as noise.
        """
        self.frame = frame_index
        self.frame_size = frame_size
        boxes = [np.asarray(b, dtype=np.float64) for b in boxes]

        if scene_cut:
            self.scene += 1
            # BoT-SORT may carry an ID across the cut onto a different
            # person; re-check every ID in the new shot.
            self.bindings.clear()
        elif warp is not None:
            self._apply_camera_motion(warp)

        assignments = [None] * len(boxes)
        seen_tracker_ids = set()

        for index, (tid, box, feat) in enumerate(
            zip(tracker_ids, boxes, features)
        ):
            tid = int(tid)
            seen_tracker_ids.add(tid)
            player_id = self.bindings.get(tid)

            if player_id is not None and self._is_switch(player_id, feat):
                del self.bindings[tid]
                player_id = None

            if player_id is not None:
                self._update_player(self.players[player_id], box, feat)
                assignments[index] = player_id
                continue

            pending = self.pending.get(tid)
            if pending is None or pending.scene != self.scene:
                pending = Pending(
                    first_frame=frame_index, first_box=box, box=box,
                    last_frame=frame_index, scene=self.scene,
                )
                self.pending[tid] = pending
            pending.box = box
            pending.last_frame = frame_index
            pending.features.append(feat)
            pending.hits += 1

        resolved = self._resolve_pending(seen_tracker_ids)

        for index, tid in enumerate(tracker_ids):
            if assignments[index] is None:
                assignments[index] = self.bindings.get(int(tid))

        self._forget_old_players()
        return assignments, resolved

    def _near_border(self, center, height):
        if self.frame_size is None:
            return True
        width, frame_height = self.frame_size
        margin = self.config.border_margin * height
        x, y = center
        return (
            x < margin or y < margin
            or x > width - margin or y > frame_height - margin
        )

    def flush(self):
        """Resolve every pending ID at the end of a video."""
        return self._resolve_pending(set(), force=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_camera_motion(self, warp):
        for player in self.players.values():
            if player.scene != self.scene:
                continue
            player.box = warp_box(player.box, warp)
            player.velocity = warp[:, :2] @ player.velocity
        for pending in self.pending.values():
            pending.first_box = warp_box(pending.first_box, warp)
            pending.box = warp_box(pending.box, warp)

    def _is_switch(self, player_id, feature):
        player = self.players[player_id]
        if feature is None or player.feature is None:
            return False
        if descriptor_distance(player.feature, feature) > self.config.switch_distance:
            player.mismatches += 1
        else:
            player.mismatches = 0
        if player.mismatches >= self.config.switch_patience:
            player.mismatches = 0
            return True
        return False

    def _update_player(self, player, box, feature):
        gap = self.frame - player.last_frame
        if player.scene == self.scene and 0 < gap <= 5:
            step = (_center(box) - _center(player.box)) / gap
            player.velocity = 0.6 * player.velocity + 0.4 * step
        else:
            player.velocity = np.zeros(2)
        player.box = box
        player.feature = blend(
            player.feature, feature, self.config.feature_momentum
        )
        player.last_frame = self.frame
        player.scene = self.scene
        player.hits += 1

    def _new_player(self, pending):
        player = Player(
            player_id=self.next_player_id,
            box=pending.box,
            feature=pending.mean_feature(),
            last_frame=pending.last_frame,
            scene=self.scene,
            hits=pending.hits,
        )
        self.players[player.player_id] = player
        self.next_player_id += 1
        return player

    def _ready(self, pending, force):
        if force or pending.hits >= self.config.confirm_frames:
            return True
        age = self.frame - pending.first_frame
        return age >= self.config.pending_timeout or pending.scene != self.scene

    def _resolve_pending(self, seen_tracker_ids, force=False):
        cfg = self.config
        resolved = {}
        ready = []

        for tid, pending in list(self.pending.items()):
            if not self._ready(pending, force):
                continue
            if pending.hits < min(cfg.min_pending_hits, cfg.confirm_frames):
                resolved[tid] = None
                del self.pending[tid]
                continue
            ready.append(tid)

        if not ready:
            return resolved

        # A player can only be re-linked if nobody holds them right now.
        active = {
            pid for tid, pid in self.bindings.items()
            if tid in seen_tracker_ids
        }
        candidates = [
            p for p in self.players.values()
            if p.player_id not in active
            and self.frame - p.last_frame <= cfg.max_missing_frames
        ]

        matches = self._match(ready, candidates)

        for tid in ready:
            pending = self.pending.pop(tid)
            player = matches.get(tid)
            if player is None:
                player = self._new_player(pending)
            else:
                # Drop any stale BoT-SORT ID still pointing at this player.
                for old_tid in [t for t, p in self.bindings.items()
                                if p == player.player_id]:
                    del self.bindings[old_tid]
                player.box = pending.box
                player.feature = blend(
                    player.feature, pending.mean_feature(),
                    cfg.feature_momentum,
                )
                player.last_frame = pending.last_frame
                player.scene = pending.scene
                player.velocity = np.zeros(2)
                player.hits += pending.hits
            self.bindings[tid] = player.player_id
            resolved[tid] = player.player_id

        return resolved

    def _match(self, pending_ids, candidates):
        cfg = self.config
        if not candidates:
            return {}

        inf = 1e6
        cost = np.full((len(pending_ids), len(candidates)), inf)
        appearance = np.ones_like(cost)
        appearance_only = np.zeros(cost.shape, dtype=bool)

        for i, tid in enumerate(pending_ids):
            pending = self.pending[tid]
            feature = pending.mean_feature()
            for j, player in enumerate(candidates):
                # Seen together in one frame means two different people.
                if player.last_frame >= pending.first_frame:
                    continue
                gap = pending.first_frame - player.last_frame
                app = descriptor_distance(feature, player.feature)
                appearance[i, j] = app

                # Positions are only comparable inside one shot and for a
                # short time; after that the player could be anywhere.
                if player.scene != pending.scene or gap > cfg.motion_horizon:
                    appearance_only[i, j] = True
                    if app <= cfg.appearance_gate_after_cut:
                        cost[i, j] = app / cfg.appearance_gate_after_cut
                    continue

                if feature is None or player.feature is None:
                    # No clean view yet (overlapping players): decide on
                    # motion alone, with a neutral appearance score.
                    app = cfg.unknown_appearance
                if app > cfg.appearance_gate:
                    continue

                steps = min(gap, cfg.max_extrapolation_frames)
                expected = _center(player.box) + player.velocity * steps
                height = _height(player.box)
                distance = np.linalg.norm(
                    _center(pending.first_box) - expected
                ) / height
                gate = min(
                    cfg.gate_max,
                    cfg.gate_base + cfg.gate_growth_per_frame * gap,
                )
                if distance > gate:
                    continue
                entering = self._near_border(
                    _center(pending.first_box), _height(pending.first_box)
                )
                if entering and not self._near_border(expected, height):
                    continue
                cost[i, j] = (
                    cfg.appearance_weight * app / cfg.appearance_gate
                    + (1 - cfg.appearance_weight) * distance / gate
                )

        # Without position information, refuse ambiguous links: the best
        # candidate must look clearly different from every other one.
        for i, j in zip(*np.nonzero(appearance_only & (cost < inf))):
            others = np.delete(appearance[i], j)
            if len(others) == 0:
                continue
            runner_up = others.min()
            if (
                appearance[i, j] > cfg.cut_ratio * runner_up
                or runner_up - appearance[i, j] < cfg.cut_margin
            ):
                cost[i, j] = inf

        rows, cols = linear_sum_assignment(cost)
        matches = {}
        for i, j in zip(rows, cols):
            if cost[i, j] <= cfg.max_cost:
                matches[pending_ids[i]] = candidates[j]
        return matches

    def _forget_old_players(self):
        limit = self.config.max_missing_frames
        old = {
            pid for pid, p in self.players.items()
            if self.frame - p.last_frame > limit
        }
        for pid in old:
            del self.players[pid]
        for tid in [t for t, pid in self.bindings.items() if pid in old]:
            del self.bindings[tid]


class StablePlayerTracker:
    """Per-frame glue: duplicate removal, camera motion, appearance and
    identity, for the detections BoT-SORT returns on one frame."""

    def __init__(self, config=None):
        self.identity = IdentityManager(config)
        self.camera = CameraMotionEstimator()
        self.last_cut = False

    def process(self, frame_index, frame, tracker_ids, boxes, confidences):
        """Return (keep, assignments, resolved).

        keep: indices of the input detections that are not duplicates.
        assignments: player ID (or None while pending) for each kept index.
        resolved: see IdentityManager.update.
        """
        cfg = self.identity.config
        duplicates = find_duplicates(
            boxes, confidences,
            cfg.duplicate_containment, cfg.duplicate_area_ratio,
        )
        keep = [i for i in range(len(boxes)) if i not in duplicates]

        warp, cut = self.camera.update(frame, [boxes[i] for i in keep])
        self.last_cut = cut
        # A box that overlaps another one mixes two players' colours;
        # leave it out of the appearance model.
        kept_boxes = [boxes[i] for i in keep]
        features = [
            None if overlaps_other(index, kept_boxes, cfg.overlap_limit)
            else extract_descriptor(frame, boxes[i])
            for index, i in enumerate(keep)
        ]

        assignments, resolved = self.identity.update(
            frame_index,
            [tracker_ids[i] for i in keep],
            [boxes[i] for i in keep],
            features,
            warp=warp,
            scene_cut=cut,
            frame_size=(frame.shape[1], frame.shape[0]),
        )
        return keep, assignments, resolved

    def flush(self):
        return self.identity.flush()
