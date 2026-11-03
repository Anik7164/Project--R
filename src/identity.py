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

When players run into each other BoT-SORT sometimes swaps their IDs or
carries an ID onto the other player. Three checks catch that:

* A player's colour model is frozen while the view disagrees with it.
* After two players touch, an ID whose view looks clearly more like the
  other player is handed back (or the two IDs are exchanged).
* A team colour model is fitted online; an ID whose player suddenly
  wears the other team's shirt is released, and pending IDs are never
  linked to a player of the other team.

It also drops boxes that duplicate a bigger box in the same frame (a
partial-body second detection of the same player).
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from appearance import blend, descriptor_distance, extract_descriptor, on_pitch
from camera_motion import CameraMotionEstimator, warp_box
from team_identification import TEAM_LABELS, TeamClassifier


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

    # Motion gate, in units of the player's box height. A sprinting player
    # covers about 0.15 of their own height per frame at 25 fps. Tuned on
    # broadcast footage (test1.mp4).
    gate_base: float = 1.5
    gate_growth_per_frame: float = 0.15
    gate_max: float = 6.0
    # Frames of constant velocity extrapolation for missing players.
    max_extrapolation_frames: int = 15
    # Missing longer than this (5 s at 25 fps), a player could be
    # anywhere, so only a distinctive appearance can re-link them.
    motion_horizon: int = 125

    # Appearance gates (Hellinger distance, 0 = identical). On real
    # broadcast crops, two views of the same player 0.4-2 s apart are
    # within 0.57 in 90% of cases (small, blurred players), so the gate
    # inside one shot is loose; it mainly keeps the other team out.
    appearance_gate: float = 0.65
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

    # Inside a shot, a link after more than ambiguity_free_gap frames is
    # refused when the runner-up candidate's cost is within this margin.
    ambiguity_margin: float = 0.1
    ambiguity_free_gap: int = 10

    # Appearance score used when a player had no clean view; such links
    # also need the player within gate_base box heights.
    unknown_appearance: float = 0.3

    appearance_weight: float = 0.5
    max_cost: float = 0.85

    # A bound BoT-SORT ID whose appearance stops matching its player for
    # this many frames in a row is treated as an ID switch and re-linked.
    switch_distance: float = 0.6
    switch_patience: int = 3

    # After two players touch, BoT-SORT often swaps their IDs. For this
    # many frames after the contact, a player whose view matches the other
    # player better (by contact_margin) is swapped back at once.
    contact_window: int = 50
    contact_margin: float = 0.1
    contact_patience: int = 2
    # Only for players whose colours differ this much (different kits);
    # two teammates look alike and colour cannot tell them apart.
    contact_min_difference: float = 0.4

    # Online team colours: fitted once this many clean views are collected,
    # refitted every team_refit_frames on the latest team_buffer views.
    team_check: bool = True
    team_warmup: int = 150
    team_refit_frames: int = 250
    team_buffer: int = 3000
    # A player's team is known after this many votes with this share.
    team_min_votes: int = 3
    team_min_share: float = 0.75
    # Clean views in a row showing the other team before an ID is released.
    team_switch_patience: int = 3
    # A view counts for a team only if it is this much closer to that
    # team's colour than to the other (goalkeepers fit neither well).
    team_ratio: float = 0.8

    # Boxes covered by another box beyond this fraction get no
    # appearance update.
    overlap_limit: float = 0.15

    # Drop people whose feet are not on grass (crowd, staff, close-ups).
    require_on_pitch: bool = True

    # Duplicate suppression inside a frame.
    duplicate_containment: float = 0.9
    duplicate_area_ratio: float = 0.6

    feature_momentum: float = 0.9

    # Deep appearance (ReID) embeddings, when a ReID model is given to
    # StablePlayerTracker. Re-linking then compares
    #   (1 - reid_weight) * colour distance + reid_weight * reid_scale * ReID distance
    # instead of colour alone. On test1.mp4 (LMBN, Market-1501) the ReID
    # cosine distance separates teammates better than colour (AUC 0.85 vs
    # 0.79); the same player's views are within 0.39 in 90% of cases,
    # hence reid_scale ~ 0.57 / 0.39 to put it on the colour scale.
    reid_weight: float = 0.5
    reid_scale: float = 1.5
    # Beyond this ReID distance two views are never the same player.
    reid_gate: float = 0.6


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
    # Other player ID -> last frame the two boxes touched.
    contacts: dict = field(default_factory=dict)
    swap_votes: dict = field(default_factory=dict)
    team_votes: Counter = field(default_factory=Counter)
    wrong_team: int = 0
    embedding: np.ndarray | None = None

    def team(self, min_votes, min_share):
        total = sum(self.team_votes.values())
        if total < min_votes:
            return None
        label, count = self.team_votes.most_common(1)[0]
        return label if count >= min_share * total else None


@dataclass
class Pending:
    first_frame: int
    first_box: np.ndarray
    box: np.ndarray
    last_frame: int
    scene: int
    features: list = field(default_factory=list)
    teams: list = field(default_factory=list)
    embeddings: list = field(default_factory=list)
    hits: int = 0

    def mean_embedding(self):
        return _blend_embedding(None, [e for e in self.embeddings if e is not None])

    def team(self):
        votes = Counter(t for t in self.teams if t in TEAM_LABELS)
        if not votes:
            return None
        label, count = votes.most_common(1)[0]
        return label if count > sum(votes.values()) / 2 else None

    def mean_feature(self):
        feats = [f for f in self.features if f is not None]
        if not feats:
            return None
        mean = np.mean(feats, axis=0)
        return mean / mean.sum()


def _blend_embedding(old, new, momentum=0.9):
    """Moving average of L2-normalised embeddings. new: one embedding,
    a list of them (averaged) or None."""
    if isinstance(new, list):
        new = np.mean(new, axis=0) if new else None
    if new is None:
        return old
    mixed = new if old is None else momentum * old + (1.0 - momentum) * new
    norm = np.linalg.norm(mixed)
    return mixed / norm if norm > 0 else old


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


def _touching(a, b, slack=0.1):
    """True if two boxes overlap or nearly do (slack in box heights)."""
    pad = slack * min(_height(a), _height(b))
    return (
        a[0] - pad < b[2] and b[0] - pad < a[2]
        and a[1] - pad < b[3] and b[1] - pad < a[3]
    )


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
        self.teams = TeamClassifier()
        self.team_samples = []
        self.team_fitted_at = None
        self.touching = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_index, tracker_ids, boxes, features,
               warp=None, scene_cut=False, frame_size=None, others=(),
               embeddings=None):
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
            others: boxes of other people in the frame that are not
                tracked (dropped duplicates, spectators); a player touching
                one is not a clean view.
            embeddings: matching L2-normalised ReID embeddings (or None),
                optional.

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

        if embeddings is None:
            embeddings = [None] * len(boxes)
        assignments = [None] * len(boxes)
        seen_tracker_ids = set()
        teams = self._predict_teams(features)
        self._record_contacts(tracker_ids, boxes, others)
        self._fix_swaps(tracker_ids, features)

        for index, (tid, box, feat, team, emb) in enumerate(
            zip(tracker_ids, boxes, features, teams, embeddings)
        ):
            tid = int(tid)
            seen_tracker_ids.add(tid)
            player_id = self.bindings.get(tid)

            if player_id is not None and (
                self._is_switch(player_id, feat)
                or self._changed_team(player_id, team)
            ):
                del self.bindings[tid]
                player_id = None

            if player_id is not None:
                self._update_player(self.players[player_id], box, feat, emb)
                if team is not None:
                    self.players[player_id].team_votes[team] += 1
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
            pending.teams.append(team)
            pending.embeddings.append(emb)
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

    def _predict_teams(self, features):
        """Team label per detection (None without a clean view or before
        the team colours are known)."""
        cfg = self.config
        if not cfg.team_check:
            return [None] * len(features)
        clean = [f for f in features if f is not None]
        self.team_samples.extend(clean)
        del self.team_samples[:-cfg.team_buffer]
        due = (
            self.team_fitted_at is None
            and len(self.team_samples) >= cfg.team_warmup
        ) or (
            self.team_fitted_at is not None
            and self.frame - self.team_fitted_at >= cfg.team_refit_frames
        )
        if due:
            self._fit_teams()
        if not self.teams.fitted:
            return [None] * len(features)
        return [self._team_of(f) for f in features]

    def _team_of(self, feature):
        label, _ = self.teams.predict(feature)
        if label not in TEAM_LABELS:
            return label
        point = np.sqrt(np.asarray(feature, dtype=np.float32))
        near, far = sorted(np.linalg.norm(self.teams.centres - point, axis=1))
        return label if near < self.config.team_ratio * far else None

    def _fit_teams(self):
        old = self.teams
        new = TeamClassifier().fit(self.team_samples)
        if old.fitted:
            # Keep team_a / team_b meaning the same shirts after a refit.
            straight = (np.linalg.norm(old.centres - new.centres, axis=1)
                        .sum())
            crossed = (np.linalg.norm(old.centres - new.centres[::-1],
                                      axis=1).sum())
            if crossed < straight:
                new.centres = new.centres[::-1].copy()
        self.teams = new
        self.team_fitted_at = self.frame

    def _changed_team(self, player_id, team):
        """True once a bound ID has shown the other team's shirt for
        team_switch_patience clean views in a row."""
        cfg = self.config
        player = self.players[player_id]
        known = player.team(cfg.team_min_votes, cfg.team_min_share)
        if team not in TEAM_LABELS or known not in TEAM_LABELS:
            return False
        if team == known:
            player.wrong_team = 0
            return False
        player.wrong_team += 1
        if player.wrong_team >= cfg.team_switch_patience:
            player.wrong_team = 0
            return True
        return False

    def _record_contacts(self, tracker_ids, boxes, others=()):
        """Remember which players touch, and which IDs touch anyone now."""
        self.touching = set()
        tids = [int(t) for t in tracker_ids]
        for tid, box in zip(tids, boxes):
            if any(_touching(box, other) for other in others):
                self.touching.add(tid)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if not _touching(boxes[i], boxes[j]):
                    continue
                self.touching.update((tids[i], tids[j]))
                pa = self.bindings.get(tids[i])
                pb = self.bindings.get(tids[j])
                if pa is not None and pb is not None:
                    self.players[pa].contacts[pb] = self.frame
                    self.players[pb].contacts[pa] = self.frame

    def _fix_swaps(self, tracker_ids, features):
        """Undo BoT-SORT ID swaps between players who just touched.

        A tracker ID whose clean view looks clearly more like a player it
        recently touched than like its own player is handed to that
        player. If that player's tracker ID shows our player, the two are
        exchanged; if that player is not in view, ours goes missing and
        is re-linked later.
        """
        cfg = self.config
        # Only views well apart from everyone else: a box that still
        # touches another player shows part of their shirt too.
        view = {
            int(t): f for t, f in zip(tracker_ids, features)
            if f is not None and int(t) in self.bindings
            and int(t) not in self.touching
        }
        tid_of = {pid: tid for tid, pid in self.bindings.items()}
        visible = {self.bindings[int(t)] for t in tracker_ids
                   if int(t) in self.bindings}
        moves = {}
        for tid, feat in view.items():
            pid = self.bindings[tid]
            player = self.players[pid]
            if player.feature is None:
                continue
            own = descriptor_distance(player.feature, feat)
            best, best_dist = None, None
            for other, when in list(player.contacts.items()):
                if self.frame - when > cfg.contact_window:
                    del player.contacts[other]
                    continue
                other_player = self.players.get(other)
                if other_player is None or other_player.feature is None:
                    continue
                if descriptor_distance(
                    player.feature, other_player.feature
                ) < cfg.contact_min_difference:
                    continue
                d = descriptor_distance(other_player.feature, feat)
                if d + cfg.contact_margin < own and (
                    best is None or d < best_dist
                ):
                    best, best_dist = other, d
            if best is None:
                player.swap_votes.clear()
                continue
            votes = player.swap_votes
            votes[best] = votes.get(best, 0) + 1
            for other in [o for o in votes if o != best]:
                del votes[other]
            if votes[best] >= cfg.contact_patience:
                moves[tid] = best

        done = set()
        for tid, target in moves.items():
            pid = self.bindings.get(tid)
            if tid in done or pid is None or target == pid:
                continue
            other_tid = tid_of.get(target)
            if other_tid is not None and moves.get(other_tid) == pid:
                # Clean swap: exchange the two IDs.
                self.bindings[tid], self.bindings[other_tid] = target, pid
                done.update((tid, other_tid))
            elif target not in visible:
                # The other player is out of view or merged into one box:
                # this ID shows them now, ours is missing.
                if other_tid is not None:
                    del self.bindings[other_tid]
                self.bindings[tid] = target
                done.add(tid)
            else:
                continue
            for p in (pid, target):
                self.players[p].swap_votes.clear()
                self.players[p].mismatches = 0

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

    def _update_player(self, player, box, feature, embedding=None):
        gap = self.frame - player.last_frame
        if player.scene == self.scene and 0 < gap <= 5:
            step = (_center(box) - _center(player.box)) / gap
            player.velocity = 0.6 * player.velocity + 0.4 * step
        else:
            player.velocity = np.zeros(2)
        player.box = box
        # Keep the colour model frozen while the view disagrees with it,
        # otherwise a swapped-in player slowly becomes "the same" player.
        if player.mismatches == 0 and not player.swap_votes:
            player.feature = blend(
                player.feature, feature, self.config.feature_momentum
            )
            player.embedding = _blend_embedding(
                player.embedding, embedding, self.config.feature_momentum)
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
            team_votes=Counter(t for t in pending.teams if t),
            embedding=pending.mean_embedding(),
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
                player.team_votes.update(t for t in pending.teams if t)
                player.embedding = _blend_embedding(
                    player.embedding, pending.mean_embedding(),
                    cfg.feature_momentum)
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
        gaps = np.zeros(cost.shape)

        for i, tid in enumerate(pending_ids):
            pending = self.pending[tid]
            feature = pending.mean_feature()
            embedding = pending.mean_embedding()
            team = pending.team()
            for j, player in enumerate(candidates):
                # Seen together in one frame means two different people.
                if player.last_frame >= pending.first_frame:
                    continue
                known = player.team(cfg.team_min_votes, cfg.team_min_share)
                if known in TEAM_LABELS and team and team != known:
                    continue
                gap = pending.first_frame - player.last_frame
                gaps[i, j] = gap
                app = descriptor_distance(feature, player.feature)
                if embedding is not None and player.embedding is not None:
                    reid = 1.0 - float(embedding @ player.embedding)
                    if reid > cfg.reid_gate:
                        continue
                    if feature is None or player.feature is None:
                        app = cfg.reid_scale * reid
                    else:
                        app = ((1 - cfg.reid_weight) * app
                               + cfg.reid_weight * cfg.reid_scale * reid)
                    feature_known = True
                else:
                    feature_known = feature is not None and player.feature is not None
                appearance[i, j] = app

                # Positions are only comparable inside one shot and for a
                # short time; after that the player could be anywhere.
                if player.scene != pending.scene or gap > cfg.motion_horizon:
                    appearance_only[i, j] = True
                    if app <= cfg.appearance_gate_after_cut:
                        cost[i, j] = app / cfg.appearance_gate_after_cut
                    continue

                unknown = not feature_known
                if unknown:
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
                if distance > gate or (unknown and distance > cfg.gate_base):
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

        # Inside a shot, after more than a moment apart, two candidates that
        # fit about equally well (teammates near each other) mean we cannot
        # tell who this is.
        for i in range(len(pending_ids)):
            finite = np.nonzero((cost[i] < inf) & ~appearance_only[i])[0]
            if len(finite) < 2:
                continue
            order = finite[np.argsort(cost[i, finite])]
            best, second = order[0], order[1]
            if (
                gaps[i, best] > cfg.ambiguity_free_gap
                and cost[i, second] - cost[i, best] < cfg.ambiguity_margin
            ):
                cost[i, finite] = inf

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

    def __init__(self, config=None, reid=None):
        """reid: optional callable (frame, boxes) -> L2-normalised
        embeddings, e.g. reid.ReidEmbedder."""
        self.identity = IdentityManager(config)
        self.camera = CameraMotionEstimator()
        self.reid = reid
        self.reid_every = 5
        self.last_cut = False

    def process(self, frame_index, frame, tracker_ids, boxes, confidences):
        """Return (keep, assignments, resolved).

        keep: indices of the input detections that are players: not
            duplicates and (by default) standing on the pitch.
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

        # People not standing on grass are spectators, staff or faces in
        # close-ups, not players.
        if cfg.require_on_pitch:
            keep = [i for i in keep if on_pitch(frame, boxes[i])]
        # A box that overlaps another one (any detection, also a dropped
        # duplicate) mixes two players' colours; leave it out of the
        # appearance model.
        features = [
            None if overlaps_other(i, boxes, cfg.overlap_limit)
            else extract_descriptor(frame, boxes[i])
            for i in keep
        ]
        kept = set(keep)
        embeddings = None
        if self.reid is not None:
            # Only clean views, as for the colour descriptors. Players
            # already bound to an ID only refresh their embedding every
            # reid_every frames; new (pending) IDs get one every frame.
            refresh = frame_index % self.reid_every == 0
            clean = [i for i, f in zip(keep, features) if f is not None
                     and (refresh or int(tracker_ids[i]) not in self.identity.bindings)]
            by_index = dict(zip(clean, self.reid(frame, [boxes[i] for i in clean])))
            embeddings = [by_index.get(i) for i in keep]

        assignments, resolved = self.identity.update(
            frame_index,
            [tracker_ids[i] for i in keep],
            [boxes[i] for i in keep],
            features,
            warp=warp,
            scene_cut=cut,
            frame_size=(frame.shape[1], frame.shape[0]),
            others=[boxes[i] for i in range(len(boxes)) if i not in kept],
            embeddings=embeddings,
        )
        return keep, assignments, resolved

    def flush(self):
        return self.identity.flush()
