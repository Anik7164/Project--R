from collections import Counter, defaultdict

import numpy as np

from broadcast_sim import simulate
from camera_motion import CameraMotionEstimator, warp_points
from identity import IdentityManager, StablePlayerTracker, find_duplicates


def run_pipeline(**sim_kwargs):
    """Run the stable tracker over a simulated clip.

    Returns rows of (frame, truth, raw tracker ID, player ID).
    """
    tracker = StablePlayerTracker()
    rows = []
    pending_rows = defaultdict(list)

    def apply_resolved(resolved):
        for tid, pid in resolved.items():
            for frame, truth in pending_rows.pop(tid, []):
                rows.append((frame, truth, tid, pid))

    for frame_index, image, dets in simulate(**sim_kwargs):
        tids = [d["tracker_id"] for d in dets]
        boxes = [d["box"] for d in dets]
        confs = [d["conf"] for d in dets]
        keep, assignments, resolved = tracker.process(
            frame_index, image, tids, boxes, confs
        )
        for i, pid in zip(keep, assignments):
            if pid is None:
                pending_rows[tids[i]].append((frame_index, dets[i]["truth"]))
            else:
                rows.append((frame_index, dets[i]["truth"], tids[i], pid))
        apply_resolved(resolved)
    apply_resolved(tracker.flush())
    return rows


def identity_stats(rows, column):
    """Number of IDs, purity, and number of truth players split across IDs."""
    rows = [r for r in rows if r[1] is not None and r[column] is not None]
    by_id = defaultdict(Counter)
    for row in rows:
        by_id[row[column]][row[1]] += 1
    correct = sum(c.most_common(1)[0][1] for c in by_id.values())
    ids_per_player = defaultdict(set)
    for row in rows:
        ids_per_player[row[1]].add(row[column])
    return {
        "ids": len(by_id),
        "purity": correct / max(1, len(rows)),
        "mean_ids_per_player": float(np.mean([len(v) for v in ids_per_player.values()])),
    }


def test_camera_motion_recovers_pan():
    rng = np.random.default_rng(1)
    world = rng.integers(0, 255, (800, 1400, 3), dtype=np.uint8)
    world = np.ascontiguousarray(
        np.repeat(np.repeat(world[::8, ::8], 8, 0), 8, 1)[:800, :1400]
    )
    estimator = CameraMotionEstimator()
    estimator.update(world[100:640, 100:1060])
    warp, cut = estimator.update(world[100:640, 112:1072])
    assert not cut
    moved = warp_points([[500, 300]], warp)[0]
    # Camera moved 12 px right, so the scene moves 12 px left.
    assert abs(moved[0] - 488) < 1.5 and abs(moved[1] - 300) < 1.5


def test_scene_cut_detected():
    rng = np.random.default_rng(2)
    estimator = CameraMotionEstimator()
    green = np.zeros((540, 960, 3), np.uint8)
    green[:] = (40, 130, 40)
    estimator.update(green)
    crowd = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    _, cut = estimator.update(crowd)
    assert cut


def test_cut_between_similar_wide_shots_detected():
    # Same colours, different picture: a cut from one wide camera to another.
    estimator = CameraMotionEstimator()
    first = np.zeros((540, 960, 3), np.uint8)
    first[:] = (40, 130, 40)
    for y in range(0, 540, 60):
        first[y:y + 30] = (230, 230, 230)
    for _ in range(5):
        _, cut = estimator.update(first)
    second = np.zeros_like(first)
    second[:] = (40, 130, 40)
    for y in range(30, 540, 60):
        second[y:y + 30] = (230, 230, 230)
    _, cut = estimator.update(second)
    assert cut


def test_spectators_are_not_on_pitch():
    from appearance import on_pitch
    frame = np.zeros((300, 300, 3), np.uint8)
    frame[150:] = (40, 130, 40)          # pitch at the bottom
    frame[:150] = (90, 60, 140)          # stands at the top
    assert on_pitch(frame, (100, 120, 120, 180))
    assert not on_pitch(frame, (100, 40, 120, 100))


def test_partial_duplicate_box_is_dropped():
    boxes = [np.array([100, 100, 130, 180]), np.array([102, 102, 128, 140]),
             np.array([300, 100, 330, 180])]
    assert find_duplicates(boxes, [0.9, 0.5, 0.9]) == {1}


def test_overlapping_players_of_same_size_are_kept():
    boxes = [np.array([100, 100, 130, 180]), np.array([108, 104, 138, 182])]
    assert find_duplicates(boxes, [0.9, 0.8]) == set()


def test_relinks_player_who_returns_near_expected_position():
    manager = IdentityManager()
    feat = np.full(10, 0.1)
    box = np.array([100.0, 100.0, 120.0, 150.0])
    for f in range(5):
        manager.update(f, [1], [box], [feat])
    # Lost for 20 frames, comes back as BoT-SORT ID 2 close by.
    for f in range(25, 30):
        assignments, _ = manager.update(f, [2], [box + 6], [feat])
    assert assignments == [1]


def test_never_merges_two_players_seen_together():
    manager = IdentityManager()
    feat = np.full(10, 0.1)
    a = np.array([100.0, 100.0, 120.0, 150.0])
    b = np.array([104.0, 100.0, 124.0, 150.0])
    for f in range(10):
        assignments, _ = manager.update(f, [1, 2], [a, b], [feat, feat])
    assert len(set(assignments)) == 2


def _kit(index):
    feat = np.full(10, 0.02)
    feat[index] = 0.82
    return feat


def test_ids_swapped_by_tracker_during_contact_are_swapped_back():
    manager = IdentityManager()
    red, blue = _kit(0), _kit(5)
    a = np.array([100.0, 100.0, 120.0, 150.0])
    b = np.array([140.0, 100.0, 160.0, 150.0])
    for f in range(10):
        ids, _ = manager.update(f, [1, 2], [a, b], [red, blue])
    red_id, blue_id = ids
    # They run into each other: no clean view while the boxes overlap.
    mid = np.array([120.0, 100.0, 140.0, 150.0])
    for f in range(10, 15):
        manager.update(f, [1, 2], [mid, mid + 4], [None, None])
    # BoT-SORT comes out of the contact with the two IDs swapped.
    for f in range(15, 20):
        ids, _ = manager.update(f, [1, 2], [b, a], [blue, red])
    assert ids == [blue_id, red_id]


def test_simulated_broadcast_ids_are_stable_and_pure():
    rows = run_pipeline(frames=900, seed=1)
    raw = identity_stats(rows, column=2)
    stable = identity_stats(rows, column=3)
    print("raw BoT-SORT-like IDs:", raw)
    print("stable player IDs:   ", stable)
    # Far fewer IDs per real player than the raw tracker ...
    assert stable["mean_ids_per_player"] < raw["mean_ids_per_player"] / 1.8
    # ... without merging many different players together. The simulated
    # teammates are pixel-identical, so the motion gates tuned on real
    # broadcast footage cost a little purity here.
    assert stable["purity"] > 0.95


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rows = run_pipeline(frames=900, seed=seed)
    print("raw   ", identity_stats(rows, 2))
    print("stable", identity_stats(rows, 3))
    by_id = defaultdict(Counter)
    for frame, truth, tid, pid in rows:
        if truth is not None:
            by_id[pid][truth] += 1
    for pid, c in sorted(by_id.items()):
        if len(c) > 1:
            print("impure", pid, c.most_common())
