from auto_label import ball_labels, pick_frames

WIDTH = 1280


def test_track_fills_misses_and_ignores_off_track_noise():
    candidates = {}
    for f in range(30):
        x = 100 + 10 * f  # ball rolling right
        found = [] if f in (10, 11) else [(x, 400, 8, 8, 0.5 if f % 4 == 0 else 0.1)]
        found.append((900, 100 + 37 * (f % 7), 10, 10, 0.12))  # jumping noise
        candidates[f] = found
    labels = ball_labels(candidates, WIDTH)
    assert set(labels) == set(range(30))
    assert labels[11][4] == "filled" and abs(labels[11][0] - 210) < 1
    assert labels[1][4] == "faint" and labels[0][4] == "detected"
    assert all(label[0] < 800 for label in labels.values())


def test_no_fill_across_a_kick_and_keep_vets_tracks():
    # The ball comes back after the gap (a kick): no guessed boxes.
    path = list(range(0, 100, 10)) + [80, 60, 40, 20, 0, -20, -40]
    candidates = {f: [] if f in (9, 10) else [(300 + x, 400, 8, 8, 0.5)]
                  for f, x in enumerate(path)}
    labels = ball_labels(candidates, WIDTH)
    assert 9 not in labels and 10 not in labels and 8 in labels
    # A "ball" that is never on the grass is not kept.
    assert ball_labels(candidates, WIDTH, keep=lambda f, box: False) == {}


def test_faint_or_still_tracks_are_not_the_ball():
    faint = {f: [(100 + 10 * f, 400, 8, 8, 0.2)] for f in range(20)}
    assert ball_labels(faint, WIDTH) == {}
    still = {f: [(500, 300, 8, 8, 0.35 if f == 0 else 0.1)] for f in range(20)}
    assert ball_labels(still, WIDTH) == {}
    short = {f: [(100 + 10 * f, 400, 8, 8, 0.9)] for f in range(4)}
    assert ball_labels(short, WIDTH) == {}


def test_pick_frames_prefers_hard_frames_and_keeps_space():
    labels = {f: (0, 0, 8, 8, "detected") for f in range(30)}
    labels[11] = (0, 0, 8, 8, "filled")
    chosen = pick_frames(labels, 6, 3)
    assert 11 in chosen and len(chosen) == 3
    assert all(abs(a - b) >= 6 for a in chosen for b in chosen if a != b)
