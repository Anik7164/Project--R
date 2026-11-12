import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_events import STEP, attacks_from_crowd, crowd_lesson, sound_numbers  # noqa: E402
from score_commentary import score  # noqa: E402
from train_events import step_labels  # noqa: E402


def _roar(seconds, strength=5.0):
    return {"time": seconds, "type": "crowd_roar", "strength": strength, "duration": 2.0}


def _match(goal_view_at, seconds=600, seed=0):
    """Picture features of a match whose camera shows the "goal end" view
    for 8 s from each time in goal_view_at, and midfield otherwise."""
    rng = np.random.default_rng(seed)
    count = int(seconds / STEP)
    midfield, goal_end = rng.normal(size=64), rng.normal(size=64)
    features = np.tile(midfield, (count, 1))
    for start in goal_view_at:
        features[int(start / STEP):int((start + 8) / STEP)] = goal_end
    return features + 0.3 * rng.normal(size=features.shape)


def test_crowd_lesson_marks_the_seconds_around_a_strong_roar():
    lesson, roars = crowd_lesson(200, [_roar(50.0), _roar(80.0, strength=1.0),
                                       {"time": 20.0, "type": "whistle", "strength": 30,
                                        "duration": 1.0}])
    assert roars == 1
    assert lesson[int(50 / STEP)] and lesson[int(46 / STEP)] and lesson[int(56 / STEP)]
    assert not lesson[int(80 / STEP)] and not lesson[int(20 / STEP)]


def test_attacks_from_crowd_finds_the_same_view_without_a_roar():
    with_roar = [60, 150, 240, 330]
    without_roar = [420, 510]
    features = _match(with_roar + without_roar)
    cues = [_roar(t + 3.0) for t in with_roar]
    found = [seconds for seconds, _ in attacks_from_crowd(features, cues)]
    for start in without_roar:
        assert any(start - 2 <= seconds <= start + 10 for seconds in found), found
    # Nothing in the long midfield stretches.
    assert not any(any(a + 20 <= s <= b - 12 for s in found)
                   for a, b in zip([0] + with_roar + without_roar,
                                   with_roar + without_roar + [600]))


def test_attacks_from_crowd_needs_a_few_roars_and_skips_quiet_stretches():
    features = _match([60, 150, 240, 330, 420])
    assert attacks_from_crowd(features, [_roar(63.0), _roar(153.0)]) == []
    cues = [_roar(t + 3.0) for t in (60, 150, 240)]
    found = attacks_from_crowd(features, cues, quiet=lambda seconds: seconds >= 300)
    assert found and all(seconds < 300 for seconds, _ in found)


def test_sound_numbers_are_empty_for_a_video_with_no_sound():
    numbers = sound_numbers(None, 40)
    assert numbers.shape[0] == 40 and np.isnan(numbers).all()


def test_step_labels_mark_the_steps_near_each_real_event():
    truth = [{"time": 30.0, "type": "shot", "team": None},
             {"time": 60.0, "type": "whistle", "team": None}]
    labels = step_labels(truth, 200)
    assert labels["attack"][int(30 / STEP)] and labels["attack"][int(33 / STEP)]
    assert not labels["attack"][int(40 / STEP)] and not labels["attack"][int(60 / STEP)]
    assert labels["stoppage"][int(60 / STEP)] and not labels["stoppage"][int(30 / STEP)]
    assert not labels["throw_in"].any()


def test_an_attack_line_is_right_near_a_real_attack():
    truth = [{"time": 100.0, "type": "attack", "team": "orange"}]
    lines, used = score([{"time": 95.0, "type": "attack", "team_colour": "orange"},
                         {"time": 300.0, "type": "attack"}], truth, {})
    assert [verdict for _, verdict, _ in lines] == ["right", "wrong"]
    assert used == {0}


def test_position_scores_pick_out_the_stretch_that_differs_like_an_attack():
    import position_events
    from match_events import MatchData

    fps, tracks, ball, seen = 25.0, {}, {}, set()
    for frame in range(int(300 * fps)):
        attack = 200 <= frame / fps < 245
        count = 3 if attack else 10
        tracks[frame] = [(i, (100.0 + 100 * i, 300.0, 140.0 + 100 * i, 400.0)) for i in range(count)]
        x = 1200.0 if attack else 640.0
        ball[frame] = (x, 360.0, x, 360.0, 1)
        seen.add(frame)
    match = MatchData(fps=fps, tracks=tracks, teams={}, ball=ball, ball_detected=seen)
    found = position_events.attacks(match, 1280, 720)
    assert found and all(195 <= seconds <= 250 for seconds, _ in found)
    assert position_events.attacks(match, 1280, 720, quiet=lambda s: True) == []
