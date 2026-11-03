import cv2
import numpy as np

from match_events import MatchData, scoring_teams
from scoreboard import find_score_changes


def _write_clip(path, goals, frames=300, fps=25):
    """Random moving background with a fixed score graphic; `goals` maps
    frame -> side (0 home, 1 away) whose digit goes up from then on."""
    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                             fps, (320, 240))
    score = [0, 0]
    for f in range(frames):
        if f in goals:
            score[goals[f]] += 1
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10, 10), (130, 40), (255, 255, 255), -1)
        cv2.rectangle(frame, (14, 14), (34, 36), (0, 0, 220), -1)    # red badge
        cv2.rectangle(frame, (106, 14), (126, 36), (200, 60, 0), -1)  # blue badge
        cv2.rectangle(frame, (40, 12), (100, 38), (20, 20, 20), -1)
        for side, x in ((0, 44), (1, 82)):
            cv2.putText(frame, str(score[side]), (x, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_score_changes_are_found(tmp_path):
    clip = tmp_path / "clip.avi"
    _write_clip(clip, {100: 0, 220: 1})
    changes = find_score_changes(clip)
    assert [c["side"] for c in changes] == ["home", "away"]
    assert [c["score"] for c in changes] == ["1-0", "1-1"]
    assert abs(changes[0]["frame"] - 100) <= 5
    assert abs(changes[1]["frame"] - 220) <= 5
    assert [c["side_colour"] for c in changes] == ["red", "blue"]


def test_board_colours_map_sides_to_teams():
    match = MatchData(fps=25, tracks={}, teams={}, ball={},
                      team_colours={"team_a": "grey", "team_b": "red"})
    changes = [{"side": "home", "side_colour": "red"},
               {"side": "away", "side_colour": "blue"}]
    assert scoring_teams(match, changes) == {"home": "team_b", "away": "team_a"}
