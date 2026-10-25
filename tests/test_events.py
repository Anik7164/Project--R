import json

import numpy as np

from ball_tracking import BallSelector, interpolate_gaps
from card_detection import detect_card
from match_events import (
    MatchData, events_from_rules, events_from_spotting, load_spotting,
)

FPS = 25.0
A1, A2, B1, GK, REF = 1, 2, 3, 4, 5
TEAMS = {
    A1: {"team": "team_a", "role": "player"},
    A2: {"team": "team_a", "role": "player"},
    B1: {"team": "team_b", "role": "player"},
    GK: {"team": "team_b", "role": "goalkeeper"},
    REF: {"team": None, "role": "referee"},
}


def box_at(x, y):
    """A 20x50 player box standing at (x, y)."""
    return (x - 10.0, y - 50.0, x + 10.0, y)


def make_match(script):
    """script: list of (first, last, positions {tid: (x, y)}, ball (x, y))."""
    tracks, ball = {}, {}
    for first, last, positions, ball_xy in script:
        for frame in range(first, last + 1):
            tracks[frame] = [(tid, box_at(*xy)) for tid, xy in positions.items()]
            if ball_xy is not None:
                x, y = ball_xy
                ball[frame] = (x, y, x, y, 1)
    return MatchData(fps=FPS, tracks=tracks, teams=TEAMS, ball=ball)


BASE = {A1: (300, 300), A2: (200, 300), B1: (330, 300), GK: (800, 300), REF: (500, 260)}


NAMES = {"A1": A1, "A2": A2, "B1": B1, "GK": GK, "REF": REF}


def scene(**moves):
    positions = dict(BASE)
    positions.update({NAMES[name]: xy for name, xy in moves.items()})
    return positions


def test_goal_scorer_and_assist():
    match = make_match([
        (0, 20, scene(), (200, 300)),        # A2 on the ball
        (21, 29, scene(), (250, 280)),       # pass in flight
        (30, 50, scene(), (300, 300)),       # A1 on the ball
        (51, 60, scene(), (700, 250)),       # shot
    ])
    events = events_from_spotting(match, [(60, "goal", 0.9)])
    roles = {p["role"]: p["track_id"] for p in events[0]["players"]}
    assert roles == {"scorer": A1, "assist": A2}
    assert events[0]["minute"] == 1


def test_foul_and_card_go_to_the_fouler():
    match = make_match([
        (0, 30, scene(), (300, 300)),                 # A1 on the ball, B1 next to him
        (31, 400, scene(REF=(900, 100)), None),       # play stopped
        (401, 420, scene(REF=(335, 290)), None),      # referee walks up to B1
    ])
    events = events_from_spotting(match, [(32, "foul", 0.8), (410, "yellow_card", 0.8)])
    foul, card = events
    roles = {p["role"]: p["track_id"] for p in foul["players"]}
    assert roles == {"fouler": B1, "fouled": A1}
    assert card["players"][0]["track_id"] in (A1, B1)  # nearest to referee
    assert card["players"][0]["role"] == "booked"


def test_card_without_referee_falls_back_to_last_fouler():
    match = make_match([
        (0, 30, scene(), (300, 300)),
        (31, 100, {A1: (300, 300), B1: (330, 300)}, None),  # referee off screen
    ])
    events = events_from_spotting(match, [(32, "foul", 0.8), (90, "red_card", 0.7)])
    assert events[1]["players"] == [{"role": "booked", "track_id": B1, "team": "team_b"}]


def test_rules_find_free_kick_foul_and_goal_kick():
    match = make_match([
        (0, 30, scene(), (300, 300)),                  # A1 on the ball, B1 close
        (31, 120, scene(A1=(400, 400)), (420, 380)),   # ball placed, still
        (121, 140, scene(A1=(420, 380)), (420, 380)),  # A1 steps up
        (141, 150, scene(A1=(420, 380)), (520, 330)),  # ball played
        (151, 300, scene(GK=(800, 300)), (800, 300)),  # still at the GK's feet
        (301, 320, scene(GK=(800, 300)), (900, 200)),  # GK kicks it away
    ])
    events = events_from_rules(match)
    by_type = {e["type"]: e for e in events}
    assert by_type["freekick"]["players"][0]["track_id"] == A1
    foul_roles = {p["role"]: p["track_id"] for p in by_type["foul"]["players"]}
    assert foul_roles == {"fouler": B1, "fouled": A1}
    assert by_type["goal_kick"]["players"][0]["track_id"] == GK


def test_load_soccernet_spotting(tmp_path):
    path = tmp_path / "results_spotting.json"
    path.write_text(json.dumps({"predictions": [
        {"gameTime": "1 - 00:02", "label": "Goal", "position": "2000", "half": "1", "confidence": "0.9"},
        {"gameTime": "1 - 00:04", "label": "Kick-off", "position": "4000", "half": "1", "confidence": "0.9"},
        {"gameTime": "2 - 00:01", "label": "Corner", "position": "1000", "half": "2", "confidence": "0.9"},
    ]}))
    assert load_spotting(path, FPS) == [(50, "goal", 0.9)]


def test_card_detection():
    frame = np.zeros((400, 400, 3), np.uint8)
    frame[:] = (40, 130, 40)
    referee = (180, 150, 220, 250)
    frame[150:250, 180:220] = (20, 20, 20)
    assert detect_card(frame, referee) is None

    shown = frame.copy()
    shown[118:130, 222:230] = (0, 230, 255)  # yellow card above the head
    assert detect_card(shown, referee) == "yellow"

    red = frame.copy()
    red[118:130, 222:230] = (0, 0, 230)
    assert detect_card(red, referee) == "red"

    # A yellow shirt of another player in the same area is not a card.
    shirt = frame.copy()
    shirt[118:130, 222:230] = (0, 230, 255)
    assert detect_card(shirt, referee, other_boxes=[(215, 110, 240, 200)]) is None


def test_ball_selector_rejects_jumps_and_interpolates():
    selector = BallSelector(frame_width=1000)
    assert selector.update([(100, 100, 8, 8, 0.6)]) is not None
    # A far-away false detection is ignored in favour of the continuation.
    picked = selector.update([(900, 500, 8, 8, 0.9), (110, 102, 8, 8, 0.3)])
    assert picked[:2] == (110, 102)
    filled = interpolate_gaps({0: (0.0, 0.0, 1.0), 4: (40.0, 0.0, 1.0)})
    assert filled[2][0] == 20.0 and filled[2][3] == "interpolated"


def _closeup_with_card(advert=False):
    import cv2
    frame = np.full((360, 640, 3), (40, 140, 40), dtype=np.uint8)  # grass
    if advert:
        cv2.rectangle(frame, (250, 40), (420, 160), (0, 230, 255), -1)  # yellow board
    cv2.rectangle(frame, (310, 60), (330, 95), (30, 30, 250), -1)       # red card
    return frame


def test_card_held_up_in_close_up_is_found():
    from card_detection import card_candidates, held_up
    candidates = card_candidates(_closeup_with_card())
    assert [c[0] for c in candidates] == ["red"]
    referee = (280.0, 110.0, 380.0, 360.0)  # standing right below the card
    assert held_up(candidates[0], [referee])
    assert not held_up(candidates[0], [(20.0, 110.0, 120.0, 360.0)])


def test_red_patch_inside_an_advert_is_not_a_card():
    from card_detection import card_candidates
    assert card_candidates(_closeup_with_card(advert=True)) == []


def test_spotted_cards_take_colour_and_time_from_the_picture():
    from match_events import merge_cards
    spotted = [(2200, "foul", 0.38), (2225, "yellow_card", 0.47),
               (2375, "yellow_card", 0.65)]
    seen = [(2378, "red_card")]
    merged = merge_cards(spotted, seen, FPS)
    assert merged == [(2200, "foul", 0.38), (2378, "red_card", 0.65)]


def test_spotting_threshold_drops_weak_predictions(tmp_path):
    path = tmp_path / "spot.json"
    path.write_text(json.dumps({"predictions": [
        {"label": "Foul", "position": "88000", "half": "1", "confidence": "0.38"},
        {"label": "Corner", "position": "53500", "half": "1", "confidence": "0.2"},
        {"label": "Shots off target", "position": "30000", "half": "1",
         "confidence": "0.40"},
    ]}))
    assert load_spotting(path, FPS, threshold=0.35) == [(2200, "foul", 0.38)]
