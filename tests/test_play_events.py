import random

from match_events import MatchData, Touch
from play_events import candidates, match_intro, thin_out
from player_identity import fill_template

FPS = 30.0


def _match(positions, ball, teams):
    """positions: {frame: {track_id: (x, feet_y)}} with 60 px tall boxes."""
    tracks = {f: [(t, (x - 10, y - 60, x + 10, y)) for t, (x, y) in p.items()]
              for f, p in positions.items()}
    return MatchData(
        fps=FPS, tracks=tracks,
        teams={t: {"team": team, "role": "player"} for t, team in teams.items()},
        ball={f: (x, y, x, y, 1) for f, (x, y) in ball.items()},
        team_colours={"team_a": "orange", "team_b": "sky blue"},
        ball_detected=set(ball))


def test_pass_between_teammates_far_apart():
    frames = range(0, 60)
    positions = {f: {1: (100, 300), 2: (500, 300)} for f in frames}
    ball = {f: (100, 300) if f < 20 else (500, 300) for f in frames if not 20 <= f < 30}
    match = _match(positions, ball, {1: "team_a", 2: "team_a"})
    touches = [Touch(0, 19, 1, "team_a"), Touch(30, 59, 2, "team_a")]
    [(frame, kind, players)] = candidates(match, touches)
    assert kind == "pass"  # 400 px = 6.7 player heights (~12 m)
    assert [p["role"] for p in players] == ["passer", "receiver"]


def test_quick_back_and_forth_is_not_a_tackle():
    frames = range(0, 90)
    positions = {f: {1: (100, 300), 2: (120, 300)} for f in frames}
    ball = {f: (110, 300) for f in frames}
    match = _match(positions, ball, {1: "team_a", 2: "team_b"})
    duel = [Touch(0, 29, 1, "team_a"), Touch(30, 59, 2, "team_b"),
            Touch(62, 89, 1, "team_a")]
    assert candidates(match, duel) == []
    won = [Touch(0, 29, 1, "team_a"), Touch(30, 89, 2, "team_b")]
    [(_, kind, players)] = candidates(match, won)
    assert kind == "tackle_won" and players[0]["track_id"] == 2


def test_thin_out_keeps_space_and_avoids_big_events():
    found = [(0, "pass", []), (30, "tackle_won", []), (400, "pass", []),
             (900, "pass", [])]
    kept = thin_out(found, FPS, busy_frames=[950])
    assert [k[0] for k in kept] == [30, 400]


def test_intro_and_two_player_templates():
    match = _match({}, {}, {})
    intro = match_intro(match)
    assert intro["team_colour"] == "orange"
    who = {"player": "A", "jersey": None, "level": "colour"}
    templates = ["{from_player} to {to_player}", "pass by {player}"]
    assert fill_template(templates, 3, who, random.Random(0),
                         {"to_player": "B"}) in {"A to B", "pass by A"}
    assert fill_template(templates, 3, who, random.Random(0)) == "pass by A"
