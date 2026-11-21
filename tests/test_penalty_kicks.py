import numpy as np

from fill_commentary import filler_kind, shootout_span
from match_events import MatchData
from penalty_kicks import SHOOTOUT_AFTER_SECONDS, find, player_speed, still_stretches
from score_commentary import score

FPS = 25.0


def _match(still, seconds=400, players=6):
    """Players 100 px tall who run about (2 body heights a second), except
    in the (start, end) second ranges of still, where they only shuffle."""
    rng = np.random.default_rng(0)
    place = rng.uniform(200, 1000, size=(players, 2))
    angle = rng.uniform(0, 2 * np.pi, size=players)
    heading = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    tracks = {}
    for frame in range(int(seconds * FPS)):
        at = frame / FPS
        speed = 5.0 if any(a <= at < b for a, b in still) else 200.0
        place += heading * speed / FPS
        heading[(place < 100) | (place > 1100)] *= -1      # turn round at the touchline
        tracks[frame] = [(i, (x - 20, y - 100, x + 20, y)) for i, (x, y) in enumerate(place)]
    return MatchData(fps=FPS, tracks=tracks, teams={}, ball={})


def _roar(seconds, strength=10.0):
    return {"time": seconds, "type": "crowd_roar", "strength": strength, "duration": 4.0}


def test_player_speed_is_low_only_where_the_players_stand():
    speed = player_speed(_match([(100, 120)]))
    assert np.nanmax(speed[101:119]) < 0.2
    assert np.nanmin(speed[:99]) > 0.5 and np.nanmin(speed[121:-1]) > 0.5


def test_still_stretches_bridge_a_short_break_and_drop_short_waits():
    speed = np.full(100, 2.0)
    speed[10:20] = speed[21:30] = 0.1      # one second of movement in between
    speed[50:55] = 0.1                     # too short: a throw-in, a free kick
    speed[70:90] = np.nan                  # nobody tracked is not "still"
    assert still_stretches(speed) == [(10.0, 30.0)]


def test_three_long_waits_in_a_row_are_a_shootout():
    match = _match([(100, 120), (160, 180), (280, 300)])
    kicks, shootouts = find(match)
    assert [kind for _, kind in kicks] == ["shootout_kick"] * 3
    assert all(abs(seconds - end) <= 2 for (seconds, _), end in zip(kicks, (120, 180, 300)))
    (start, end), = shootouts
    assert abs(start - 100) <= 2 and abs(end - (300 + SHOOTOUT_AFTER_SECONDS)) <= 2


def test_a_roar_far_from_every_kick_in_a_shootout_is_one_more_kick():
    match = _match([(100, 120), (160, 180), (280, 300)])
    cues = [_roar(121.0), _roar(230.0), _roar(310.0), _roar(232.0, strength=2.0),
            _roar(50.0)]
    kicks, _ = find(match, cues)
    assert [round(seconds) for seconds, _ in kicks if abs(seconds - 230) < 5] == [230]
    assert len(kicks) == 4      # not the roars at the kicks, right after one, or in play


def test_one_long_wait_is_a_penalty_only_if_the_crowd_roars_at_its_end():
    match = _match([(100, 120)])
    assert find(match) == ([], [])
    assert find(match, [_roar(150.0)]) == ([], [])
    kicks, shootouts = find(match, [_roar(121.0)])
    assert [kind for _, kind in kicks] == ["penalty_kick"] and shootouts == []
    assert abs(kicks[0][0] - 120) <= 2


def test_nothing_is_found_in_open_play_or_where_there_is_no_play():
    assert find(_match([]), [_roar(50.0), _roar(200.0)]) == ([], [])
    # The line-up before kick-off, and the players waiting for kick-off itself.
    match = _match([(10, 40), (60, 80), (100, 121)])
    assert find(match, [_roar(41.0), _roar(81.0), _roar(121.0)],
                quiet=lambda seconds: seconds < 120) == ([], [])


def test_filler_lines_in_a_shootout_are_about_the_shootout():
    lines = [{"time": 30.0, "type": "attack"}, {"time": 100.0, "type": "shootout_start"},
             {"time": 130.0, "type": "shootout_kick"}, {"time": 200.0, "type": "shootout_kick"}]
    span = shootout_span(lines)
    assert span == (100.0, 200.0)
    assert filler_kind(50.0, span) == "play" and filler_kind(150.0, span) == "shootout"
    assert filler_kind(220.0, span) is None
    assert shootout_span(lines[:1]) is None and filler_kind(50.0, None) == "play"


def test_a_shootout_kick_line_is_right_near_a_real_penalty():
    truth = [{"time": 100.0, "type": "penalty", "team": "black"},
             {"time": 300.0, "type": "shot", "team": "black"}]
    lines, used = score([{"time": 80.0, "type": "shootout_start"},
                         {"time": 102.0, "type": "shootout_kick"},
                         {"time": 301.0, "type": "shootout_kick"}], truth, {})
    assert [verdict for _, verdict, _ in lines] == ["right", "right", "wrong"]
    assert used == {0}
