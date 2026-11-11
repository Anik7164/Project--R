from score_commentary import score


def test_lines_scored_against_real_events():
    truth = [{"time": 100.0, "type": "foul", "team": None},
             {"time": 200.0, "type": "goal", "team": "orange"},
             {"time": 300.0, "type": "attack", "team": "white"}]
    events = [
        {"time": 101.0, "type": "whistle_stop", "players": []},          # right
        {"time": 198.0, "type": "goal", "template_type": "goal_celebration",
         "team_colour": "orange", "players": []},                         # right
        {"time": 301.0, "type": "big_chance", "team_colour": "orange",
         "players": []},                                                  # wrong team
        {"time": 400.0, "type": "freekick", "players": []},              # wrong
        {"time": 50.0, "type": "pass", "players": []},                   # not checked
    ]
    lines, used = score(events, truth, {})
    assert [v for _, v, _ in lines] == ["right", "right", "wrong team", "wrong", "not checked"]
    assert used == {0, 1, 2}
