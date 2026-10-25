from collections import defaultdict

import pytest

from appearance import extract_descriptor, shirt_colour_name
from broadcast_sim import KITS, simulate
from identity import overlaps_other
from team_identification import TeamClassifier, TeamVotes

def sim_samples(frames=900, seed=3, step=5):
    """Per-frame (true player IDs, boxes, descriptors) from the simulator."""
    samples = []
    for frame, image, dets in simulate(frames=frames, seed=seed, duplicate_prob=0):
        if frame % step or not dets:
            continue
        boxes = [d["box"] for d in dets]
        descriptors = [
            None if overlaps_other(i, boxes, 0.15) else extract_descriptor(image, b)
            for i, b in enumerate(boxes)
        ]
        samples.append(([d["truth"] for d in dets], boxes, descriptors))
    return samples


def role_of(player_id):
    # Same order as broadcast_sim.make_players.
    if player_id <= 10:
        return "home"
    if player_id == 11:
        return "home_gk"
    if player_id <= 21:
        return "away"
    if player_id == 22:
        return "away_gk"
    return "referee"


@pytest.mark.parametrize("seed", [0, 3, 4, 7])
def test_teams_goalkeepers_and_referee(seed):
    samples = sim_samples(seed=seed)
    classifier = TeamClassifier().fit([d for _, _, ds in samples for d in ds])
    votes = TeamVotes()
    for ids, boxes, descriptors in samples:
        votes.add_frame(
            ids, boxes, [classifier.predict(d)[0] for d in descriptors], descriptors
        )
    decisions = votes.decide()

    team_of_role = defaultdict(set)
    for player_id, decision in decisions.items():
        role = role_of(player_id)
        if role in ("home", "away"):
            assert decision["role"] == "player", (player_id, decision)
            team_of_role[role].add(decision["team"])
    # Each outfield team maps to exactly one, different, label.
    assert len(team_of_role["home"]) == 1 and len(team_of_role["away"]) == 1
    assert team_of_role["home"] != team_of_role["away"]

    if 23 in decisions:
        assert decisions[23]["role"] == "referee"
    for gk, side in ((11, "home"), (22, "away")):
        if gk in decisions:
            assert decisions[gk]["role"] == "goalkeeper", decisions[gk]
            assert {decisions[gk]["team"]} == team_of_role[side]

    home_label = next(iter(team_of_role["home"]))
    away_label = next(iter(team_of_role["away"]))
    assert classifier.colours[home_label] == "red"
    assert classifier.colours[away_label] == "blue"


def test_colour_names():
    import numpy as np
    for role, expected in (("home", "red"), ("away", "blue"),
                           ("home_gk", "yellow"), ("referee", "black")):
        image = np.zeros((100, 60, 3), np.uint8)
        image[:] = KITS[role][0]
        name = shirt_colour_name(extract_descriptor(image, (0, 0, 60, 100)))
        assert name == expected, (role, name)
