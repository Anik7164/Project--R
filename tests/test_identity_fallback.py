import json
import random
from pathlib import Path

import numpy as np

from jersey_reading import apply_roster, parse_name, parse_number, read_player, vote
from player_identity import commentary, describe, fill_template, video_time

DATASET = json.loads((Path(__file__).resolve().parent.parent / "dataset.json").read_text(encoding="utf-8"))
TEMPLATES = {item["event_type"]: item["templates"] for item in DATASET}


def test_number_and_name_parsing():
    assert parse_number([(None, "07", 0.9)]) == ("7", 0.9)
    assert parse_number([(None, "123", 0.9), (None, "0", 0.9)]) is None
    assert parse_number([(None, "10", 0.2)]) is None
    assert parse_name([(None, "rashford", 0.8), (None, "AB", 0.9)]) == ("RASHFORD", 0.8)


def test_vote_needs_agreement():
    assert vote([("7", 0.9), ("7", 0.6), ("1", 0.5)]) == ("7", 2)
    assert vote([("7", 0.9)]) == (None, 0)            # one crop is not enough
    assert vote([("7", 0.5), ("1", 0.5), ("4", 0.5), ("7", 0.4)]) == (None, 0)


class FakeReader:
    """Stands in for EasyOCR: returns scripted text per call."""

    def __init__(self, numbers, names):
        self.numbers, self.names = list(numbers), list(names)

    def readtext(self, image, allowlist=""):
        queue = self.numbers if allowlist.startswith("0") else self.names
        text = queue.pop(0)
        return [] if text is None else [(None, text, 0.8)]


def test_read_player_votes_over_crops():
    crops = [np.zeros((60, 40, 3), np.uint8)] * 3
    reader = FakeReader(["10", "10", "16"], ["MESSI", None, "MESS1"])
    assert read_player(reader, crops) == {
        "jersey_number": "10", "number_votes": 2, "name": None, "name_votes": 0,
    }


def test_roster_turns_number_into_name_and_fixes_spelling():
    roster = {"team_a": {"10": "Messi", "7": "Rashford"}}
    by_number = apply_roster({"jersey_number": "10", "name": None}, "team_a", roster)
    assert by_number["name"] == "Messi" and by_number["name_source"] == "roster"
    by_name = apply_roster({"jersey_number": None, "name": "RASHFRD"}, "team_a", roster)
    assert by_name["name"] == "Rashford" and by_name["jersey_number"] == "7"


def test_fallback_chain():
    assert describe({"name": "Messi", "jersey_number": "10"})["level"] == "name"
    by_number = describe({"name": "", "jersey_number": "7"}, {"shirt_colour": "red"})
    assert by_number["player"] == "৭ নম্বর জার্সির খেলোয়াড়"
    by_colour = describe({}, {"shirt_colour": "red"})
    assert by_colour["player"] == "লাল জার্সির খেলোয়াড়"
    assert describe(None, None)["level"] == "unknown"


def test_templates_only_use_slots_we_can_fill():
    rng = random.Random(0)
    colour_only = describe({}, {"shirt_colour": "blue"})
    for _ in range(20):
        line = fill_template(TEMPLATES["goal"], 23, colour_only, rng)
        assert "নীল জার্সির খেলোয়াড়" in line
        assert "নম্বর জার্সি" not in line  # no empty {jersey} slot
        assert "{" not in line

    named = describe({"name": "Messi", "jersey_number": "10"})
    lines = {fill_template(TEMPLATES["goal"], 23, named, rng) for _ in range(40)}
    assert any("১০ নম্বর" in line for line in lines)


def test_commentary_names_the_right_player():
    events = [{
        "minute": 12, "type": "yellow_card",
        "players": [{"role": "booked", "track_id": 3, "team": "team_b"}],
    }]
    identities = {3: {"name": "", "jersey_number": "4"}}
    teams = {3: {"shirt_colour": "blue"}}
    [(_, line)] = commentary(events, identities, teams, TEMPLATES, random.Random(1))
    assert "৪ নম্বর জার্সির খেলোয়াড়" in line


def test_team_colour_wins_over_own_crop_colour():
    who = describe({}, {"shirt_colour": "orange", "team_colour": "red"})
    assert who["player"] == "লাল জার্সির খেলোয়াড়"


def test_video_time_format():
    assert video_time(82.24) == "1:22.2"
    assert video_time(5) == "0:05.0"


def test_goal_without_scorer_uses_team_colour():
    event = {"type": "goal", "minute": 2, "players": [], "team": "team_b"}
    teams = {9: {"team": "team_b", "role": "player", "team_colour": "blue"}}
    [(_, line)] = commentary([event], {}, teams, TEMPLATES, random.Random(0))
    assert "নীল জার্সির খেলোয়াড়" in line
