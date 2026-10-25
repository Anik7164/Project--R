"""Football detector support: SoccerNet labels -> YOLO, class lookup,
roles from the detector."""

import csv

from detector_classes import ball_class, is_football_model, people_classes
from identify_teams import detector_roles
from team_identification import TeamVotes
import train_detector as td

COCO = {0: "person", 32: "sports ball", 2: "car"}
FOOTBALL = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}


def test_class_lookup():
    assert people_classes(COCO) == [0] and ball_class(COCO) == 32
    assert people_classes(FOOTBALL) == [0, 1, 2] and ball_class(FOOTBALL) == 3
    assert is_football_model(FOOTBALL) and not is_football_model(COCO)


def make_clip(folder):
    (folder / "gt").mkdir(parents=True)
    (folder / "seqinfo.ini").write_text(
        "[Sequence]\nimWidth=1000\nimHeight=500\nseqLength=4\n")
    (folder / "gameinfo.ini").write_text(
        "[Sequence]\nnum_tracklets=5\ntrackletID_1= ball;1\n"
        "trackletID_2= player team left;7\ntrackletID_3= goalkeepers team right;1\n"
        "trackletID_4= referee;main\ntrackletID_5= other;coach\n")
    rows = []
    for frame in range(1, 5):
        rows += [f"{frame},1,500,250,10,10,1,-1,-1,-1",
                 f"{frame},2,100,100,50,100,1,-1,-1,-1",
                 f"{frame},3,-20,100,60,100,1,-1,-1,-1",  # partly outside
                 f"{frame},4,700,100,50,100,1,-1,-1,-1",
                 f"{frame},5,900,100,50,100,1,-1,-1,-1"]
    (folder / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")


def test_soccernet_labels_to_yolo(tmp_path):
    clip = tmp_path / "SNMOT-001"
    make_clip(clip)
    assert td.tracklet_classes(clip) == {1: 3, 2: 0, 3: 1, 4: 2}
    labels = td.yolo_labels(clip, stride=3)
    assert sorted(labels) == [1, 4]  # every 3rd frame
    rows = [r.split() for r in labels[1]]
    assert sorted(int(r[0]) for r in rows) == [0, 1, 2, 3]  # coach left out
    keeper = next(r for r in rows if r[0] == "1")
    # Clipped to the image: x from 0 to 40 of 1000.
    assert abs(float(keeper[1]) - 0.02) < 1e-6 and abs(float(keeper[3]) - 0.04) < 1e-6


def test_roles_from_detector(tmp_path):
    path = tmp_path / "tracks.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "track_id", "class_name"])
        for frame in range(10):
            writer.writerow([frame, 1, "referee"])
            writer.writerow([frame, 2, "player" if frame < 8 else "goalkeeper"])
            writer.writerow([frame, 3, "player" if frame < 5 else "referee"])
    assert detector_roles(path) == {1: "referee", 2: "player"}

    votes = TeamVotes()
    votes.votes[1].update({"team_a": 9, "other": 1})   # colour says player
    votes.votes[2].update({"other": 8, "team_b": 2})   # colour says outlier
    decisions = votes.decide({1: "referee", 2: "player"})
    assert decisions[1] == {"team": None, "role": "referee", "confidence": 1.0}
    assert decisions[2]["role"] == "player" and decisions[2]["team"] == "team_b"
