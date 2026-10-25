"""Smoke tests for the benchmark scripts on a synthetic SoccerNet-style
tracking clip (the real data is downloaded on the user's PC)."""

import json

import cv2
import numpy as np
import pytest

from broadcast_sim import simulate

pytest.importorskip("trackeval")


def make_sequence(folder, frames=120):
    images = folder / "img1"
    images.mkdir(parents=True)
    gt, det = [], []
    for index, image, detections in simulate(frames=frames, cut_ranges=()):
        cv2.imwrite(str(images / f"{index + 1:06d}.jpg"), image)
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            det.append(f"{index + 1},-1,{x1},{y1},{x2 - x1},{y2 - y1},{d['conf']},-1,-1,-1")
            if d["truth"] is not None:
                gt.append(f"{index + 1},{d['truth'] + 1},{x1},{y1},{x2 - x1},{y2 - y1},1,-1,-1,-1")
        # A ball that must be left out of the ground truth.
        gt.append(f"{index + 1},99,10,10,5,5,1,-1,-1,-1")
    (folder / "gt").mkdir()
    (folder / "gt" / "gt.txt").write_text("\n".join(gt) + "\n")
    (folder / "det").mkdir()
    (folder / "det" / "det.txt").write_text("\n".join(det) + "\n")
    (folder / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-900\nimDir=img1\nframeRate=25\n"
        f"seqLength={frames}\nimWidth=960\nimHeight=540\nimExt=.jpg\n")
    (folder / "gameinfo.ini").write_text(
        "[Sequence]\nnum_tracklets=2\ntrackletID_1= player team left;1\n"
        "trackletID_99= ball;1\n")


def test_tracking_benchmark_runs(tmp_path, monkeypatch):
    import tracking_benchmark as tb

    make_sequence(tmp_path / "test" / "SNMOT-900")
    monkeypatch.setattr(tb, "SEQUENCES", tmp_path / "test")
    monkeypatch.setattr(tb, "WORK", tmp_path / "work")
    monkeypatch.setattr(tb, "RESULTS", tmp_path / "results")
    (tmp_path / "results").mkdir()
    assert tb.ball_ids(tmp_path / "test" / "SNMOT-900") == {99}

    monkeypatch.setattr("sys.argv", ["tracking_benchmark.py", "--detectors",
                                     "det", "--trackers", "bytetrack"])
    tb.main()
    rows = json.loads((tmp_path / "results" / "tracking_results.json").read_text())
    assert [r["ID layer"] for r in rows] == ["none", "ours"]
    for row in rows:
        assert 30 < row["HOTA"] <= 100
        assert row["tracker FPS"] > 0
    gt = (tmp_path / "work" / "gt" / "SNMOT-900" / "gt" / "gt.txt").read_text()
    assert ",99," not in gt


def test_calf_runs_on_features():
    import spotting_benchmark as sb
    from common import SN_SPOTTING

    if not (SN_SPOTTING / sb.CALF_WEIGHTS).exists():
        pytest.skip("sn-spotting not downloaded")
    rng = np.random.default_rng(0)
    detections = sb.CALF(SN_SPOTTING).detections(
        rng.normal(size=(500, 512)).astype(np.float32))
    assert detections.shape == (500, 17)
    predictions = sb.calf_predictions(detections, half=2)
    assert predictions and all(p["half"] == "2" for p in predictions)
