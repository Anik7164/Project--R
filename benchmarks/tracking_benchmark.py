"""
Player tracking benchmark on SoccerNet-Tracking (2023 test split: 49
thirty-second broadcast clips with every player, goalkeeper and referee
boxed and identified in every frame).

Every detector x tracker pair is run twice: the tracker alone, and the
tracker followed by our stable-ID layer (src/identity.py). So the table
shows both how our full pipeline compares to the trackers on the market
and how much the ID layer adds to each of them.

Detectors (person class, COCO-trained; downloaded on first use into
.cache/models):
  det          the public detections shipped with SoccerNet (same boxes
               for every tracker, the usual way to compare trackers)
  yolo26n/s/m/l/x, yolo11n/s/m/l/x, rtdetr-l, rtdetr-x   (Ultralytics)
  any .pt path  e.g. a football fine-tuned model

Trackers:
  bytetrack, botsort        Ultralytics (botsort uses our botsort_custom.yaml,
                            as in track_players.py)
  ocsort, deepocsort, strongsort, boosttrack, hybridsort, botsort-reid
                            BoxMOT (pip install boxmot); the ReID ones use
                            OSNet x0.25 (MSMT17) appearance features

Metrics (TrackEval, the code SoccerNet uses): HOTA, DetA, AssA, IDF1,
MOTA, ID switches; plus tracker speed (frames per second, detection time
not included) and detector speed. The ball is removed from the ground
truth, since the trackers follow people only.

Usage (after python benchmarks/download_data.py --tracking):
    python benchmarks/tracking_benchmark.py                       # all trackers, public detections
    python benchmarks/tracking_benchmark.py --trackers botsort \
        --detectors yolo26n yolo26s yolo26m yolo26l yolo26x yolo11x rtdetr-l rtdetr-x
    python benchmarks/tracking_benchmark.py --sequences 5 \\
        --detectors det yolo26s --trackers botsort bytetrack      # quick look
Results: outputs/benchmarks/tracking_results.{md,json,csv}. Finished runs
are reused; --force runs them again.
"""

import argparse
import configparser
import csv
import inspect
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from common import CACHE, RESULTS, ROOT, SOCCERNET, markdown_table

from identity import StablePlayerTracker

SEQUENCES = SOCCERNET / "tracking-2023" / "test"
WORK = RESULTS / "tracking"
MODELS = CACHE / "models"
REID_WEIGHTS = CACHE / "boxmot" / "osnet_x0_25_msmt17.pt"
OUR_BOTSORT = ROOT / "botsort_custom.yaml"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DETECTORS = ["det", "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
             "yolo11x", "rtdetr-l", "rtdetr-x"]
# The full grid takes days; by default all trackers run on the public
# detections (see benchmarks/README.md for the detector comparison).
DEFAULT_DETECTORS = ["det"]
TRACKERS = ["bytetrack", "botsort", "ocsort", "deepocsort", "strongsort",
            "boosttrack", "hybridsort", "botsort-reid"]
# Low, so ByteTrack-style trackers get their second-stage boxes; each
# tracker applies its own thresholds.
DETECTION_CONFIDENCE = 0.1
IMAGE_SIZE = 1280


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def sequence_names(limit=None):
    if not SEQUENCES.exists():
        raise SystemExit(f"No tracking data in {SEQUENCES}. Run: "
                         "python benchmarks/download_data.py --tracking")
    names = sorted(p.name for p in SEQUENCES.iterdir()
                   if (p / "gt" / "gt.txt").exists())
    return names[:limit] if limit else names


def sequence_info(folder):
    info = configparser.ConfigParser()
    info.read(folder / "seqinfo.ini")
    section = info["Sequence"]
    return {
        "length": int(section["seqLength"]),
        "fps": float(section.get("frameRate", 25)),
        "images": folder / section.get("imDir", "img1"),
        "extension": section.get("imExt", ".jpg"),
    }


def ball_ids(folder):
    """Tracklet IDs labelled as the ball in gameinfo.ini."""
    path = folder / "gameinfo.ini"
    if not path.exists():
        return set()
    info = configparser.ConfigParser()
    info.read(path)
    ids = set()
    for key, value in info["Sequence"].items():
        if key.lower().startswith("trackletid_") and \
                value.strip().lower().startswith("ball"):
            ids.add(int(key.split("_")[1]))
    return ids


def frames(info):
    for index in range(info["length"]):
        path = info["images"] / f"{index + 1:06d}{info['extension']}"
        yield index, cv2.imread(str(path))


# ----------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------

def load_detector(name):
    from ultralytics import RTDETR, YOLO

    path = Path(name)
    if path.suffix != ".pt":
        path = MODELS / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    # A full path makes Ultralytics download the weights there, not into
    # the current folder.
    return (RTDETR if "rtdetr" in path.name else YOLO)(str(path))


def public_detections(folder):
    rows = defaultdict(list)
    path = folder / "det" / "det.txt"
    for line in path.read_text().splitlines():
        values = line.split(",")
        frame, x, y, w, h, conf = int(values[0]) - 1, *map(float, values[2:7])
        rows[frame].append((x, y, x + w, y + h, conf))
    return rows


def detect(detector_name, sequence, info, force=False):
    """{frame: (N, 5) x1, y1, x2, y2, conf}, cached per sequence; and the
    detector's frames per second (None when cached or public)."""
    folder = SEQUENCES / sequence
    if detector_name == "det":
        rows = public_detections(folder)
        return {f: np.array(rows.get(f, []), np.float32).reshape(-1, 5)
                for f in range(info["length"])}, None

    cache = WORK / "detections" / Path(detector_name).stem / f"{sequence}.npz"
    if cache.exists() and not force:
        data = np.load(cache)
        return {int(k): data[k] for k in data.files}, None

    model = _detector_models.setdefault(detector_name, load_detector(detector_name))
    output, started = {}, time.perf_counter()
    for index, image in frames(info):
        result = model.predict(image, classes=[0], conf=DETECTION_CONFIDENCE,
                               imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)[0]
        boxes = result.boxes
        output[index] = np.concatenate(
            [boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None]],
            axis=1).astype(np.float32)
    fps = info["length"] / (time.perf_counter() - started)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{str(k): v for k, v in output.items()})
    return output, fps


_detector_models = {}


# ----------------------------------------------------------------------
# Trackers: each returns (M, >=6) rows x1, y1, x2, y2, id, conf
# ----------------------------------------------------------------------

class UltralyticsTracker:
    def __init__(self, config, fps):
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import IterableSimpleNamespace, YAML

        args = IterableSimpleNamespace(**YAML.load(config))
        tracker_class = TRACKER_MAP[args.tracker_type]
        if "frame_rate" in inspect.signature(tracker_class.__init__).parameters:
            self.tracker = tracker_class(args=args, frame_rate=int(round(fps)))
        else:
            self.tracker = tracker_class(args=args)

    def update(self, detections, image):
        from ultralytics.engine.results import Boxes

        data = np.zeros((len(detections), 6), np.float32)
        data[:, :5] = detections
        tracks = self.tracker.update(Boxes(data, image.shape[:2]), image)
        return np.asarray(tracks).reshape(-1, 8)[:, :6]


class BoxmotTracker:
    def __init__(self, name, fps):
        import boxmot

        reid = {"reid_weights": REID_WEIGHTS, "device": DEVICE,
                "half": DEVICE != "cpu"}
        REID_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        if name == "ocsort":
            self.tracker = boxmot.OcSort()
        elif name == "deepocsort":
            self.tracker = boxmot.DeepOcSort(**reid)
        elif name == "strongsort":
            self.tracker = boxmot.StrongSort(**reid)
        elif name == "boosttrack":
            self.tracker = boxmot.BoostTrack(use_embeddings=True, **reid)
        elif name == "hybridsort":
            self.tracker = boxmot.HybridSort(**reid)
        elif name == "botsort-reid":
            self.tracker = boxmot.BotSort(frame_rate=int(round(fps)), **reid)
        else:
            raise ValueError(name)

    def update(self, detections, image):
        data = np.zeros((len(detections), 6), np.float32)
        data[:, :5] = detections
        tracks = self.tracker.update(data, image)
        return np.asarray(tracks).reshape(-1, 8)[:, :6] if len(tracks) else \
            np.zeros((0, 6))


def make_tracker(name, fps):
    if name == "bytetrack":
        from ultralytics.utils.checks import check_yaml
        return UltralyticsTracker(check_yaml("bytetrack.yaml"), fps)
    if name == "botsort":
        return UltralyticsTracker(OUR_BOTSORT, fps)
    return BoxmotTracker(name, fps)


# ----------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------

def track_sequence(detections, info, tracker_name, with_layer):
    """Return ({frame: [(id, x1, y1, x2, y2, conf)]}, tracker seconds)."""
    tracker = make_tracker(tracker_name, info["fps"])
    stable = StablePlayerTracker() if with_layer else None
    output = defaultdict(list)
    pending = defaultdict(list)  # tracker id -> rows waiting for a player ID
    seconds = 0.0
    for index, image in frames(info):
        started = time.perf_counter()
        tracks = tracker.update(detections[index], image)
        if stable is None:
            for x1, y1, x2, y2, track_id, conf in tracks:
                output[index].append([int(track_id), x1, y1, x2, y2, conf])
            seconds += time.perf_counter() - started
            continue

        height, width = image.shape[:2]
        boxes = [(max(0, int(x1)), max(0, int(y1)),
                  min(width - 1, int(x2)), min(height - 1, int(y2)))
                 for x1, y1, x2, y2 in tracks[:, :4]]
        ids = [int(t) for t in tracks[:, 4]]
        keep, assignments, resolved = stable.process(
            index, image, ids, boxes, [float(c) for c in tracks[:, 5]])
        _resolve(pending, resolved)
        for i, player_id in zip(keep, assignments):
            row = [player_id, *boxes[i], float(tracks[i, 5])]
            if player_id is None:
                pending[ids[i]].append(row)
            output[index].append(row)
        seconds += time.perf_counter() - started
    if stable is not None:
        _resolve(pending, stable.flush())
    final = {f: [r for r in rows if r[0] is not None and r[0] >= 0]
             for f, rows in output.items()}
    return final, seconds


def _resolve(pending, resolved):
    for tracker_id, player_id in resolved.items():
        for row in pending.pop(tracker_id, []):
            row[0] = player_id if player_id is not None else -1


def write_mot(path, tracks):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for frame in sorted(tracks):
            for track_id, x1, y1, x2, y2, conf in tracks[frame]:
                file.write(f"{frame + 1},{track_id},{x1:.1f},{y1:.1f},"
                           f"{x2 - x1:.1f},{y2 - y1:.1f},{conf:.3f},-1,-1,-1\n")


def write_ground_truth(sequences):
    """Ground truth without the ball, for TrackEval."""
    folder = WORK / "gt"
    for sequence in sequences:
        source = SEQUENCES / sequence
        balls = ball_ids(source)
        lines = [line for line in (source / "gt" / "gt.txt").read_text().splitlines()
                 if line and int(line.split(",")[1]) not in balls]
        target = folder / sequence / "gt" / "gt.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n")
    return folder


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate(run_names, sequences, lengths):
    # TrackEval still uses the NumPy aliases removed in NumPy 1.24.
    for alias, builtin in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, alias):
            setattr(np, alias, builtin)
    import trackeval

    gt_folder = write_ground_truth(sequences)
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({"PRINT_RESULTS": False, "PRINT_CONFIG": False,
                        "TIME_PROGRESS": False, "OUTPUT_DETAILED": False,
                        "PLOT_CURVES": False, "DISPLAY_LESS_PROGRESS": True})
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(gt_folder),
        "TRACKERS_FOLDER": str(WORK / "runs"),
        "OUTPUT_FOLDER": str(WORK / "trackeval"),
        "TRACKERS_TO_EVAL": run_names,
        "SEQ_INFO": {s: lengths[s] for s in sequences},
        "SKIP_SPLIT_FOL": True,
        "DO_PREPROC": False,  # as in the SoccerNet tracking evaluation
        "PRINT_CONFIG": False,
    })
    evaluator = trackeval.Evaluator(eval_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
               trackeval.metrics.Identity()]
    results, _ = evaluator.evaluate(
        [trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics)
    scores = {}
    for name, result in results["MotChallenge2DBox"].items():
        combined = result["COMBINED_SEQ"]["pedestrian"]
        scores[name] = {
            "HOTA": 100 * float(np.mean(combined["HOTA"]["HOTA"])),
            "DetA": 100 * float(np.mean(combined["HOTA"]["DetA"])),
            "AssA": 100 * float(np.mean(combined["HOTA"]["AssA"])),
            "IDF1": 100 * float(combined["Identity"]["IDF1"]),
            "MOTA": 100 * float(combined["CLEAR"]["MOTA"]),
            "IDSW": int(combined["CLEAR"]["IDSW"]),
        }
    return scores


def run_name(detector, tracker, with_layer):
    return f"{Path(detector).stem}__{tracker}{'__ours' if with_layer else ''}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--detectors", nargs="+", default=DEFAULT_DETECTORS,
                        help=f"any of {DETECTORS} or a .pt path")
    parser.add_argument("--trackers", nargs="+", default=TRACKERS,
                        choices=TRACKERS)
    parser.add_argument("--sequences", type=int,
                        help="only the first N test clips (quick runs)")
    parser.add_argument("--no-layer", action="store_true",
                        help="skip the runs with our stable-ID layer")
    parser.add_argument("--force", action="store_true",
                        help="run again even if results exist")
    args = parser.parse_args()

    sequences = sequence_names(args.sequences)
    infos = {s: sequence_info(SEQUENCES / s) for s in sequences}
    timing_path = WORK / "timings.json"
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    layers = (False,) if args.no_layer else (False, True)

    run_names = []
    redetected = set()  # with --force, detect each clip again only once
    for detector in args.detectors:
        detector_frames = detector_seconds = 0
        for tracker in args.trackers:
            for with_layer in layers:
                name = run_name(detector, tracker, with_layer)
                run_names.append(name)
                folder = WORK / "runs" / name / "data"
                done = all((folder / f"{s}.txt").exists() for s in sequences)
                if done and not args.force:
                    continue
                print(f"{name}: tracking {len(sequences)} clips")
                frames_done, seconds = 0, 0.0
                for number, sequence in enumerate(sequences, 1):
                    force = args.force and (detector, sequence) not in redetected
                    redetected.add((detector, sequence))
                    detections, fps = detect(detector, sequence, infos[sequence],
                                             force)
                    if fps:
                        detector_frames += infos[sequence]["length"]
                        detector_seconds += infos[sequence]["length"] / fps
                    tracks, spent = track_sequence(detections, infos[sequence],
                                                   tracker, with_layer)
                    write_mot(folder / f"{sequence}.txt", tracks)
                    frames_done += infos[sequence]["length"]
                    seconds += spent
                    print(f"  {number}/{len(sequences)} {sequence}", end="\r")
                print()
                timings[name] = frames_done / max(seconds, 1e-9)
        if detector_seconds:
            timings[f"detector:{Path(detector).stem}"] = \
                detector_frames / detector_seconds
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(timings, indent=2))

    # Report every finished run (also from earlier invocations), so the
    # tracker and detector comparisons end up in one table.
    run_names = [d.name for d in sorted((WORK / "runs").iterdir())
                 if all((d / "data" / f"{s}.txt").exists() for s in sequences)]
    scores = evaluate(run_names, sequences,
                      {s: infos[s]["length"] for s in sequences})
    rows = []
    for name in run_names:
        detector, tracker, *layer = name.split("__")
        rows.append({
            "detector": detector, "tracker": tracker,
            "ID layer": "ours" if layer else "none",
            **scores[name],
            "tracker FPS": timings.get(name),
            "detector FPS": timings.get(f"detector:{detector}"),
        })
    columns = ["detector", "tracker", "ID layer", "HOTA", "DetA", "AssA",
               "IDF1", "MOTA", "IDSW", "tracker FPS", "detector FPS"]
    text = (f"# Player tracking on SoccerNet-Tracking test "
            f"({len(sequences)} clips)\n\n"
            "TrackEval, ball removed from the ground truth. Scores in %. "
            f"Speeds measured on {DEVICE}.\n\n"
            + markdown_table(rows, columns) + "\n")
    (RESULTS / "tracking_results.md").write_text(text, encoding="utf-8")
    (RESULTS / "tracking_results.json").write_text(json.dumps(rows, indent=2))
    with (RESULTS / "tracking_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        writer.writerows(rows)
    print(text)
    print("Saved:", RESULTS / "tracking_results.md")


if __name__ == "__main__":
    main()
