"""
Event spotting benchmark: our NetVLAD++ port against the official
SoccerNet-v2 baselines, scored with the official SoccerNet evaluation.

Methods (all trained by the SoccerNet team on the 300 SoccerNet-v2
training matches, all 17 classes):

  maxpool    MaxPool baseline (Deliège et al., CVPRW 2021); official test
             predictions published in sn-spotting
  netvlad    NetVLAD baseline (same paper); official test predictions
  calf       CALF (Cioppa et al., CVPR 2020); official weights, run here
             with the official CALF code on ResNet PCA-512 features
  netvlad++  NetVLAD++ (Giancola & Ghanem, CVSports 2021); official
             weights through our port in src/action_spotting.py, run with
             the official test settings (30 s NMS, no threshold)
  ours       the same NetVLAD++ scores with the NMS window our live
             pipeline uses (10 s), to show what that choice costs on full
             matches

Two modes:

1. SoccerNet test set (100 matches, needs download_data.py --spotting):
       python benchmarks/spotting_benchmark.py
   Writes outputs/benchmarks/spotting/<method>/... prediction folders and
   outputs/benchmarks/spotting_results.{md,json} with average-mAP
   (loose: 5-60 s tolerance, tight: 1-5 s) overall and per class.

2. One of your own videos (no labels needed, compares what each model
   finds):
       python benchmarks/spotting_benchmark.py --video data/videos/test1.mp4
   Add --labels my_labels.json (SoccerNet Labels-v2 format, half 1) to
   also score them.

Published methods we cannot run here (no released weights for these
features, or need raw video and a GPU for training) are listed in
benchmarks/README.md with their published numbers and sources.
"""

import argparse
import importlib.util
import json
import pickle
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from common import (RESULTS, ROOT, SOCCERNET, ensure_sn_spotting,
                    markdown_table)

import action_spotting as ours

SPOTTING_RESULTS = RESULTS / "spotting"
FRAMERATE = 2
OFFICIAL_ZIPS = {
    "maxpool": "Benchmarks/Pooling/models/MAXPOOL_v2/results_spotting_test.zip",
    "netvlad": "Benchmarks/Pooling/models/NETVLAD_v2/results_spotting_test.zip",
}
CALF_WEIGHTS = "Benchmarks/CALF/models/CALF_benchmark/model.pth.tar"
PCA_FILE = "Features/pca_512_TF2.pkl"
AVERAGE_FILE = "Features/average_512_TF2.pkl"
METHODS = ["maxpool", "netvlad", "calf", "netvlad++", "ours"]
OFFICIAL_NMS_SECONDS = 30.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# CALF, through the official code
# ----------------------------------------------------------------------

def _load_module(name, path):
    # The CALF files have generic names (model.py, preprocessing.py), so
    # load them under a unique name instead of putting them on sys.path.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CALF:
    """Official CALF (SoccerNet-v2 benchmark weights): 120 s chunks,
    40 s receptive field, 15 detections per chunk, 20 s NMS."""

    chunk_seconds = 120
    receptive_seconds = 40

    def __init__(self, repo):
        source = repo / "Benchmarks" / "CALF" / "src"
        self.model_code = _load_module("calf_model", source / "model.py")
        self.preprocessing = _load_module("calf_preprocessing",
                                          source / "preprocessing.py")
        self.metrics = _load_module("calf_metrics",
                                    source / "metrics_visibility_fast.py")
        self.chunk = self.chunk_seconds * FRAMERATE
        self.receptive = self.receptive_seconds * FRAMERATE
        self.model = self.model_code.ContextAwareModel(
            weights=None, input_size=512, num_classes=17,
            chunk_size=self.chunk, dim_capsule=16,
            receptive_field=self.receptive, num_detections=15,
            framerate=FRAMERATE)
        checkpoint = torch.load(repo / CALF_WEIGHTS, map_location="cpu",
                                weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(DEVICE).eval()

    def _clips(self, features):
        # Same as SoccerNetClipsTesting.feats2clip in the CALF code.
        if len(features) < self.chunk:  # short videos: repeat the last frame
            pad = np.repeat(features[-1:], self.chunk - len(features), axis=0)
            features = np.concatenate([features, pad])
        feats = torch.from_numpy(features)
        stride = self.chunk - self.receptive
        idx = torch.arange(0, feats.shape[0] - 1, stride)
        idx = torch.stack([idx + i for i in range(self.chunk)], dim=1)
        idx = idx.clamp(0, feats.shape[0] - 1)
        idx[-1] = torch.arange(self.chunk) + feats.shape[0] - self.chunk
        return feats[idx], feats.shape[0]

    def detections(self, features):
        """(frames, 17) array: confidence at spotted frames, -1 elsewhere."""
        length = len(features)
        clips, padded = self._clips(np.asarray(features, dtype=np.float32))
        with torch.no_grad():
            _, spotting = self.model(clips.unsqueeze(1).to(DEVICE))
        long = self.preprocessing.timestamps2long(
            spotting.cpu(), padded, self.chunk, self.receptive)
        nms = self.metrics.NMS(long.numpy(), 20 * FRAMERATE)
        return nms[:length]


def calf_predictions(detections, half):
    predictions = []
    frames, classes = np.where(detections >= 0)
    for frame, label in zip(frames, classes):
        seconds = frame / FRAMERATE
        predictions.append({
            "gameTime": f"{half} - {int(seconds // 60)}:{int(seconds % 60):02d}",
            "label": ours.LABELS[label],
            "position": str(int(seconds * 1000)),
            "half": str(half),
            "confidence": str(float(detections[frame, label])),
        })
    return predictions


def to_pca512(features, repo):
    """ResNet-152 2048-d features -> the PCA-512 features CALF expects,
    with the PCA SoccerNet published (needs scikit-learn)."""
    with open(repo / PCA_FILE, "rb") as file:
        pca = pickle.load(file)
    with open(repo / AVERAGE_FILE, "rb") as file:
        average = pickle.load(file)
    return pca.transform(features - average).astype(np.float32)


# ----------------------------------------------------------------------
# NetVLAD++ (our port)
# ----------------------------------------------------------------------

def netvlad_scores(model, features):
    return ours.score_frames(model, features, device=DEVICE)


# ----------------------------------------------------------------------
# SoccerNet test set
# ----------------------------------------------------------------------

def test_games():
    from SoccerNet.utils import getListGames
    return getListGames(split="test", task="spotting")


def write_game(folder, game, predictions):
    path = folder / game / "results_spotting.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"UrlLocal": game, "predictions": predictions}),
                    encoding="utf-8")


def run_test_set(methods, repo):
    games = test_games()
    missing = [g for g in games
               if not (SOCCERNET / g / "Labels-v2.json").exists()]
    if missing:
        sys.exit(f"{len(missing)} of {len(games)} test games have no "
                 f"Labels-v2.json in {SOCCERNET}. Run: "
                 "python benchmarks/download_data.py --spotting")

    timings = {}
    for method in methods:
        folder = SPOTTING_RESULTS / method
        if method in OFFICIAL_ZIPS:
            if folder.exists():
                shutil.rmtree(folder)
            with zipfile.ZipFile(repo / OFFICIAL_ZIPS[method]) as archive:
                archive.extractall(folder)
            continue

        print(f"Running {method} on {len(games)} test games ({DEVICE})")
        calf = CALF(repo) if method == "calf" else None
        netvlad = ours.load_model() if method != "calf" else None
        nms = OFFICIAL_NMS_SECONDS if method == "netvlad++" else ours.NMS_SECONDS
        started, frames = time.perf_counter(), 0
        for number, game in enumerate(games, 1):
            predictions = []
            for half in (1, 2):
                if calf is not None:
                    name = f"{half}_ResNET_TF2_PCA512.npy"
                else:
                    name = f"{half}_ResNET_TF2.npy"
                features = np.load(SOCCERNET / game / name)
                frames += len(features)
                if calf is not None:
                    predictions += calf_predictions(calf.detections(features), half)
                else:
                    scores = netvlad_scores(netvlad, features)
                    predictions += ours.spot(scores, 0.0, nms, half)
            write_game(folder, game, predictions)
            print(f"  {number}/{len(games)} {game}", end="\r")
        seconds = time.perf_counter() - started
        # Video seconds handled per second of compute (features given).
        timings[method] = (frames / FRAMERATE) / seconds
        print()
    return timings


def evaluate_all(methods, timings):
    from SoccerNet.Evaluation.ActionSpotting import evaluate

    rows, per_class = [], {}
    for method in methods:
        row = {"method": method}
        for metric in ("loose", "tight"):
            result = evaluate(SoccerNet_path=str(SOCCERNET),
                              Predictions_path=str(SPOTTING_RESULTS / method),
                              split="test", version=2, prediction_file=
                              "results_spotting.json", metric=metric)
            row[f"{metric} a-mAP"] = 100 * result["a_mAP"]
            row[f"{metric} visible"] = 100 * result["a_mAP_visible"]
            row[f"{metric} unshown"] = 100 * result["a_mAP_unshown"]
            if metric == "loose":
                per_class[method] = [100 * v for v in result["a_mAP_per_class"]]
        row["video s / s"] = timings.get(method)
        rows.append(row)
        print(row)

    columns = ["method", "loose a-mAP", "loose visible", "loose unshown",
               "tight a-mAP", "tight visible", "tight unshown", "video s / s"]
    class_rows = [{"class": label,
                   **{m: per_class[m][i] for m in methods}}
                  for i, label in enumerate(ours.LABELS)]
    text = ("# Event spotting on the SoccerNet-v2 test set (100 matches)\n\n"
            "Average-mAP in %, official SoccerNet evaluation. "
            "'video s / s' = seconds of video per second of compute, "
            "features already extracted (blank = published predictions).\n\n"
            + markdown_table(rows, columns)
            + "\n\n## Loose average-mAP per class\n\n"
            + markdown_table(class_rows, ["class"] + methods) + "\n")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "spotting_results.md").write_text(text, encoding="utf-8")
    (RESULTS / "spotting_results.json").write_text(
        json.dumps({"overall": rows, "per_class_loose": class_rows}, indent=2),
        encoding="utf-8")
    print(text)
    print("Saved:", RESULTS / "spotting_results.md")


# ----------------------------------------------------------------------
# One video
# ----------------------------------------------------------------------

def run_video(video, labels, repo, threshold):
    name = video.stem
    features_path = ours.TRACKING_DIR / f"{name}_resnet152.npy"
    if features_path.exists():
        features = np.load(features_path)
    else:
        features = ours.extract_features(video)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(features_path, features)

    netvlad = ours.load_model()
    scores = netvlad_scores(netvlad, features)
    predictions = {
        "netvlad++ (30 s NMS)": ours.spot(scores, threshold, OFFICIAL_NMS_SECONDS),
        "ours (10 s NMS)": ours.spot(scores, threshold, ours.NMS_SECONDS),
        "calf": [p for p in calf_predictions(
            CALF(repo).detections(to_pca512(features, repo)), 1)
            if float(p["confidence"]) >= threshold],
    }

    folder = RESULTS / "video" / name
    folder.mkdir(parents=True, exist_ok=True)
    for method, found in predictions.items():
        print(f"\n== {method}: {len(found)} events (confidence >= {threshold})")
        for p in sorted(found, key=lambda p: int(p["position"])):
            seconds = int(p["position"]) / 1000
            print(f"  {int(seconds // 60)}:{seconds % 60:04.1f}  "
                  f"{p['label']:<20} {float(p['confidence']):.2f}")
        safe = method.split()[0].replace("+", "p")
        (folder / f"{safe}.json").write_text(
            json.dumps({"UrlLocal": str(video), "predictions": found}, indent=2),
            encoding="utf-8")
    if labels:
        score_video(predictions, labels)
    print("\nSaved predictions in", folder)


def score_video(predictions, labels_path):
    """Average-mAP of each method against your own labels for the video
    (SoccerNet Labels-v2 format, every annotation in half 1), with the
    official evaluation."""
    from SoccerNet.Evaluation.ActionSpotting import evaluate
    from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2

    work = RESULTS / "video" / "_score"
    if work.exists():
        shutil.rmtree(work)
    (work / "labels" / "video").mkdir(parents=True)
    shutil.copy(labels_path, work / "labels" / "video" / "Labels-v2.json")
    for method, found in predictions.items():
        write_game(work / "predictions" / method, "video", found)
        line = [method]
        for metric in ("loose", "tight"):
            result = evaluate(SoccerNet_path=str(work / "labels"),
                              Predictions_path=str(work / "predictions" / method),
                              prediction_file="results_spotting.json",
                              label_files="Labels-v2.json", dataset=None,
                              EVENT_DICTIONARY=EVENT_DICTIONARY_V2,
                              metric=metric)
            line.append(f"{metric} a-mAP {100 * result['a_mAP']:.1f}")
        print("  ".join(line))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS)
    parser.add_argument("--video", type=Path,
                        help="compare the models on one of your own videos")
    parser.add_argument("--labels", type=Path,
                        help="with --video: your labels (Labels-v2 format)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="with --video: lowest confidence shown")
    parser.add_argument("--skip-run", action="store_true",
                        help="only evaluate predictions already written")
    args = parser.parse_args()

    repo = ensure_sn_spotting()
    if args.video:
        run_video(args.video, args.labels, repo, args.threshold)
        return
    timings = {} if args.skip_run else run_test_set(args.methods, repo)
    evaluate_all(args.methods, timings)


if __name__ == "__main__":
    main()
