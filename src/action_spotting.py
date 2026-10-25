"""
Spot match events with the trained SoccerNet-v2 NetVLAD++ model.

NetVLAD++ (Giancola & Ghanem, CVSports 2021, trained by the SoccerNet
team on 500 full matches) spots 17 event types in broadcast video,
including goals, fouls, yellow/red cards, corners, direct/indirect free
kicks, clearances (goal kicks), penalties, offsides and throw-ins.

Steps:
1. Features: 2 frames per second, each resized to 224 px high and
   centre-cropped to 224x224, go through ResNet-152 (ImageNet, Keras):
   one 2048-number vector per half second. This is exactly how the
   SoccerNet features the model was trained on were made.
2. Spotting: for every half second the model looks at the 15 seconds
   around it and gives a score per event type; the peaks of each score
   (non-maximum suppression) are the events.

Output: outputs/tracking/<name>_spotting.json in the SoccerNet
"results_spotting" format, which match_events.py --spotting reads.

Models are downloaded on first use into <project>/.cache (see
project_cache.py): ResNet-152 ImageNet weights (~240 MB, Keras) and the
NetVLAD++ weights (~20 MB, from github.com/SoccerNet/sn-spotting,
Apache-2.0).

Needs: pip install tensorflow  (CPU is fine: ~1 s per second of video)

Usage:
    python src/action_spotting.py --video data/videos/test1.mp4
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import json
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = ROOT / "outputs" / "tracking"
MODEL_URL = (
    "https://raw.githubusercontent.com/SoccerNet/sn-spotting/main/"
    "Benchmarks/TemporallyAwarePooling/models/NetVLAD++/model.pth.tar"
)
MODEL_PATH = project_cache.CACHE / "soccernet" / "NetVLAD++" / "model.pth.tar"

FEATURE_FPS = 2.0
WINDOW_SECONDS = 15
# Peaks of one event type closer than this are one event. SoccerNet uses
# 30 s for full matches; highlight reels pack events much closer.
NMS_SECONDS = 10.0

# SoccerNet-v2 classes, in the model's output order.
LABELS = [
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside",
    "Shots on target", "Shots off target", "Clearance", "Ball out of play",
    "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick", "Corner",
    "Yellow card", "Red card", "Yellow->red card",
]


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------

def sample_frames(video_path, fps=FEATURE_FPS):
    """Frames at `fps` per second, 224x224, as SoccerNet's FrameCV makes
    them (OpenCV BGR, resized to 224 high, centre crop)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    drop = video_fps / fps
    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if index % drop < 1:
            height, width = frame.shape[:2]
            new_width = int(round(width * 224.0 / height))
            frame = cv2.resize(frame, (new_width, 224),
                               interpolation=cv2.INTER_AREA)
            side = (new_width - 224) // 2
            frames.append(frame[:, side:side + 224])
    capture.release()
    return np.array(frames), video_fps


def extract_features(video_path, batch_size=32):
    """(N, 2048) ResNet-152 features, one per half second."""
    from tensorflow import keras
    from tensorflow.keras.applications.resnet import preprocess_input

    frames, _ = sample_frames(video_path)
    base = keras.applications.resnet.ResNet152(include_top=True,
                                               weights="imagenet")
    model = keras.Model(base.input, base.get_layer("avg_pool").output)
    return model.predict(preprocess_input(frames.astype(np.float32)),
                         batch_size=batch_size, verbose=1)


# ----------------------------------------------------------------------
# NetVLAD++ (same layers as the sn-spotting code, so the weights load)
# ----------------------------------------------------------------------

class NetVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        scale = 1 / math.sqrt(feature_size)
        self.clusters = nn.Parameter(scale * torch.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter(scale * torch.randn(1, feature_size, cluster_size))

    def forward(self, x):
        frames = x.size(1)
        x = F.normalize(x, p=2, dim=2).reshape(-1, self.feature_size)
        assignment = F.softmax(x @ self.clusters, dim=1)
        assignment = assignment.view(-1, frames, self.cluster_size)
        a = assignment.sum(-2, keepdim=True) * self.clusters2
        x = x.view(-1, frames, self.feature_size)
        vlad = (assignment.transpose(1, 2) @ x).transpose(1, 2) - a
        vlad = F.normalize(vlad)
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        return F.normalize(vlad)


class NetVLADPlusPlus(nn.Module):
    def __init__(self, input_size=2048, num_classes=17, vocab_size=64):
        super().__init__()
        self.feature_extractor = nn.Linear(input_size, 512)
        self.pool_layer_before = NetVLAD(vocab_size // 2, 512)
        self.pool_layer_after = NetVLAD(vocab_size // 2, 512)
        self.fc = nn.Linear(512 * vocab_size, num_classes + 1)

    def forward(self, inputs):
        batch, frames, channels = inputs.shape
        x = self.feature_extractor(inputs.reshape(batch * frames, channels))
        x = x.reshape(batch, frames, -1)
        half = frames // 2
        pooled = torch.cat((self.pool_layer_before(x[:, :half]),
                            self.pool_layer_after(x[:, half:])), dim=1)
        return torch.sigmoid(self.fc(pooled))


def load_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading NetVLAD++ weights to {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    model = NetVLADPlusPlus()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def clips(features, window, start=0, stop=None):
    """One window of `window` feature frames centred on each frame from
    start to stop (edges repeat the first/last frame)."""
    stop = len(features) if stop is None else min(stop, len(features))
    offsets = np.arange(-(window // 2), window - window // 2)
    index = np.clip(np.arange(start, stop)[:, None] + offsets, 0,
                    len(features) - 1)
    return features[index]


def score_frames(model, features, batch_size=256, device="cpu"):
    """(N, 17) event scores, one row per half second. Windows are built
    one batch at a time, so a full match half fits in memory."""
    features = np.asarray(features, dtype=np.float32)
    window = int(WINDOW_SECONDS * FEATURE_FPS)
    model.to(device)
    scores = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = clips(features, window, start, start + batch_size)
            output = model(torch.from_numpy(batch).to(device))
            scores.append(output.cpu().numpy()[:, 1:])  # column 0: background
    return np.concatenate(scores) if scores else np.zeros((0, len(LABELS)))


def peaks(scores, window, threshold):
    """Non-maximum suppression on one score curve: [(index, score)]."""
    scores = np.array(scores, dtype=np.float64)
    found = []
    while len(scores) and scores.max() >= threshold:
        index = int(scores.argmax())
        found.append((index, float(scores[index])))
        scores[max(0, index - window // 2):index + window // 2] = -1
    return found


def spot(scores, threshold=0.0, nms_seconds=NMS_SECONDS, half=1):
    """SoccerNet results_spotting predictions from frame scores."""
    window = int(nms_seconds * FEATURE_FPS)
    predictions = []
    for label_index, label in enumerate(LABELS):
        for index, confidence in peaks(scores[:, label_index], window, threshold):
            seconds = index / FEATURE_FPS
            predictions.append({
                "gameTime": f"{half} - {int(seconds // 60)}:{int(seconds % 60):02d}",
                "label": label,
                "position": str(int(seconds * 1000)),
                "half": str(half),
                "confidence": str(round(confidence, 4)),
            })
    predictions.sort(key=lambda p: int(p["position"]))
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="lowest confidence kept in the JSON")
    parser.add_argument("--nms-seconds", type=float, default=NMS_SECONDS)
    args = parser.parse_args()

    name = args.video.stem
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    features_path = TRACKING_DIR / f"{name}_resnet152.npy"
    if features_path.exists():
        features = np.load(features_path)
    else:
        features = extract_features(args.video)
        np.save(features_path, features)

    scores = score_frames(load_model(), features)
    np.save(TRACKING_DIR / f"{name}_spotting_scores.npy", scores)
    predictions = spot(scores, args.threshold, args.nms_seconds)

    output = TRACKING_DIR / f"{name}_spotting.json"
    output.write_text(json.dumps({"UrlLocal": str(args.video),
                                  "predictions": predictions}, indent=2),
                      encoding="utf-8")
    for p in predictions:
        seconds = int(p["position"]) / 1000
        print(f"{int(seconds // 60)}:{seconds % 60:04.1f}  "
              f"{p['label']:<20} {float(p['confidence']):.2f}")
    print("Spotting JSON:", output)


if __name__ == "__main__":
    main()
