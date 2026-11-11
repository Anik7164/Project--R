"""
Train a football detector: player, goalkeeper, referee and ball.

The COCO model (yolo26n.pt) only knows "person" and "sports ball". It
cannot tell a referee from a player, and on broadcast video it misses
the small ball or finds it on shoes, lines and adverts. This script
trains a YOLO26 model on SoccerNet-Tracking, where every player,
goalkeeper, referee and the ball is boxed in every frame of 57
thirty-second broadcast clips.

    python src/train_detector.py download          # SoccerNet-Tracking train (~12 GB)
    python src/train_detector.py prepare           # YOLO dataset in data/football_det
    python src/train_detector.py train             # ~3-8 h on one GPU
    python src/train_detector.py train --model yolo26m --epochs 60 --batch 8

The best weights are copied to models/football_<model>.pt; give that file
to the other scripts with --model or the YOLO_MODEL variable.

Clips: the 57 train clips are split by clip, 50 for training and 7 for
validation. The 49 test clips are never used, so they stay free for a
fair benchmark later. Every 3rd frame is used (neighbouring frames are
almost the same). Images are hard-linked, not copied, so the dataset
takes almost no extra space (same drive needed).
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import configparser
import os
import shutil
import zipfile
from pathlib import Path

ROOT = project_cache.ROOT
SOCCERNET = ROOT / "data" / "SoccerNet"
TRACKING = SOCCERNET / "tracking-2023"
DATASET = ROOT / "data" / "football_det"
MODELS = ROOT / "models"
RUNS = ROOT / "outputs" / "training"
# Your own footage, labelled by hand (label-frames / prepare-local).
LOCAL = ROOT / "data" / "local_det"
LOCAL_DATASET = ROOT / "data" / "football_local"

CLASSES = ["player", "goalkeeper", "referee", "ball"]
VALIDATION_CLIPS = 7
FRAME_STRIDE = 3
# Each class is a big object except the ball: train on large images.
IMAGE_SIZE = 1280


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def download():
    from SoccerNet.Downloader import SoccerNetDownloader

    SOCCERNET.mkdir(parents=True, exist_ok=True)
    SoccerNetDownloader(LocalDirectory=str(SOCCERNET)).downloadDataTask(
        task="tracking-2023", split=["train"])
    archive = TRACKING / "train.zip"
    if archive.exists() and not (TRACKING / "train").exists():
        print(f"Unzipping {archive}")
        with zipfile.ZipFile(archive) as file:
            file.extractall(TRACKING)
        print(f"You can delete {archive} now to free ~12 GB.")


def tracklet_classes(clip):
    """{tracklet id: class index} from the clip's gameinfo.ini."""
    info = configparser.ConfigParser()
    info.read(clip / "gameinfo.ini")
    classes = {}
    for key, value in info["Sequence"].items():
        if not key.lower().startswith("trackletid_"):
            continue
        role = value.split(";")[0].strip().lower()
        if role.startswith("ball"):
            index = 3
        elif role.startswith("goalkeeper"):
            index = 1
        elif role.startswith("referee"):
            index = 2
        elif role.startswith("player"):
            index = 0
        else:
            continue  # "other": staff, unknown
        classes[int(key.split("_")[1])] = index
    return classes


def clip_size(clip):
    info = configparser.ConfigParser()
    info.read(clip / "seqinfo.ini")
    section = info["Sequence"]
    return int(section["imWidth"]), int(section["imHeight"])


def yolo_labels(clip, stride):
    """{frame number (1-based): ["class cx cy w h", ...]} for every
    stride-th frame."""
    classes = tracklet_classes(clip)
    width, height = clip_size(clip)
    labels = {}
    for line in (clip / "gt" / "gt.txt").read_text().splitlines():
        if not line.strip():
            continue
        values = line.split(",")
        frame, tracklet = int(values[0]), int(values[1])
        if (frame - 1) % stride or tracklet not in classes:
            continue
        x, y, w, h = map(float, values[2:6])
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(width), x + w), min(float(height), y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        labels.setdefault(frame, []).append(
            f"{classes[tracklet]} {(x0 + x1) / 2 / width:.6f} "
            f"{(y0 + y1) / 2 / height:.6f} {(x1 - x0) / width:.6f} "
            f"{(y1 - y0) / height:.6f}")
    return labels


def _link(source, target):
    try:
        os.link(source, target)  # no extra space, no admin rights needed
    except OSError:
        shutil.copy2(source, target)  # other drive


def prepare(source=None, stride=FRAME_STRIDE, validation=VALIDATION_CLIPS):
    source = Path(source or TRACKING / "train")
    clips = sorted(p for p in source.iterdir() if (p / "gt" / "gt.txt").exists())
    if not clips:
        raise SystemExit(f"No clips in {source}. Run: python src/train_detector.py download")
    validation_clips = set(c.name for c in clips[-validation:]) if validation else set()

    if DATASET.exists():
        shutil.rmtree(DATASET / "images", ignore_errors=True)
        shutil.rmtree(DATASET / "labels", ignore_errors=True)
    counts = {split: [0] * len(CLASSES) for split in ("train", "val")}
    images = {"train": 0, "val": 0}
    for clip in clips:
        split = "val" if clip.name in validation_clips else "train"
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)
        for frame, rows in yolo_labels(clip, stride).items():
            name = f"{clip.name}_{frame:06d}"
            _link(clip / "img1" / f"{frame:06d}.jpg",
                  DATASET / "images" / split / f"{name}.jpg")
            (DATASET / "labels" / split / f"{name}.txt").write_text(
                "\n".join(rows) + "\n")
            images[split] += 1
            for row in rows:
                counts[split][int(row.split()[0])] += 1

    config = DATASET / "football.yaml"
    config.write_text(
        f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8")
    for split in ("train", "val"):
        boxes = ", ".join(f"{n} {c}" for n, c in zip(CLASSES, counts[split]))
        print(f"{split}: {images[split]} images; {boxes}")
    print("Dataset:", config)
    return config


# ----------------------------------------------------------------------
# Your own footage
# ----------------------------------------------------------------------

# Ball-only labelling: player boxes above this confidence are taken from
# the model as they are (they are already good: mAP50 0.97).
AUTO_PEOPLE_CONFIDENCE = 0.5


def label_frames(videos, every=2.0, limit=400, model=None, conf=0.15,
                 ball_only=True):
    """Save frames from your videos with the current model's boxes as
    YOLO labels, to be corrected by hand.

    Frames are taken every `every` seconds (neighbouring frames are almost
    the same). The model's guesses use a low confidence so that a missed
    ball is more often there to fix than to draw. Output:
    data/local_det/images/*.jpg, data/local_det/labels/*.txt and
    data/local_det/classes.txt (the class order the labels use).

    ball_only (default): the labels to correct hold only the ball; the
    model's player/goalkeeper/referee boxes are kept aside in
    data/local_det/people/ and added back by prepare-local. Only the ball
    has to be checked, a few seconds per frame.
    """
    import cv2
    from ultralytics import YOLO

    from detector_classes import default_model

    model = YOLO(str(model or default_model()))
    names = model.names
    index = {name: CLASSES.index(name) for name in names.values() if name in CLASSES}
    if len(index) != len(CLASSES):
        raise SystemExit("Pre-labelling needs the football model "
                         "(player/goalkeeper/referee/ball); give --model.")
    (LOCAL / "images").mkdir(parents=True, exist_ok=True)
    (LOCAL / "labels").mkdir(parents=True, exist_ok=True)
    (LOCAL / "people").mkdir(parents=True, exist_ok=True)
    (LOCAL / "classes.txt").write_text("\n".join(CLASSES) + "\n")

    per_video = max(1, limit // len(videos))
    saved = 0
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(every * fps))
        wanted = list(range(step // 2, frames, step))
        if len(wanted) > per_video:  # spread evenly over the whole video
            wanted = [wanted[int(i * len(wanted) / per_video)] for i in range(per_video)]
        for frame_number in wanted:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                continue
            name = f"{Path(video).stem}_{frame_number:06d}"
            if (LOCAL / "labels" / f"{name}.txt").exists():
                continue  # already there, maybe corrected by hand: keep it
            height, width = frame.shape[:2]
            result = model.predict(frame, imgsz=IMAGE_SIZE, conf=conf, verbose=False)[0]
            rows, people = [], []
            for box, cls, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                       result.boxes.conf.tolist()):
                x1, y1, x2, y2 = box
                class_index = index[names[int(cls)]]
                row = (f"{class_index} {(x1 + x2) / 2 / width:.6f} "
                       f"{(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} "
                       f"{(y2 - y1) / height:.6f}")
                if not ball_only or class_index == CLASSES.index("ball"):
                    rows.append(row)
                elif score >= AUTO_PEOPLE_CONFIDENCE:
                    people.append(row)
            cv2.imwrite(str(LOCAL / "images" / f"{name}.jpg"), frame)
            (LOCAL / "labels" / f"{name}.txt").write_text("\n".join(rows) + "\n" if rows else "")
            if ball_only:
                (LOCAL / "people" / f"{name}.txt").write_text(
                    "\n".join(people) + "\n" if people else "")
            saved += 1
        capture.release()
    print(f"{saved} frames with pre-labels in {LOCAL}")
    print("Correct them in a YOLO-format labelling tool (classes.txt gives the "
          "class order), then run: python src/train_detector.py prepare-local")
    return saved


# Frames labelled by auto-label; only these may be overwritten by a later
# auto-label run (frames you corrected by hand are never touched).
AUTO_LIST = LOCAL / "auto_labelled.txt"
# Share of grass-coloured pixels around an auto-labelled ball.
GRASS_SHARE = 0.4


# Frames you labelled by hand with click_balls.py: validation uses these,
# as they are the only real labels of your own footage.
HAND_LIST = LOCAL / "hand_labelled.txt"


def _hand_stems():
    if not HAND_LIST.exists():
        return set()
    return set(HAND_LIST.read_text().split())


def _auto_stems():
    if not AUTO_LIST.exists():
        return set()
    return set(AUTO_LIST.read_text().split())


def _yolo_row(class_index, x1, y1, x2, y2, width, height):
    return (f"{class_index} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def auto_label(videos, model=None, window=3.0, every=6.0, frame_gap=6,
               limit=1500, batch=8, device=None):
    """Label frames of your videos with no hand work (see auto_label.py).

    Windows of `window` seconds every `every` seconds are run through the
    detector frame by frame; inside a window frames are saved at least
    `frame_gap` frames apart, at most `limit` frames over all videos.
    Writes data/local_det/images + labels like label-frames, with the
    ball and the people in the label; frames you corrected by hand are
    never overwritten.
    """
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    from auto_label import LOW_CONFIDENCE, ball_labels, pick_frames
    from ball_tracking import compose, invert
    from camera_motion import CameraMotionEstimator, warp_points
    from detector_classes import default_model

    model_path = Path(model or default_model())
    detector = YOLO(str(model_path))
    names = detector.names
    index = {i: CLASSES.index(n) for i, n in names.items() if n in CLASSES}
    if len(index) != len(CLASSES):
        raise SystemExit("Auto-labelling needs the football model "
                         "(player/goalkeeper/referee/ball); give --model.")
    ball = CLASSES.index("ball")
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    for folder in ("images", "labels"):
        (LOCAL / folder).mkdir(parents=True, exist_ok=True)
    (LOCAL / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    auto = _auto_stems()
    print(f"Auto-labelling with {model_path.name}")

    per_video = max(1, limit // len(videos))
    stats = {"frames": 0, "detected": 0, "faint": 0, "filled": 0, "no_ball": 0}
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise SystemExit(f"Cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        span, step = max(2, int(window * fps)), max(1, int(every * fps))
        starts = list(range(0, max(1, total - span), step))
        per_window = max(1, per_video // max(1, len(starts)))
        saved_here = 0
        for number, start in enumerate(starts):
            if saved_here >= per_video:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for _ in range(span):
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            if len(frames) < 2:
                continue
            height, width = frames[0].shape[:2]

            # Camera motion: every frame mapped back to the first frame of
            # its shot, so a panning camera does not bend the ball's path.
            camera = CameraMotionEstimator()
            shot, cumulative = 0, np.eye(3)[:2]
            shots, to_frame = [], []
            for frame in frames:
                warp, cut = camera.update(frame)
                if cut:
                    shot, cumulative = shot + 1, np.eye(3)[:2]
                else:
                    cumulative = compose(cumulative, warp)
                shots.append(shot)
                to_frame.append(cumulative)

            candidates, people = {}, {}
            for first in range(0, len(frames), batch):
                results = detector.predict(frames[first:first + batch], imgsz=IMAGE_SIZE,
                                           conf=LOW_CONFIDENCE, device=device, verbose=False)
                for offset, result in enumerate(results):
                    i = first + offset
                    candidates[i], people[i] = [], []
                    for box, cls, score in zip(result.boxes.xyxy.tolist(),
                                               result.boxes.cls.tolist(),
                                               result.boxes.conf.tolist()):
                        x1, y1, x2, y2 = box
                        if index[int(cls)] == ball:
                            sx, sy = warp_points([[(x1 + x2) / 2, (y1 + y2) / 2]],
                                                 invert(to_frame[i]))[0]
                            candidates[i].append((float(sx), float(sy), x2 - x1, y2 - y1, score))
                        elif score >= AUTO_PEOPLE_CONFIDENCE:
                            people[i].append(_yolo_row(index[int(cls)], x1, y1, x2, y2,
                                                       width, height))

            hsv = {}

            def on_grass(i, box):
                """Is there grass around the ball? (not a cap in the crowd)"""
                if i not in hsv:
                    hsv[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
                sx, sy, w, h = box
                x, y = warp_points([[sx, sy]], to_frame[i])[0]
                r = 2.5 * max(w, h, 6)
                patch = hsv[i][int(max(0, y - r)):int(min(height, y + r)),
                               int(max(0, x - r)):int(min(width, x + r))]
                if patch.size == 0:
                    return False
                green = ((patch[..., 0] >= 30) & (patch[..., 0] <= 90)
                         & (patch[..., 1] >= 40))
                return green.mean() >= GRASS_SHARE

            labels = {}
            for s in set(shots):
                labels.update(ball_labels(
                    {i: c for i, c in candidates.items() if shots[i] == s}, width,
                    keep=on_grass))
            if not labels:
                stats["no_ball"] += 1
                continue
            for i in pick_frames(labels, frame_gap, per_window):
                name = f"{Path(video).stem}_{start + i:06d}"
                label_file = LOCAL / "labels" / f"{name}.txt"
                if label_file.exists() and name not in auto:
                    continue  # labelled (or corrected) by hand: keep it
                sx, sy, w, h, source = labels[i]
                x, y = warp_points([[sx, sy]], to_frame[i])[0]
                rows = [_yolo_row(ball, x - w / 2, y - h / 2, x + w / 2, y + h / 2,
                                  width, height)] + people[i]
                cv2.imwrite(str(LOCAL / "images" / f"{name}.jpg"), frames[i])
                label_file.write_text("\n".join(rows) + "\n")
                auto.add(name)
                stats["frames"] += 1
                stats[source] += 1
                saved_here += 1
            if (number + 1) % 10 == 0:
                print(f"{Path(video).name}: window {number + 1}/{len(starts)}, "
                      f"{saved_here} frames labelled")
        capture.release()

    AUTO_LIST.write_text("\n".join(sorted(auto)) + "\n")
    print(f"\n{stats['frames']} frames labelled automatically in {LOCAL}:")
    print(f"  ball seen clearly by the model:      {stats['detected']}")
    print(f"  faint ball confirmed by its motion:  {stats['faint']}")
    print(f"  missed ball placed from its motion:  {stats['filled']}")
    print(f"  ({stats['no_ball']} windows where the ball could not be followed were skipped)")
    return stats


def import_labels(export_dir):
    """Copy corrected labels from a labelling tool's YOLO export back to
    data/local_det/labels.

    Handles what Roboflow does to an export: renamed files
    ("match2_000300_jpg.rf.<hash>.txt"), train/valid/test folders, and
    classes in another order (its data.yaml says which).
    """
    import re

    import yaml

    export_dir = Path(export_dir)
    remap = None
    exported = next(export_dir.rglob("data.yaml"), None)
    if exported is not None:
        names = yaml.safe_load(exported.read_text(encoding="utf-8")).get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        unknown = [n for n in names if n not in CLASSES]
        if unknown:
            raise SystemExit(f"{exported} has classes {unknown}; expected {CLASSES}")
        remap = {i: CLASSES.index(n) for i, n in enumerate(names)}
        print(f"Class order in the export: {names}")
    else:
        print("No data.yaml in the export: assuming the order", CLASSES)

    known = {p.stem for p in (LOCAL / "images").glob("*.jpg")}
    imported, unknown_files = 0, []
    corrected = set()
    for label in export_dir.rglob("*.txt"):
        if label.name == "classes.txt":
            continue
        stem = re.sub(r"_(jpg|jpeg|png)(\.rf\.[0-9A-Za-z]+)?$", "", label.stem)
        stem = re.sub(r"\.rf\.[0-9A-Za-z]+$", "", stem)
        if stem not in known:
            unknown_files.append(label.name)
            continue
        lines = [line.split() for line in label.read_text().splitlines() if line.strip()]
        if remap:
            lines = [[str(remap[int(p[0])])] + p[1:] for p in lines]
        (LOCAL / "labels" / f"{stem}.txt").write_text("".join(" ".join(p) + "\n" for p in lines))
        corrected.add(stem)
        imported += 1
    auto = _auto_stems()
    if auto & corrected:  # now corrected by hand: auto-label must keep them
        AUTO_LIST.write_text("\n".join(sorted(auto - corrected)) + "\n")
    print(f"Imported {imported} corrected label files into {LOCAL / 'labels'}")
    if unknown_files:
        print(f"{len(unknown_files)} files did not match a frame from label-frames, "
              f"e.g. {unknown_files[:3]}")
    return imported


def prepare_local(validation_share=0.15, soccernet_share=0.5):
    """YOLO dataset from your corrected frames (plus some SoccerNet frames,
    so the model does not forget broadcast footage).

    Validation frames are the last part of each video, not random frames,
    so they are not near-copies of training frames.
    """
    import random

    images = sorted((LOCAL / "images").glob("*.jpg"))
    labelled = [i for i in images if (LOCAL / "labels" / f"{i.stem}.txt").exists()]
    if not labelled:
        raise SystemExit(f"No labelled frames in {LOCAL}. Run label-frames first.")
    by_video = {}
    for image in labelled:
        by_video.setdefault(image.stem.rsplit("_", 1)[0], []).append(image)

    for split in ("train", "val"):
        shutil.rmtree(LOCAL_DATASET / "images" / split, ignore_errors=True)
        shutil.rmtree(LOCAL_DATASET / "labels" / split, ignore_errors=True)
        (LOCAL_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (LOCAL_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)
    remap = None
    counts = {"train": 0, "val": 0}
    balls = {"train": 0, "val": 0}
    # With hand-labelled frames, validation is the last part of those only
    # (automatic labels are not a fair test); otherwise of all frames.
    hand = _hand_stems()
    for frames in by_video.values():
        frames = sorted(frames)
        checkable = [f for f in frames if f.stem in hand] if hand else frames
        cut = len(checkable) - max(1, int(round(len(checkable) * validation_share)))
        val = set(checkable[cut:]) if checkable else set()
        for image in frames:
            split = "val" if image in val else "train"
            label = LOCAL / "labels" / f"{image.stem}.txt"
            _link(image, LOCAL_DATASET / "images" / split / image.name)
            lines = [line.split() for line in label.read_text().splitlines() if line.strip()]
            if remap:
                lines = [[str(remap[int(p[0])])] + p[1:] for p in lines]
            people = LOCAL / "people" / f"{image.stem}.txt"
            if people.exists() and all(p[0] == "3" for p in lines):
                # Ball-only labelling: add the model's player boxes.
                lines += [line.split() for line in people.read_text().splitlines() if line.strip()]
            (LOCAL_DATASET / "labels" / split / label.name).write_text(
                "".join(" ".join(p) + "\n" for p in lines))
            counts[split] += 1
            balls[split] += sum(1 for p in lines if p[0] == "3")

    train = ["images/train"]
    soccernet = DATASET / "images" / "train"
    if soccernet_share > 0 and soccernet.exists():
        pool = sorted(soccernet.glob("*.jpg"))
        random.Random(0).shuffle(pool)
        extra = pool[:int(counts["train"] * soccernet_share / (1 - soccernet_share))]
        (LOCAL_DATASET / "images" / "soccernet").mkdir(parents=True, exist_ok=True)
        (LOCAL_DATASET / "labels" / "soccernet").mkdir(parents=True, exist_ok=True)
        for image in extra:
            _link(image, LOCAL_DATASET / "images" / "soccernet" / image.name)
            shutil.copy2(DATASET / "labels" / "train" / f"{image.stem}.txt",
                         LOCAL_DATASET / "labels" / "soccernet" / f"{image.stem}.txt")
        train.append("images/soccernet")
        print(f"Mixed in {len(extra)} SoccerNet frames")

    config = LOCAL_DATASET / "local.yaml"
    config.write_text(
        f"path: {LOCAL_DATASET.as_posix()}\ntrain: [{', '.join(train)}]\n"
        "val: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)), encoding="utf-8")
    for split in ("train", "val"):
        print(f"{split}: {counts[split]} of your frames, {balls[split]} ball boxes")
    if hand:
        print("Validation: your hand-labelled frames only")
    print("Dataset:", config)
    return config


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def auto_finetune(videos, model=None, limit=1500, every=6.0, rounds=1,
                  epochs=30, batch=-1, device=None, workers=4):
    """auto-label -> prepare-local -> train, with no hand work.

    Always fine-tunes the starting model (not the previous round's), so a
    second round only changes the labels: the better model finds more
    balls, and those become the next round's labels.
    """
    start = Path(model) if model else MODELS / "football_yolo26s.pt"
    if not start.exists():
        from detector_classes import default_model
        start = default_model()
    labeller = start
    for round_number in range(1, rounds + 1):
        print(f"\n=== Round {round_number}/{rounds}: labelling with {Path(labeller).name} ===")
        auto_label(videos, labeller, every=every, limit=limit, device=device)
        prepare_local()
        labeller = train(str(start), epochs, batch, device=device, workers=workers,
                         data="local")
    print("\nDone. The scripts now use", labeller, "(the newest models/football_*.pt).")
    print("Check it: run ball_tracking.py on your video and compare the "
          "'Ball in N/M frames' line with the old model's.")
    return labeller


def train(model="yolo26s", epochs=50, batch=-1, image_size=IMAGE_SIZE,
          device=None, workers=4, name=None, data="soccernet"):
    import torch
    from ultralytics import YOLO

    config = (LOCAL_DATASET / "local.yaml" if data == "local"
              else DATASET / "football.yaml")
    if not config.exists():
        step = "prepare-local" if data == "local" else "prepare"
        raise SystemExit(f"No dataset yet. Run: python src/train_detector.py {step}")
    weights = Path(model)
    if weights.suffix != ".pt":
        weights = project_cache.CACHE / "models" / f"{model}.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    name = name or (f"{weights.stem}_local" if data == "local"
                    else f"football_{weights.stem}")
    result = YOLO(str(weights)).train(
        data=str(config), epochs=epochs, imgsz=image_size, batch=batch,
        device=device, workers=workers, project=str(RUNS), name=name,
        exist_ok=True,
        # Broadcast football: no vertical flips, mild colour changes (kit
        # colours matter later), keep small objects: no strong zoom-out.
        fliplr=0.5, flipud=0.0, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
        scale=0.3, mosaic=1.0, close_mosaic=10, patience=15,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    MODELS.mkdir(exist_ok=True)
    target = MODELS / f"{name}.pt"
    shutil.copy2(best, target)
    print("\nFootball detector:", target)
    print(f"Use it with:  $env:YOLO_MODEL = \"{target}\"")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="SoccerNet-Tracking train clips")
    p = sub.add_parser("prepare", help="build the YOLO dataset")
    p.add_argument("--source", type=Path, help="folder of SNMOT-* clips")
    p.add_argument("--stride", type=int, default=FRAME_STRIDE)
    p.add_argument("--validation-clips", type=int, default=VALIDATION_CLIPS)
    lf = sub.add_parser("label-frames",
                        help="frames from your videos, pre-labelled, to correct")
    lf.add_argument("videos", nargs="+", type=Path)
    lf.add_argument("--every", type=float, default=2.0, help="seconds between frames")
    lf.add_argument("--limit", type=int, default=400, help="frames in all")
    lf.add_argument("--model", type=Path, help="default: the newest models/football_*.pt")
    lf.add_argument("--all-classes", action="store_true",
                    help="correct players too (default: only the ball)")
    il = sub.add_parser("import-labels",
                        help="corrected labels from a YOLO export (e.g. Roboflow)")
    il.add_argument("export_dir", type=Path, help="the unzipped export folder")
    al = sub.add_parser("auto-label",
                        help="label the ball in your videos automatically (no hand work)")
    al.add_argument("videos", nargs="+", type=Path)
    al.add_argument("--model", type=Path, help="default: the newest models/football_*.pt")
    al.add_argument("--limit", type=int, default=1500, help="frames in all")
    al.add_argument("--every", type=float, default=6.0,
                    help="seconds between the 3-second windows that are read")
    af = sub.add_parser("auto-finetune",
                        help="auto-label, build the dataset and fine-tune, in one go")
    af.add_argument("videos", nargs="+", type=Path)
    af.add_argument("--model", type=Path, help="model to start from "
                    "(default: models/football_yolo26s.pt, else the newest football model)")
    af.add_argument("--limit", type=int, default=1500, help="frames in all")
    af.add_argument("--every", type=float, default=6.0)
    af.add_argument("--rounds", type=int, default=1,
                    help="2: label again with the fine-tuned model and train once more")
    af.add_argument("--epochs", type=int, default=30)
    af.add_argument("--batch", type=int, default=-1)
    af.add_argument("--device", default=None)
    af.add_argument("--workers", type=int, default=4)
    ft = sub.add_parser("finetune",
                        help="prepare-local + train from models/football_yolo26s.pt (after click_balls.py)")
    ft.add_argument("--model", type=Path, help="model to start from")
    ft.add_argument("--epochs", type=int, default=40)
    ft.add_argument("--batch", type=int, default=-1)
    ft.add_argument("--device", default=None)
    ft.add_argument("--workers", type=int, default=4)
    pl = sub.add_parser("prepare-local", help="dataset from your corrected frames")
    pl.add_argument("--soccernet-share", type=float, default=0.5,
                    help="share of SoccerNet frames mixed into training (0 = none)")
    t = sub.add_parser("train", help="train the detector")
    t.add_argument("--data", choices=["soccernet", "local"], default="soccernet",
                   help="local: your corrected frames (prepare-local)")
    t.add_argument("--model", default="yolo26s",
                   help="yolo26n/s/m/l/x or a .pt file to fine-tune")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch", type=int, default=-1,
                   help="-1: as many as fit in GPU memory")
    t.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    t.add_argument("--device", default=None)
    t.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.command == "download":
        download()
    elif args.command == "prepare":
        prepare(args.source, args.stride, args.validation_clips)
    elif args.command == "label-frames":
        label_frames(args.videos, args.every, args.limit, args.model,
                     ball_only=not args.all_classes)
    elif args.command == "auto-label":
        auto_label(args.videos, args.model, every=args.every, limit=args.limit)
    elif args.command == "auto-finetune":
        auto_finetune(args.videos, args.model, args.limit, args.every, args.rounds,
                      args.epochs, args.batch, args.device, args.workers)
    elif args.command == "finetune":
        start = args.model or MODELS / "football_yolo26s.pt"
        prepare_local()
        train(str(start), args.epochs, args.batch, device=args.device,
              workers=args.workers, data="local")
    elif args.command == "import-labels":
        import_labels(args.export_dir)
    elif args.command == "prepare-local":
        prepare_local(soccernet_share=args.soccernet_share)
    else:
        train(args.model, args.epochs, args.batch, args.imgsz, args.device,
              args.workers, data=args.data)


if __name__ == "__main__":
    main()
