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
# Training
# ----------------------------------------------------------------------

def train(model="yolo26s", epochs=50, batch=-1, image_size=IMAGE_SIZE,
          device=None, workers=4, name=None):
    import torch
    from ultralytics import YOLO

    config = DATASET / "football.yaml"
    if not config.exists():
        raise SystemExit("No dataset yet. Run: python src/train_detector.py prepare")
    weights = Path(model)
    if weights.suffix != ".pt":
        weights = project_cache.CACHE / "models" / f"{model}.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    name = name or f"football_{weights.stem}"
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
    t = sub.add_parser("train", help="train the detector")
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
    else:
        train(args.model, args.epochs, args.batch, args.imgsz, args.device,
              args.workers)


if __name__ == "__main__":
    main()
