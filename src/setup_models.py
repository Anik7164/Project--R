"""
Install the extra packages and download every model the project uses,
all inside the project folder (.cache/ and models/), never on C:.

    python src/setup_models.py --install      # packages, then models
    python src/setup_models.py                # models only
    python src/setup_models.py --benchmarks   # also what benchmarks/ needs

Models (~1.2 GB; ~2 GB with --benchmarks):
  YOLO26 n, s, m (COCO)       start points for train_detector.py   .cache/models
  PARSeq                      shirt names and numbers              .cache/parseq
  EasyOCR (CRAFT detector)    finds the text on the shirt          .cache/easyocr
  LMBN ReID (Market-1501)     appearance features for the ID layer .cache/boxmot
  ResNet-152 (ImageNet)       features for event spotting          .cache/keras
  NetVLAD++ (SoccerNet)       event spotting                       .cache/soccernet
  MMS-TTS Bengali (Meta)      Bengali commentary voice             .cache/huggingface
--benchmarks adds YOLO26 l/x, YOLO11x, RT-DETR l/x, OSNet ReID and the
SoccerNet sn-spotting repository (CALF and the official baselines).

The football detector itself is trained, not downloaded:
    python src/train_detector.py download / prepare / train
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
import subprocess
import sys

ROOT = project_cache.ROOT
CACHE = project_cache.CACHE
MODELS = CACHE / "models"
REID_WEIGHTS = CACHE / "boxmot" / "lmbn_n_market.pt"
NO_DEPS = ["easyocr", "boxmot==25.0.0"]


def install(benchmarks=False):
    # project_cache.py points PIP_CACHE_DIR and TEMP into .cache; pip
    # inherits that.
    pip = [sys.executable, "-m", "pip", "install"]
    subprocess.run(pip + ["-r", str(ROOT / "requirements-models.txt")], check=True)
    subprocess.run(pip + ["--no-deps"] + NO_DEPS, check=True)
    if benchmarks:
        subprocess.run(pip + ["-r", str(ROOT / "requirements-benchmarks.txt")],
                       check=True)


def step(name, action):
    print(f"\n== {name}")
    try:
        action()
        print("   ok")
        return True
    except Exception as error:  # keep going, report at the end
        print(f"   FAILED: {type(error).__name__}: {error}")
        return False


def detectors(names):
    from ultralytics import RTDETR, YOLO

    MODELS.mkdir(parents=True, exist_ok=True)
    for name in names:
        print("  ", name)
        path = MODELS / f"{name}.pt"
        (RTDETR if name.startswith("rtdetr") else YOLO)(str(path))


def parseq():
    from parseq_reader import load_parseq
    load_parseq()


def easyocr():
    import easyocr as ocr
    ocr.Reader(["en"], gpu=False, verbose=False)


def reid(weights=REID_WEIGHTS):
    import numpy as np
    from boxmot.reid.core import ReID

    weights.parent.mkdir(parents=True, exist_ok=True)
    model = ReID(weights=weights, device="cpu", half=False).model
    model.get_features(np.array([[10, 10, 50, 100]], float),
                       np.zeros((200, 200, 3), np.uint8))


def resnet():
    from tensorflow import keras
    keras.applications.resnet.ResNet152(include_top=True, weights="imagenet")


def netvlad():
    import action_spotting
    action_spotting.load_model()


def mms_tts():
    from transformers import AutoTokenizer, VitsModel
    from speak_commentary import MMS_MODEL
    AutoTokenizer.from_pretrained(MMS_MODEL)
    VitsModel.from_pretrained(MMS_MODEL)


def sn_spotting():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from common import ensure_sn_spotting
    ensure_sn_spotting()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--install", action="store_true",
                        help="pip install the extra packages first")
    parser.add_argument("--benchmarks", action="store_true",
                        help="also the packages and models benchmarks/ uses")
    args = parser.parse_args()
    if args.install:
        install(args.benchmarks)

    steps = [
        ("YOLO26 n/s/m", lambda: detectors(["yolo26n", "yolo26s", "yolo26m"])),
        ("PARSeq shirt text", parseq),
        ("EasyOCR text detector", easyocr),
        ("LMBN ReID", reid),
        ("ResNet-152", resnet),
        ("NetVLAD++", netvlad),
        ("MMS-TTS Bengali voice", mms_tts),
    ]
    if args.benchmarks:
        steps += [
            ("Benchmark detectors", lambda: detectors(
                ["yolo26l", "yolo26x", "yolo11x", "rtdetr-l", "rtdetr-x"])),
            ("OSNet ReID", lambda: reid(CACHE / "boxmot" / "osnet_x0_25_msmt17.pt")),
            ("SoccerNet sn-spotting", sn_spotting),
        ]
    failed = [name for name, action in steps if not step(name, action)]
    print("\nAll models ready." if not failed else f"\nFailed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
