"""
Install the benchmark packages and download every model the project and
the benchmarks use, all inside the project folder (.cache/), so nothing
lands on the C: drive.

    python benchmarks/setup_models.py --install    # pip packages, then models
    python benchmarks/setup_models.py              # models only

Models (about 1.5 GB in total):
  YOLO26 n/s/m/l/x, YOLO11 x, RT-DETR l/x   person detectors      .cache/models
  OSNet x0.25 (MSMT17)                      ReID for BoxMOT       .cache/boxmot
  ResNet-152 (ImageNet, Keras)              spotting features     .cache/keras
  NetVLAD++                                 our spotting model    .cache/soccernet
  sn-spotting (CALF weights, PCA, official  baselines             .cache/repos
  MaxPool/NetVLAD test predictions)
  EasyOCR English                           shirt numbers         .cache/easyocr
"""

import argparse
import subprocess
import sys

from common import ROOT, ensure_sn_spotting

import action_spotting
import tracking_benchmark as tb

BOXMOT = "boxmot==25.0.0"


def install():
    # project_cache.py has pointed PIP_CACHE_DIR and TEMP into .cache, and
    # the child process inherits that.
    pip = [sys.executable, "-m", "pip", "install"]
    subprocess.run(pip + ["-r", str(ROOT / "requirements-benchmarks.txt")],
                   check=True)
    # Without its dependencies: boxmot asks for opencv-python<5 and would
    # replace the OpenCV 5 and CUDA torch from requirements.txt.
    subprocess.run(pip + ["--no-deps", BOXMOT], check=True)


def step(name, action):
    print(f"\n== {name}")
    try:
        action()
        print("   ok")
        return True
    except Exception as error:  # keep going, report at the end
        print(f"   FAILED: {error}")
        return False


def detectors():
    for name in tb.DETECTORS:
        if name != "det":
            print("  ", name)
            tb.load_detector(name)


def reid():
    import boxmot
    import numpy as np
    tracker = boxmot.StrongSort(reid_weights=tb.REID_WEIGHTS, device="cpu",
                                half=False)
    # The weights are fetched on the first update.
    tracker.update(np.array([[10, 10, 50, 100, 0.9, 0]], np.float32),
                   np.zeros((200, 200, 3), np.uint8))
    if not tb.REID_WEIGHTS.exists():
        raise RuntimeError(f"{tb.REID_WEIGHTS} was not downloaded")


def resnet():
    from tensorflow import keras
    keras.applications.resnet.ResNet152(include_top=True, weights="imagenet")


def easyocr():
    try:
        import easyocr as ocr
    except ImportError:
        print("   skipped: easyocr is not installed (only jersey_reading.py "
              "needs it)")
        return
    ocr.Reader(["en"], gpu=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--install", action="store_true",
                        help="pip install the benchmark packages first")
    args = parser.parse_args()
    if args.install:
        install()

    results = {
        "detectors": step("Detectors (YOLO26, YOLO11, RT-DETR)", detectors),
        "reid": step("OSNet ReID weights", reid),
        "resnet": step("ResNet-152 ImageNet weights", resnet),
        "netvlad++": step("NetVLAD++ weights", action_spotting.load_model),
        "sn-spotting": step("SoccerNet sn-spotting (CALF, baselines)",
                            ensure_sn_spotting),
        "easyocr": step("EasyOCR English model", easyocr),
    }
    failed = [name for name, ok in results.items() if not ok]
    print("\nAll models ready." if not failed else f"\nFailed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
