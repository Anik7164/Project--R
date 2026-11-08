"""
Which detector classes are people and which is the ball.

The project works with two kinds of models:
- COCO models (yolo26n.pt, ...): "person" and "sports ball";
- our football model (train_detector.py): "player", "goalkeeper",
  "referee" and "ball", which also says who is a goalkeeper or referee.
"""

PEOPLE_NAMES = {"person", "player", "goalkeeper", "referee"}
BALL_NAMES = {"sports ball", "ball"}


def _names(model_or_names):
    names = getattr(model_or_names, "names", model_or_names)
    return dict(enumerate(names)) if isinstance(names, (list, tuple)) else names


def people_classes(model_or_names):
    return [i for i, n in _names(model_or_names).items() if n in PEOPLE_NAMES]


def ball_class(model_or_names):
    for i, n in _names(model_or_names).items():
        if n in BALL_NAMES:
            return i
    raise ValueError("this model has no ball class")


def is_football_model(model_or_names):
    return "referee" in _names(model_or_names).values()


def default_model():
    """The detector every script uses unless --model is given:
    $YOLO_MODEL, else the newest trained football model in models/,
    else the COCO yolo26n.pt in the project folder."""
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if os.environ.get("YOLO_MODEL"):
        return Path(os.environ["YOLO_MODEL"])
    trained = sorted((root / "models").glob("football_*.pt"),
                     key=lambda p: p.stat().st_mtime)
    return trained[-1] if trained else root / "yolo26n.pt"


# Our football model learnt players from TV broadcasts, where they are
# small. On phone video filmed close from the touchline (match5: players
# 150-350 px tall) it can find nobody at all, while the COCO model does.
PROBE_FRAMES = 20
MIN_SHARE_OF_COCO = 0.4


def check_people_model(model, video, coco_path, image_size=960, device="cpu"):
    """Return (model to use for people, message). Compares the people the
    given model finds on PROBE_FRAMES frames of the video with the COCO
    model's, and falls back to COCO if it finds far fewer."""
    import cv2
    from pathlib import Path
    from ultralytics import YOLO

    if not is_football_model(model) or not Path(coco_path).exists():
        return model, None
    coco = YOLO(str(coco_path))
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ours = theirs = 0
    for k in range(PROBE_FRAMES):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / PROBE_FRAMES))
        ok, frame = capture.read()
        if not ok:
            continue
        for m, counter in ((model, "ours"), (coco, "theirs")):
            found = len(m.predict(frame, imgsz=image_size, conf=0.35, device=device,
                                  classes=people_classes(m), verbose=False)[0].boxes)
            if counter == "ours":
                ours += found
            else:
                theirs += found
    capture.release()
    if theirs >= PROBE_FRAMES and ours < MIN_SHARE_OF_COCO * theirs:
        return coco, (f"The football model found {ours} people on {PROBE_FRAMES} test "
                      f"frames, the COCO model {theirs}: using {Path(coco_path).name} "
                      "for the players of this video (give --model to choose yourself).")
    return model, None
