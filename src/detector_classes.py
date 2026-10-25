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
