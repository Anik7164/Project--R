"""
Deep appearance (ReID) embeddings for the stable-ID layer.

Colour histograms cannot tell two teammates apart. A person
re-identification network looks at body shape, hair, skin, boots and
socks too. We use LMBN (Lightweight Multi-Branch Network, trained on
Market-1501) through BoxMOT; its weights (~37 MB, from BoxMOT's GitHub
releases) are downloaded once into .cache/boxmot. Any other BoxMOT ReID
weights file works too (e.g. osnet_x1_0_msmt17.pt, or a model fine-tuned
on SoccerNet-ReID).

Needs boxmot (python src/setup_models.py --install). Without it the
identity layer simply works on colour alone.
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache

import numpy as np

DEFAULT_WEIGHTS = project_cache.CACHE / "boxmot" / "lmbn_n_market.pt"


class ReidEmbedder:
    def __init__(self, weights=DEFAULT_WEIGHTS, device="cpu"):
        from boxmot.reid.core import ReID

        weights.parent.mkdir(parents=True, exist_ok=True)
        half = str(device) not in ("cpu", "")
        self.model = ReID(weights=weights, device=str(device), half=half).model

    def __call__(self, frame, boxes):
        """One L2-normalised embedding per (x1, y1, x2, y2) box."""
        if not boxes:
            return []
        features = np.asarray(self.model.get_features(
            np.asarray(boxes, dtype=np.float64), frame), dtype=np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return list(features / np.maximum(norms, 1e-9))


def load_default(device="cpu"):
    """A ReidEmbedder, or None (with a note) if boxmot is missing."""
    try:
        return ReidEmbedder(device=device)
    except ImportError:
        print("ReID off: boxmot is not installed "
              "(python src/setup_models.py --install). Using colour only.")
    except Exception as error:  # e.g. no internet for the first download
        print(f"ReID off ({type(error).__name__}: {error}). Using colour only.")
    return None
