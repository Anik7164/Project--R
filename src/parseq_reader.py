"""
Shirt text recognition with PARSeq.

PARSeq (Bautista & Atienza, ECCV 2022) is a scene-text recogniser that
reads short, distorted, low-resolution text far better than EasyOCR's
recogniser. It is what the best published shirt-number readers build on
(e.g. Koshkina & Elder, CVPRW 2024).

PARSeq reads one line of text, so text regions are found first with
EasyOCR's CRAFT detector when EasyOCR is installed; otherwise the whole
torso crop and its upper and lower halves (name above, number below on
the back) are read.

The official code and weights (Apache-2.0) are downloaded on first use
into <project>/.cache: github.com/baudm/parseq and its v1.0.0 release
(parseq-bb5792a6.pt, ~100 MB). Only torch and timm are needed (not
PyTorch Lightning).

ParseqReader has the same readtext(image, allowlist=...) ->
[(box, text, confidence)] interface as easyocr.Reader, so
jersey_reading.py can use either.
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import subprocess
import sys

import cv2
import numpy as np
import torch

REPO_URL = "https://github.com/baudm/parseq.git"
REPO = project_cache.CACHE / "repos" / "parseq"
WEIGHTS = {
    "parseq": "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt",
    "parseq-tiny": "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt",
}
ARCHITECTURES = {
    "parseq": {"embed_dim": 384, "enc_num_heads": 6, "dec_num_heads": 12},
    "parseq-tiny": {"embed_dim": 192, "enc_num_heads": 3, "dec_num_heads": 6},
}
CHARSET = ("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
IMAGE_SIZE = (32, 128)  # height, width


def _strhub():
    if not (REPO / "strhub").exists():
        REPO.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)],
                       check=True)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from strhub.data.utils import Tokenizer
    from strhub.models.parseq.model import PARSeq
    return PARSeq, Tokenizer


def load_parseq(name="parseq", device="cpu"):
    PARSeq, Tokenizer = _strhub()
    tokenizer = Tokenizer(CHARSET)
    model = PARSeq(
        num_tokens=len(tokenizer), max_label_length=25, img_size=IMAGE_SIZE,
        patch_size=(4, 8), enc_mlp_ratio=4, enc_depth=12, dec_mlp_ratio=4,
        dec_depth=1, decode_ar=True, refine_iters=1, dropout=0.1,
        **ARCHITECTURES[name])
    state = torch.hub.load_state_dict_from_url(
        WEIGHTS[name], model_dir=str(project_cache.CACHE / "parseq"),
        map_location="cpu", check_hash=True)
    model.load_state_dict(state)
    return model.to(device).eval(), tokenizer


class ParseqReader:
    def __init__(self, name="parseq", gpu=False, detector=None):
        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = load_parseq(name, self.device)
        self.detector = detector
        if self.detector is None:
            try:
                import easyocr
                self.detector = easyocr.Reader(["en"], gpu=gpu, recognizer=False)
            except ImportError:
                self.detector = None
        self._mask = None
        self.model.head.register_forward_hook(self._apply_mask)

    # Only the allowed characters (and end of text) can be predicted,
    # also inside the step-by-step decoding.
    def _apply_mask(self, module, inputs, output):
        return output if self._mask is None else output + self._mask

    def _set_allowlist(self, allowlist):
        if not allowlist:
            self._mask = None
            return
        allowed = {self.tokenizer.eos_id}
        for char in set(allowlist) | set(allowlist.lower()):
            if char in self.tokenizer._stoi:
                allowed.add(self.tokenizer._stoi[char])
        mask = torch.full((len(self.tokenizer) - 2,), float("-inf"))
        mask[sorted(allowed)] = 0.0
        self._mask = mask.to(self.device)

    def regions(self, image):
        """Text boxes [(x1, y1, x2, y2)] in the image."""
        height, width = image.shape[:2]
        boxes = []
        if self.detector is not None:
            # Lower thresholds than EasyOCR's defaults: shirt text in a
            # broadcast is small and blurred. No region means no text.
            horizontal, free = self.detector.detect(
                image, text_threshold=0.5, low_text=0.3, mag_ratio=1.5)
            for x1, x2, y1, y2 in horizontal[0]:
                boxes.append((max(0, int(x1)), max(0, int(y1)),
                              min(width, int(x2)), min(height, int(y2))))
            for points in free[0]:
                points = np.array(points)
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                boxes.append((max(0, int(x1)), max(0, int(y1)),
                              min(width, int(x2)), min(height, int(y2))))
        elif not boxes:
            boxes = [(0, 0, width, height), (0, 0, width, height // 2),
                     (0, height // 3, width, height)]
        return [b for b in boxes if b[2] - b[0] > 2 and b[3] - b[1] > 2]

    def recognise(self, crops):
        """[(text, confidence)] for BGR crops of single text lines."""
        if not crops:
            return []
        batch = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, IMAGE_SIZE[::-1], interpolation=cv2.INTER_CUBIC)
            batch.append((rgb.astype(np.float32) / 255.0 - 0.5) / 0.5)
        images = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2)
        with torch.no_grad():
            logits = self.model(self.tokenizer, images.to(self.device))
        texts, probabilities = self.tokenizer.decode(logits.softmax(-1).cpu())
        return [(text, float(p.prod())) for text, p in zip(texts, probabilities)]

    def readtext(self, image, allowlist=None):
        boxes = self.regions(image)
        self._set_allowlist(allowlist)
        try:
            results = self.recognise([image[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes])
        finally:
            self._mask = None
        return [(box, text, confidence)
                for box, (text, confidence) in zip(boxes, results) if text]
