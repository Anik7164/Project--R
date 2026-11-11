"""
Keep every download and cache inside the project folder.

By default PyTorch, Ultralytics, EasyOCR, Keras, Hugging Face, matplotlib and
temporary files go to the user's home or temp folder (on Windows, the C:
drive). Importing this module first points all of them to
<project>/.cache instead, so nothing is written outside the project
drive. Variables the user already set are left alone.

    import project_cache  # noqa: F401  (before torch / ultralytics / easyocr)
"""

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"

_LOCATIONS = {
    "TORCH_HOME": "torch",
    "HF_HOME": "huggingface",
    "XDG_CACHE_HOME": "xdg",
    "YOLO_CONFIG_DIR": "ultralytics",
    "EASYOCR_MODULE_PATH": "easyocr",
    "KERAS_HOME": "keras",
    "MPLCONFIGDIR": "matplotlib",
    "PIP_CACHE_DIR": "pip",
    "TEMP": "tmp",
    "TMP": "tmp",
    "TMPDIR": "tmp",
}

for _variable, _folder in _LOCATIONS.items():
    _path = CACHE / _folder
    _path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_variable, str(_path))

tempfile.tempdir = os.environ["TEMP"]
