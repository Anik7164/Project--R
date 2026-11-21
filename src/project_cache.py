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


def _writable(folder):
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write-test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


# Installed where this user cannot write (Program Files): the user's own
# app-data folder instead, so the app still starts.
if not _writable(CACHE):
    CACHE = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Project R" / "cache"

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


def _no_console_windows():
    """Windows: when this program has no console of its own on screen (the
    desktop app, the setup's shortcuts), the tools it starts (ffmpeg for the
    microphone, camera listing) would each open a black console window on
    top of the app. Start them without one. From a terminal nothing changes."""
    import subprocess
    import sys

    if sys.platform != "win32":
        return
    try:
        import ctypes

        window = ctypes.windll.kernel32.GetConsoleWindow()
        if window and ctypes.windll.user32.IsWindowVisible(window):
            return
    except Exception:
        pass
    hidden = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    original = subprocess.Popen.__init__

    def __init__(self, *args, **kwargs):
        if not kwargs.get("creationflags"):
            kwargs["creationflags"] = hidden
        original(self, *args, **kwargs)

    subprocess.Popen.__init__ = __init__


_no_console_windows()
