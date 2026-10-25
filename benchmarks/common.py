"""Shared paths and helpers for the benchmarks (everything stays inside the
project folder: models in .cache/, data in data/, results in
outputs/benchmarks/)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import project_cache  # noqa: E402,F401  keeps downloads/caches in <project>/.cache

CACHE = project_cache.CACHE
DATA = ROOT / "data"
SOCCERNET = DATA / "SoccerNet"
RESULTS = ROOT / "outputs" / "benchmarks"
REPOS = CACHE / "repos"
SN_SPOTTING = REPOS / "sn-spotting"
SN_SPOTTING_URL = "https://github.com/SoccerNet/sn-spotting.git"


def ensure_repo(url, path):
    """Shallow-clone a public repository once."""
    path = Path(path)
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {url} -> {path}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(path)],
                       check=True)
    return path


def ensure_sn_spotting():
    """Official SoccerNet spotting code, weights and published predictions."""
    return ensure_repo(SN_SPOTTING_URL, SN_SPOTTING)


def markdown_table(rows, columns):
    """rows: list of dicts. Returns a GitHub markdown table."""
    def cell(value):
        if isinstance(value, float):
            return f"{value:.2f}"
        return "" if value is None else str(value)
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)
