"""
Install and download everything the benchmarks need. This is
src/setup_models.py with --benchmarks:

    python benchmarks/setup_models.py --install
"""

import subprocess
import sys
from pathlib import Path

SETUP = Path(__file__).resolve().parent.parent / "src" / "setup_models.py"

if __name__ == "__main__":
    sys.exit(subprocess.run([sys.executable, str(SETUP), "--benchmarks",
                             *sys.argv[1:]]).returncode)
