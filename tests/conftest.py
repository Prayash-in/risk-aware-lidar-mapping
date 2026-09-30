"""Make the package importable from a source checkout (no install needed)."""

from __future__ import annotations

import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)          # repository root
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Tests, examples and benchmarks load configs relative to the repository root.
os.chdir(ROOT)
