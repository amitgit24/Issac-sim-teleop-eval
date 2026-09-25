"""Pure-Python unit tests (numpy/scipy only, no Isaac Sim): run anywhere, including CI.

The Isaac Sim integration checks live in tests/check_*.py and need the simulator and a GPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scenes"))
