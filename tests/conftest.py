"""Pytest setup: make the repo root and phase1_training importable."""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "phase1_training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
