"""Pytest setup: make the repo root, phase1_training, and phase2_attacks importable."""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "phase1_training"),
           os.path.join(_ROOT, "phase2_attacks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
