#!/usr/bin/env python3
"""
diag_common.py — shared foundation for the NXA Generation-1 forensic suite.
================================================================================
Directive (2026-10-03): "PRODUCTION RUN CANCELLATION + FORENSIC TRAINING
FAILURE DIAGNOSIS". Everything here runs against the REAL repo code paths
(no reimplementation): the pinned dataset loader, the FoundationModel,
the ported Gen-0 curriculum, the registry. All outputs are quarantined
under diagnosis_artifacts/ — preserved evidence is never touched.

Sections (each a runnable module in this package):
  4  dataset forensics            -> diag_dataset_forensics.py
  3  clean-CE control (matched)   -> diag_clean_control.py
  5  tiny-set overfit gate        -> diag_tiny_overfit.py  (STOP gate)
  6  input/glimpse sanity         -> diag_glimpse_sanity.py
  7  output sanity / collapse     -> diag_output_sanity.py
  8  gradient/update forensics    -> diag_gradient_forensics.py
  9  objective decomposition      -> diag_objective_decomp.py
  10 adversarial isolation probe  -> diag_adv_isolation.py
"""
from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ART = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ART, "out")
os.makedirs(OUT, exist_ok=True)

DATA_ROOT = os.path.join(REPO_ROOT, "data", "imagenet100")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 96

#: The pinned production fingerprint (scripts/prepare_imagenet100.py).
PINNED_FINGERPRINT_SHA = "0b06779f7b892183ad207ac9808b8a9fb122bda25ef9cebd666c1ee9f0c6db09"


def save_json(name: str, obj) -> str:
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  [out] {path}")
    return path


def out_path(name: str) -> str:
    return os.path.join(OUT, name)
