"""Central configuration for Cognitive Vision Lab v2.

All paths are resolved relative to the repository root so the lab works
regardless of the current working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAB_DIR = PROJECT_ROOT / "cognitive_vision_lab"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_TIER2_DIR = PROJECT_ROOT / "checkpoints_tier2"
REPORT_DIR = PROJECT_ROOT / "report"
TIER1_RESULTS = PROJECT_ROOT / "tier1" / "results"
PHASE5_SDT_RESULTS = PROJECT_ROOT / "phase5_sdt" / "results"
HUMAN_STUDY_DIR = PROJECT_ROOT / "phase3_human_study"

COMPARISON_TABLE = TIER1_RESULTS / "comparison_table.csv"
SDT_RESULTS = PHASE5_SDT_RESULTS / "sdt_results.csv"

# Legacy sweep artifacts (scripts/generate_full_sweep.py)
SWEEP_PATH = REPORT_DIR / "final_sweep_results_stl10.json"
PRIOR_SWEEP_PATH = REPORT_DIR / "prior_results.json"

# Kaggle isolation sweep (Run A / Run B vs TRADES Large baseline) — live source
# for the isolation profiles surfaced on the benchmark dashboards.
ISOLATION_SWEEP = REPORT_DIR / "isolation_sweep_results.json"
DRIFT_STATS = TIER1_RESULTS / "representation_drift_stats.csv"
HUMAN_RESPONSES = HUMAN_STUDY_DIR / "human_responses_reconstructed_wide.csv"
HUMAN_MANIFEST = HUMAN_STUDY_DIR / "manifest.csv"

CACHE_DIR = LAB_DIR / "cache"
ASSETS_DIR = LAB_DIR / "assets"
DATA_DIR = LAB_DIR / "data"
EXPERIMENTS_FILE = CACHE_DIR / "experiments.json"

# ── App metadata ──────────────────────────────────────────────────────────────
APP_TITLE = "Cognitive Vision Lab"
APP_SUBTITLE = "RHAN research workbench"
APP_VERSION = "2.0.0"
ORG_NAME = "Adversarial Cognitive Model Project"
GITHUB_URL = "https://github.com/FerrariKazu/Adversarial-Cognitive-Model"

# ── Runtime ───────────────────────────────────────────────────────────────────
def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


CUDA_AVAILABLE = _cuda_available()
DEFAULT_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

# ── STL-10 (the project's primary benchmark, 96×96) ──────────────────────────
STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)
STL10_SIZE = 96
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

# ── CIFAR-10 ──────────────────────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ── ImageNet (torchvision convention) ─────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_SIZE = 224

# ── Default experiment grid (Finding-17 matched convention) ───────────────────
DEFAULT_EPS_GRID = [0.0, 0.031, 0.062, 0.094]
DEFAULT_PGD_STEPS = 50
DEFAULT_SEED = 42

# ── Theme ─────────────────────────────────────────────────────────────────────
PRIMARY_COLOR = "#2563EB"
ACCENT_COLOR = "#0EA5E9"
SUCCESS_COLOR = "#16A34A"
WARNING_COLOR = "#D97706"
DANGER_COLOR = "#DC2626"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
