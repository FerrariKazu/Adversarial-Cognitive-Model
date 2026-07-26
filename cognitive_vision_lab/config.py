import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_TIER2_DIR = PROJECT_ROOT / "checkpoints_tier2"
REPORT_DIR = PROJECT_ROOT / "report"
FIGURES_DIR = PROJECT_ROOT / "figures_v3"
DATA_DIR = PROJECT_ROOT / "cognitive_vision_lab" / "data"

SWEEP_PATH = REPORT_DIR / "final_sweep_results_stl10.json"
PRIOR_SWEEP_PATH = REPORT_DIR / "prior_results.json"

# Backend
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

# Database (optional — falls back to JSON if not configured)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    os.getenv("POSTGRES_URL", ""),
)

# Available models (display_name -> (module_path, class_name, checkpoint))
MODEL_REGISTRY = {
    "ResNet-18": {
        "module": "torchvision.models.resnet18",
        "kwargs": {"weights": "DEFAULT"},
        "checkpoint": None,
        "stl10": False,
        "description": "Standard ResNet-18 — 11.2M params. Served as the classic CNN baseline.",
    },
    "EfficientNet-B0": {
        "module": "torchvision.models.efficientnet_b0",
        "kwargs": {"weights": "DEFAULT"},
        "checkpoint": None,
        "stl10": False,
        "description": "EfficientNet-B0 — 5.3M params. Highest clean accuracy on ImageNet among efficient CNNs.",
    },
    "ViT-B-16": {
        "module": "torchvision.models.vit_b_16",
        "kwargs": {"weights": "DEFAULT"},
        "checkpoint": None,
        "stl10": False,
        "description": "Vision Transformer — 85.8M params. Pure attention-based architecture.",
    },
    "Swin-Tiny": {
        "module": "torchvision.models.swin_t",
        "kwargs": {"weights": "DEFAULT"},
        "checkpoint": None,
        "stl10": False,
        "description": "Swin Transformer Tiny — 28.3M params. Hierarchical shifted-window attention.",
    },
}

RHAN_MODELS = {
    "RHAN-Large (ep45)": {
        "checkpoint": "rhan_stl10_large_ep45_best.pth",
        "architecture": "rhan_stl10_large",
        "stl10": True,
    },
    "RHAN-v10 (Final)": {
        "checkpoint": "rhan_v10_final.pth",
        "architecture": "rhan_v10",
        "stl10": True,
    },
    "RHAN-v11 (Best)": {
        "checkpoint": "rhan_stl10_v11_best.pth",
        "architecture": "rhan_v11",
        "stl10": True,
    },
}

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
