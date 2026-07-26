import sys
import os
import re
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "phase1_training"))

from cognitive_vision_lab.config import (
    MODEL_REGISTRY,
    RHAN_MODELS,
    CHECKPOINTS_DIR,
    CHECKPOINTS_TIER2_DIR,
    STL10_MEAN,
    STL10_STD,
    STL10_CLASSES,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_imagenet_labels():
    import json
    import urllib.request
    url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
    try:
        with urllib.request.urlopen(url) as f:
            data = json.load(f)
        return [data[str(k)][1] for k in range(1000)]
    except Exception:
        return [f"class_{i}" for i in range(1000)]


IMAGENET_LABELS = load_imagenet_labels()
_cache = {}


def get_transform(stl10: bool = False):
    if stl10:
        return T.Compose([
            T.Resize((96, 96)),
            T.ToTensor(),
            T.Normalize(STL10_MEAN, STL10_STD),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


ARCHITECTURE_MAP = {
    "rhan":              ("model_rhan", "RHAN"),
    "rhan_adaptive":     ("model_rhan_adaptive", "AdaptiveRHAN"),
    "rhan_aligned":      ("model_rhan_aligned", "RHANAligned"),
    "rhan_clip":         ("model_rhan_clip", "RHANWithCLIP"),
    "rhan_dualstream":   ("model_rhan_dualstream", "RHAN_DualStream"),
    "rhan_full":         ("model_rhan_full", "RHAN_Full"),
    "rhan_predcoding":   ("model_rhan_predcoding", "RHAN_PredCoding"),
    "rhan_predictive":   ("model_rhan_predictive", "RHANPredictive"),
    "rhan_split":        ("model_rhan_split", "RHANSplit"),
    "rhan_stl10":        ("model_rhan_stl10", "RHANSTL10"),
    "rhan_stl10_large":  ("model_rhan_stl10_large", "RHANLargeSTL10"),
    "rhan_stl10_pretrained": ("model_rhan_stl10_pretrained", "RHANUnifiedSTL10"),
    "rhan_unified":      ("model_rhan_unified", "RHANUnified"),
    "rhan_v3_adaptive":  ("model_rhan_v3_adaptive", "AdaptiveRHANSplit"),
    "rhan_v4":           ("model_rhan_v4", "RHANv4"),
    "rhan_v5":           ("model_rhan_v5", "RHANv5"),
    "rhan_v6":           ("model_rhan_v6", "RHANv6"),
    "rhan_v7":           ("model_rhan_v7", "RHANv7"),
    "rhan_v9":           ("model_rhan_v9", "RHANv9"),
    "rhan_v10":          ("model_rhan_v10", "RHANv10"),
    "rhan_v11":          ("model_rhan_v11", "RHANv11"),
}

ARCHITECTURE_LOOKUP = {}
for prefix_key, (mod_name, cls_name) in ARCHITECTURE_MAP.items():
    ARCHITECTURE_LOOKUP[prefix_key] = (mod_name, cls_name)

CKPT_SUFFIXES = [
    "_best", "_final", "_rolling", "_checkpoint",
    "_ep10", "_ep45", "_start",
    "_clip_init", "_decoder_warm",
    "_phase0", "_phase1_ep10", "_phase1_final",
    "_phase2_ep10", "_phase2_final",
    "_phase3_ep10", "_phase3_final",
    "_phase4_ep10", "_phase4_final",
    "_phase5_ep10", "_phase5_final",
    "_phase_a_final", "_phase_b_final", "_phase_c_final",
    "_labeled", "_labeled_rolling",
    "_pretrained", "_pretrained_rolling",
    "_trades", "_trades_actual", "_trades_clean_consistency", "_trades_rolling",
]


def _stem(filename: str) -> str:
    name = filename.replace(".pth", "").replace(":Zone.Identifier", "")
    for suffix in CKPT_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _resolve_architecture(filename: str):
    stem_name = _stem(filename)
    if stem_name in ARCHITECTURE_LOOKUP:
        return ARCHITECTURE_LOOKUP[stem_name]
    if stem_name.startswith("rhan_stl10_"):
        for prefix in ["rhan_stl10_large", "rhan_stl10_pretrained", "rhan_stl10"]:
            if prefix in ARCHITECTURE_LOOKUP and stem_name.startswith(prefix):
                return ARCHITECTURE_LOOKUP[prefix]
    if stem_name.startswith("rhan_v"):
        for v in ["v11", "v10", "v9", "v7", "v6", "v5", "v4", "v3_adaptive"]:
            key = "rhan_" + v
            if key in ARCHITECTURE_LOOKUP and stem_name.startswith(key):
                return ARCHITECTURE_LOOKUP[key]
    if stem_name.startswith("rhan_unified"):
        return ARCHITECTURE_LOOKUP["rhan_unified"]
    if stem_name.startswith("rhan_stl_pretrained"):
        return ARCHITECTURE_LOOKUP["rhan_stl10_pretrained"]
    if stem_name.startswith("rhan_stl10"):
        return ARCHITECTURE_LOOKUP["rhan_stl10"]
    if stem_name.startswith("rhan_"):
        return ARCHITECTURE_LOOKUP["rhan"]
    return None


def discover_checkpoints():
    discovered = {}
    dirs = [CHECKPOINTS_DIR]
    if CHECKPOINTS_TIER2_DIR.exists():
        dirs.append(CHECKPOINTS_TIER2_DIR)
    for ckpt_dir in dirs:
        for fpath in ckpt_dir.iterdir():
            if not fpath.name.endswith(".pth"):
                continue
            if ":Zone.Identifier" in fpath.name:
                continue
            arch_info = _resolve_architecture(fpath.name)
            if arch_info is None:
                continue
            display = _stem(fpath.name).replace("_", " ").title().replace("Stl", "STL").replace("Rhan", "RHAN")
            if fpath.name in discovered:
                continue
            discovered[fpath.name] = {
                "checkpoint": fpath.name,
                "architecture": arch_info,
                "stl10": True,
                "display": display,
                "path": fpath,
            }
    return discovered


DESCOVERED_CACHE = None


def get_all_checkpoint_models():
    global DESCOVERED_CACHE
    if DESCOVERED_CACHE is None:
        DESCOVERED_CACHE = discover_checkpoints()
    return DESCOVERED_CACHE


def load_model(
    model_id: str,
    use_cpu: bool = False,
    force_reload: bool = False,
):
    if model_id in _cache and not force_reload:
        return _cache[model_id]

    target_device = torch.device("cpu") if use_cpu else device

    if model_id in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[model_id]
        try:
            import importlib
            module_path = entry["module"]
            parts = module_path.split(".")
            mod = importlib.import_module(".".join(parts[:-1]))
            cls = getattr(mod, parts[-1])
            kwargs = entry.get("kwargs", {}).copy()
            try:
                model = cls(**kwargs)
            except Exception:
                kwargs.pop("weights", None)
                model = cls(**kwargs)
        except Exception:
            from torchvision.models import resnet18
            model = resnet18(weights=None)
        model.eval()
        model = model.to(target_device)
        tfm = get_transform(False)
        _cache[model_id] = (model, tfm, False)
        return _cache[model_id]

    if model_id in RHAN_MODELS:
        entry = RHAN_MODELS[model_id]
        ckpt_path = CHECKPOINTS_DIR / entry["checkpoint"]
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint {ckpt_path} not found. "
                "Download from HuggingFace or train first."
            )
        mod_name, cls_name = entry["architecture"]
        arch_module = __import__(mod_name, fromlist=[cls_name])
        model_class = getattr(arch_module, cls_name)
        if mod_name == "model_rhan_v11":
            model = model_class(max_foraging_steps=4, fovea_size=48, metabolic_cost=0.05)
        else:
            model = model_class()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ["model_state_dict", "model", "state_dict"]:
                if key in state:
                    state = state[key]
                    break
        model.load_state_dict(state, strict=False)
        model.eval()
        model = model.to(target_device)
        tfm = get_transform(True)
        _cache[model_id] = (model, tfm, True)
        return _cache[model_id]

    discovered = get_all_checkpoint_models()
    if model_id in discovered:
        entry = discovered[model_id]
        ckpt_path = entry["path"]
        mod_name, cls_name = entry["architecture"]
        arch_module = __import__(mod_name, fromlist=[cls_name])
        model_class = getattr(arch_module, cls_name)
        if mod_name == "model_rhan_v11":
            model = model_class(max_foraging_steps=4, fovea_size=48, metabolic_cost=0.05)
        else:
            model = model_class()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ["model_state_dict", "model", "state_dict"]:
                if key in state:
                    state = state[key]
                    break
        model.load_state_dict(state, strict=False)
        model.eval()
        model = model.to(target_device)
        tfm = get_transform(True)
        _cache[model_id] = (model, tfm, True)
        return _cache[model_id]

    raise ValueError(f"Unknown model_id: {model_id}")


def list_available_models():
    models = []
    for mid, entry in MODEL_REGISTRY.items():
        models.append({
            "id": mid,
            "description": entry["description"],
            "checkpoint": False,
            "stl10": False,
        })
    for mid, entry in RHAN_MODELS.items():
        ckpt = CHECKPOINTS_DIR / entry["checkpoint"]
        models.append({
            "id": mid,
            "description": f"RHAN ({entry['architecture'][0]})",
            "checkpoint": ckpt.exists(),
            "stl10": True,
        })
    discovered = get_all_checkpoint_models()
    for ckpt_name, entry in discovered.items():
        if ckpt_name in {e["checkpoint"] for e in RHAN_MODELS.values()}:
            continue
        models.append({
            "id": ckpt_name,
            "description": f"{entry['display']} ({entry['architecture'][0]})",
            "checkpoint": True,
            "stl10": True,
        })
    return models


def predict(model, image_tensor, stl10: bool = False):
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            output = model(image_tensor.unsqueeze(0))
            if isinstance(output, (tuple, list)):
                output = output[0]
            probs = torch.softmax(output, dim=1).squeeze(0)
    if stl10:
        labels = STL10_CLASSES
    else:
        labels = IMAGENET_LABELS
    pred_idx = probs.argmax().item()
    return {
        "predicted_class": labels[pred_idx] if pred_idx < len(labels) else str(pred_idx),
        "predicted_idx": pred_idx,
        "confidence": probs[pred_idx].item(),
        "all_probs": probs.tolist(),
    }
