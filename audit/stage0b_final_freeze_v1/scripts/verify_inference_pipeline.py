#!/usr/bin/env python3
"""Deterministic Stage 0B inference-pipeline verification.

This script verifies the raw-image-to-logit path on a fixed, small sample from
the calibration split.  It does not load evaluation images, fit a temperature,
compute calibration metrics, or inspect Stage 0B outcomes.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
FREEZE = ROOT / "audit" / "stage0b_final_freeze_v1"
DATA_GATE = ROOT / "audit" / "stage0b_data_gate_v1"
STAGE0A = ROOT / "audit" / "stage0a_closeout_v1"
DATASET_ROOT = DATA_GATE / "work" / "extracted" / "imagenetv2-matched-frequency-format-val"
PARTITION = DATA_GATE / "PARTITION_MANIFEST.json"
CONFIG = FREEZE / "STAGE0B_INFERENCE_CONFIG.json"
BACKBONE_FILE = FREEZE / "work" / "backbone_cache" / "vit_base_patch14_dinov2.lvd142m" / "model.safetensors"
SAMPLE_CLASSES = 8
OUT_JSON = FREEZE / "work" / "inference_pipeline_test.json"

sys.path.insert(0, str(ROOT))
from stage0a.data import make_transform  # noqa: E402
from stage0a.utils import state_dict_sha256  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return sha256_bytes(array.tobytes())


class LinearHead(torch.nn.Module):
    """The Stage 0A head contract: one linear layer named fc."""

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = torch.nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(PARTITION.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise SystemExit("partition manifest is not PASS")
    if manifest["verification"]["manifest_sha256"] != config["data_binding"]["partition_canonical_sha256"]:
        raise SystemExit("partition canonical hash does not match inference config")

    assignments = sorted(
        (row for row in manifest["assignments"] if row["split"] == "calibration"),
        key=lambda row: (row["class_index"], row["within_class_rank"]),
    )
    chosen = []
    seen = set()
    for row in assignments:
        if row["class_index"] in seen:
            continue
        seen.add(row["class_index"])
        chosen.append(row)
        if len(chosen) == SAMPLE_CLASSES:
            break
    if len(chosen) != SAMPLE_CLASSES:
        raise SystemExit(f"could not select {SAMPLE_CLASSES} sample images")

    # Independent restatement of the documented Stage 0A transform.
    import torchvision.transforms as T

    mean = tuple(config["preprocessing"]["normalization_mean"])
    std = tuple(config["preprocessing"]["normalization_std"])
    documented_transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    stage0a_transform = make_transform(224)

    image_rows = []
    tensors = []
    documented_tensors = []
    for row in chosen:
        path = DATASET_ROOT / row["relative_path"]
        if not path.is_file():
            raise SystemExit(f"missing sample image: {path}")
        content_sha = sha256_file(path)
        if content_sha != row["content_sha256"]:
            raise SystemExit(f"image content hash mismatch: {row['relative_path']}")
        with Image.open(path) as image:
            original_format = image.format
            original_mode = image.mode
            original_size = list(image.size)
            rgb = image.convert("RGB")
            tensor = documented_transform(rgb)
            stage0a_tensor = stage0a_transform(rgb.copy())
        if tensor.shape != (3, 224, 224):
            raise SystemExit(f"unexpected tensor shape: {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            raise SystemExit(f"unexpected tensor dtype: {tensor.dtype}")
        if not torch.equal(tensor, stage0a_tensor):
            max_diff = float((tensor - stage0a_tensor).abs().max())
            raise SystemExit(f"documented transform differs from Stage 0A transform by {max_diff}")
        tensors.append(tensor)
        documented_tensors.append(tensor)
        image_rows.append(
            {
                "class_index": row["class_index"],
                "wnid": row["wnid"],
                "relative_path": row["relative_path"],
                "content_sha256": content_sha,
                "original_format": original_format,
                "original_mode": original_mode,
                "original_size": {"width": original_size[0], "height": original_size[1]},
                "transformed_shape": list(tensor.shape),
                "transformed_dtype": str(tensor.dtype),
                "transformed_tensor_sha256": tensor_sha256(tensor),
                "transform_matches_stage0a_source": True,
            }
        )

    batch = torch.stack(tensors, dim=0)
    if len(documented_tensors) != len(tensors):
        raise SystemExit("internal tensor accounting error")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required by the frozen inference configuration")
    device = torch.device("cuda:0")
    torch.manual_seed(int(config["runtime"]["manual_seed"]))
    torch.backends.cudnn.benchmark = bool(config["runtime"]["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(config["runtime"]["cudnn_deterministic"])

    import timm

    backbone_path = BACKBONE_FILE
    if not backbone_path.is_file():
        raise SystemExit(f"missing backbone weights: {backbone_path}")
    backbone_sha = sha256_file(backbone_path)
    if backbone_sha != config["backbone"]["weights_file_sha256"]:
        raise SystemExit("backbone weights file hash mismatch")

    model = timm.create_model(
        config["backbone"]["timm_name"],
        pretrained=True,
        pretrained_cfg_overlay={"file": str(backbone_path)},
        num_classes=0,
        img_size=int(config["preprocessing"]["input_resolution"]),
        dynamic_img_size=True,
    )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    actual_state_sha = state_dict_sha256(model.state_dict())
    if actual_state_sha != config["backbone"]["state_dict_sha256"]:
        raise SystemExit(
            f"backbone state-dict hash mismatch: {actual_state_sha} != "
            f"{config['backbone']['state_dict_sha256']}"
        )
    architecture = {
        "patch_size": list(model.patch_embed.patch_size),
        "embed_dim": int(model.embed_dim),
        "depth": int(len(model.blocks)),
        "num_heads": int(model.blocks[0].attn.num_heads),
        "mlp_ratio": float(model.blocks[0].mlp.fc1.out_features / model.blocks[0].mlp.fc1.in_features),
        "global_pool": str(model.global_pool),
        "num_classes": int(model.num_classes),
        "pos_embed_shape": list(model.pos_embed.shape),
    }
    if architecture["embed_dim"] != 768 or architecture["global_pool"] != "token":
        raise SystemExit(f"unexpected backbone architecture: {architecture}")
    if architecture["pos_embed_shape"] != config["backbone"]["architecture"]["positional_embedding_runtime_shape"]:
        raise SystemExit(f"unexpected positional embedding shape: {architecture['pos_embed_shape']}")

    with torch.inference_mode():
        x = batch.to(device, non_blocking=False)
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            features = model(x)
            direct = model.forward_head(model.forward_features(x), pre_logits=True)
        if features.shape != (SAMPLE_CLASSES, 768):
            raise SystemExit(f"unexpected feature shape: {tuple(features.shape)}")
        if not torch.equal(features, direct):
            max_diff = float((features.float() - direct.float()).abs().max())
            raise SystemExit(f"forward_features/forward_head path differs by {max_diff}")
        features_f16 = features.float().cpu().numpy().astype(np.float16)
        features_f32 = features_f16.astype(np.float32)

    head_rows = []
    sample_logits = {}
    for spec in config["classification_heads"]["seeds"]:
        head_path = ROOT / spec["path"]
        head_sha = sha256_file(head_path)
        sidecar_path = head_path.with_suffix(head_path.suffix + ".sha256")
        sidecar_sha = sidecar_path.read_text(encoding="utf-8").strip().split()[0]
        if head_sha != spec["sha256"] or sidecar_sha != spec["sha256"]:
            raise SystemExit(f"head hash mismatch for seed {spec['seed']}")
        payload = torch.load(head_path, map_location="cpu", weights_only=False)
        state = payload.get("model")
        if not isinstance(state, dict) or set(state) != {"fc.weight", "fc.bias"}:
            raise SystemExit(f"unexpected head payload keys for seed {spec['seed']}: {state.keys() if isinstance(state, dict) else type(state)}")
        if list(state["fc.weight"].shape) != [1000, 768] or list(state["fc.bias"].shape) != [1000]:
            raise SystemExit(f"unexpected head tensor shapes for seed {spec['seed']}")
        if state["fc.weight"].dtype != torch.float32 or state["fc.bias"].dtype != torch.float32:
            raise SystemExit(f"unexpected head dtype for seed {spec['seed']}")
        head = LinearHead(768, 1000)
        head.load_state_dict(state)
        head.eval()
        with torch.inference_mode():
            logits = head(torch.from_numpy(features_f32))
        if logits.shape != (SAMPLE_CLASSES, 1000) or logits.dtype != torch.float32:
            raise SystemExit(f"unexpected logits shape/dtype for seed {spec['seed']}")
        if not torch.isfinite(logits).all():
            raise SystemExit(f"non-finite logits for seed {spec['seed']}")
        sample_logits[str(spec["seed"])] = logits.numpy().copy()
        head_rows.append(
            {
                "seed": spec["seed"],
                "path": spec["path"],
                "sha256": head_sha,
                "sidecar_sha256": sidecar_sha,
                "sidecar_matches": head_sha == sidecar_sha,
                "model_keys": sorted(state),
                "fc_weight_shape": list(state["fc.weight"].shape),
                "fc_bias_shape": list(state["fc.bias"].shape),
                "logits_shape": list(logits.shape),
                "logits_dtype": str(logits.dtype),
                "all_finite": True,
                "logits_sha256": tensor_sha256(logits),
                "logits_min": float(logits.min()),
                "logits_max": float(logits.max()),
            }
        )

    result = {
        "schema": "rhan-stage0b-inference-pipeline-test/v1",
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "protocol_version": config["protocol_version"],
        "inference_config_sha256": sha256_file(CONFIG),
        "partition_manifest_sha256": sha256_file(PARTITION),
        "partition_canonical_sha256": manifest["verification"]["manifest_sha256"],
        "sample_selection": "first calibration image (within_class_rank=0) for class indices 0-7; no evaluation image used",
        "sample_count": SAMPLE_CLASSES,
        "dataset_root": str(DATASET_ROOT.relative_to(ROOT)),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "timm": timm.__version__,
            "numpy": np.__version__,
            "pillow": __import__("PIL").__version__,
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0),
        },
        "transform": {
            "documented_equals_stage0a_make_transform": True,
            "input_resolution": 224,
            "resize_short_side": 256,
            "resize_interpolation": "bicubic",
            "resize_antialias": True,
            "center_crop": 224,
            "normalization": "ImageNet mean/std",
        },
        "backbone": {
            "timm_name": config["backbone"]["timm_name"],
            "hf_revision": config["backbone"]["hf_revision"],
            "weights_file_sha256": backbone_sha,
            "state_dict_sha256": actual_state_sha,
            "state_dict_matches_frozen": True,
            "architecture": architecture,
        },
        "features": {
            "shape": list(features_f16.shape),
            "dtype": "float16",
            "cache_upcast_dtype": "float32",
            "feature_sha256": sha256_bytes(features_f16.tobytes()),
            "forward_features_forward_head_identical": True,
        },
        "images": image_rows,
        "heads": head_rows,
        "guards": {
            "evaluation_images_used": 0,
            "temperature_fitted": False,
            "calibration_metrics_computed": False,
            "aggregate_outcomes_inspected": False,
            "stage0b_launch_authorized": False,
        },
    }
    if len(sample_logits) != 3:
        raise SystemExit("expected three head result sets")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"status=PASS images={SAMPLE_CLASSES} features={tuple(features_f16.shape)} "
        f"heads={len(head_rows)} real_temperature_fitted=False"
    )
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
