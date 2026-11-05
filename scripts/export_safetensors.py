#!/usr/bin/env python3
"""Export the official feedforward-112 checkpoint to safetensors + config.json.

The export is deterministic: rerunning it on the same source checkpoint and
source tree produces byte-identical artifacts. No timestamp is embedded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gen2_foundation.model import Gen2FoundationModel
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD

OFFICIAL_CHECKPOINT = REPO_ROOT / "checkpoints" / "foundation_ff_112_seed17_final_d00fb750.pth"
OFFICIAL_SHA256 = "d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc"
DEFAULT_OUTPUT = REPO_ROOT / "release_staging" / "rhan-nxa-a-ff112"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_config(phase: str, img_size: int) -> Dict[str, Any]:
    return {
        "class": "gen2_foundation.model.Gen2FoundationModel",
        "kwargs": {
            "phase": phase,
            "arm": "v1",
            "fovea_size": img_size,
            "num_classes": 100,
            "d_z": 384,
            "num_glimpses": 4,
            "within_glimpse_iters": 2,
            "use_layer_scale": True,
            "center_crop_mode": False,
        },
    }


def _build_model(config: Dict[str, Any]) -> Gen2FoundationModel:
    kwargs = dict(config["kwargs"])
    return Gen2FoundationModel(**kwargs)


def load_exported_model(export_dir: Path) -> Gen2FoundationModel:
    """Load model.safetensors using the sibling config.json."""
    from safetensors.torch import load_file

    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    model = _build_model(config["model"])
    state = load_file(str(export_dir / config["state_dict_file"]), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export_checkpoint(
    checkpoint: Path = OFFICIAL_CHECKPOINT,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    phase: str = "ff_112",
    img_size: int = 112,
    verify_source_hash: bool = True,
) -> Dict[str, Any]:
    from safetensors.torch import save_file

    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    source_sha256 = sha256_file(checkpoint)
    if verify_source_hash and source_sha256 != OFFICIAL_SHA256:
        raise RuntimeError(
            f"source checkpoint sha256 mismatch: expected {OFFICIAL_SHA256}, got {source_sha256}"
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError("checkpoint does not contain a 'model' state dict")
    if payload.get("phase") != phase:
        raise RuntimeError(f"checkpoint phase {payload.get('phase')!r} != {phase!r}")
    if payload.get("epoch") != 15:
        raise RuntimeError(f"checkpoint epoch {payload.get('epoch')!r} != 15")

    model = _build_model(_model_config(phase, img_size))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    # The gist encoder intentionally aliases backbone.patch_embed. Clone every
    # tensor so safetensors never sees shared storage (the alias is structural,
    # and load_state_dict re-establishes it).
    state = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in model.state_dict().items()
    }
    param_count = sum(parameter.numel() for parameter in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.safetensors"
    config_path = output_dir / "config.json"
    tmp_weights = output_dir / ".model.safetensors.tmp"
    tmp_config = output_dir / ".config.json.tmp"

    metadata = {
        "provenance": json.dumps(
            {
                "format": "pt",
                "architecture": "Gen2FoundationModel",
                "phase": phase,
                "source_sha256": source_sha256,
            },
            sort_keys=True,
        )
    }
    save_file(state, str(tmp_weights), metadata=metadata)
    os.replace(tmp_weights, weights_path)

    config = {
        "format": "safetensors",
        "model_name": "RHAN-NXA, feedforward-112",
        "state_dict_file": "model.safetensors",
        "param_count": param_count,
        "input": {"height": img_size, "width": img_size, "channels": 3},
        "num_classes": 100,
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "space": "imagenet_normalized",
        },
        "training_epsilon_norm": 0.094,
        "model": _model_config(phase, img_size),
        "source_checkpoint": {
            "filename": checkpoint.name,
            "sha256": source_sha256,
            "phase": payload.get("phase"),
            "epoch": payload.get("epoch"),
        },
    }
    tmp_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_config, config_path)

    exported_sha256 = sha256_file(weights_path)
    reloaded = load_exported_model(output_dir)
    generator = torch.Generator().manual_seed(0)
    batch = torch.randn(2, 3, img_size, img_size, generator=generator)
    with torch.no_grad():
        reference = model(batch)
        candidate = reloaded(batch)
    parity_max_abs = (reference - candidate).abs().max().item()
    if parity_max_abs != 0.0:
        raise RuntimeError(f"CPU eval parity failed: max_abs={parity_max_abs}")

    return {
        "weights_path": weights_path,
        "config_path": config_path,
        "source_sha256": source_sha256,
        "weights_sha256": exported_sha256,
        "config_sha256": sha256_file(config_path),
        "param_count": param_count,
        "parity_max_abs": parity_max_abs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=OFFICIAL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", default="ff_112")
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--skip-source-hash", action="store_true")
    args = parser.parse_args()

    result = export_checkpoint(
        args.checkpoint,
        args.output_dir,
        phase=args.phase,
        img_size=args.img_size,
        verify_source_hash=not args.skip_source_hash,
    )
    print(f"SOURCE_SHA256={result['source_sha256']}")
    print(f"WEIGHTS_SHA256={result['weights_sha256']}")
    print(f"CONFIG_SHA256={result['config_sha256']}")
    print(f"PARAM_COUNT={result['param_count']}")
    print(f"PARITY_MAX_ABS={result['parity_max_abs']}")
    print(f"WEIGHTS={result['weights_path']}")
    print(f"CONFIG={result['config_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
