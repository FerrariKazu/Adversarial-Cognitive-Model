#!/usr/bin/env python3
"""Stage 0B experiment runner.

Commands:
  preflight  Final execution preflight only. Does not fit temperatures or score.
  fit        Temperature fitting. Requires a separate execution approval.
  score      One-shot evaluation. Requires a separate execution approval and
             a completed, verified fit.

This runner is intentionally separate from the frozen Stage 0B bundle.
It reads the frozen protocol and verifies it, but never edits it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FREEZE_DIR = PROJECT_ROOT / "audit" / "stage0b_final_freeze_v1"
DATA_GATE_DIR = PROJECT_ROOT / "audit" / "stage0b_data_gate_v1"
STAGE0A_DIR = PROJECT_ROOT / "audit" / "stage0a_closeout_v1"
EXEC_DIR = PROJECT_ROOT / "audit" / "stage0b_execution_v1"
DATASET_ROOT = DATA_GATE_DIR / "work" / "extracted" / "imagenetv2-matched-frequency-format-val"
ARCHIVE = DATA_GATE_DIR / "work" / "source" / "imagenetv2-matched-frequency.tar.gz"

APPROVED_PAYLOAD = "0bcf8bf8e37e3d8929bdf778bd443f41517e54597afb15a31d8e869367959672"
APPROVED_MANIFEST_SHA = "47374b8f232868e95697fdecad7442ae6077b41f02063ccb8356e853cd8564e3"
APPROVED_PARTITION_CANONICAL = "557e10f1b94c161bc5e57926c137faaa27208410743a0a039b5b43be6da52578"
APPROVED_PARTITION_JSON = "fa7f23032d6b953664bc573a53d7d90b4d84be3d46fe7faed445e288168135e6"
APPROVED_INFERENCE_CONFIG = "d678558dbc5a53505f688e6411c43536fd3f7f5dbd8166decc31ce96c5bdda67"
APPROVED_ENV_FINGERPRINT = "893427754f21fbe0e2a4c3b0ec31f74dd0ca620f77b5b8f3292bdc78d1ac9e29"
APPROVED_HEADS = {
    17: "320f79a79b7110c7f8120eaedaa122049f8e2b12ba2fe765e867f1e1f3af308e",
    42: "f5e7d42d3a901b93b0fb0d67d2cf3d25a0fa34f8dc5d59b9530c2ff40b23c357",
    99: "cb91a2b14c96b514c977e2c09a87deb7002ea3d87450f1a04d2de45a7aac452d",
}
APPROVED_BACKBONE_STATE = "375b4c89151489e61df6f1f3b3fe1f9fa8b815c52ae0f7117b23e2e201fbb4fe"
ARCHIVE_BYTES = 1264079360
ARCHIVE_SHA = "f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c"

DATA_USE_CONFIRMATION_PATH = EXEC_DIR / "data_use_confirmation.json"
APPROVED_DATA_USE_SCHEMA = "rhan-stage0b-data-use-confirmation/v1"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FREEZE_DIR / "scripts"))
import stage0b_evaluation_core as EC  # noqa: E402

from stage0a.data import make_transform  # noqa: E402

sys.path.insert(0, str(DATA_GATE_DIR / "scripts"))
import calibration_core as CC  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2) + "\n")


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return obj


def load_protocol() -> tuple[dict, dict]:
    """Load and verify the frozen protocol. Raises on any mismatch."""
    # 1. MANIFEST.sha256
    manifest_path = FREEZE_DIR / "MANIFEST.sha256"
    if sha256_file(manifest_path) != APPROVED_MANIFEST_SHA:
        raise RuntimeError("MANIFEST.sha256 file hash mismatch")
    entries = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.strip()
        entries.append((digest, rel))
    if len(entries) != 27:
        raise RuntimeError(f"MANIFEST has {len(entries)} entries, expected 27")
    for digest, rel in entries:
        p = FREEZE_DIR / rel
        if not p.is_file():
            raise RuntimeError(f"missing frozen artifact: {rel}")
        if sha256_file(p) != digest:
            raise RuntimeError(f"hash mismatch for frozen artifact: {rel}")
    # 2. protocol manifest payload
    protocol = json.loads((FREEZE_DIR / "STAGE0B_PROTOCOL_MANIFEST.json").read_text())
    if protocol["manifest_payload_sha256"] != APPROVED_PAYLOAD:
        raise RuntimeError("protocol manifest payload field mismatch")
    # Recompute payload hash from the full protocol manifest excluding the hash field.
    body = dict(protocol)
    body.pop("manifest_payload_sha256", None)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != APPROVED_PAYLOAD:
        raise RuntimeError("recomputed protocol payload mismatch")
    # 3. partition
    partition_file = DATA_GATE_DIR / "PARTITION_MANIFEST.json"
    if sha256_file(partition_file) != APPROVED_PARTITION_JSON:
        raise RuntimeError("partition JSON hash mismatch")
    partition = json.loads(partition_file.read_text())
    if partition["verification"]["manifest_sha256"] != APPROVED_PARTITION_CANONICAL:
        raise RuntimeError("partition canonical hash mismatch")
    # 4. inference config
    cfg_path = FREEZE_DIR / "STAGE0B_INFERENCE_CONFIG.json"
    if sha256_file(cfg_path) != APPROVED_INFERENCE_CONFIG:
        raise RuntimeError("inference config hash mismatch")
    # 5. heads
    for hc in protocol["classification_heads"]:
        seed = int(hc["seed"])
        rel = hc["artifact"]
        p = PROJECT_ROOT / rel
        if not p.is_file():
            raise RuntimeError(f"missing head: {rel}")
        if sha256_file(p) != APPROVED_HEADS[seed]:
            raise RuntimeError(f"head hash mismatch seed {seed}")
    # 6. backbone state dict
    backbone_file = PROJECT_ROOT / protocol["backbone"]["weights_file"]
    if not backbone_file.is_file():
        raise RuntimeError("backbone weights missing")
    if sha256_file(backbone_file) != protocol["backbone"]["weights_file_sha256"]:
        raise RuntimeError("backbone file hash mismatch")
    return protocol, partition


def assignments(partition: dict, split: str) -> list[dict]:
    rows = [r for r in partition["assignments"] if r["split"] == split]
    rows.sort(key=lambda r: (int(r["class_index"]), int(r["within_class_rank"]), r["relative_path"]))
    return rows


def make_head(seed: int):
    import torch.nn as nn

    class LinearHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(768, 1000)

        def forward(self, x):
            return self.fc(x)

    hc = None
    protocol, _ = load_protocol()
    for h in protocol["classification_heads"]:
        if int(h["seed"]) == seed:
            hc = h
            break
    head = LinearHead()
    ckpt = torch.load(PROJECT_ROOT / hc["artifact"], map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["model"])
    head.eval()
    return head


def load_backbone():
    import timm
    from stage0a.utils import state_dict_sha256

    protocol, _ = load_protocol()
    b = protocol["backbone"]
    path = PROJECT_ROOT / b["weights_file"]
    model = timm.create_model(
        b["timm_name"],
        pretrained=True,
        pretrained_cfg_overlay={"file": str(path)},
        num_classes=0,
        img_size=224,
        dynamic_img_size=True,
    )
    model.eval()
    sd = model.state_dict()
    actual = state_dict_sha256(sd)
    if actual != APPROVED_BACKBONE_STATE:
        raise RuntimeError(f"backbone state dict hash mismatch: {actual}")
    if sha256_file(path) != b["weights_file_sha256"]:
        raise RuntimeError("backbone file hash mismatch")
    return model, b


def runtime_record() -> dict:
    import torch
    rec = {
        "captured_at_utc": utc_now(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
        },
        "gpu": {},
        "software": {},
        "disk": {},
        "memory": {},
    }
    # GPU
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(dev)
        rec["gpu"] = {
            "available": True,
            "name": torch.cuda.get_device_name(dev),
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": f"{props.major}.{props.minor}",
            "driver_version": torch.version.cuda,
        }
    else:
        rec["gpu"] = {"available": False}
    # software
    import torchvision, timm, numpy, scipy, PIL
    rec["software"] = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pillow": PIL.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    # memory
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem = {}
        for line in lines:
            key, val = line.split(":", 1)
            mem[key.strip()] = int(val.strip().split()[0]) * 1024
        rec["memory"] = {
            "mem_total_bytes": mem.get("MemTotal"),
            "mem_available_bytes": mem.get("MemAvailable"),
        }
    except Exception as e:
        rec["memory"] = {"error": str(e)}
    # disk
    st = os.statvfs(str(PROJECT_ROOT))
    rec["disk"] = {
        "path": str(PROJECT_ROOT),
        "total_bytes": int(st.f_blocks * st.f_frsize),
        "free_bytes": int(st.f_bavail * st.f_frsize),
    }
    return rec


def verify_archive() -> dict:
    if not ARCHIVE.is_file():
        raise RuntimeError("archive missing")
    size = ARCHIVE.stat().st_size
    if size != ARCHIVE_BYTES:
        raise RuntimeError(f"archive size mismatch: {size} != {ARCHIVE_BYTES}")
    h = sha256_file(ARCHIVE)
    if h != ARCHIVE_SHA:
        raise RuntimeError("archive sha mismatch")
    return {"path": str(ARCHIVE), "size_bytes": size, "sha256": h, "status": "PASS"}


def verify_dataset(partition: dict) -> dict:
    if not DATASET_ROOT.is_dir():
        raise RuntimeError("dataset root missing")
    all_files = [p for p in DATASET_ROOT.rglob("*") if p.is_file()]
    actual_rel = sorted(str(p.relative_to(DATASET_ROOT)) for p in all_files)
    expected_rel = sorted(r["relative_path"] for r in partition["assignments"])
    if actual_rel != expected_rel:
        missing = sorted(set(expected_rel) - set(actual_rel))
        extra = sorted(set(actual_rel) - set(expected_rel))
        raise RuntimeError(f"dataset file set mismatch; missing={len(missing)} extra={len(extra)}")
    # counts
    class_dirs = [d for d in DATASET_ROOT.iterdir() if d.is_dir()]
    if len(class_dirs) != 1000:
        raise RuntimeError(f"expected 1000 class dirs, got {len(class_dirs)}")
    per_class = {}
    for d in class_dirs:
        n = sum(1 for p in d.iterdir() if p.is_file())
        per_class[d.name] = n
    bad = {k: v for k, v in per_class.items() if v != 10}
    if bad:
        raise RuntimeError(f"classes without 10 images: {len(bad)}")
    # split separation
    cal = [r for r in partition["assignments"] if r["split"] == "calibration"]
    ev = [r for r in partition["assignments"] if r["split"] == "evaluation"]
    if len(cal) != 5000 or len(ev) != 5000:
        raise RuntimeError("split counts wrong")
    overlap = set(r["relative_path"] for r in cal) & set(r["relative_path"] for r in ev)
    if overlap:
        raise RuntimeError(f"split overlap: {len(overlap)}")
    # calibration content hash check (evaluation images are not opened)
    cal_hashes = {}
    for r in cal:
        cal_hashes[r["relative_path"]] = sha256_file(DATASET_ROOT / r["relative_path"])
    hash_mismatches = [
        r["relative_path"]
        for r in cal
        if cal_hashes[r["relative_path"]] != r["content_sha256"]
    ]
    if hash_mismatches:
        raise RuntimeError(f"calibration content hash mismatches: {len(hash_mismatches)}")
    return {
        "root": str(DATASET_ROOT),
        "file_count": len(all_files),
        "class_dir_count": len(class_dirs),
        "images_per_class": 10,
        "calibration_assignments": len(cal),
        "evaluation_assignments": len(ev),
        "split_overlap": 0,
        "calibration_content_hashes_checked": len(cal),
        "calibration_content_hash_mismatches": len(hash_mismatches),
        "evaluation_images_opened_for_integrity": 0,
        "status": "PASS",
    }


def verify_class_mapping(partition: dict) -> dict:
    mapping_path = DATA_GATE_DIR / "CLASS_MAPPING_VERIFICATION.json"
    if not mapping_path.is_file():
        raise RuntimeError("class mapping verification missing")
    m = json.loads(mapping_path.read_text())
    if m.get("status") != "PASS":
        raise RuntimeError("class mapping status not PASS")
    # verify folder index == class index for all rows
    bad = []
    for r in partition["assignments"]:
        folder = int((DATASET_ROOT / r["relative_path"]).parent.name)
        if folder != int(r["class_index"]):
            bad.append(r["relative_path"])
    if bad:
        raise RuntimeError(f"folder/class index mismatches: {len(bad)}")
    return {
        "status": "PASS",
        "class_count": m.get("class_count"),
        "canonical_classes": m.get("canonical_classes"),
        "stage0a_classes": m.get("stage0a_classes"),
        "class_order_matches_canonical": m.get("class_order_matches_canonical"),
        "folder_equals_class_index": True,
        "rows_checked": len(partition["assignments"]),
    }


def verify_heads(protocol: dict) -> dict:
    out = {}
    for hc in protocol["classification_heads"]:
        seed = int(hc["seed"])
        p = PROJECT_ROOT / hc["artifact"]
        sidecar = p.with_name(p.name + ".sha256")
        if not sidecar.is_file():
            raise RuntimeError(f"missing head sidecar seed {seed}")
        sidecar_text = sidecar.read_text().strip().split()[0]
        if sidecar_text != hc["sha256"]:
            raise RuntimeError(f"head sidecar mismatch seed {seed}")
        if sha256_file(p) != hc["sha256"]:
            raise RuntimeError(f"head hash mismatch seed {seed}")
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if set(ckpt["model"].keys()) != {"fc.weight", "fc.bias"}:
            raise RuntimeError(f"unexpected head keys seed {seed}")
        if tuple(ckpt["model"]["fc.weight"].shape) != (1000, 768):
            raise RuntimeError(f"bad head weight shape seed {seed}")
        if tuple(ckpt["model"]["fc.bias"].shape) != (1000,):
            raise RuntimeError(f"bad head bias shape seed {seed}")
        out[str(seed)] = {
            "path": hc["artifact"],
            "sha256": hc["sha256"],
            "sidecar_matches": True,
            "keys": sorted(ckpt["model"].keys()),
            "weight_shape": list(ckpt["model"]["fc.weight"].shape),
            "bias_shape": list(ckpt["model"]["fc.bias"].shape),
            "status": "PASS",
        }
    return out


def verify_backbone(protocol: dict) -> dict:
    from stage0a.utils import state_dict_sha256

    b = protocol["backbone"]
    path = PROJECT_ROOT / b["weights_file"]
    if sha256_file(path) != b["weights_file_sha256"]:
        raise RuntimeError("backbone file hash mismatch")
    model = None
    try:
        model, _ = load_backbone()
        sd_hash = state_dict_sha256(model.state_dict())
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if sd_hash != APPROVED_BACKBONE_STATE:
        raise RuntimeError(f"backbone state dict hash mismatch: {sd_hash}")
    return {
        "path": b["weights_file"],
        "file_sha256": b["weights_file_sha256"],
        "state_dict_sha256": sd_hash,
        "status": "PASS",
    }


def pipeline_smoke() -> dict:
    protocol, partition = load_protocol()
    model, b = load_backbone()
    model = model.cuda().eval()
    heads = {s: make_head(s) for s in (17, 42, 99)}
    transform = make_transform(224)
    # first calibration image of classes 0..7
    rows = [r for r in partition["assignments"] if r["split"] == "calibration"]
    rows.sort(key=lambda r: (int(r["class_index"]), int(r["within_class_rank"])))
    chosen = []
    seen = set()
    for r in rows:
        ci = int(r["class_index"])
        if ci < 8 and ci not in seen:
            chosen.append(r)
            seen.add(ci)
        if len(chosen) == 8:
            break
    if len(chosen) != 8:
        raise RuntimeError("could not select 8 smoke images")
    imgs = []
    for r in chosen:
        with Image.open(DATASET_ROOT / r["relative_path"]) as im:
            imgs.append(im.convert("RGB"))
    batch = torch.stack([transform(im) for im in imgs])
    with torch.inference_mode():
        batch = batch.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            features = model(batch.cuda())
        features = features.float().cpu()
        logits_by_seed = {}
        for seed, head in heads.items():
            head = head.cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                logits = head(features.cuda()).float().cpu()
            logits_by_seed[str(seed)] = {
                "shape": list(logits.shape),
                "finite": bool(torch.isfinite(logits).all().item()),
            }
            head.cpu()
    del model
    for h in heads.values():
        del h
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "image_count": len(chosen),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "logits": logits_by_seed,
        "status": "PASS",
    }


def cmd_preflight(args) -> int:
    out_dir = Path(args.out) if args.out else EXEC_DIR / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Verifying frozen protocol bundle...")
    protocol, partition = load_protocol()
    print("Verifying archive...")
    archive_info = verify_archive()
    print("Verifying dataset (calibration hashes only; evaluation images not opened)...")
    dataset_info = verify_dataset(partition)
    print("Verifying class mapping...")
    mapping_info = verify_class_mapping(partition)
    print("Verifying three heads...")
    heads_info = verify_heads(protocol)
    print("Verifying backbone...")
    backbone_info = verify_backbone(protocol)
    print("Running 8-image inference smoke test...")
    smoke_info = pipeline_smoke()
    print("Capturing runtime...")
    rt = runtime_record()

    # execution approval status
    approval_path = EXEC_DIR / "execution_approval.json"
    approval_present = approval_path.is_file()
    approval_valid = False
    approval_errors = []
    if approval_present:
        try:
            ap = json.loads(approval_path.read_text())
            if ap.get("schema") != "rhan-stage0b-execution-approval/v1":
                approval_errors.append("bad schema")
            if not (ap.get("approved") is True and ap.get("execution_approved") is True):
                approval_errors.append("approval flags not true")
            if ap.get("protocol_manifest_payload_sha256") != APPROVED_PAYLOAD:
                approval_errors.append("payload hash mismatch")
            approval_valid = (len(approval_errors) == 0)
        except Exception as e:
            approval_errors.append(str(e))
    else:
        approval_errors.append("execution_approval.json not found")

    # data-use confirmation status
    data_use_path = DATA_USE_CONFIRMATION_PATH
    data_use_present = data_use_path.is_file()
    data_use_valid = False
    data_use_errors = []
    if data_use_present:
        try:
            validate_data_use_confirmation(data_use_path)
            data_use_valid = True
        except Exception as e:
            data_use_errors.append(str(e))
    else:
        data_use_errors.append("data_use_confirmation.json not found")

    technical_status = "PASS"
    local_execution_ready = bool(
        technical_status == "PASS" and approval_valid and data_use_valid
    )
    publication_ready = False

    execution_blockers = []
    if not approval_valid:
        execution_blockers.append("separate execution approval not recorded or invalid")
    if not data_use_valid:
        execution_blockers.append("supervisor/institutional data-use confirmation not recorded or invalid")
    publication_blockers = [
        "publication/upload rights for image-derived artifacts unresolved"
    ]

    preflight = {
        "schema": "rhan-stage0b-preflight/v1",
        "created_at_utc": utc_now(),
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol["protocol_version"],
            "payload_sha256": protocol["manifest_payload_sha256"],
            "manifest_sha256": APPROVED_MANIFEST_SHA,
            "inference_config_sha256": APPROVED_INFERENCE_CONFIG,
            "partition_canonical_sha256": APPROVED_PARTITION_CANONICAL,
            "partition_json_sha256": APPROVED_PARTITION_JSON,
        },
        "archive": archive_info,
        "dataset": dataset_info,
        "class_mapping": mapping_info,
        "heads": heads_info,
        "backbone": backbone_info,
        "pipeline_smoke": smoke_info,
        "runtime": rt,
        "runtime_comparison": {
            "frozen_environment_fingerprint": APPROVED_ENV_FINGERPRINT,
            "note": "Frozen environment fingerprint includes capture-time volatile fields and is not recomputed identically. Static software/GPU fields are recorded for manual comparison.",
            "software_matches_frozen": (
                rt["software"]["torch"] == "2.9.1+cu128"
                and rt["software"]["torchvision"] == "0.24.1+cu128"
                and rt["software"]["timm"] == "1.0.27"
                and rt["software"]["numpy"] == "2.2.6"
                and rt["software"]["scipy"] == "1.15.3"
                and rt["software"]["pillow"] == "12.2.0"
                and rt["software"]["cuda_runtime"] == "12.8"
                and rt["software"]["cudnn_version"] == 91002
            ),
            "gpu_matches_frozen": (
                rt["gpu"].get("name") == "NVIDIA GeForce RTX 4060"
                and rt["gpu"].get("compute_capability") == "8.9"
            ),
        },
        "execution_approval": {
            "path": str(approval_path),
            "present": approval_present,
            "valid": approval_valid,
            "errors": approval_errors,
        },
        "data_use_confirmation": {
            "path": str(data_use_path),
            "present": data_use_present,
            "valid": data_use_valid,
            "errors": data_use_errors,
        },
        "evaluation_images_opened_for_outcomes": 0,
        "evaluation_images_opened_for_integrity": 0,
        "calibration_images_opened_for_smoke": smoke_info["image_count"],
        "technical_status": technical_status,
        "local_execution_ready": local_execution_ready,
        "publication_ready": publication_ready,
        "execution_blockers": execution_blockers,
        "publication_blockers": publication_blockers,
    }
    write_json(out_dir / "preflight.json", preflight)
    write_json(out_dir / "environment.json", rt)
    # markdown summary
    md = []
    md.append("# Stage 0B Execution Preflight")
    md.append("")
    md.append(f"- Created: {preflight['created_at_utc']}")
    md.append(f"- Technical status: {preflight['technical_status']}")
    md.append(f"- Local execution ready: {preflight['local_execution_ready']}")
    md.append(f"- Publication ready: {preflight['publication_ready']}")
    md.append("- Evaluation images opened for outcomes: 0")
    md.append("- Evaluation images opened for integrity: 0")
    md.append("")
    md.append("## Execution blockers")
    for b in execution_blockers:
        md.append(f"- {b}")
    md.append("")
    md.append("## Publication blockers")
    for b in publication_blockers:
        md.append(f"- {b}")
    md.append("")
    md.append("## Key verification")
    md.append(f"- Archive: {archive_info['status']} ({archive_info['size_bytes']} bytes)")
    md.append(f"- Dataset: {dataset_info['status']} ({dataset_info['file_count']} files, {dataset_info['class_dir_count']} classes)")
    md.append(f"- Class mapping: {mapping_info['status']}")
    md.append(f"- Heads: {', '.join(heads_info[k]['status'] for k in heads_info)}")
    md.append(f"- Backbone: {backbone_info['status']}")
    md.append(f"- Pipeline smoke: {smoke_info['status']} ({smoke_info['image_count']} images, features {smoke_info['feature_shape']})")
    (out_dir / "PREFLIGHT.md").write_text("\n".join(md) + "\n")
    print(
        f"Preflight written to {out_dir}; technical={technical_status}; "
        f"local_execution_ready={local_execution_ready}; publication_ready={publication_ready}; "
        f"execution_blockers={len(execution_blockers)}"
    )
    return 0


def validate_execution_approval(approval_path: Path | None = None) -> dict:
    if approval_path is None:
        approval_path = EXEC_DIR / "execution_approval.json"
    approval_path = Path(approval_path)
    if not approval_path.is_file():
        raise RuntimeError(
            "execution approval required: create audit/stage0b_execution_v1/execution_approval.json "
            "with schema rhan-stage0b-execution-approval/v1 and approved=true"
        )
    ap = json.loads(approval_path.read_text())
    errors = []
    if ap.get("schema") != "rhan-stage0b-execution-approval/v1":
        errors.append("schema must be rhan-stage0b-execution-approval/v1")
    if ap.get("approved") is not True or ap.get("execution_approved") is not True:
        errors.append("approved and execution_approved must be true")
    if ap.get("protocol_manifest_payload_sha256") != APPROVED_PAYLOAD:
        errors.append("protocol payload hash mismatch")
    if errors:
        raise RuntimeError("invalid execution approval: " + "; ".join(errors))
    return ap


def validate_data_use_confirmation(path: Path | None = None) -> dict:
    if path is None:
        path = DATA_USE_CONFIRMATION_PATH
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(
            "data-use confirmation required: create "
            + str(path)
            + " with schema "
            + APPROVED_DATA_USE_SCHEMA
            + " and confirmation_status=confirmed"
        )
    try:
        doc = json.loads(path.read_text())
    except Exception as e:
        raise RuntimeError(f"invalid data-use confirmation: {e}")
    errors = []
    if doc.get("schema") != APPROVED_DATA_USE_SCHEMA:
        errors.append("schema must be " + APPROVED_DATA_USE_SCHEMA)
    if doc.get("confirmation_status") != "confirmed" or doc.get("data_use_confirmed") is not True:
        errors.append("confirmation_status must be confirmed and data_use_confirmed must be true")
    if not doc.get("confirmer_role"):
        errors.append("confirmer_role is required")
    if not doc.get("confirmation_date_utc"):
        errors.append("confirmation_date_utc is required")
    scope = doc.get("approved_scope")
    if not isinstance(scope, list) or not scope:
        errors.append("approved_scope must be a non-empty list")
    if doc.get("protocol_manifest_payload_sha256") != APPROVED_PAYLOAD:
        errors.append("protocol payload hash mismatch")
    dataset = doc.get("dataset_identity")
    if not isinstance(dataset, dict):
        errors.append("dataset_identity is required")
        dataset = {}
    if dataset.get("partition_canonical_sha256") != APPROVED_PARTITION_CANONICAL:
        errors.append("partition canonical hash mismatch")
    if dataset.get("partition_json_sha256") != APPROVED_PARTITION_JSON:
        errors.append("partition JSON hash mismatch")
    if not dataset.get("dataset_name"):
        errors.append("dataset_identity.dataset_name is required")
    restrictions = doc.get("restrictions")
    if not isinstance(restrictions, list) or not restrictions:
        errors.append("restrictions must be a non-empty list")
    if not doc.get("evidence_reference"):
        errors.append("evidence_reference is required")
    if doc.get("publication_rights_granted") is True:
        errors.append("data-use confirmation must not grant publication rights")
    if errors:
        raise RuntimeError("invalid data-use confirmation: " + "; ".join(errors))
    return doc


def validate_execution_gates(
    execution_approval_path: Path | None = None,
    data_use_confirmation_path: Path | None = None,
) -> dict:
    if execution_approval_path is None:
        execution_approval_path = EXEC_DIR / "execution_approval.json"
    if data_use_confirmation_path is None:
        data_use_confirmation_path = DATA_USE_CONFIRMATION_PATH
    approval = validate_execution_approval(execution_approval_path)
    data_use = validate_data_use_confirmation(data_use_confirmation_path)
    return {"execution_approval": approval, "data_use_confirmation": data_use}


def extract_features(split: str, out_dir: Path) -> dict:
    protocol, partition = load_protocol()
    rows = assignments(partition, split)
    if not rows:
        raise RuntimeError(f"no assignments for split {split}")
    model, b = load_backbone()
    heads = None
    transform = make_transform(224)
    # determinism
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    class FeatDS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            r = self.rows[idx]
            with Image.open(DATASET_ROOT / r["relative_path"]) as im:
                img = im.convert("RGB")
            x = transform(img)
            return x, int(r["class_index"]), r["relative_path"]

    ds = FeatDS(rows)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    n = len(rows)
    features = np.zeros((n, 768), dtype=np.float16)
    labels = np.zeros(n, dtype=np.int64)
    paths = []
    model = model.cuda()
    model.eval()
    idx = 0
    with torch.inference_mode():
        for batch_x, batch_y, batch_paths in loader:
            batch_x = batch_x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                feats = model(batch_x)
            feats = feats.float().cpu().numpy()
            bs = feats.shape[0]
            features[idx : idx + bs] = feats.astype(np.float16)
            labels[idx : idx + bs] = batch_y.numpy()
            paths.extend(list(batch_paths))
            idx += bs
    if idx != n:
        raise RuntimeError(f"feature extraction incomplete: {idx}/{n}")
    # verify labels correspond to rows
    for i, r in enumerate(rows):
        if int(labels[i]) != int(r["class_index"]):
            raise RuntimeError(f"label mismatch at row {i}")
        if paths[i] != r["relative_path"]:
            raise RuntimeError(f"path mismatch at row {i}")
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / f"features_{split}.npy"
    labels_path = out_dir / f"labels_{split}.npy"
    paths_path = out_dir / f"paths_{split}.json"
    np.save(feat_path, features)
    np.save(labels_path, labels)
    paths_path.write_text(json.dumps(paths, indent=2) + "\n")
    meta = {
        "split": split,
        "row_count": n,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "labels_shape": list(labels.shape),
        "feature_file_sha256": sha256_file(feat_path),
        "labels_file_sha256": sha256_file(labels_path),
        "paths_file_sha256": sha256_file(paths_path),
        "row_order": "frozen partition order (class_index, within_class_rank, relative_path)",
        "batch_size": 8,
        "num_workers": 0,
        "amp_dtype": "float16",
        "created_at_utc": utc_now(),
    }
    write_json(out_dir / f"features_{split}_meta.json", meta)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def cmd_fit(args) -> int:
    validate_execution_gates()
    protocol, partition = load_protocol()
    run_dir = Path(args.out) if args.out else EXEC_DIR / "stage0b_run_v1"
    if run_dir.exists() and not args.force:
        # allow reuse only if no completion marker
        pass
    run_dir.mkdir(parents=True, exist_ok=True)
    # refuse if already completed
    if (run_dir / "fit_complete.json").is_file():
        raise RuntimeError("fit already completed; refusing to overwrite")
    # extract calibration features only
    cal_dir = run_dir / "features"
    print("Extracting calibration features (fit phase only; no evaluation images)...")
    cal_meta = extract_features("calibration", cal_dir)
    print(f"Calibration features: {cal_meta['row_count']} rows")

    # load features
    features = np.load(cal_dir / "features_calibration.npy")
    labels = np.load(cal_dir / "labels_calibration.npy")
    feats_f32 = features.astype(np.float32)
    feat_t = torch.from_numpy(feats_f32)

    fits = []
    for seed in (17, 42, 99):
        head = make_head(seed).cuda()
        with torch.inference_mode():
            logits = head(feat_t.cuda()).float().cpu().numpy()
        head.cpu()
        del head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        T, result = CC.fit_temperature(logits, labels.astype(np.int64))
        converged = bool(result.success)
        at_boundary = bool(T <= 0.05 + 1e-12 or T >= 20.0 - 1e-12)
        fit_rec = {
            "seed": seed,
            "temperature": float(T),
            "converged": converged,
            "at_boundary": at_boundary,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "function_evaluations": int(result.nfev),
            "final_nll": float(result.fun),
            "initial_temperature": 1.0,
            "bounds": [0.05, 20.0],
            "dtype": "float64",
            "calibration_images_used": int(labels.shape[0]),
            "evaluation_images_used": 0,
        }
        fits.append(fit_rec)
        print(f"seed {seed}: T={T:.6f} converged={converged} boundary={at_boundary}")
    all_converged = all(f["converged"] for f in fits)
    any_boundary = any(f["at_boundary"] for f in fits)

    # write temperatures
    temps = {str(f["seed"]): f["temperature"] for f in fits}
    write_json(run_dir / "temperatures.json", {
        "schema": "rhan-stage0b-temperatures/v1",
        "created_at_utc": utc_now(),
        "protocol_payload_sha256": APPROVED_PAYLOAD,
        "temperatures": temps,
        "fits": fits,
        "all_fits_converged": all_converged,
        "any_fit_at_boundary": any_boundary,
    })
    # fit_complete marker (required before scoring)
    fit_complete = {
        "schema": "rhan-stage0b-fit-complete/v1",
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "calibration_feature_meta_sha256": sha256_file(cal_dir / "features_calibration_meta.json"),
        "calibration_features_sha256": sha256_file(cal_dir / "features_calibration.npy"),
        "calibration_labels_sha256": sha256_file(cal_dir / "labels_calibration.npy"),
        "temperatures_sha256": sha256_file(run_dir / "temperatures.json"),
        "all_fits_converged": all_converged,
        "any_fit_at_boundary": any_boundary,
        "evaluation_images_opened": 0,
        "evaluation_images_opened_for_scoring": 0,
    }
    write_json(run_dir / "fit_complete.json", fit_complete)
    print(f"Fit complete. all_converged={all_converged} any_boundary={any_boundary}")
    print(f"fit_complete.json written to {run_dir}")
    return 0


def cmd_score(args) -> int:
    validate_execution_gates()
    protocol, partition = load_protocol()
    run_dir = Path(args.out) if args.out else EXEC_DIR / "stage0b_run_v1"
    if not (run_dir / "fit_complete.json").is_file():
        raise RuntimeError("fit_complete.json missing; run fit first")
    fit_complete = json.loads((run_dir / "fit_complete.json").read_text())
    temps_doc = json.loads((run_dir / "temperatures.json").read_text())
    temperatures = {int(k): float(v) for k, v in temps_doc["temperatures"].items()}
    # verify fit_complete hashes still match
    cal_dir = run_dir / "features"
    if sha256_file(cal_dir / "features_calibration_meta.json") != fit_complete["calibration_feature_meta_sha256"]:
        raise RuntimeError("calibration feature meta changed since fit")
    if sha256_file(cal_dir / "features_calibration.npy") != fit_complete["calibration_features_sha256"]:
        raise RuntimeError("calibration features changed since fit")
    if sha256_file(cal_dir / "labels_calibration.npy") != fit_complete["calibration_labels_sha256"]:
        raise RuntimeError("calibration labels changed since fit")
    if sha256_file(run_dir / "temperatures.json") != fit_complete["temperatures_sha256"]:
        raise RuntimeError("temperatures changed since fit")
    # refuse if score already completed
    if (run_dir / "score_complete.json").is_file():
        raise RuntimeError("score already completed; refusing to overwrite")
    # mark score started (before opening eval images)
    score_started = {
        "schema": "rhan-stage0b-score-started/v1",
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "score_started.json", score_started)

    # extract evaluation features (first time eval images are opened)
    print("Extracting evaluation features (one-shot scoring)...")
    eval_dir = run_dir / "features"
    eval_meta = extract_features("evaluation", eval_dir)
    print(f"Evaluation features: {eval_meta['row_count']} rows")

    # load features
    cal_features = np.load(cal_dir / "features_calibration.npy").astype(np.float32)
    cal_labels = np.load(cal_dir / "labels_calibration.npy").astype(np.int64)
    eval_features = np.load(eval_dir / "features_evaluation.npy").astype(np.float32)
    eval_labels = np.load(eval_dir / "labels_evaluation.npy").astype(np.int64)

    # compute logits per seed
    raw_logits_by_seed = {}
    for seed in (17, 42, 99):
        head = make_head(seed)
        head.eval()
        with torch.inference_mode():
            raw_logits_by_seed[seed] = head(torch.from_numpy(eval_features)).float().cpu().numpy()
        del head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # raw metrics
    raw_scores = []
    for seed in (17, 42, 99):
        s = EC.score_logits(raw_logits_by_seed[seed], eval_labels, 1.0)
        raw_scores.append(s)
    # calibrated logits + metrics
    cal_scores = []
    calibrated_logits_by_seed = {}
    for seed in (17, 42, 99):
        T = temperatures[seed]
        calibrated = raw_logits_by_seed[seed] / T
        calibrated_logits_by_seed[seed] = calibrated
        s = EC.score_logits(calibrated, eval_labels, 1.0)
        cal_scores.append(s)

    # paired deltas
    deltas = []
    for i, seed in enumerate((17, 42, 99)):
        d = EC.paired_delta(
            raw_logits_by_seed[seed],
            calibrated_logits_by_seed[seed],
            raw_scores[i],
            cal_scores[i],
        )
        d["seed"] = seed
        deltas.append(d)

    # bootstrap on three-seed mean ECE reduction
    stats = json.loads((FREEZE_DIR / "work" / "statistical_config.json").read_text())
    n_boot = int(stats["bootstrap"]["n_boot"])
    boot_seed = int(stats["bootstrap"]["seed"])
    bootstrap = EC.stratified_paired_bootstrap(
        [raw_logits_by_seed[s] for s in (17, 42, 99)],
        [calibrated_logits_by_seed[s] for s in (17, 42, 99)],
        eval_labels,
        n_boot=n_boot,
        seed=boot_seed,
    )
    decision = EC.acceptance_decision(
        deltas,
        bootstrap,
        all_fits_converged=bool(fit_complete["all_fits_converged"]),
        any_fit_at_boundary=bool(fit_complete["any_fit_at_boundary"]),
    )

    # aggregate
    def mean_metric(scores, key):
        return float(np.mean([s[key] for s in scores]))

    result = {
        "schema": "rhan-stage0b-score/v1",
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "evaluation_rows": int(eval_labels.shape[0]),
        "temperatures": temperatures,
        "raw_scores": raw_scores,
        "calibrated_scores": cal_scores,
        "deltas": deltas,
        "bootstrap": bootstrap,
        "decision": decision,
        "aggregate": {
            "raw_ece_mean": mean_metric(raw_scores, "ece_15"),
            "calibrated_ece_mean": mean_metric(cal_scores, "ece_15"),
            "raw_nll_mean": mean_metric(raw_scores, "nll"),
            "calibrated_nll_mean": mean_metric(cal_scores, "nll"),
            "raw_brier_mean": mean_metric(raw_scores, "brier"),
            "calibrated_brier_mean": mean_metric(cal_scores, "brier"),
            "raw_top1_mean": mean_metric(raw_scores, "top1"),
            "calibrated_top1_mean": mean_metric(cal_scores, "top1"),
            "raw_top5_mean": mean_metric(raw_scores, "top5"),
            "calibrated_top5_mean": mean_metric(cal_scores, "top5"),
        },
        "evaluation_images_opened": int(eval_labels.shape[0]),
    }
    write_json(run_dir / "score_result.json", result)
    score_complete = {
        "schema": "rhan-stage0b-score-complete/v1",
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "score_result_sha256": sha256_file(run_dir / "score_result.json"),
        "decision_accepted": bool(decision["accepted"]),
        "evaluation_images_opened": int(eval_labels.shape[0]),
    }
    write_json(run_dir / "score_complete.json", score_complete)
    print(f"Scoring complete. accepted={decision['accepted']}")
    print(f"score_result.json written to {run_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 0B experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight", help="final execution preflight only")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_preflight)
    f = sub.add_parser("fit", help="temperature fitting (requires execution approval)")
    f.add_argument("--out", default=None)
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fit)
    s = sub.add_parser("score", help="one-shot evaluation (requires execution approval)")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_score)
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())