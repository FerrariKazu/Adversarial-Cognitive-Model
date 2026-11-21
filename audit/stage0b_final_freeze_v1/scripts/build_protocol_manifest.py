#!/usr/bin/env python3
"""Build the Stage 0B protocol manifest from verified artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FREEZE = ROOT / "audit" / "stage0b_final_freeze_v1"
DATA_GATE = ROOT / "audit" / "stage0b_data_gate_v1"
STAGE0A = ROOT / "audit" / "stage0a_closeout_v1"
PREP = ROOT / "audit" / "stage0b_preparation_v1"
OUT = FREEZE / "STAGE0B_PROTOCOL_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def entry(path: Path, *, schema: str | None = None) -> dict:
    record = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if schema:
        record["schema"] = schema
    return record


def main() -> int:
    inference_config_path = FREEZE / "STAGE0B_INFERENCE_CONFIG.json"
    stats_path = FREEZE / "work" / "statistical_config.json"
    env_path = FREEZE / "work" / "environment.json"
    partition_path = DATA_GATE / "PARTITION_MANIFEST.json"
    inference_config = load(inference_config_path)
    stats = load(stats_path)
    env = load(env_path)
    partition = load(partition_path)
    if partition.get("status") != "PASS":
        raise SystemExit("partition is not PASS")
    if inference_config["data_binding"]["partition_canonical_sha256"] != partition["verification"]["manifest_sha256"]:
        raise SystemExit("partition canonical hash mismatch")

    head_specs = inference_config["classification_heads"]["seeds"]
    heads = []
    for spec in head_specs:
        path = ROOT / spec["path"]
        sidecar = path.with_suffix(path.suffix + ".sha256")
        actual = sha256_file(path)
        sidecar_value = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if actual != spec["sha256"] or sidecar_value != spec["sha256"]:
            raise SystemExit(f"head hash mismatch for seed {spec['seed']}")
        heads.append(
            {
                "seed": spec["seed"],
                "artifact": spec["path"],
                "sha256": actual,
                "sidecar_sha256": sidecar_value,
                "sidecar_matches": True,
                "model_keys": ["fc.bias", "fc.weight"],
                "fc_weight_shape": [1000, 768],
                "fc_bias_shape": [1000],
                "checkpoint_role": "final.pt; reported model",
                "best_pt_used": False,
            }
        )

    backbone_file = FREEZE / "work" / "backbone_cache" / "vit_base_patch14_dinov2.lvd142m" / "model.safetensors"
    backbone_config = backbone_file.parent / "config.json"
    if sha256_file(backbone_file) != inference_config["backbone"]["weights_file_sha256"]:
        raise SystemExit("backbone file hash mismatch")

    validation_files = {
        "inference_pipeline_test": FREEZE / "work" / "inference_pipeline_test.json",
        "compute_probe": FREEZE / "work" / "compute_probe.json",
        "synthetic_evaluation_test": FREEZE / "work" / "synthetic_evaluation_test.json",
        "synthetic_calibration_test": DATA_GATE / "work" / "synthetic_calibration_test.json",
    }
    validation = {}
    for key, path in validation_files.items():
        data = load(path)
        validation[key] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "status": data.get("status"),
        }

    calibration_core = DATA_GATE / "scripts" / "calibration_core.py"
    evaluation_core = FREEZE / "scripts" / "stage0b_evaluation_core.py"
    inference_script = FREEZE / "scripts" / "verify_inference_pipeline.py"
    stats_file_hash = sha256_file(stats_path)
    env_file_hash = sha256_file(env_path)

    artifacts = {}
    artifact_paths = {
        "stage0a_closeout_report": STAGE0A / "STAGE0A_CLOSEOUT_REPORT.md",
        "stage0a_independent_audit": STAGE0A / "STAGE0A_INDEPENDENT_AUDIT.json",
        "stage0a_protocol_compliance": STAGE0A / "STAGE0A_PROTOCOL_COMPLIANCE.md",
        "stage0a_unresolved_blockers": STAGE0A / "UNRESOLVED_BLOCKERS.md",
        "stage0b_historical_proposal": PREP / "STAGE0B_FINAL_PREREGISTRATION_PROPOSAL.md",
        "data_license_review": DATA_GATE / "DATA_LICENSE_AND_USAGE_REVIEW.md",
        "dataset_download_verification": DATA_GATE / "DATASET_DOWNLOAD_VERIFICATION.json",
        "real_dataset_audit": DATA_GATE / "REAL_DATASET_AUDIT.json",
        "real_dataset_audit_md": DATA_GATE / "REAL_DATASET_AUDIT.md",
        "class_mapping_verification": DATA_GATE / "CLASS_MAPPING_VERIFICATION.json",
        "partition_manifest": partition_path,
        "partition_verification": DATA_GATE / "PARTITION_VERIFICATION.md",
        "data_gate_verdict": DATA_GATE / "STAGE0B_DATA_GATE_VERDICT.md",
        "data_gate_environment": DATA_GATE / "ENVIRONMENT_RECORD.json",
        "stage0a_data_source": ROOT / "stage0a" / "data.py",
        "stage0a_features_source": ROOT / "stage0a" / "features.py",
        "stage0a_config_source": ROOT / "stage0a" / "config.py",
        "stage0a_evaluation_source": ROOT / "stage0a" / "evaluation.py",
        "stage0b_inference_config": inference_config_path,
        "stage0b_inference_script": inference_script,
        "stage0b_calibration_core": calibration_core,
        "stage0b_evaluation_core": evaluation_core,
        "stage0b_statistical_config": stats_path,
        "stage0b_environment": env_path,
        "stage0b_preregistration": FREEZE / "STAGE0B_FINAL_PREREGISTRATION.md",
        "stage0b_inference_audit": FREEZE / "STAGE0B_INFERENCE_PIPELINE_AUDIT.md",
        "stage0b_data_provenance": FREEZE / "STAGE0B_DATA_PROVENANCE.md",
        "stage0b_statistical_plan": FREEZE / "STAGE0B_STATISTICAL_ANALYSIS_PLAN.md",
        "stage0b_execution_plan": FREEZE / "STAGE0B_EXECUTION_PLAN.md",
        "stage0b_launch_checklist": FREEZE / "STAGE0B_LAUNCH_CHECKLIST.md",
        "stage0b_unresolved_blockers": FREEZE / "STAGE0B_UNRESOLVED_BLOCKERS.md",
        "stage0b_freeze_verdict": FREEZE / "STAGE0B_FREEZE_VERDICT.md",
    }
    for key, path in artifact_paths.items():
        if not path.exists():
            raise SystemExit(f"missing artifact {key}: {path}")
        artifacts[key] = entry(path)

    payload = {
        "schema": "rhan-stage0b-protocol-manifest/v1",
        "protocol_id": inference_config["protocol_id"],
        "protocol_version": inference_config["protocol_version"],
        "freeze_status": "READY_FOR_HUMAN_PROTOCOL_FREEZE",
        "execution_authorized": False,
        "real_temperature_fitted": False,
        "evaluation_outcomes_inspected": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope_note": "Binds the frozen Stage 0B design, data partition, model artifacts, implementation, environment, and statistics. It does not authorize execution.",
        "supersedes_for_stage0b": {
            "path": str((PREP / "STAGE0B_FINAL_PREREGISTRATION_PROPOSAL.md").relative_to(ROOT)),
            "sha256": sha256_file(PREP / "STAGE0B_FINAL_PREREGISTRATION_PROPOSAL.md"),
            "reason": "The historical proposal uses the obsolete path-based split and unstratified image bootstrap. The frozen Stage 0B protocol uses the verified content-hash partition and class-stratified paired bootstrap.",
        },
        "parent_stage0a": {
            "protocol_id": "rhan-stage0a-dinov2/1",
            "protocol_version": "1.0.0",
            "closeout_verdict": "CONDITIONAL PASS",
            "training_code_commit": "a23dc0f76f5ba2514a3c6a42035d28b44a4ae17e",
            "artifact_finalization_commit": "d610dd4d8bdbf4b51f34f1175e57902de64687af",
            "closeout_report": artifacts["stage0a_closeout_report"],
            "independent_audit": artifacts["stage0a_independent_audit"],
            "image_level_replay": "BLOCKED — not completed; not claimed by Stage 0B",
        },
        "backbone": {
            "timm_name": inference_config["backbone"]["timm_name"],
            "hf_repo": inference_config["backbone"]["hf_repo"],
            "hf_revision": inference_config["backbone"]["hf_revision"],
            "weights_file": str(backbone_file.relative_to(ROOT)),
            "weights_file_sha256": sha256_file(backbone_file),
            "hf_config_sha256": sha256_file(backbone_config),
            "state_dict_sha256": inference_config["backbone"]["state_dict_sha256"],
            "architecture": inference_config["backbone"]["architecture"],
            "feature_layer": inference_config["backbone"]["feature_layer"],
            "pooling": inference_config["backbone"]["pooling"],
            "feature_dimension": 768,
        },
        "classification_heads": heads,
        "data_binding": {
            "dataset": "ImageNetV2 MatchedFrequency",
            "source_revision": "d626240be2538720e83103a0e1178d24aca8b12c",
            "archive_sha256": "f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c",
            "archive_size_bytes": 1264079360,
            "archive_container_detected": "uncompressed POSIX tar",
            "archive_extension_matches_container": False,
            "real_dataset_audit": artifacts["real_dataset_audit"],
            "class_mapping_verification": artifacts["class_mapping_verification"],
            "download_verification": artifacts["dataset_download_verification"],
            "partition": {
                "manifest_file": str(partition_path.relative_to(ROOT)),
                "manifest_file_sha256": sha256_file(partition_path),
                "canonical_sha256": partition["verification"]["manifest_sha256"],
                "repeat_canonical_sha256": partition["verification"]["repeat_manifest_sha256"],
                "seed": partition["seed"],
                "calibration": partition["counts"]["calibration"],
                "evaluation": partition["counts"]["evaluation"],
                "per_class_calibration": partition["counts"]["per_class_calibration"],
                "per_class_evaluation": partition["counts"]["per_class_evaluation"],
                "overlap_count": partition["verification"]["overlap_count"],
                "disjoint": partition["verification"]["disjoint"],
                "algorithm": partition["algorithm"],
            },
        },
        "implementations": {
            "inference": {
                "config": artifacts["stage0b_inference_config"],
                "script": artifacts["stage0b_inference_script"],
                "transform_source": artifacts["stage0a_data_source"],
                "feature_source": artifacts["stage0a_features_source"],
                "head_source": artifacts["stage0a_config_source"],
                "evaluation_source": artifacts["stage0a_evaluation_source"],
            },
            "calibration": {
                "path": str(calibration_core.relative_to(ROOT)),
                "sha256": sha256_file(calibration_core),
                "scope": "temperature fitting and metric primitives; synthetic-tested; no real temperature fitted",
            },
            "evaluation": {
                "path": str(evaluation_core.relative_to(ROOT)),
                "sha256": sha256_file(evaluation_core),
                "scope": "frozen metric and class-stratified paired-bootstrap implementation; synthetic-tested",
            },
            "statistics": {
                "path": str(stats_path.relative_to(ROOT)),
                "sha256": stats_file_hash,
                "config": stats,
            },
            "environment": {
                "path": str(env_path.relative_to(ROOT)),
                "sha256": env_file_hash,
                "fingerprint_sha256": env["fingerprint_sha256"],
                "software": env["software"],
                "gpu": env["gpu"],
                "platform": env["platform"],
                "git_commit": env["git_commit"],
            },
        },
        "validation": validation,
        "deliverables": [
            "STAGE0B_FINAL_PREREGISTRATION.md",
            "STAGE0B_PROTOCOL_MANIFEST.json",
            "STAGE0B_INFERENCE_PIPELINE_AUDIT.md",
            "STAGE0B_INFERENCE_CONFIG.json",
            "STAGE0B_DATA_PROVENANCE.md",
            "STAGE0B_STATISTICAL_ANALYSIS_PLAN.md",
            "STAGE0B_EXECUTION_PLAN.md",
            "STAGE0B_LAUNCH_CHECKLIST.md",
            "STAGE0B_UNRESOLVED_BLOCKERS.md",
            "STAGE0B_FREEZE_VERDICT.md",
            "MANIFEST.sha256",
        ],
        "launch_blockers": [
            "human protocol-freeze approval not recorded",
            "separate execution approval not recorded",
            "supervisor/institutional data-use confirmation not recorded",
            "no upload/publication rights for image-derived artifacts",
            "runtime must be re-verified on the execution host",
        ],
        "artifacts": artifacts,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"protocol={payload['protocol_id']} version={payload['protocol_version']}")
    print(f"manifest_payload_sha256={payload['manifest_payload_sha256']}")
    print(f"artifacts={len(artifacts)} heads={len(heads)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
