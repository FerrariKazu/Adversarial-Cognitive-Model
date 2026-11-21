#!/usr/bin/env python3
"""Static validation of the Stage 0B freeze bundle.

No images, evaluation labels, temperatures, or Stage 0B outcomes are loaded.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FREEZE = ROOT / "audit" / "stage0b_final_freeze_v1"
DATA_GATE = ROOT / "audit" / "stage0b_data_gate_v1"
OUT = FREEZE / "work" / "static_validation.json"

REQUIRED = [
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
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    checks = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    for relative in REQUIRED:
        path = FREEZE / relative
        check(f"required:{relative}", path.is_file(), str(path.relative_to(ROOT)))

    json_paths = sorted(FREEZE.rglob("*.json"))
    json_errors = []
    for path in json_paths:
        try:
            load(path)
        except Exception as exc:  # noqa: BLE001
            json_errors.append(f"{path.relative_to(FREEZE)}: {type(exc).__name__}: {exc}")
    check("all JSON parses", not json_errors, f"{len(json_paths)} files; errors={json_errors}")

    manifest = load(FREEZE / "STAGE0B_PROTOCOL_MANIFEST.json")
    body = dict(manifest)
    claimed = body.pop("manifest_payload_sha256")
    actual_payload = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    check("manifest payload hash", claimed == actual_payload, f"{claimed} vs {actual_payload}")

    artifact_bad = []
    for name, record in manifest["artifacts"].items():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            artifact_bad.append(name)
    check("manifest artifact hashes", not artifact_bad, f"{len(manifest['artifacts'])} artifacts; bad={artifact_bad}")

    partition_path = DATA_GATE / "PARTITION_MANIFEST.json"
    partition = load(partition_path)
    check(
        "partition PASS and counts",
        partition["status"] == "PASS"
        and partition["counts"]["calibration"] == 5000
        and partition["counts"]["evaluation"] == 5000
        and partition["verification"]["overlap_count"] == 0,
        f"status={partition['status']} counts={partition['counts']}",
    )
    check(
        "partition canonical hash bound",
        manifest["data_binding"]["partition"]["canonical_sha256"]
        == partition["verification"]["manifest_sha256"]
        == "557e10f1b94c161bc5e57926c137faaa27208410743a0a039b5b43be6da52578",
        manifest["data_binding"]["partition"]["canonical_sha256"],
    )
    check(
        "partition file hash bound",
        manifest["data_binding"]["partition"]["manifest_file_sha256"] == sha256_file(partition_path),
        sha256_file(partition_path),
    )

    config = load(FREEZE / "STAGE0B_INFERENCE_CONFIG.json")
    check(
        "inference config partition binding",
        config["data_binding"]["partition_canonical_sha256"]
        == partition["verification"]["manifest_sha256"],
        config["data_binding"]["partition_canonical_sha256"],
    )
    check(
        "inference config head count",
        len(config["classification_heads"]["seeds"]) == 3,
        str(len(config["classification_heads"]["seeds"])),
    )

    head_bad = []
    for head in manifest["classification_heads"]:
        path = ROOT / head["artifact"]
        if sha256_file(path) != head["sha256"]:
            head_bad.append(head["seed"])
    check("three head hashes", len(manifest["classification_heads"]) == 3 and not head_bad, f"bad={head_bad}")

    backbone_path = ROOT / manifest["backbone"]["weights_file"]
    check(
        "backbone file hash",
        sha256_file(backbone_path) == manifest["backbone"]["weights_file_sha256"],
        manifest["backbone"]["weights_file_sha256"],
    )
    check(
        "backbone state hash",
        manifest["backbone"]["state_dict_sha256"]
        == "375b4c89151489e61df6f1f3b3fe1f9fa8b815c52ae0f7117b23e2e201fbb4fe",
        manifest["backbone"]["state_dict_sha256"],
    )

    inference_test = load(FREEZE / "work" / "inference_pipeline_test.json")
    check(
        "inference test PASS and no evaluation",
        inference_test["status"] == "PASS"
        and inference_test["guards"]["evaluation_images_used"] == 0
        and inference_test["guards"]["temperature_fitted"] is False,
        f"status={inference_test['status']} guards={inference_test['guards']}",
    )
    check(
        "inference transform equality",
        inference_test["transform"]["documented_equals_stage0a_make_transform"] is True,
        "documented transform equals Stage 0A make_transform",
    )
    check(
        "three finite head outputs",
        len(inference_test["heads"]) == 3
        and all(row["all_finite"] for row in inference_test["heads"])
        and all(row["logits_shape"] == [8, 1000] for row in inference_test["heads"]),
        "3 heads, 8x1000 finite float32",
    )

    synthetic_cal = load(DATA_GATE / "work" / "synthetic_calibration_test.json")
    synthetic_eval = load(FREEZE / "work" / "synthetic_evaluation_test.json")
    check(
        "synthetic calibration 17/17",
        synthetic_cal["status"] == "PASS" and synthetic_cal["tests_passed"] == 17,
        f"{synthetic_cal['tests_passed']}/{synthetic_cal['tests_total']}",
    )
    check(
        "synthetic evaluation 7/7",
        synthetic_eval["status"] == "PASS" and synthetic_eval["tests_passed"] == 7,
        f"{synthetic_eval['tests_passed']}/{synthetic_eval['tests_total']}",
    )

    stats = load(FREEZE / "work" / "statistical_config.json")
    check(
        "primary endpoint frozen",
        stats["metrics"]["primary"]["name"] == "mean_15bin_ECE_reduction"
        and stats["bootstrap"]["n_boot"] == 10000
        and stats["bootstrap"]["seed"] == 20261121
        and stats["bootstrap"]["method"] == "class-stratified paired percentile bootstrap",
        stats["metrics"]["primary"]["name"],
    )
    check(
        "nine-part decision rule frozen",
        len(stats["decision_rule"]["accept_h1_only_if_all"]) == 9,
        str(len(stats["decision_rule"]["accept_h1_only_if_all"])),
    )

    docs = {
        "prereg": (FREEZE / "STAGE0B_FINAL_PREREGISTRATION.md").read_text(encoding="utf-8"),
        "inference": (FREEZE / "STAGE0B_INFERENCE_PIPELINE_AUDIT.md").read_text(encoding="utf-8"),
        "stats": (FREEZE / "STAGE0B_STATISTICAL_ANALYSIS_PLAN.md").read_text(encoding="utf-8"),
        "checklist": (FREEZE / "STAGE0B_LAUNCH_CHECKLIST.md").read_text(encoding="utf-8"),
        "blockers": (FREEZE / "STAGE0B_UNRESOLVED_BLOCKERS.md").read_text(encoding="utf-8"),
        "verdict": (FREEZE / "STAGE0B_FREEZE_VERDICT.md").read_text(encoding="utf-8"),
    }
    non_launch_markers = ("NOT LAUNCHED", "NOT AUTHORIZED", "NOT EXECUTED")
    check(
        "no-launch statements present",
        all(any(marker in text for marker in non_launch_markers) for text in docs.values()),
        "all primary docs state a non-launch status",
    )
    check(
        "class-stratified bootstrap documented",
        "class-stratified" in docs["prereg"].lower()
        and "class-stratified" in docs["stats"].lower(),
        "prereg and stats use class-stratified language",
    )
    check(
        "evaluation leakage guard documented",
        "evaluation set is not used" in docs["prereg"].lower()
        and "fit_complete.json" in docs["checklist"].lower(),
        "fit-before-score order present",
    )

    # Explicitly ensure no real Stage 0B run output exists in this bundle.
    forbidden = [
        path
        for path in FREEZE.rglob("*")
        if path.is_file()
        and any(part in {"temperatures", "metrics", "logits", "run"} for part in path.parts)
    ]
    check("no Stage 0B run output", not forbidden, f"forbidden={forbidden}")

    status = "PASS" if all(row["passed"] for row in checks) else "FAIL"
    result = {
        "schema": "rhan-stage0b-static-validation/v1",
        "status": status,
        "checks_passed": sum(bool(row["passed"]) for row in checks),
        "checks_total": len(checks),
        "no_real_temperature_fitted": True,
        "no_evaluation_outcomes_loaded": True,
        "checks": checks,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"status={status} checks={result['checks_passed']}/{result['checks_total']}")
    print(f"wrote {OUT}")
    if status != "PASS":
        for row in checks:
            if not row["passed"]:
                print(f"FAIL: {row['name']}: {row['detail']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
