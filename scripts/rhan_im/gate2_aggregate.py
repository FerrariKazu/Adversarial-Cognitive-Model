#!/usr/bin/env python3
"""Aggregate the three frozen Gate-2 seed runs with the v1.2 2-of-3 rule.

The aggregator refuses runs whose protocol id, code hashes, seeds, or provenance sidecars do
not match.  It does not measure anything and does not read images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rhan_im import gate2 as g2  # noqa: E402


def _read(path: pathlib.Path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _validate_run(run: dict, rules: dict) -> list:
    """Return protocol deviations; an empty list means the run is a valid v1.2 protocol run."""
    err = []
    if run.get("protocol", {}).get("protocol_id") != g2.PROTOCOL_ID:
        err.append("protocol_id")
    if run.get("data", {}).get("mode") != "class_split_file":
        err.append("data.mode")
    if run.get("config", {}).get("self_test"):
        err.append("config.self_test")
    trunk = run.get("trunk", {})
    if trunk.get("sha256") != rules["trunk"]["sha256"] or trunk.get("sha256") != g2.TRUNK_SHA256:
        err.append("trunk.sha256")
    cfg = run.get("config", {})
    checks = {
        "canvas": 224, "window": 112, "stride_units": 2, "op_size": 4,
        "sizes": [2, 4, 6], "fit_null": 850,
        "n_cal_target": 425, "n_cal_null": 425, "n_test_target": 400, "n_test_null": 400,
        "n_dprime_target": 800, "n_dprime_null": 800,
        "n_cal_subset": 300, "bootstrap": 500, "alpha": 0.10, "delta": 0.10, "k_min": 1, "min_pos": 30,
        "lambdas": list(g2.LTT_LAMBDAS),
        "steps": 600, "batch": 48, "lr": 2e-3,
        "head_sizes": [2, 4, 6], "head_alphas": [1.0, 0.6], "p_null": 0.3,
    }
    for key, expected in checks.items():
        if cfg.get(key) != expected:
            err.append(f"config.{key}")
    if int(run.get("seed", -1)) not in (17, 42, 99):
        err.append("seed")
    data = run.get("data", {})
    class_split = data.get("class_split") or {}
    val_split = data.get("val_split") or {}
    holdout = data.get("train_holdout") or {}
    if class_split.get("class_split_sha256") != rules["data_bindings"]["class_split"]["class_split_sha256"]:
        err.append("data.class_split")
    if val_split.get("file_sha256") != rules["data_bindings"]["val_split"]["sha256"]:
        err.append("data.val_split")
    if holdout.get("file_sha256") != rules["data_bindings"]["train_holdout"]["sha256"]:
        err.append("data.train_holdout")
    if holdout.get("holdout_assignment_sha256") != rules["data_bindings"]["train_holdout"]["holdout_assignment_sha256"]:
        err.append("data.holdout_assignment")
    if data.get("head_training_images") != rules["data_bindings"]["train_holdout"]["head_training_images"]:
        err.append("data.head_training_images")
    counts = run.get("counts", {})
    expected_counts = {"fit": (850, 0, 850), "cal": (850, 425, 425), "test": (800, 400, 400), "dprime": (1600, 800, 800)}
    for name, (n, nt, nn) in expected_counts.items():
        c = counts.get(name, {})
        if (c.get("n_scenes"), c.get("n_target"), c.get("n_null")) != (n, nt, nn):
            err.append(f"counts.{name}")
    sensor = run.get("sensor", {})
    if sensor.get("op_size_units") != 4 or sensor.get("op_alpha") != 1.0 or sensor.get("unit_cap") is not None:
        err.append("sensor.operating_point")
    if sensor.get("combination_mode") not in ("sum", "best"):
        err.append("sensor.combination_mode")
    for key in ("d_gist", "kappa", "rho_null", "vif_tok_used", "vif_gist_used"):
        if not _finite(sensor.get(key)):
            err.append(f"sensor.{key}")
    e2 = run.get("e2", {})
    if e2.get("n_scenes") != 800 or not _finite(e2.get("ece_15_K8")) or e2.get("status") not in ("PASS", "INCONCLUSIVE", "FAIL"):
        err.append("e2")
    e5 = run.get("e5", {}).get("selective", {})
    if (e5.get("n_cal_pool"), e5.get("n_cal_subset"), e5.get("bootstrap"), e5.get("k_min")) != (850, 300, 500, 1):
        err.append("e5.counts")
    if e5.get("alpha") != 0.10 or e5.get("delta") != 0.10 or e5.get("lambda_grid") != list(g2.LTT_LAMBDAS):
        err.append("e5.thresholds")
    return err


def _check_sidecar(run_path: pathlib.Path, run: dict) -> dict:
    prov = run.get("provenance", {})
    name = prov.get("file")
    if not name:
        return {"ok": False, "reason": "run has no provenance sidecar file"}
    side_path = run_path.with_name(name)
    if not side_path.is_file():
        return {"ok": False, "reason": f"missing {side_path.name}"}
    side, sha = _read(side_path)
    if sha != prov.get("sidecar_sha256"):
        return {"ok": False, "reason": "sidecar sha256 mismatch"}
    if side.get("link_sha256") != prov.get("link_sha256"):
        return {"ok": False, "reason": "sidecar link_sha256 mismatch"}
    if int(side.get("link", {}).get("seed", -1)) != int(run.get("seed", -2)):
        return {"ok": False, "reason": "sidecar seed does not match the run"}
    return {"ok": True, "file": side_path.name, "sha256": sha, "link_sha256": side.get("link_sha256")}


def aggregate(paths) -> dict:
    rules = json.loads((ROOT / "prereg/gate2_rules_v1_2.json").read_text())
    runs, files, sidecars = [], [], []
    for path in paths:
        p = pathlib.Path(path)
        run, sha = _read(p)
        runs.append(run)
        files.append({"file": str(p), "sha256": sha})
        sidecars.append(_check_sidecar(p, run))
    if any(not s["ok"] for s in sidecars):
        return {"status": "INVALID", "reason": "provenance sidecar check failed", "files": files, "sidecars": sidecars}
    for i, run in enumerate(runs):
        deviations = _validate_run(run, rules)
        if deviations:
            return {"status": "INVALID", "reason": f"run {i} protocol deviation: {deviations}", "files": files, "sidecars": sidecars}
    protocol_ids = {r.get("protocol", {}).get("protocol_id") for r in runs}
    if protocol_ids != {"rhan-im-gate2/1.2"}:
        return {"status": "INVALID", "reason": f"protocol ids {sorted(map(str, protocol_ids))}", "files": files}
    rules_sha = g2.file_sha256(ROOT / "prereg/gate2_rules_v1_2.json")
    if any(r.get("protocol", {}).get("rules_sha256") != rules_sha for r in runs):
        return {"status": "INVALID", "reason": "run rules_sha256 does not match the current v1.2 rules", "files": files}
    code_maps = [r.get("code_sha256", {}) for r in runs]
    if any(cm != code_maps[0] for cm in code_maps[1:]):
        return {"status": "INVALID", "reason": "the three runs used different code hashes", "files": files}
    current_code = {f: g2.file_sha256(ROOT / f) for f in g2.GATE2_CODE_FILES}
    if code_maps[0] != current_code:
        return {"status": "INVALID", "reason": "run code hashes do not match the current measurement code", "files": files}
    seeds = sorted(int(r.get("seed", -1)) for r in runs)
    if seeds != [17, 42, 99]:
        return {"status": "INVALID", "reason": f"expected seeds [17, 42, 99]; got {seeds}", "files": files}
    out = g2.aggregate_runs(runs)
    out.update({
        "protocol_id": "rhan-im-gate2/1.2",
        "rules_sha256": rules_sha,
        "aggregator_sha256": g2.file_sha256(ROOT / "scripts/rhan_im/gate2_aggregate.py"),
        "files": files,
        "sidecars": sidecars,
        "code_sha256": code_maps[0],
        "sensor_summary": [{"seed": r.get("seed"), "combination_mode": r.get("sensor", {}).get("combination_mode"),
                            "kappa": r.get("sensor", {}).get("kappa"), "rho_null": r.get("sensor", {}).get("rho_null"),
                            "vif_tok_used": r.get("sensor", {}).get("vif_tok_used"), "vif_gist_used": r.get("sensor", {}).get("vif_gist_used")}
                           for r in runs],
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=3, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = aggregate(args.runs)
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
