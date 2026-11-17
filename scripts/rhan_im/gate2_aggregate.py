#!/usr/bin/env python3
"""Aggregate the three frozen Gate-2 seed runs with the v1.2 2-of-3 rule.

The aggregator refuses runs whose protocol id, code hashes, seeds, or provenance sidecars do
not match.  It does not measure anything and does not read images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rhan_im import gate2 as g2  # noqa: E402


def _read(path: pathlib.Path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


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
    runs, files, sidecars = [], [], []
    for path in paths:
        p = pathlib.Path(path)
        run, sha = _read(p)
        runs.append(run)
        files.append({"file": str(p), "sha256": sha})
        sidecars.append(_check_sidecar(p, run))
    if any(not s["ok"] for s in sidecars):
        return {"status": "INVALID", "reason": "provenance sidecar check failed", "files": files, "sidecars": sidecars}
    protocol_ids = {r.get("protocol", {}).get("protocol_id") for r in runs}
    if protocol_ids != {"rhan-im-gate2/1.2"}:
        return {"status": "INVALID", "reason": f"protocol ids {sorted(map(str, protocol_ids))}", "files": files}
    rules_sha = g2.file_sha256(ROOT / "prereg/gate2_rules_v1_2.json")
    if any(r.get("protocol", {}).get("rules_sha256") != rules_sha for r in runs):
        return {"status": "INVALID", "reason": "run rules_sha256 does not match the current v1.2 rules", "files": files}
    code_maps = [r.get("code_sha256", {}) for r in runs]
    if any(cm != code_maps[0] for cm in code_maps[1:]):
        return {"status": "INVALID", "reason": "the three runs used different code hashes", "files": files}
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
