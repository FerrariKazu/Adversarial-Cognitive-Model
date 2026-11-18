"""Generate the machine-readable and human-readable run reports (docs/12 §3)."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rhan_bi import __version__
from rhan_bi.config import Config
from rhan_bi.util import canonical_json, sha256_text


def git_commit(cwd: str | Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True,
                             text=True, check=False)
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def manifest(config: Config, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "package_version": __version__,
        "config_hash": config.hash(),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config.to_dict(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }


def write_reports(results: dict[str, Any], out_dir: str | Path, config: Config) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest(config), **results}
    (out / "demo_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md = render_markdown(payload)
    (out / "demo_report.md").write_text(md)
    (out / "episodes.jsonl").write_text(
        "\n".join(json.dumps(ep, sort_keys=True) for ep in results.get("episodes", [])) + "\n"
    )
    return {"json": str(out / "demo_report.json"), "md": str(out / "demo_report.md"),
            "episodes": str(out / "episodes.jsonl")}


def render_markdown(payload: dict[str, Any]) -> str:
    m = payload["manifest"]
    a = payload.get("gate_a", {})
    b = payload.get("gate_b", {})
    c = payload.get("gate_c", {})
    d = payload.get("gate_d", {})
    e = payload.get("gate_e", {})

    def fmt(x: Any, nd: int = 3) -> str:
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    lines: list[str] = []
    lines.append("# RHAN-Bi v1 — demo report\n")
    lines.append("> **Claim level: feasibility.** This is a demo-scale run, not the pre-registered "
                 "confirmatory N=400 Gate E. No result here is a confirmed claim.\n")
    lines.append(f"- package `{m['package_version']}`, config `{m['config_hash'][:12]}`, "
                 f"git `{m['git_commit'][:12]}`, python {m['python']}")
    lines.append(f"- seeds: {m['config']['run']['seed']}, "
                 f"N train/cal/dev/test = {m['config']['run']['n_train']}/"
                 f"{m['config']['run']['n_cal']}/{m['config']['run']['n_dev']}/"
                 f"{m['config']['run']['n_test']}, budget {m['config']['run']['budget']}")
    lines.append("")

    lines.append("## Gate A — single-observation competence\n")
    lines.append(f"- accuracy **{fmt(a.get('accuracy'))}**, balanced **{fmt(a.get('balanced_accuracy'))}**")
    lines.append(f"- defect-vs-reflection balanced accuracy **{fmt(a.get('defect_vs_reflection_balanced_accuracy'))}** "
                 f"(required band 0.55–0.80: the task must remain ambiguous)")
    lines.append(f"- confusion (rows truth, cols predicted, labels {a.get('confusion', {}).get('labels')}): "
                 f"`{a.get('confusion', {}).get('matrix')}`")
    lines.append(f"- gate: **{'PASS' if a.get('pass') else 'FAIL'}**\n")

    lines.append("## Gate B — spatial evidence\n")
    lines.append(f"- IoU@0.3 **{fmt(b.get('iou_at_0_3'))}**, mean IoU {fmt(b.get('mean_iou'))}, "
                 f"pointing **{fmt(b.get('pointing_accuracy'))}**, localization-conditioned "
                 f"{fmt(b.get('localization_conditioned_accuracy'))}")
    lines.append(f"- gate: **{'PASS' if b.get('pass') else 'FAIL'}**\n")

    lines.append("## Gate C — reliability and coverage\n")
    lines.append(f"- temperature **{fmt(c.get('temperature'))}**; ECE {fmt(c.get('ece_before'))} → "
                 f"**{fmt(c.get('ece_after'))}**; Brier {fmt(c.get('brier'))}; NLL {fmt(c.get('nll'))}")
    lines.append(f"- selective risk at coverage 0.5: **{fmt(c.get('selective_risk_at_coverage_0_5'))}**")
    lines.append(f"- conformal threshold {fmt(c.get('conformal_threshold'))}: calibration coverage "
                 f"{fmt(c.get('calibration_coverage_at_threshold'))}, dev coverage "
                 f"**{fmt(c.get('dev_coverage_at_threshold'))}**, dev selective risk "
                 f"{fmt(c.get('dev_selective_risk_at_threshold'))}")
    lines.append(f"- gate: **{'PASS' if c.get('pass') else 'FAIL'}** "
                 f"(coverage floor 0.40 at risk target {fmt(c.get('risk_target'))})\n")

    lines.append("## Gate D — belief revision and dependence\n")
    for arm, vals in d.get("arms", {}).items():
        lines.append(f"- `{arm}`: accuracy {fmt(vals['accuracy'])}, ECE {fmt(vals['ece'])}, "
                     f"revision accuracy {fmt(vals['revision_accuracy'])}, "
                     f"mean confidence {fmt(vals['mean_confidence'])}")
    lines.append(f"- revision gain vs pooled: **{fmt(d.get('revision_gain_vs_pooled'))}**; "
                 f"confidence gap (ledger − no_dep) {fmt(d.get('confidence_gap_ledger_minus_no_dep'))}; "
                 f"median n_eff {fmt(d.get('n_eff_median'))}")
    lines.append(f"- gate: **{'PASS' if d.get('pass') else 'FAIL'}**\n")

    lines.append("## Gate E — matched-budget policy comparison\n")
    lines.append("| policy | error | accuracy | mean cost | mean confidence | ECE |")
    lines.append("|---|---|---|---|---|---|")
    for pol, vals in e.get("policies", {}).items():
        lines.append(f"| {pol} | {fmt(vals['error'])} | {fmt(vals['accuracy'])} | "
                     f"{fmt(vals['mean_cost'], 2)} | {fmt(vals['mean_confidence'])} | {fmt(vals['ece'])} |")
    lines.append("")
    for other, cmp in e.get("comparisons", {}).items():
        lines.append(f"- falsification − {other}: **{fmt(cmp['diff'])}** "
                     f"(95% CI [{fmt(cmp['ci'][0])}, {fmt(cmp['ci'][1])}]), "
                     f"McNemar p = {fmt(cmp['mcnemar_p'], 4)}")
    lines.append(f"- oracle error {fmt(e.get('headroom', {}).get('oracle_error'))}, "
                 f"best deployable {fmt(e.get('headroom', {}).get('best_deployable_error'))}, "
                 f"headroom gap {fmt(e.get('headroom', {}).get('gap'))}")
    lines.append(f"- gate: **{e.get('status', 'PASS' if e.get('pass') else 'FAIL')}** "
                 f"(feasibility, not confirmatory)\n")

    lines.append("## Limitations (required)\n")
    lines.append("- synthetic laboratory only; no real-image result in this run")
    lines.append("- defect-vs-reflection ambiguity is a property of the simulator's action set")
    lines.append("- abstention is counted as an error in the primary endpoint; coverage is reported separately")
    lines.append("- the planner's forward model and stopping rule share errors (risk R5)")
    lines.append("- the evidence ledger records provenance, not truth")
    lines.append("")
    return "\n".join(lines)
