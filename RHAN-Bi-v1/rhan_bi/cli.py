"""Command line interface: demo, verify, stage.

    python -m rhan_bi.cli demo [--config configs/demo.yaml] [--n-train N ...]
    python -m rhan_bi.cli verify
    python -m rhan_bi.cli stage {a,b,c,d,e}
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from rhan_bi import __version__
from rhan_bi.belief import BeliefState
from rhan_bi.config import Config, load_config
from rhan_bi.evaluation.report import write_reports
from rhan_bi.evaluation.runner import run_episode
from rhan_bi.evidence import EvidenceBuilder
from rhan_bi.ledger import EvidenceLedger
from rhan_bi.perception.backbone import HandcraftedBackbone
from rhan_bi.simulator.lab import InspectionLab
from rhan_bi.simulator.scene import make_scene
from rhan_bi.training.data import collect_base_rows, collect_calibration, scene_list
from rhan_bi.training.stages import fit_all, stage_a, stage_b, stage_c, stage_d, stage_e


def _config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    overrides = {}
    for key in ("seed", "n_train", "n_cal", "n_dev", "n_test", "n_gate_e", "budget", "bootstrap"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    if overrides:
        cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, **overrides))
    return cfg


def _demo(args: argparse.Namespace) -> int:
    cfg = _config(args)
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else root / "results"
    print(f"RHAN-Bi {__version__} demo | config {cfg.hash()[:12]} | N={cfg.run.n_train}/"
          f"{cfg.run.n_cal}/{cfg.run.n_dev}/{cfg.run.n_test}")

    backbone = HandcraftedBackbone(cfg.features)
    train = collect_base_rows("train", cfg.run.n_train, cfg, backbone)
    dev = collect_base_rows("dev", cfg.run.n_dev, cfg, backbone)
    cal = collect_calibration(cfg.run.n_cal, cfg, backbone)
    model, forward = fit_all(cal, cfg, backbone)
    print(f"[fit] evidence model {model.hash()} | forward {forward.hash()}")

    a = stage_a(train, dev, cfg)
    print(f"[A] accuracy {a['accuracy']:.3f}  def/ref {a['defect_vs_reflection_balanced_accuracy']:.3f}  "
          f"{'PASS' if a['pass'] else 'FAIL'}")
    b = stage_b(train, dev, cfg, a)
    print(f"[B] IoU@0.3 {b['iou_at_0_3']:.3f}  pointing {b['pointing_accuracy']:.3f}  "
          f"{'PASS' if b['pass'] else 'FAIL'}")
    c = stage_c(cal, dev, model, cfg)
    print(f"[C] ECE {c['ece_after']:.3f}  coverage@risk {c['dev_coverage_at_threshold']:.3f}  "
          f"{'PASS' if c['pass'] else 'FAIL'}")
    d = stage_d(cfg.run.n_dev, cfg, backbone, model)
    print(f"[D] revision {d['arms']['ledger']['revision_accuracy']:.3f} vs pooled "
          f"{d['arms']['pooled']['revision_accuracy']:.3f}  {'PASS' if d['pass'] else 'FAIL'}")
    e = stage_e(cfg, backbone, model, forward, n=cfg.run.n_gate_e, seed=cfg.run.seed)
    print(f"[E] falsification error {e['policies']['falsification']['error']:.3f}  "
          f"EIG {e['policies']['eig']['error']:.3f}  diff {e['comparisons']['eig']['diff']:+.3f}  "
          f"CI [{e['comparisons']['eig']['ci'][0]:+.3f}, {e['comparisons']['eig']['ci'][1]:+.3f}]  "
          f"{e.get('status', 'PASS' if e['pass'] else 'FAIL')}")

    episodes = []
    for scene in scene_list("test", min(12, cfg.run.n_test), cfg):
        ep = run_episode(scene, "falsification", cfg, backbone, model, forward, seed=cfg.run.seed)
        ep.pop("trace", None)
        episodes.append(ep)

    results = {
        "gate_a": a,
        "gate_b": b,
        "gate_c": {k: v for k, v in c.items() if k != "risk_coverage"},
        "gate_d": d,
        "gate_e": e,
        "episodes": episodes,
        "split": {"train": cfg.run.n_train, "cal": cfg.run.n_cal, "dev": cfg.run.n_dev,
                  "test": cfg.run.n_test},
    }
    paths = write_reports(results, out, cfg)
    print(f"[report] {paths['md']}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    cfg = _config(args)
    backbone = HandcraftedBackbone(cfg.features)
    scene = make_scene("dev", 3, cfg.sim)
    lab = InspectionLab(scene, cfg)
    ledger = EvidenceLedger()
    builder = EvidenceBuilder(cfg, backbone, model=None)
    obs = lab.reset()
    ledger.add(builder.build(obs, ledger, lab.episode_id), image=obs.image)
    obs = lab.observe(__import__("rhan_bi.types", fromlist=["Action"]).Action(kind="light", light=2))
    ledger.add(builder.build(obs, ledger, lab.episode_id), image=obs.image)
    report = ledger.verify()
    belief = BeliefState.from_ledger(ledger)
    total = sum(belief.posterior.values())
    assert abs(total - 1.0) < 1e-9, total
    assert "cause" not in ledger.records[0].to_json()
    print(json.dumps({"ledger": report, "posterior_sum": total,
                      "n_eff": ledger.n_eff(), "weights": [r.weight() for r in ledger.records]}, indent=2))
    return 0


def _stage(args: argparse.Namespace) -> int:
    cfg = _config(args)
    backbone = HandcraftedBackbone(cfg.features)
    train = collect_base_rows("train", cfg.run.n_train, cfg, backbone)
    dev = collect_base_rows("dev", cfg.run.n_dev, cfg, backbone)
    cal = collect_calibration(cfg.run.n_cal, cfg, backbone)
    model, forward = fit_all(cal, cfg, backbone)
    if args.name == "a":
        out = stage_a(train, dev, cfg)
        out.pop("predictions", None)
        out.pop("labels", None)
    elif args.name == "b":
        a = stage_a(train, dev, cfg)
        out = stage_b(train, dev, cfg, a)
    elif args.name == "c":
        out = stage_c(cal, dev, model, cfg)
        out.pop("risk_coverage", None)
    elif args.name == "d":
        out = stage_d(cfg.run.n_dev, cfg, backbone, model)
    else:
        out = stage_e(cfg, backbone, model, forward, n=cfg.run.n_gate_e, seed=cfg.run.seed)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rhan_bi", description="RHAN-Bi v1 reference implementation")
    p.add_argument("--config", default=None, help="YAML config (defaults are in rhan_bi/config.py)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-train", dest="n_train", type=int, default=None)
    p.add_argument("--n-cal", dest="n_cal", type=int, default=None)
    p.add_argument("--n-dev", dest="n_dev", type=int, default=None)
    p.add_argument("--n-test", dest="n_test", type=int, default=None)
    p.add_argument("--n-gate-e", dest="n_gate_e", type=int, default=None)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=None)
    sub = p.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run Gates A-E at demo scale and write results")
    demo.add_argument("--out", default=None)
    demo.set_defaults(func=_demo)

    verify = sub.add_parser("verify", help="ledger, hash and posterior checks")
    verify.set_defaults(func=_verify)

    stage = sub.add_parser("stage", help="run a single gate")
    stage.add_argument("name", choices=["a", "b", "c", "d", "e"])
    stage.set_defaults(func=_stage)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
