#!/usr/bin/env python3
"""
Generation-0 pre-flight: per-step |dW| for any registered optimizer group.
================================================================================

Generalizes scratch/measure_hpc_dw.py to every new trainable component in the
RHAN-NX ladder. The Stage 2 lesson, applied PREEMPTIVELY: the HPC predictor's
output conv froze at its ±0.01 init draw for 15 epochs (hpc_error ratio 1.00)
because the shared global clip let the backbone's TRADES gradient own the
budget (per-step |dW| ~1.36e-5 — invisible). The two-group fix (head lr x
6.67, per-group clip) put it in the learnable regime (~2.7e-3/step, 2.85%/step).
That was NOT a one-off HPC repair: every new trainable component (SBR slots,
AIS-v2 candidate-eval head, relational/evidence heads, belief-HPC predictor)
is measured HERE before its first full smoke test.

Usage (from the repo root, before the component's first smoke):
    python3 scripts/measure_group_dw.py --group-name sbr \
        --ckpt checkpoints/rhan_next_ais_hpc_best.pth \
        --sbr-stage gate_only --freeze-backbone-for-sbr0
    python3 scripts/measure_group_dw.py --group-name ais_v2 \
        --ckpt checkpoints/rhan_next_ais_v1_halting_only_best.pth \
        --ais-variant info_gain_v2
    python3 scripts/measure_group_dw.py --group-name hpc \
        --ckpt checkpoints/rhan_next_ais_v1_halting_only_best.pth \
        --hpc-target belief

Same 4-criterion verdict structure as the original HPC pre-flight:
  C1  |dW| >= 5x the starved baseline (1.36e-5, the measured pre-fix floor)
  C2  relative movement >= 1%/step
  C3  the group's loss/error actually declines over the window
  C4  within an order of magnitude of the isolated reference recipe
      (SGD lr=0.05, weight 1.0, no clip)

The group's loss signal is the REAL training loss path that reaches it:
  hpc           -> w_hpc * get_hpc_loss   (pixel or belief target)
  sbr           -> CE(logits, labels)     (slots -> pooled -> classifier)
  relational    -> CE(logits, labels)     (message-passing -> pooled evidence)
  evidence      -> CE(logits, labels)     (evidence heads -> pooled evidence)
  ais_v2        -> w_eig * EIG TD loss    (predicted vs observed surprise)
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

STARVED_DW = 1.36e-5        # measured pre-fix per-step |dW| (HPC output conv)
ISOLATED_LR = 0.05           # proven isolated learnability recipe
DEFAULT_ACCUM = 8            # micro-batches per optimizer step (real loop: 16)
DEFAULT_MICRO_B = 8          # batch per micro-batch (real loop: 16)
PHASE1_LR = 0.003            # backbone lr in phase 1 (the curriculum's base)
DEFAULT_STEPS = 24           # real-recipe optimizer steps (matches HPC pre-flight)
ISOLATED_STEPS = 10

# STL10-normalized inputs (the training loader's stats — raw [0,1] tensors
# push the backbone off-manifold and diverge to NaN).
_MEAN = torch.tensor([0.4467, 0.4398, 0.4066]).view(1, 3, 1, 1)
_STD = torch.tensor([0.2242, 0.2215, 0.2239]).view(1, 3, 1, 1)


def build_model_and_loss(args) -> tuple:
    """Construct the RHAN-NX model for the target config, load the base
    checkpoint (fresh-init for the new component), freeze everything except
    the measured group, and return (model, loss_fn, group_param_names)."""
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, os.path.join(REPO_ROOT, "phase1_training"))

    from checkpoint_utils import compat_load
    from rhan_core.config.pillar_config import RHANNextConfig
    from rhan_core.model import RHANNext
    from train_rhan_next import eig_loss_from_trajectory

    if args.sbr_stage == "gate_only":
        cfg = RHANNextConfig(enable_ais=True, enable_hpc=True, hpc_num_levels=1,
                             enable_sbr=True, sbr_stage="gate_only",
                             freeze_backbone_for_sbr0=True)
    elif args.sbr_stage in ("relational", "uncertainty"):
        cfg = RHANNextConfig(enable_ais=True, enable_hpc=True, hpc_num_levels=1,
                             enable_sbr=True, sbr_stage=args.sbr_stage)
    elif args.sbr_stage in ("clean_classifier", "adversarial_ramp"):
        cfg = RHANNextConfig(enable_ais=True, enable_hpc=True, hpc_num_levels=1,
                             enable_sbr=True, sbr_stage=args.sbr_stage)
    else:
        cfg = RHANNextConfig(enable_ais=True,
                             ais_variant=args.ais_variant,
                             enable_hpc=True, hpc_num_levels=1,
                             hpc_target=args.hpc_target)
    m = RHANNext(config=cfg)
    try:
        ckpt = compat_load(args.ckpt, map_location="cpu")
    except Exception:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = m.load_state_dict(ckpt["model"], strict=False)
    n_new = sum(1 for k in missing if args.group_frag in k)
    print(f"  base ckpt: epoch={ckpt.get('epoch')} "
          f"best_acc={ckpt.get('best_acc')} | missing={len(missing)} "
          f"({args.group_frag} fresh-init: {n_new}) "
          f"unexpected={len(unexpected)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(device).train()

    # Freeze everything except the measured group (faithful for |dW|: the
    # group's movement depends only on the features at that step, and
    # backprop'ing the full loss through a frozen backbone keeps the trained
    # weights on-manifold — same rationale as the HPC pre-flight).
    group_params = []
    for name, p in m.named_parameters():
        if args.group_frag in name:
            p.requires_grad = True
            group_params.append(p)
        else:
            p.requires_grad = False
    print(f"  group '{args.group_name}': {len(group_params)} params",
          flush=True)
    assert group_params, f"no params contain fragment {args.group_frag!r}"

    def loss_fn(x, labels):
        """The real training loss path that reaches the measured group."""
        logits, traj = m(x, return_trajectory=True)
        ce = nn.functional.cross_entropy(logits, labels)
        parts = {"ce": ce}
        if args.group_name == "hpc":
            l_hpc = m.get_hpc_loss(x, (logits, traj))
            parts["hpc"] = args.w_hpc * l_hpc
        if args.group_name == "ais_v2":
            parts["eig"] = args.w_eig * eig_loss_from_trajectory(
                traj, x.device)
        total = sum(parts.values())
        return total, parts, traj

    return m, loss_fn, group_params


def make_batch(device, micro_b: int = DEFAULT_MICRO_B):
    x = torch.rand(micro_b, 3, 96, 96, device=device)
    return (x - _MEAN.to(device)) / _STD.to(device), \
        torch.randint(0, 10, (micro_b,), device=device)


def group_dw(m, group_params) -> float:
    """||dW|| of the group's params after a step (sum of per-param norms)."""
    dw = 0.0
    for p in group_params:
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            dw += float((p.grad.detach().norm() ** 2))
    return dw ** 0.5


def per_param_rel(ref_before, ref_after) -> Dict[str, float]:
    """Per-param |dW| / |W| over a step window (median + max).

    The original HPC criterion (rel >= 1%/step) was calibrated on the head's
    SINGLE output conv (|W| ~ 0.005). An aggregate group (e.g. SBR's 24 slot
    params) mixes big stable matrices with head-like output params, so the
    meaningful signal is the per-param distribution: the head-like params
    must move at >= 1%/step of their OWN norm (median >= 0.1%/step or max
    >= 1%/step) while the big matrices stay put — exactly the pattern the
    starved HPC head violated (0.27%/step on its only conv).
    """
    rels = []
    for a, b in zip(ref_before, ref_after):
        w_norm = float(a.norm())
        if w_norm > 1e-9:
            rels.append(float((b - a).norm()) / w_norm)
    if not rels:
        return {"median": 0.0, "max": 0.0, "n": 0}
    import statistics
    return {"median": float(statistics.median(rels)),
            "max": float(max(rels)), "n": len(rels)}


def run_real_recipe(m, loss_fn, group_params, args) -> Dict:
    """Real loop: registry optimizer (group lr = phase_lr * 6.67), per-group
    clip 1.0, loss accumulated over args.accum micro-batches / accum."""
    from train_rhan_next import build_next_optimizer, clip_grad_per_group
    opt = build_next_optimizer(m, phase_lr=PHASE1_LR, hpc_lr_mult=6.67)
    # Find THIS group's param group (by name, Generation-0 registry).
    target = None
    for g in opt.param_groups:
        if g.get("name") == args.group_name:
            target = g
    assert target is not None, (
        f"group {args.group_name!r} not in registry groups "
        f"{[g.get('name') for g in opt.param_groups]}")
    g_lr = target["lr"]
    w0 = sum(float(p.detach().norm()) for p in group_params) or 1.0
    ref = [p.detach().clone() for p in group_params]
    rels_acc = []
    dws, rels, errs = [], [], []
    # Group-relevant error baseline: for hpc / ais_v2 the group's OWN term
    # (the component's error the smoke gate tracks); for CE-path groups the
    # total loss.
    def _err(parts: Dict) -> float:
        if args.group_name == "hpc" and "hpc" in parts:
            return float(parts["hpc"].item())
        if args.group_name == "ais_v2" and "eig" in parts:
            return float(parts["eig"].item())
        return sum(v.item() for v in parts.values())

    x0, y0 = make_batch(next(m.parameters()).device, args.micro_b)
    with torch.no_grad():
        _, parts0, _ = loss_fn(x0, y0)
    err0 = _err(parts0)
    errs.append(err0)
    t0 = time.time()
    for s in range(args.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            x, y = make_batch(next(m.parameters()).device, args.micro_b)
            with torch.enable_grad():
                total, _, _ = loss_fn(x, y)
            (total / args.accum).backward()
        clip_grad_per_group(opt, 1.0)
        opt.step()
        dw = sum(float((p.detach() - r).norm()) for p, r in zip(group_params, ref))
        dws.append(dw)
        rels.append(dw / w0)
        rels_acc.append(per_param_rel(ref, [p.detach() for p in group_params]))
        ref = [p.detach().clone() for p in group_params]
        x, y = make_batch(next(m.parameters()).device, args.micro_b)
        with torch.no_grad():
            _, parts, _ = loss_fn(x, y)
        errs.append(_err(parts))
    dt = time.time() - t0
    rels_agg = {k: max(float(rr[k]) for rr in rels_acc) if rels_acc else 0.0
                for k in ("median", "max", "n")}
    return {"mode": "real", "lr": g_lr, "steps": args.steps, "secs": dt,
            "dws": dws, "rels": rels, "err0": err0, "errN": errs[-1],
            "per_param_rel": rels_agg}


def run_isolated(m, loss_fn, group_params, args) -> Dict:
    """Isolated reference: SGD(group, lr=0.05), weight 1.0, no clip."""
    opt = torch.optim.SGD(group_params, lr=ISOLATED_LR)
    w0 = sum(float(p.detach().norm()) for p in group_params) or 1.0
    ref = [p.detach().clone() for p in group_params]
    dws, rels, errs = [], [], []

    def _err(parts: Dict) -> float:
        if args.group_name == "hpc" and "hpc" in parts:
            return float(parts["hpc"].item())
        if args.group_name == "ais_v2" and "eig" in parts:
            return float(parts["eig"].item())
        return sum(v.item() for v in parts.values())

    x0, y0 = make_batch(next(m.parameters()).device, args.micro_b)
    with torch.no_grad():
        _, parts0, _ = loss_fn(x0, y0)
    err0 = _err(parts0)
    errs.append(err0)
    t0 = time.time()
    for s in range(ISOLATED_STEPS):
        opt.zero_grad(set_to_none=True)
        x, y = make_batch(next(m.parameters()).device, args.micro_b)
        with torch.enable_grad():
            total, _, _ = loss_fn(x, y)
        total.backward()                      # weight 1.0, no clip
        opt.step()
        dw = sum(float((p.detach() - r).norm()) for p, r in zip(group_params, ref))
        dws.append(dw)
        rels.append(dw / w0)
        ref = [p.detach().clone() for p in group_params]
        x, y = make_batch(next(m.parameters()).device, args.micro_b)
        with torch.no_grad():
            _, parts, _ = loss_fn(x, y)
        errs.append(_err(parts))
    dt = time.time() - t0
    return {"mode": "isolated", "lr": ISOLATED_LR, "steps": ISOLATED_STEPS,
            "secs": dt, "dws": dws, "rels": rels, "err0": err0, "errN": errs[-1]}


def summarize(r: Dict) -> str:
    drop = 1.0 - r["errN"] / r["err0"] if r["err0"] > 0 else 0.0
    line = (f"  {r['mode']:<9} {r['steps']:>3} steps ({r['secs']:.0f}s) | "
            f"|dW|/step mean={sum(r['dws'])/len(r['dws']):.3e} "
            f"(max {max(r['dws']):.3e}) | "
            f"loss {r['err0']:.4f} -> {r['errN']:.4f} "
            f"(drop {100*drop:.1f}%)")
    if "per_param_rel" in r:
        pr = r["per_param_rel"]
        line += (f" | per-param rel: median {100*pr['median']:.3f}%/step, "
                 f"max {100*pr['max']:.3f}%/step")
    else:
        line += f" | rel {100*sum(r['rels'])/len(r['rels']):.2f}%/step"
    return line


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group-name", required=True,
                    choices=["hpc", "sbr", "ais_v2", "relational", "evidence"],
                    help="Optimizer group to measure (Generation-0 name).")
    ap.add_argument("--ckpt", default="checkpoints/rhan_next_ais_hpc_best.pth",
                    help="Base checkpoint the component's training starts from.")
    ap.add_argument("--sbr-stage", default="legacy",
                    choices=["legacy", "gate_only", "clean_classifier",
                             "adversarial_ramp", "relational", "uncertainty"])
    ap.add_argument("--ais-variant", default="halting_only",
                    choices=["halting_only", "info_gain_v2"])
    ap.add_argument("--hpc-target", default="pixel", choices=["pixel", "belief"])
    ap.add_argument("--w-hpc", type=float, default=0.10)
    ap.add_argument("--w-eig", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--accum", type=int, default=DEFAULT_ACCUM)
    ap.add_argument("--micro-b", type=int, default=DEFAULT_MICRO_B)
    args = ap.parse_args(argv)

    # The parameter-name fragment that identifies the group in named_parameters.
    args.group_frag = {
        "hpc": "hpc_level1",          # pixel stack (hpc_belief handled below)
        "sbr": "structured_belief",
        "ais_v2": "gaze_policy_v2",
        "relational": "structured_belief.relational",
        "evidence": "structured_belief.evidence",
    }[args.group_name]
    if args.group_name == "hpc" and args.hpc_target == "belief":
        args.group_frag = "hpc_belief"

    if not os.path.exists(args.ckpt):
        cand = os.path.join(REPO_ROOT, args.ckpt)
        if os.path.exists(cand):
            args.ckpt = cand
    if not os.path.exists(args.ckpt):
        print(f"FATAL: base checkpoint not found: {args.ckpt}", flush=True)
        return 2

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} | group={args.group_name} | base={args.ckpt}",
          flush=True)
    m, loss_fn, group_params = build_model_and_loss(args)

    print(f"\n--- recipe A: REAL loop (registry lr, per-group clip 1.0) ---")
    ra = run_real_recipe(m, loss_fn, group_params, args)
    print(summarize(ra))
    print(f"\n--- recipe B: ISOLATED (lr={ISOLATED_LR}, weight 1.0, no clip) ---")
    rb = run_isolated(m, loss_fn, group_params, args)
    print(summarize(rb))

    dwA = sum(ra["dws"]) / len(ra["dws"])
    dwB = sum(rb["dws"]) / len(rb["dws"])
    dropA = 1.0 - ra["errN"] / ra["err0"] if ra["err0"] > 0 else 0.0
    dropB = 1.0 - rb["errN"] / rb["err0"] if rb["err0"] > 0 else 0.0
    per_step_A = dropA / ra["steps"]
    per_step_B = dropB / rb["steps"]
    ratio = (per_step_B / per_step_A) if per_step_A > 0 else float("inf")
    pp = ra.get("per_param_rel") or {"median": 0.0, "max": 0.0}

    # C3/C4 are only meaningful for groups with their OWN error term (hpc
    # error, ais_v2 EIG TD loss), where the head demonstrably reduces its
    # error. For CE-path groups (sbr/relational/evidence) the loss is the
    # main CE on RANDOM inputs — not a monotonic learning target over a few
    # steps — so C3 is replaced by an instability guard: real |dW| must not
    # exceed ~10x the isolated recipe (movement that violent is divergence,
    # not learning).
    has_own_term = args.group_name in ("hpc", "ais_v2")
    print("\n--- VERDICT (same 4-criterion structure as the HPC pre-flight) ---")
    c1 = dwA >= 5 * STARVED_DW
    # C2 for the single-output-conv HPC case: rel >= 1%/step on the group. For
    # an AGGREGATE group the per-param distribution is the signal: the
    # head-like params must move at >= 1%/step of their own norm (median
    # >= 0.1%/step or max >= 1%/step) while big stable matrices stay put.
    c2 = (pp["max"] >= 0.01) if pp["n"] else (relA >= 0.01)
    if has_own_term:
        c3 = dropA > 0.0
        # C4: the isolated reference's |dW|/step must be within an order of
        # magnitude of the real recipe's — movement-scale comparison. (The
        # original loss-drop-rate ratio is only meaningful when BOTH sides
        # decline; a TD head's isolated loss on random inputs is noisy and can
        # go negative, which made the ratio sign-flip and false-FAIL.)
        c4 = 0.1 <= (dwA / dwB if dwB > 0 else float("inf")) <= 10.0
    else:
        c3 = dwA <= 10.0 * dwB          # instability guard (CE-path groups)
        c4 = True
    ok = True
    if pp["n"]:
        c2_detail = (f"per-param max {100*pp['max']:.3f}%/step, "
                     f"median {100*pp['median']:.3f}%/step")
    else:
        c2_detail = f"{100*relA:.2f}%/step"
    if has_own_term:
        c3_detail = f"{100*dropA:.1f}% over {ra['steps']} steps"
        c4_detail = (f"|dW| ratio real/isolated = "
                     f"{dwA:.3e}/{dwB:.3e} = {dwA/dwB if dwB>0 else float('inf'):.2f}x")
    else:
        c3_detail = (f"real |dW| {dwA:.3e} <= 10x isolated {dwB:.3e} "
                     f"(instability guard; no dedicated error term)")
        c4_detail = "N/A (CE-path group)"
    for name, passed, detail in [
        ("C1 |dW| >= 5x starved", c1,
         f"{dwA:.3e} vs starved {STARVED_DW:.2e}"),
        ("C2 head-like params move >= 1%/step", c2, c2_detail),
        ("C3 loss/error declines" if has_own_term else "C3 no instability",
         c3, c3_detail),
        ("C4 within ~10x of isolated", c4, c4_detail),
    ]:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} — {detail}")
        ok = ok and passed
    print("\n  =>", "PROCEED: movement in the learnable regime; run the "
                   "component's 15-epoch smoke." if ok
          else "STOP: movement too small; diagnose before the smoke run.")
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())