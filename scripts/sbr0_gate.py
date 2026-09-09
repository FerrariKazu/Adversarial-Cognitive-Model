#!/usr/bin/env python3
"""
SBR-0 structural-convergence gate — the four pre-registered criteria.
================================================================================

SBR-0 is the frozen-backbone, clean-only structural gate that BLOCKS
everything else in the SBR ladder. All four metrics are written to
report/sbr0_gate_verdict.json regardless of pass/fail — a FAIL is a complete,
valid, reportable outcome (it would mean the slot architecture itself needs
redesign before any further SBR work is worth attempting).

Pre-registered criteria (ALL must pass):

  1. Slot occupancy entropy: Shannon entropy of the per-slot attention mass,
     normalized by log(K), >= 0.6 — no collapse to one dominant slot.
  2. Pairwise slot attention-map cosine similarity: DECREASING trend over
     the training run (measured every 5 epochs), not flat or increasing —
     fit a linear trend over the series, require negative slope.
  3. Per-slot linear-probe decodability: a frozen-features logistic
     regression on EACH slot's D-dim vector alone (10-way STL-10) must
     exceed 25% accuracy (2.5x random) for >= 4 of K=16 slots — real
     discriminative content in individual slots, not just the aggregate.
  4. "Everything slot" ablation: zero out the single highest-norm slot and
     re-run the classifier; retained accuracy >= 70% of the full-slots
     accuracy — no single slot silently carries the whole representation.

The linear probe is torch-based (logistic head over frozen slot features —
no sklearn dependency in the gate path), so it runs identically on Colab
and locally. Criterion 2 needs a SERIES of measurements (>= 2 checkpoints,
preferably every 5 epochs); a single point is INSUFFICIENT_DATA (the gate
cannot pass on one measurement).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: Pre-registered constants (SBR-0 ladder, 2026-09-08).
NUM_SLOTS = 16
SLOT_DIM = 512
NUM_CLASSES = 10
OCCUPANCY_ENTROPY_FLOOR = 0.6          # normalized by log(K)
COSINE_TREND_SLOPE_MAX = 0.0           # need strictly negative slope
PROBE_ACC_FLOOR = 0.25                 # 2.5x random chance (10 classes)
MIN_SLOTS_ABOVE_FLOOR = 4              # of 16
ABLATION_RETAIN_FLOOR = 0.70           # retained acc / full acc
MIN_COSINE_POINTS = 2                  # trend needs >= 2 measurements

VERDICT_PATH = os.path.join(REPO_ROOT, "report", "sbr0_gate_verdict.json")


# ── Criterion 1: slot occupancy entropy ─────────────────────────────────────
def occupancy_entropy(attn: torch.Tensor) -> torch.Tensor:
    """(B,) normalized Shannon entropy of the per-slot attention mass.

    attn: (B, K, N) slot attention maps. Per-slot mass = attention each slot
    claims summed over input positions, normalized to a distribution over K
    slots; entropy normalized by log(K) -> 1.0 = perfectly spread, 0.0 =
    collapse to one slot.
    """
    mass = attn.sum(dim=-1).clamp(min=1e-8)             # (B, K)
    p = mass / mass.sum(dim=-1, keepdim=True)
    ent = -(p * p.log()).sum(dim=-1)                    # (B,)
    return ent / np.log(p.shape[-1])


def criterion1_passes(entropy: torch.Tensor) -> Tuple[bool, Dict]:
    mean = float(entropy.mean())
    passed = mean >= OCCUPANCY_ENTROPY_FLOOR
    return passed, {"mean_normalized_entropy": round(mean, 6),
                    "floor": OCCUPANCY_ENTROPY_FLOOR}


# ── Criterion 2: pairwise attention-map cosine trend ────────────────────────
def attention_pairwise_cosine(attn: torch.Tensor) -> float:
    """Mean pairwise cosine similarity between slot attention maps.

    attn: (B, K, N). Each slot's attention map (over N positions) is
    flattened to a vector, L2-normalized; cosine is averaged over all slot
    pairs and batch. High = slots attend to the same places (redundant);
    decreasing over training = slots specialize.
    """
    B, K, N = attn.shape
    maps = attn.reshape(B, K, -1)                        # (B, K, N)
    maps = maps / (maps.norm(dim=-1, keepdim=True) + 1e-8)
    cos = torch.einsum("bkd,bld->bkl", maps, maps)       # (B, K, K)
    iu = torch.triu_indices(K, K, offset=1, device=attn.device)
    return float(cos[:, iu[0], iu[1]].mean())


def fit_trend_slope(series: Sequence[Tuple[float, float]]) -> Optional[float]:
    """OLS slope of y vs x over the (epoch, cosine) series. None if < 2 pts."""
    xs = np.array([float(x) for x, _ in series])
    ys = np.array([float(y) for _, y in series])
    if len(xs) < 2 or np.ptp(xs) == 0:
        return None
    xm, ym = xs.mean(), ys.mean()
    slope = float(((xs - xm) * (ys - ym)).sum() / ((xs - xm) ** 2).sum())
    return slope


def criterion2_passes(cosine_series: Sequence[Tuple[float, float]]) \
        -> Tuple[bool, Dict]:
    """cosine_series: [(epoch, cosine), ...] across the SBR-0 training run."""
    if len(cosine_series) < MIN_COSINE_POINTS:
        return False, {"insufficient_data": True,
                       "n_points": len(cosine_series),
                       "required": MIN_COSINE_POINTS}
    slope = fit_trend_slope(cosine_series)
    passed = slope is not None and slope < COSINE_TREND_SLOPE_MAX
    return passed, {"n_points": len(cosine_series),
                    "slope": round(slope, 8) if slope is not None else None,
                    "max_slope": COSINE_TREND_SLOPE_MAX,
                    "series": [(round(float(e), 1), round(float(c), 6))
                               for e, c in cosine_series]}


# ── Criterion 3: per-slot linear-probe decodability ─────────────────────────
class _LinearProbe(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def train_linear_probe(features: torch.Tensor, labels: torch.Tensor,
                       dim: int = SLOT_DIM, n_classes: int = NUM_CLASSES,
                       steps: int = 400, lr: float = 1e-2,
                       seed: int = 0) -> _LinearProbe:
    """Frozen-features logistic probe: one linear head, CE, Adam.

    Features: (N, D) frozen slot vectors; labels: (N,) ints. Returns the
    trained probe (call .eval() and score on a HELD-OUT split — probes are
    never scored in-sample, see per_slot_probe_accuracies).
    """
    torch.manual_seed(seed)
    probe = _LinearProbe(dim, n_classes).to(features.device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    x = features.detach()
    y = labels.to(features.device)
    probe.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(x), y)
        loss.backward()
        opt.step()
    probe.eval()
    return probe


def probe_accuracy(probe: nn.Module, features: torch.Tensor,
                   labels: torch.Tensor) -> float:
    with torch.no_grad():
        pred = probe(features.detach()).argmax(dim=1)
    return float((pred == labels.to(features.device)).float().mean())


def probe_split(features: torch.Tensor, labels: torch.Tensor,
                train_frac: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor, torch.Tensor]:
    """Deterministic train/eval split of a probe batch.

    The gate scores probes on data they were NOT trained on — without this,
    a 400-step Adam probe on 256 frozen vectors memorizes and random slots
    would read as "decodable" (~90% in-sample vs ~10% held-out).
    """
    N = features.shape[0]
    n_train = max(8, int(N * train_frac))
    return (features[:n_train].detach(), labels[:n_train],
            features[n_train:].detach(), labels[n_train:])


def per_slot_probe_accuracies(slots: torch.Tensor, labels: torch.Tensor,
                              num_slots: int = NUM_SLOTS,
                              dim: int = SLOT_DIM,
                              n_classes: int = NUM_CLASSES,
                              seed: int = 0) -> List[float]:
    """One logistic probe per slot on its own D-dim vector, scored on a
    held-out split (never in-sample).

    slots: (N, K, D) collected slot states; labels: (N,). Returns K accs.
    """
    N = slots.shape[0]
    accs = []
    for k in range(num_slots):
        feats = slots[:, k].reshape(N, dim).detach()
        tr, tr_l, ev, ev_l = probe_split(feats, labels)
        probe = train_linear_probe(tr, tr_l, dim=dim, n_classes=n_classes,
                                   seed=seed + k)
        accs.append(probe_accuracy(probe, ev, ev_l))
    return accs


def criterion3_passes(accs: Sequence[float]) -> Tuple[bool, Dict]:
    above = [i for i, a in enumerate(accs) if a >= PROBE_ACC_FLOOR]
    passed = len(above) >= MIN_SLOTS_ABOVE_FLOOR
    return passed, {
        "per_slot_acc": [round(float(a), 4) for a in accs],
        "slots_above_floor": above,
        "n_above": len(above),
        "floor": PROBE_ACC_FLOOR,
        "min_required": MIN_SLOTS_ABOVE_FLOOR,
    }


# ── Criterion 4: "everything slot" ablation ─────────────────────────────────
def everything_slot_ablation(slots: torch.Tensor, labels: torch.Tensor,
                             probe: Optional[nn.Module] = None,
                             num_slots: int = NUM_SLOTS,
                             dim: int = SLOT_DIM,
                             n_classes: int = NUM_CLASSES,
                             seed: int = 0) -> Dict:
    """Zero the single highest-norm slot and measure retained accuracy.

    slots: (N, K, D). The classifier is a linear probe over the POOLED slots
    (the belief the model's own head consumes — the task's "re-run the
    classifier head"): either the caller's trained probe or one trained
    internally. Both the full-pool and ablated-pool accuracies are scored on
    the probe's held-out split, so the retained ratio compares like with
    like. Retained = acc(pooled without highest-norm slot) / acc(full).
    """
    N = slots.shape[0]
    pooled_all = slots.mean(dim=1)                       # (N, D)
    norms = slots.detach().norm(dim=-1).mean(dim=0)      # (K,)
    victim = int(norms.argmax())

    # Held-out split (same as the per-slot probes): train on the first half,
    # score both full and ablated pooling on the second half.
    tr, tr_l, ev, ev_l = probe_split(pooled_all, labels)
    if probe is None:
        probe = train_linear_probe(tr, tr_l, dim=dim, n_classes=n_classes,
                                   seed=seed)
    with torch.no_grad():
        pred_full = probe(ev).argmax(dim=1)
    full_acc = float((pred_full == ev_l.to(ev.device)).float().mean())

    slots_zeroed = slots.detach().clone()
    slots_zeroed[:, victim] = 0.0
    pooled_abl = slots_zeroed.mean(dim=1)[N - ev.shape[0]:]   # eval half
    with torch.no_grad():
        pred_abl = probe(pooled_abl).argmax(dim=1)
    abl_acc = float((pred_abl == ev_l.to(pooled_abl.device)).float().mean())
    retained = abl_acc / full_acc if full_acc > 0 else 0.0
    return {"victim_slot": victim,
            "victim_norm_share": round(float(norms[victim] / norms.sum()), 6),
            "full_acc": round(full_acc, 6),
            "ablated_acc": round(abl_acc, 6),
            "retained": round(float(retained), 6)}


def criterion4_passes(abl: Dict) -> Tuple[bool, Dict]:
    passed = abl["retained"] >= ABLATION_RETAIN_FLOOR
    return passed, dict(abl, floor=ABLATION_RETAIN_FLOOR)


# ── Aggregate gate evaluation ───────────────────────────────────────────────
def evaluate_sbr0_gate(
        attn: torch.Tensor,
        slots: torch.Tensor,
        labels: torch.Tensor,
        model: Optional[nn.Module] = None,
        cosine_series: Optional[Sequence[Tuple[float, float]]] = None,
        num_slots: int = NUM_SLOTS,
        dim: int = SLOT_DIM,
        n_classes: int = NUM_CLASSES,
        probe_seed: int = 0,
        return_probe: bool = False) -> Dict:
    """Run all four pre-registered criteria on a held-out probe batch.

    Args:
        attn: (N, K, N_pos) collected slot attention maps over the probe set.
        slots: (N, K, D) collected slot states (attached or detached — the
            probe trains on detached features).
        labels: (N,) STL-10 labels for the probe set.
        model: the full RHANNext model (for criterion 4's "re-run the
            classifier head" when no probe is passed).
        cosine_series: [(epoch, mean_pairwise_cosine), ...] accumulated over
            the training run (every 5 epochs). None/[] => insufficient data.
        return_probe: also return the trained pooled-slot probe (used by
            criterion 4 when model is None).
    """
    labels = labels.to(attn.device)
    ent = occupancy_entropy(attn)
    c1, m1 = criterion1_passes(ent)

    c2, m2 = criterion2_passes(list(cosine_series) if cosine_series else [])

    accs = per_slot_probe_accuracies(slots, labels, num_slots=num_slots,
                                     dim=dim, n_classes=n_classes,
                                     seed=probe_seed)
    c3, m3 = criterion3_passes(accs)

    # Criterion 4: zero the highest-norm slot in the pooled representation
    # (the classifier input) and re-score — the module trains its own
    # held-out-split probe so full/ablated accuracies are like-for-like.
    abl = everything_slot_ablation(slots.detach(), labels,
                                   num_slots=num_slots, dim=dim,
                                   n_classes=n_classes, seed=probe_seed + 1000)
    c4, m4 = criterion4_passes(abl)

    # Diagnostic-only pooled probe (train on the split's train half) —
    # returned when requested, never used for the verdict.
    pooled = slots.mean(dim=1).detach()
    tr, tr_l, _, _ = probe_split(pooled, labels)
    probe = train_linear_probe(tr, tr_l, dim=dim, n_classes=n_classes,
                               seed=probe_seed + 2000)

    passed = c1 and c2 and c3 and c4
    return {
        "passed": passed,
        "criteria": {
            "1_slot_occupancy_entropy": {"passed": c1, **m1},
            "2_pairwise_cosine_trend": {"passed": c2, **m2},
            "3_per_slot_probes": {"passed": c3, **m3},
            "4_everything_slot_ablation": {"passed": c4, **m4},
        },
        "probe": probe if return_probe else None,
    }


def write_verdict(verdict: Dict, path: str = VERDICT_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(verdict, f, indent=2)
        f.write("\n")
    return path


# ── CLI: evaluate a trained SBR-0 checkpoint over a data loader ────────────
def _collect_probe_batch(model, loader, device, max_samples: int = 512):
    """One forward pass over the loader, collecting attn/slots/logits/labels."""
    model.eval()
    attns, slotss, labelss = [], [], []
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, traj = model(x, return_trajectory=True)
            attns.append(traj["sbr_attn"][-1].cpu())       # last step
            slotss.append(traj["sbr_slots"][-1].cpu())
            labelss.append(y)
            n += y.shape[0]
            if n >= max_samples:
                break
    attn = torch.cat(attns)[:max_samples]
    slots = torch.cat(slotss)[:max_samples]
    labels = torch.cat(labelss)[:max_samples]
    return attn, slots, labels


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="SBR-0 checkpoint path (or 'best'/'rolling' name "
                         "under checkpoints/)")
    ap.add_argument("--cosine-series", default=None,
                    help="Optional JSON: list of [epoch, cosine] pairs from "
                         "the training run (criterion 2).")
    ap.add_argument("--series-out", default=None,
                    help="JSON file to APPEND this checkpoint's [epoch, "
                         "cosine] pair to (the notebook maintains the series "
                         "across the 5-epoch milestones); the updated series "
                         "is then used for criterion 2.")
    ap.add_argument("--epoch", type=int, default=None,
                    help="Current epoch number (required for series-out to "
                         "produce unique entries across gate runs).")
    ap.add_argument("--out", default=VERDICT_PATH)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, os.path.join(REPO_ROOT, "phase1_training"))
    import torchvision
    from torch.utils.data import DataLoader

    from checkpoint_utils import compat_load
    from rhan_core.config.pillar_config import RHANNextConfig
    from rhan_core.model import RHANNext
    from dataset_stl10 import get_stl10_loaders  # repo's STL-10 loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt
    if not os.path.exists(ckpt):
        cand = os.path.join(REPO_ROOT, "checkpoints", ckpt)
        if os.path.exists(cand):
            ckpt = cand
    print(f"[sbr0_gate] loading {ckpt}", flush=True)
    state = compat_load(ckpt, map_location="cpu")
    cfg = RHANNextConfig.from_dict(state["config"])
    if cfg.sbr_stage != "gate_only":
        print(f"  WARNING: checkpoint sbr_stage={cfg.sbr_stage!r}, expected "
              f"'gate_only'", flush=True)
    model = RHANNext(config=cfg).to(device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    print(f"  loaded: {len(state['model'])-len(missing)}/{len(state['model'])} "
          f"keys (missing={len(missing)} unexpected={len(unexpected)})",
          flush=True)
    model.eval()

    _, test_loader = get_stl10_loaders(batch_size=args.batch_size,
                                       data_root=os.path.join(REPO_ROOT, "data"))
    loader = test_loader

    cosine_series = None
    if args.cosine_series:
        with open(args.cosine_series) as f:
            cosine_series = [tuple(p) for p in json.load(f)]

    attn, slots, labels = _collect_probe_batch(model, loader, device,
                                               max_samples=args.samples)

    # Maintain the criterion-2 series: append THIS checkpoint's pairwise
    # attention-map cosine (measured every 5 epochs per the pre-registered
    # gate) so the trend spans the whole run, then re-evaluate with it.
    epoch_now = args.epoch if args.epoch is not None else int(state.get("epoch", 0))
    if args.series_out:
        series = list(cosine_series or [])
        cos_now = attention_pairwise_cosine(attn.to(device))
        # Replace any existing entry for the same epoch (idempotent re-runs).
        series = [p for p in series if abs(float(p[0]) - epoch_now) > 1e-6]
        series.append((float(epoch_now), cos_now))
        series.sort(key=lambda p: float(p[0]))
        os.makedirs(os.path.dirname(args.series_out) or ".", exist_ok=True)
        with open(args.series_out, "w") as f:
            json.dump(series, f)
        print(f"  criterion-2 series appended: epoch {epoch_now} cosine "
              f"{cos_now:.4f} -> {args.series_out} ({len(series)} points)",
              flush=True)
        cosine_series = series

    verdict = evaluate_sbr0_gate(
        attn.to(device), slots.to(device), labels.to(device), model=model,
        cosine_series=cosine_series)
    verdict.update({
        "schema": "sbr0_gate_v1",
        "ckpt": ckpt,
        "samples": int(labels.shape[0]),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "FAIL is a complete, valid, reportable outcome — it means "
                "the slot architecture itself (not curriculum/optimizer) "
                "needs redesign before further SBR work.",
    })
    path = write_verdict(verdict, args.out)
    print(json.dumps(verdict, indent=2))
    print(f"[sbr0_gate] verdict written to {path}", flush=True)
    print("  =>", "PASS — proceed to SBR-1" if verdict["passed"]
          else "FAIL — STOP the SBR ladder and report honestly", flush=True)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())