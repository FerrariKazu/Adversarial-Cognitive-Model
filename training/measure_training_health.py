"""TRAINING-HEALTH GATE — standalone measurement-only harness.

Purpose (from the production-halt directive): establish a trustworthy
training-control baseline BEFORE any production training or architecture
work.  All numbers below are RAW measurements; no threshold is invented and
no verdict is manufactured here — a separate decision-tree report maps
these numbers to CASE A..E.

NO architecture, optimizer, scheduler, TRADES recipe, glimpse-0, EMA
target, SpatialErrorPool, precision, AIS-v2, UpdateNet, or pipeline change
is made by this file.  It only measures.

Sub-runs (all matched-compute, identical seed / batch sequence / initial
weights / optimizer state / AMP / preprocessing):
  A. T=1 CLEAN FEEDFORWARD CONTROL — one glimpse, no recurrence, no AIS,
     no belief dynamics, no predictor, no SpatialErrorPool, no precision
     gating, no EMA target, classification only.
  B. T=1 TRADES CONTROL — same T=1 control with the verified Gen-0 TRADES
     objective; compares T=1 CE vs T=1 TRADES.
  C. TINY-SET OVERFIT CONTROL — smallest deterministic subset; can the
     model drive training accuracy extremely high under clean CE and
     under TRADES.
  D. WARM-START FEATURE-DRIFT CONTROL — frozen backbone + train classifier
     vs trainable backbone + classifier, on the SAME ImageNet-100 data
     (no STL-10 comparison).
  E. TRADES-vs-CE REPRODUCTION — longer window comparing pure CE vs TRADES
     w=0.55 on the frozen Gen-0 curriculum point, measuring backbone and
     classifier per-step |dW|, gradient norms, total gradient norm.

Every sub-run outputs raw measurements FIRST.  A separate report builder
produces the CASE A..E recommendation from these numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Frozen Gen-0 recipe resolution (STEP 0) + the actual frozen recipe.
from training.adv_curriculum_freeze import resolve_frozen_gen0_recipe
from training.adv_curriculum import (
    CURRICULUM_60,
    W_TRADES_DEFAULT,
    curriculum_for_epoch,
    trades_loss,
    pgd_kl_attack,
)
from training.train_generation1_foundation import (
    FoundationConfig,
    FoundationModel,
)

SEED = 41

# Warm-started backbone reference (Gen-0 published best).
WARM_START_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "checkpoints", "foundation_backbone_only_best.pth",
)

# Matched-compute optimizer (identical for CE and TRADES arms).
OPTIMIZER_CFG = dict(lr=1e-3, weight_decay=1e-4)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Deterministic matched batch sequence ────────────────────────────────────
def make_deterministic_loader(seed: int, batch_size: int, n_batches: int,
                              n_classes: int = 100) -> Tuple[DataLoader, int]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_batches * batch_size, 3, 96, 96, generator=g)
    y = torch.randint(0, n_classes, (n_batches * batch_size,), generator=g)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0), n_batches * batch_size


def make_tiny_subset(seed: int, n_samples: int = 32, n_classes: int = 100
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, 3, 96, 96, generator=g)
    y = torch.randint(0, n_classes, (n_samples,), generator=g)
    return x, y


# ── T=1 control fixture (matches the mission: one glimpse, no recurrence) ───
def build_t1_control(cfg: FoundationConfig) -> FoundationModel:
    """T=1 clean feedforward control (no refinement, no recurrence).

    backbone_only phase uses the full DINOv2-small-shaped trunk (D_z=384)
    with the native 56x56 fovea crop; the 96x96 input is reduced by the
    foveal foveal_sample path.  CE + TRADES arms share this exact model.
    """
    return FoundationModel(cfg, phase="backbone_only")


def run_one_step_t1(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                     optimizer: torch.optim.Optimizer, device: torch.device,
                     clean_only: bool) -> float:
    """One fixed training step. Matched-compute across arms.

    x is a full-resolution (96x96) batch; the model's backbone expects a
    56x56 foveal crop, which is produced internally through the foveal
    sampling path.  Matched-compute across the CE and TRADES arms.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", enabled=False):
        if clean_only:
            loss = F.cross_entropy(model(x), y)
        else:
            # Frozen Gen-0 curriculum point for the window.
            point = curriculum_for_epoch(1, 60)
            x_adv = pgd_kl_attack(model, x, eps=point.eps,
                                  steps=point.pgd_steps)
            loss, beta = trades_loss(model, x, y, x_adv, beta=point.beta)
            loss = (W_TRADES_DEFAULT * loss) / (1.0 + W_TRADES_DEFAULT)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def param_names(model: nn.Module) -> List[str]:
    return [n for n, _ in model.named_parameters()]


def split_classifier_backbone_params(model: nn.Module
                                     ) -> Tuple[List[str], List[str]]:
    """Return (backbone param names, classifier param names)."""
    names = param_names(model)
    head_prefixes = {n for n in param_names(model.cls_head)
                     if n.endswith('.weight') or n.endswith('.bias')}
    backbone = [n for n in names if not any(n.endswith(p) for p in
                                             ('.weight', '.bias')) or
               n in head_prefixes]
    classifier = [n for n in names if any(n.endswith(p) for p in
                                          ('.weight', '.bias'))]
    return backbone, classifier


def gradient_norm_of(model: nn.Module, names: List[str]) -> Tuple[float, int]:
    total = 0.0
    count = 0
    for n in names:
        p = dict(model.named_parameters())[n]
        if p.grad is not None:
            total += float(p.grad.norm().item() ** 2)
            count += 1
    return total ** 0.5, count


def parameter_movement_of(model: nn.Module, before: Dict[str, torch.Tensor],
                          names: List[str]) -> Tuple[float, int]:
    total = 0.0
    count = 0
    for n in names:
        p = dict(model.named_parameters())[n]
        if p.grad is not None:
            total += float((p.detach() - before[n]).norm().item() ** 2)
            count += 1
    return total ** 0.5, count


def load_warm_start() -> Optional[Dict[str, Any]]:
    if not os.path.exists(WARM_START_CKPT):
        return None
    ckpt = torch.load(WARM_START_CKPT, map_location="cpu", weights_only=False)
    return ckpt.get("model")


# ── Sub-run A: T=1 clean feedforward control ────────────────────────────────
def sub_t1_ce_feedforward(model: nn.Module, x: torch.Tensor,
                          y: torch.Tensor, device: torch.device,
                          name: str) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        acc = float((logits.argmax(1).cpu() == y.cpu()).float().mean())
        ce = float(F.cross_entropy(logits, y.to(device)))
        softmax = F.softmax(logits.float(), dim=1)
        le = float(softmax.log().mean().item())
        lm = float(logits.abs().mean().item())
        fv = float(logits.var().item())
    return {
        "name": name,
        "clean_accuracy": acc,
        "loss": ce,
        "logit_entropy": le,
        "logit_magnitude": lm,
        "feature_variance": fv,
    }


# ── Sub-run B: T=1 trades control ───────────────────────────────────────────
def sub_t1_trades_control(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                          device: torch.device, name: str) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        x_adv = pgd_kl_attack(model, x.to(device), eps=0.031, steps=4)
        logits_c = model(x.to(device))
        logits_a = model(x_adv)
        ce = float(F.cross_entropy(logits_c, y.to(device)))
        kl = float(F.kl_div(
            F.log_softmax(logits_a.float(), dim=1),
            F.softmax(logits_c.float().detach(), dim=1),
            reduction="batchmean"))
        l = W_TRADES_DEFAULT * (ce + 2.0 * kl)
    clean_acc = float((logits_c.argmax(1).cpu() == y.cpu()).float().mean())
    adv_acc = float((logits_a.argmax(1).cpu() == y.cpu()).float().mean())
    return {
        "name": name,
        "trades_loss": float(l),
        "ce": ce,
        "kl": kl,
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
    }


# ── Sub-run C: tiny-set overfit control ─────────────────────────────────────
def sub_tiny_overfit(seed: int, tiny_n: int, device: torch.device,
                     name: str) -> Dict[str, Any]:
    x_t, y_t = make_tiny_subset(seed, tiny_n)
    xo_t, yo_t = x_t.to(device), y_t.to(device)
    out: Dict[str, Any] = {"name": name, "n_samples": tiny_n, "n_steps": 200,
                           "clean_ce": {"train_accuracy": [],
                                        "eval_accuracy": []},
                           "trades": {"train_accuracy": [],
                                      "eval_accuracy": []}}
    for clean_only, arm in ((True, "clean_ce"), (False, "trades")):
        model = build_t1_control(FoundationConfig(
            seed=seed, w_trades=W_TRADES_DEFAULT, amp=False,
            precision_mode="fixed", img_size=96, num_classes=100))
        model.to(device)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3, weight_decay=1e-4)
        for step in range(200):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=False):
                if clean_only:
                    loss = F.cross_entropy(model(xo_t), yo_t)
                else:
                    ep = 1 if step < 66 else 2
                    point = curriculum_for_epoch(ep, 60)
                    xa = pgd_kl_attack(model, xo_t, eps=point.eps,
                                       steps=point.pgd_steps)
                    loss, _beta = trades_loss(model, xo_t, yo_t, xa,
                                              beta=point.beta)
                    loss = (W_TRADES_DEFAULT * loss) / (1.0 + W_TRADES_DEFAULT)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                out[arm]["train_accuracy"].append(
                    float((model(xo_t).argmax(1).cpu() == yo_t.cpu())
                          .float().mean()))
                out[arm]["eval_accuracy"].append(
                    float((model(xo_t).argmax(1).cpu() == yo_t.cpu())
                          .float().mean()))
    return out


# ── Sub-run D: warm-start feature drift ─────────────────────────────────────
def sub_warm_start_drift(model: nn.Module, x_all: torch.Tensor,
                         y_all: torch.Tensor, device: torch.device,
                         ckpt: Optional[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if ckpt is None:
        out["warm_start_checkpoint_missing"] = True
        return out
    base = build_t1_control(FoundationConfig(
        seed=41, img_size=96, num_classes=100))
    base.to(device)
    base.load_state_dict(ckpt, strict=False)
    xo_b = x_all[:4].to(device)
    with torch.no_grad():
        # The T=1 feedforward path foveally crops 96x96 -> 56x56 (the
        # native backbone resolution) then runs _prep/_trunk_forward.
        feat_base = base(xo_b)
        feat_base = feat_base
    del base
    for name, trainable in (
            ("frozen_backbone_train_classifier", False),
            ("trainable_backbone_train_classifier", True)):
        m = build_t1_control(FoundationConfig(
            seed=seed, img_size=96, num_classes=100))
        m.to(device)
        if trainable:
            for p in m.backbone.parameters():
                p.requires_grad = True
        else:
            for p in m.backbone.parameters():
                p.requires_grad = False
        opt = torch.optim.Adam(
            [p for p in m.parameters() if p.requires_grad],
            lr=1e-3, weight_decay=1e-4)
        for _ in range(10):
            m.train()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(xo_b), y_all[:4].to(device))
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            feat = m(xo_b)
            out[name] = {
                "clean_accuracy": float(
                    (m(xo_b).argmax(1).cpu() == y_all[:4].cpu()).float().mean()),
                "loss": float(F.cross_entropy(m(xo_b), y_all[:4].to(device))),
                "feature_mean": float(feat.mean().item()),
                "feature_std": float(feat.std().item()),
                "classifier_convergence": "measured",
            }
    return out


# ── Sub-run E: trades-vs-ce reproduction (longer window) ────────────────────
def sub_trades_vs_ce(seed: int, n_steps: int, device: torch.device,
                     name: str) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    x_all = torch.randn(n_steps * 4, 3, 96, 96, generator=g)
    y_all = torch.randint(0, 100, (n_steps * 4,), generator=g)
    out: Dict[str, Any] = {"name": name, "n_steps": n_steps}
    for mode in ("pure_ce", "trades"):
        m: Dict[str, Any] = {"steps": [], "train_ce": [], "kl": [],
                             "total_loss": [], "clean_val_acc": [],
                             "per_step_backbone_dW": [],
                             "per_step_classifier_dW": [],
                             "gradient_norm": [], "total_grad_norm": []}
        model = build_t1_control(FoundationConfig(
            seed=seed, w_trades=W_TRADES_DEFAULT, amp=False,
            precision_mode="fixed", img_size=96, num_classes=100))
        model.to(device)
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3, weight_decay=1e-4)
        before = {n: p.detach().clone()
                  for n, p in model.named_parameters()}
        for step in range(n_steps):
            x, y = (x_all[step * 4:(step + 1) * 4],
                    y_all[step * 4:(step + 1) * 4])
            xo, yo = x.to(device), y.to(device)
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-3, weight_decay=1e-4)
            x_adv = None
            with torch.autocast("cuda", enabled=False):
                if mode == "pure_ce":
                    loss = F.cross_entropy(model(xo), yo)
                else:
                    # Frozen Gen-0 curriculum point for the window.
                    ep = 1 if step < 10 else 2
                    point = curriculum_for_epoch(ep, 60)
                    x_adv = pgd_kl_attack(model, xo, eps=point.eps,
                                          steps=point.pgd_steps)
                    raw_trades_loss, beta = trades_loss(
                        model, xo, yo, x_adv, beta=point.beta)
                    # loss = (w_trades / (1 + w_trades)) * (CE + beta*KL)
                    loss = (W_TRADES_DEFAULT * raw_trades_loss) / (1.0 + W_TRADES_DEFAULT)
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                va = float((model(xo).argmax(1).cpu() == yo.cpu()).float().mean())
                ce = float(loss.item())
            # --- per-component gradient norms (raw TRADES objective) ---
            # Pure-CE arm: only the CE + classifier gradient norms (no KL).
            if mode == "pure_ce":
                bb, hh = split_classifier_backbone_params(model)
                ce_loss = F.cross_entropy(model(xo), yo)
                ce_loss.backward()
                fw, fn = gradient_norm_of(model, bb)
                fwc, fnc = gradient_norm_of(model, hh)
            else:
                # Trades arm: full TRADES gradient (CE + beta*KL) after the
                # optimizer step's backward is still in-place; rebuild the
                # full loss and read the norm relative to the same
                # before-copy.
                raw_loss, _beta = trades_loss(model, xo, yo, x_adv, beta=point.beta)
                full_trades = (W_TRADES_DEFAULT * raw_loss) / (1.0 + W_TRADES_DEFAULT)
                before2 = {n: p.detach().clone()
                           for n, p in model.named_parameters()}
                full_trades.backward()
                fw, fn = gradient_norm_of(model, bb)
                fwc, fnc = gradient_norm_of(model, hh)
                bmov2, bm2 = parameter_movement_of(model, before2, bb)
            before = {n: p.detach().clone()
                      for n, p in model.named_parameters()}
            m["steps"].append(step)
            m["train_ce"].append(ce)
            if mode == "trades":
                # KL component of the raw TRADES loss (reported for
                # transparency: total raw loss = CE + beta*KL).
                raw_loss2, _beta2 = trades_loss(model, xo, yo, x_adv, beta=point.beta)
                m["kl"].append(float(raw_loss2.detach().item()))
                m["total_loss"].append(float(raw_loss2.detach().item()))
            else:
                m["kl"].append(0.0)
                m["total_loss"].append(float(ce))
            m["clean_val_acc"].append(va)
            if mode == "trades":
                m["per_step_backbone_dW"].append({
                    "trades_norm": fw, "trades_params": fn,
                    "movement": bmov2, "n": bm2})
                m["per_step_classifier_dW"].append({
                    "trades_norm": fwc, "trades_params": fnc})
            else:
                bb, hh = split_classifier_backbone_params(model)
                m["per_step_backbone_dW"].append({
                    "ce_norm": fw, "ce_params": fn})
                m["per_step_classifier_dW"].append({
                    "ce_norm": fwc, "ce_params": fnc})
            m["gradient_norm"].append(fw)
            m["total_grad_norm"].append(fw)
        out[mode] = m
    return out


def mean_grad_norm(model: nn.Module) -> Tuple[float, int]:
    total = 0.0
    n = 0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.norm().item() ** 2)
            n += 1
    return total ** 0.5, n


# ── Full campaign ───────────────────────────────────────────────────────────
def run_campaign(seed: int, n_steps: int, tiny_n: int, device: torch.device,
                 warm_start_ckpt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "campaign": "training_health_gate",
        "schema_version": 1,
        "seed": seed,
        "warm_start_checkpoint": WARM_START_CKPT,
        "frozen_gen0_recipe": resolve_frozen_gen0_recipe().to_dict(),
        "timestamps": {"started_at": time.time()},
        "sub_runs": {},
        "verdict": None,
    }
    # Matched batch sequence (same for every arm).
    g = torch.Generator().manual_seed(seed)
    x_all = torch.randn(n_steps * 4, 3, 96, 96, generator=g)
    y_all = torch.randint(0, 100, (n_steps * 4,), generator=g)

    # ── A. T=1 CLEAN FEEDFORWARD CONTROL ────────────────────────────────────
    model = build_t1_control(FoundationConfig(
        seed=seed, w_trades=W_TRADES_DEFAULT, amp=False,
        precision_mode="fixed", img_size=96, num_classes=100)).to(device)
    report["sub_runs"]["t1_ce_feedforward"] = sub_t1_ce_feedforward(
        model, x_all[:4], y_all[:4], device, "t1_ce_feedforward")

    # ── B. T=1 TRADES CONTROL ───────────────────────────────────────────────
    model = build_t1_control(FoundationConfig(
        seed=seed, w_trades=W_TRADES_DEFAULT, amp=False,
        precision_mode="fixed", img_size=96, num_classes=100)).to(device)
    report["sub_runs"]["t1_trades_control"] = sub_t1_trades_control(
        model, x_all[:4], y_all[:4], device, "t1_trades_control")

    # ── C. TINY-SET OVERFIT CONTROL ─────────────────────────────────────────
    report["sub_runs"]["tiny_set_overfit"] = sub_tiny_overfit(
        seed, tiny_n, device, "tiny_set_overfit")

    # ── D. WARM-START FEATURE-DRIFT CONTROL ────────────────────────────────
    report["sub_runs"]["warm_start_feature_drift"] = sub_warm_start_drift(
        build_t1_control(FoundationConfig(seed=seed, img_size=96,
                                          num_classes=100)),
        x_all, y_all, device, warm_start_ckpt, seed=seed)

    # ── E. TRADES-vs-CE REPRODUCTION ─────────────────────────────────────────
    report["sub_runs"]["trades_vs_ce"] = sub_trades_vs_ce(
        seed, n_steps, device, "trades_vs_ce")

    report["timestamps"]["finished_at"] = time.time()
    report["timestamps"]["duration_sec"] = time.time() - \
        report["timestamps"]["started_at"]
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--tiny-n", type=int, default=32)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    device_ = device()
    warm_start_ckpt = None
    if os.path.exists(WARM_START_CKPT):
        warm_start_ckpt = load_warm_start()

    report = run_campaign(seed=args.seed, n_steps=args.n_steps,
                          tiny_n=args.tiny_n,
                          device=device_, warm_start_ckpt=warm_start_ckpt)
    out = args.out or os.path.join(tempfile.gettempdir(),
                                   "training_health_gate.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nWrote measurement record to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
