"""
Stage-1 candidate benchmark (audit rev-1), one variant per process.

    python3 scratch/audit_rev1/stage1_candidates.py baseline   # writes reference
    python3 scratch/audit_rev1/stage1_candidates.py A
    python3 scratch/audit_rev1/stage1_candidates.py B
    python3 scratch/audit_rev1/stage1_candidates.py C

The production step is:

    x_adv = run_pgd_kl_attack(model, x, eps, steps)     # 1 no-grad clean fwd
                                                        # + 4 grad adv fwd
    loss  = compute_trades_loss(model, x, y, x_adv)     # 1 grad clean fwd
                                                        # + 1 grad adv fwd
    loss.backward()

so the trunk runs 7 times per step.

Candidates:
  A  shared clean forward: one grad-enabled clean forward, whose detached
     softmax is the PGD-KL target; TRADES reuses the same logits (6 trunk runs).
  B  A + fused 2B trunk pass for the TRADES clean/adv forwards (5 trunk runs).
  C  torch.compile on the unmodified baseline step.

Everything runs with the SAME fixed noise seed so the PGD random start is
identical across variants; the reference x_adv and logits are dumped by the
baseline process and reloaded by the others.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen2_foundation.model import Gen2FoundationModel  # noqa: E402
from training.train_gen2_foundation import (  # noqa: E402
    compute_trades_loss,
    run_pgd_kl_attack,
)

REPORT_DIR = os.path.join(REPO_ROOT, "report", "audit_rev1")
os.makedirs(REPORT_DIR, exist_ok=True)

BATCH = 48
EPS = 0.094
STEPS = 4
BETA = 2.5
W_TRADES = 0.55
WEIGHT_SEED = 7          # seeds model construction
DATA_SEED = 7            # seeds x, y
NOISE_SEED = 12345       # seeds the PGD random start, identical in every variant
N_WARMUP = 3
N_TIMED = 5

REF_PATH = os.path.join(REPORT_DIR, "stage1_reference.pt")

OUT: List[str] = []


def line(s: str = "") -> None:
    OUT.append(s)
    print(s)


def build() -> Tuple[Gen2FoundationModel, torch.Tensor, torch.Tensor]:
    torch.manual_seed(WEIGHT_SEED)
    m = (
        Gen2FoundationModel(phase="recurrence_only", arm="v1", fovea_size=112)
        .eval()
        .cuda()
    )
    m.backbone.load_dino_warm_start(
        os.path.join(REPO_ROOT, "checkpoints", "dinov2_vits14_pretrain.pth"),
        strict_band=0.90,
    )
    g = torch.Generator(device="cuda").manual_seed(DATA_SEED)
    x = torch.randn(BATCH, 3, 112, 112, device="cuda", generator=g)
    y = torch.randint(0, 100, (BATCH,), device="cuda", generator=g)
    return m, x, y


def _eval_logits(m, inp) -> torch.Tensor:
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        return m(inp.clone().detach()).float()


# ── baseline: production path verbatim ──────────────────────────────────────
def step_baseline(m, x, y) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(NOISE_SEED)
    x_adv = run_pgd_kl_attack(m, x.clone().detach(), eps=EPS, steps=STEPS)
    with torch.amp.autocast("cuda", enabled=True):
        loss, _, _ = compute_trades_loss(
            m, x.clone().detach(), y.clone().detach(), x_adv, beta=BETA, w_trades=W_TRADES
        )
    loss.backward()
    m.zero_grad(set_to_none=True)
    lc = _eval_logits(m, x)
    la = _eval_logits(m, x_adv)
    return x_adv, lc, la


# ── shared-clean PGD (used by A and B) ──────────────────────────────────────
def _pgd_shared_clean(m, x) -> Tuple[torch.Tensor, torch.Tensor]:
    """PGD-KL that reuses ONE fp32 clean forward as both the attack target and
    the TRADES clean logits.

    The production code runs the fp32 clean forward inside run_pgd_kl_attack
    (no-grad, outside autocast) and then runs the clean forward a second time
    under autocast inside compute_trades_loss. Here the fp32 forward is
    grad-enabled and shared, which keeps the attack target numerically
    identical to production while removing the redundant autocast clean pass.
    """
    torch.manual_seed(NOISE_SEED)
    logits_clean = m(x)                      # fp32, as production's attack pass
    probs_clean = F.softmax(logits_clean.float().detach(), dim=1)

    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-EPS, EPS)
    x_adv = torch.clamp(x_adv, -4.0, 4.0)
    alpha = (2.5 * EPS) / STEPS
    for _ in range(STEPS):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_a = m(x_adv)
            loss_kl = F.kl_div(
                F.log_softmax(logits_a.float(), dim=1), probs_clean, reduction="batchmean"
            )
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - EPS), x + EPS)
            x_adv = x_adv.clamp(-4.0, 4.0)
    return logits_clean, x_adv.detach()


# ── A: shared clean forward ─────────────────────────────────────────────────
def step_shared_clean(m, x, y) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_clean, x_adv = _pgd_shared_clean(m, x.clone().detach())
    # CE is taken from the shared fp32 clean logits (production takes it from the
    # autocast clean pass); the KL target is the same fp32 softmax production's
    # attacker uses.
    loss_ce = F.cross_entropy(logits_clean.float(), y)
    with torch.amp.autocast("cuda", enabled=True):
        logits_adv = m(x_adv)
    loss_kl = F.kl_div(
        F.log_softmax(logits_adv.float(), dim=1),
        F.softmax(logits_clean.float().detach(), dim=1),
        reduction="batchmean",
    )
    loss = W_TRADES * (loss_ce + BETA * loss_kl)
    loss.backward()
    m.zero_grad(set_to_none=True)
    lc = logits_clean.detach().float()
    la = _eval_logits(m, x_adv)
    return x_adv, lc, la


# ── B: fused 2B trunk pass, math identical to production ───────────────────
def step_batched(m, x, y) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attack exactly as production, then ONE 2B autocast forward serving both
    the TRADES clean and adversarial branches (same math as baseline, one trunk
    pass fewer, wider GEMM)."""
    torch.manual_seed(NOISE_SEED)
    x_adv = run_pgd_kl_attack(m, x.clone().detach(), eps=EPS, steps=STEPS)
    n = x.shape[0]
    with torch.amp.autocast("cuda", enabled=True):
        both = m(torch.cat([x.clone().detach(), x_adv], dim=0))
        clean = both[:n]
        adv = both[n:]
        loss_ce = F.cross_entropy(clean.float(), y)
        loss_kl = F.kl_div(
            F.log_softmax(adv.float(), dim=1),
            F.softmax(clean.float().detach(), dim=1),
            reduction="batchmean",
        )
        loss = W_TRADES * (loss_ce + BETA * loss_kl)
    loss.backward()
    m.zero_grad(set_to_none=True)
    lc = _eval_logits(m, x)
    la = _eval_logits(m, x_adv)
    return x_adv, lc, la


def time_step(fn, m, x, y) -> Dict[str, Any]:
    for _ in range(N_WARMUP):
        fn(m, x, y)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times: List[float] = []
    for _ in range(N_TIMED):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(m, x, y)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return {
        "n_warmup": N_WARMUP,
        "n_timed": N_TIMED,
        "median_ms": round(statistics.median(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "all_ms": [round(t, 3) for t in times],
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
    }


def compare(ref: Dict[str, torch.Tensor], got: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
    x_adv, lc, la = got
    out: Dict[str, Any] = {}
    out["x_adv_max_abs_diff"] = float((ref["x_adv"] - x_adv).abs().max().item())
    for tag, a, b in (("clean", ref["logits_clean"], lc), ("adv", ref["logits_adv"], la)):
        out[f"logits_{tag}_max_abs_diff"] = float((a - b).abs().max().item())
        pa = F.softmax(a.float(), dim=1).clamp_min(1e-12)
        pb = F.softmax(b.float(), dim=1).clamp_min(1e-12)
        out[f"softmax_{tag}_kl"] = float(F.kl_div(pa.log(), pb, reduction="batchmean").item())
        out[f"argmax_{tag}_match_frac"] = float(
            (a.argmax(1) == b.argmax(1)).float().mean().item()
        )
    return out


def main() -> None:
    variant = (sys.argv[1] if len(sys.argv) > 1 else "baseline").lower()
    torch.backends.cudnn.benchmark = True

    line(f"=== Stage-1 candidate benchmark :: variant={variant} ===")
    line("device: " + torch.cuda.get_device_name(0))
    line(f"torch {torch.__version__} cuda {torch.version.cuda}")
    line(f"batch={BATCH} img=112 eps={EPS} pgd_steps={STEPS} beta={BETA} "
         f"w_trades={W_TRADES} noise_seed={NOISE_SEED} warmup={N_WARMUP} timed={N_TIMED}")

    m, x, y = build()

    if variant == "baseline":
        fn: Callable = step_baseline
    elif variant == "a":
        fn = step_shared_clean
    elif variant == "b":
        fn = step_batched
    elif variant == "c":
        line("compiling model...")
        t0 = time.time()
        m = torch.compile(m, dynamic=False)
        fn = step_baseline
    else:
        raise SystemExit(f"unknown variant {variant}")

    # First call (also triggers compilation for C).
    t0 = time.time()
    got = fn(m, x, y)
    first_call_s = round(time.time() - t0, 1)
    line(f"first call (incl. any compile): {first_call_s} s")

    if variant == "baseline":
        torch.save(
            {"x_adv": got[0].cpu(), "logits_clean": got[1].cpu(), "logits_adv": got[2].cpu()},
            REF_PATH,
        )
        line("wrote reference: " + REF_PATH)
        agreement: Dict[str, Any] = {"note": "reference"}
    else:
        ref = torch.load(REF_PATH, map_location="cpu")
        ref = {k: v.cuda() for k, v in ref.items()}
        agreement = compare(ref, got)
        line("agreement: " + json.dumps(agreement))

    timing = time_step(fn, m, x, y)
    line("")
    line(f"median {timing['median_ms']:.1f} ms  min {timing['min_ms']:.1f} ms  "
         f"max {timing['max_ms']:.1f} ms  peak_mem {timing['peak_mem_mb']:.0f} MB")
    line("all: " + json.dumps(timing["all_ms"]))

    payload = {
        "variant": variant,
        "machine": {
            "device": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
        },
        "config": {
            "batch": BATCH, "img_size": 112, "eps": EPS, "pgd_steps": STEPS,
            "beta": BETA, "w_trades": W_TRADES, "weight_seed": WEIGHT_SEED,
            "data_seed": DATA_SEED, "noise_seed": NOISE_SEED,
            "n_warmup": N_WARMUP, "n_timed": N_TIMED, "phase": "recurrence_only",
        },
        "first_call_s": first_call_s,
        "timing": timing,
        "agreement": agreement,
    }
    out_json = os.path.join(REPORT_DIR, f"stage1_{variant}.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(REPORT_DIR, f"stage1_{variant}.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")
    line("")
    line("Wrote: " + out_json)


if __name__ == "__main__":
    main()
