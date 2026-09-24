"""
train_generation1_foundation.py — Agent J1. Part 2 steps 1-4, one trainer.
================================================================================

Wires Agents A/B/C/D/E (+ I's harness) into the PRE-AIS-v2 foundation
sequence (Part 2 steps 1-4). Agent F's POLICY is deliberately NOT wired
here: step 5 (AIS-v2 integration) is J2's file, gated separately. The
canonical GazeState (noesis_vision.gaze.gaze_state) IS used — it is the
A_t record carrier, not the AIS-v2 selection mechanism. Extending this
trainer's phase list is a FAILURE CONDITION, not a shortcut.

Phases (training/stage_state_machine.py is the orchestration truth):
  backbone_only    step 1 — substrate + ONE fixed center fixation +
                   classifier head. NO tied refinement, NO T=4 loop.
                   Establishes the param/FLOP baseline (Part 3): Agent
                   I's compactness_report runs after EVERY step.
  recurrence_only  step 2 — + the T=4 fixed-schedule glimpse loop and
                   the tied within-glimpse refinement. NO belief, NO
                   uncertainty (the plan's explicit "NO F" step).
  belief_no_f      step 3 — + belief carrier with U_t (Agent B's
                   S=None VectorBeliefState, Agent D's EvidentialHead);
                   IDENTITY update: each glimpse's pooled observation
                   seeds/refreshes z_t unchanged — belief and
                   uncertainty live, no learned dynamics.
  belief_with_f    step 4 — + Agent E's belief dynamics: the shared
                   predictor predicts the NEXT glimpse's features
                   (error_target = latent_next_glimpse, LOCKED default),
                   precision from U_t, z_{t+1} = z_t + Pi*UpdateNet(z,E)
                   — still fixed/heuristic gaze.

GAZE SCHEME FOR ALL PHASES — PLACEHOLDER, NEVER "AIS-v2": a fixed
deterministic 4-point schedule built from Agent C's foveation crop
mechanism directly (no learned selection, no candidate scoring, no
policy). Every artifact labels it PLACEHOLDER_FIXED_GAZE.

Resume discipline (Gen-0 rules, carried): mandatory HF rolling resume
via Agent A's resume_or_abort (never a silent restart); best/rolling
parity verified before any artifact is cited; provenance manifest per
cold start (Agent A's write_manifest); checkpoints go to the DEDICATED
Gen-1 namespace (FerrariKazu/rhan-nxa-checkpoints[-rolling]) — writing
into Gen-0's rhan-checkpoints repos is structurally impossible here.

Platform portability (local RTX 4060 / Kaggle / Colab — same file):
  * single-process, single-GPU-first (CUDA if available, else CPU);
  * no platform-specific imports; HF token resolved from --hf-token,
    $HF_TOKEN, or .env (python-dotenv if present);
  * --smoke runs the ENTIRE four-phase chain on Agent I's synthetic
    loaders (tiny subset, CPU-able, NO HF writes) — the orchestration
    proof that costs seconds, run before spending any GPU-hours;
  * real launches REQUIRE a structurally valid ImageNet-100 root
    (Agent I's validate_imagenet100_root) — checked BEFORE training.

Launch examples:
  # the orchestration proof (seconds, CPU-able):
  python3 training/train_generation1_foundation.py --smoke
  # one phase, real data, 8 seeds (local/Kaggle/Colab identical):
  python3 training/train_generation1_foundation.py --phase backbone_only \
      --data-root data/imagenet100 --epochs 60 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Repo root on sys.path whether run as a script or imported as a module.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.clean_and_robust import run_clean_and_robust  # noqa: E402
from evaluation.compactness_report import compactness_report  # noqa: E402
from evaluation.imagenet100_loader import (  # noqa: E402
    IMAGENET100_IMG_SIZE,
    IMAGENET100_NUM_CLASSES,
    make_imagenet100_loaders,
    make_synthetic_loaders,
    validate_imagenet100_root,
)
from noesis_vision.beliefs.factory import populate_belief  # noqa: E402
from noesis_vision.core.checkpoint import (  # noqa: E402
    CheckpointResumeError,
    resume_or_abort,
    save_best,
    save_rolling,
    verify_best_rolling_parity,
)
from noesis_vision.core.multi_group_optimizer import (  # noqa: E402
    OptimizerGroupRegistry,
)
from noesis_vision.core.provenance import (  # noqa: E402
    config_sha256, write_manifest)
from noesis_vision.gaze.gaze_state import GazeState  # noqa: E402 (canonical A_t)
from noesis_vision.models.backbone import CompactViT  # noqa: E402
from noesis_vision.models.foveation import foveal_sample  # noqa: E402
from noesis_vision.predictive_coding.glimpse_predictor import (  # noqa: E402
    ConcreteGlimpseFeaturePredictor,
)
from noesis_vision.predictive_coding.precision import PrecisionFunction  # noqa: E402
from noesis_vision.predictive_coding.update_net import (  # noqa: E402
    ConcreteUpdateNet,
    belief_update,
)
from noesis_vision.uncertainty.evidential_head import (  # noqa: E402
    DirichletParams,
    EvidentialHead,
)
from training.stage_state_machine import (  # noqa: E402
    FOUNDATION_PHASES,
    GRADIENT_REQUIRED,
    advance,
    ensure_foundation_state,
    get_next_action,
    report_state,
)

# ── HF namespace: Gen-1 DEDICATED repos (never Gen-0's rhan-checkpoints) ────
HF_REPO = "FerrariKazu/rhan-nxa-checkpoints"
HF_REPO_ROLLING = "FerrariKazu/rhan-nxa-checkpoints-rolling"
ROADMAP_NAME = "generation1_foundation_roadmap.json"

PLACEHOLDER_GAZE_LABEL = "PLACEHOLDER_FIXED_GAZE (not AIS-v2; AIS-v2 is J2/step 5)"


# ═══════════════════════════════════════════════════════════════════════════
# HF sync helpers (per-file upload; roadmap rev-guarded down/up)
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_hf_token(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:  # optional local .env (never required on Kaggle/Colab)
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO_ROOT, ".env"))
        return os.environ.get("HF_TOKEN")
    except Exception:
        return None


def _hf_upload(local_path: str, repo_path: str, repo_id: str,
               token: Optional[str]) -> bool:
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=local_path, path_in_repo=repo_path,
            repo_id=repo_id, repo_type="dataset", token=token)
        return True
    except Exception as e:  # noqa: BLE001 — durability is best-effort per file
        print(f"  WARNING: HF upload failed ({repo_path}): {e}", flush=True)
        return False


def _hf_download(repo_id: str, filename: str, token: Optional[str]) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename,
                           repo_type="dataset", token=token,
                           local_dir=os.path.join(REPO_ROOT, "checkpoints"))


def _hf_roadmap_rev(path: str) -> int:
    try:
        with open(path) as f:
            return int(json.load(f).get("roadmap_rev", 0))
    except Exception:
        return 0


def sync_roadmap_down(roadmap_path: str, token: Optional[str]) -> bool:
    """Restore the HF roadmap over the local copy (rev-guarded: a stale HF
    copy never clobbers newer local state — the Gen-0 convention)."""
    try:
        p = _hf_download(HF_REPO_ROLLING, ROADMAP_NAME, token)
    except Exception:
        return False
    if _hf_roadmap_rev(p) < _hf_roadmap_rev(roadmap_path):
        print("  roadmap: HF copy older than local — keeping local", flush=True)
        return False
    import shutil
    os.makedirs(os.path.dirname(roadmap_path) or ".", exist_ok=True)
    shutil.copy(p, roadmap_path)
    print("  ✓ foundation roadmap restored from HF", flush=True)
    return True


def sync_roadmap_up(roadmap_path: str, token: Optional[str]) -> None:
    if _hf_upload(roadmap_path, ROADMAP_NAME, HF_REPO_ROLLING, token):
        print("  ✓ foundation roadmap synced to HF", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# The placeholder gaze schedule (NEVER "AIS-v2")
# ═══════════════════════════════════════════════════════════════════════════
def build_fixed_gaze_schedule(batch: int, num_glimpses: int,
                              device: torch.device,
                              reach: float = 0.6) -> torch.Tensor:
    """(B, T, 2) deterministic 4-point 'corner-ish' schedule, clamped to
    +/-`reach` (stays inside the frame with margin at any fovea size).
    T > 4 tiles the base grid row-major. No parameters, no randomness,
    no learned selection — the J1 placeholder, in every log/ckpt label.
    """
    base = [(-reach, -reach), (reach, -reach), (-reach, reach), (reach, reach)]
    pts = [base[i % 4] for i in range(num_glimpses)]
    sched = torch.tensor(pts, dtype=torch.float32, device=device)  # (T, 2)
    return sched.unsqueeze(0).expand(batch, num_glimpses, 2)


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class FoundationConfig:
    # data
    data_root: str = os.path.join("data", "imagenet100")
    batch_size: int = 64
    num_workers: int = 4
    num_classes: int = IMAGENET100_NUM_CLASSES
    # substrate / loop (schema-locked shapes)
    img_size: int = IMAGENET100_IMG_SIZE      # loader operating point (96)
    fovea_size: int = 56                      # the crop the trunk sees (4x14)
    d_z: int = 384
    num_glimpses: int = 4                     # LOCKED T=4 (Part 1.C)
    within_glimpse_iters: int = 2             # LOCKED range 2-3 (Part 1.C)
    # optimization
    epochs: int = 60
    lr: float = 0.003
    momentum: float = 0.9
    weight_decay: float = 1e-4
    seed: int = 41
    amp: bool = True
    roll_every: int = 1
    # eval (Agent I harness; 8-seed floor per Part 3 policy)
    eval_seeds: Tuple[int, ...] = tuple(range(41, 49))
    n_eval_samples: int = 300
    pgd_steps: int = 10
    eps_list: Tuple[float, ...] = (0.0, 0.031, 0.062, 0.094)
    # infra
    ckpt_dir: str = os.path.join(REPO_ROOT, "checkpoints")
    report_dir: str = os.path.join(REPO_ROOT, "report")
    runs_dir: str = os.path.join(REPO_ROOT, "runs")
    hf_token: Optional[str] = None
    use_hf: bool = True
    force_fresh: bool = False
    # smoke mode (set by --smoke; synthetic loaders, quick eval, NO HF)
    smoke: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["eval_seeds"] = list(self.eval_seeds)
        d["eps_list"] = list(self.eps_list)
        d["hf_token"] = None            # never serialized
        d["gaze_scheme"] = PLACEHOLDER_GAZE_LABEL
        return d


# ═══════════════════════════════════════════════════════════════════════════
# The foundation model. forward(x) -> logits (Agent I harness contract:
# the model takes the IMAGE and returns class logits; the fixed-gaze loop
# and the belief mechanics live inside, per phase).
# ═══════════════════════════════════════════════════════════════════════════
class FoundationModel(nn.Module):
    """Phase-parameterized RHAN-NXA foundation (steps 1-4 only).

    Phase flags (fixed at construction; never toggled mid-run):
      use_refinement  — tied within-glimpse refinement (step 2+)
      use_recurrence  — T=4 fixed-schedule glimpse loop (step 2+)
      carry_belief    — belief carrier + EvidentialHead (step 3+)
      belief_dynamics — predictor + precision + UpdateNet (step 4)
    """

    def __init__(self, cfg: FoundationConfig, phase: str):
        super().__init__()
        if phase not in FOUNDATION_PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        self.cfg = cfg
        self.phase = phase
        self.use_refinement = phase != "backbone_only"
        self.use_recurrence = phase in ("recurrence_only", "belief_no_f",
                                        "belief_with_f")
        self.carry_belief = phase in ("belief_no_f", "belief_with_f")
        self.belief_dynamics = phase == "belief_with_f"
        self.gaze_scheme = ("single_center_fixation" if not self.use_recurrence
                            else PLACEHOLDER_GAZE_LABEL)

        # Agent C's substrate — the ONLY backbone (shared, never duplicated).
        self.backbone = CompactViT(img_size=cfg.fovea_size)
        # Readout width is PHASE-appropriate: belief phases concatenate the
        # ONE uncertainty representation (content d_z + evidence
        # num_classes); steps 1-2 have no evidence yet and read out pooled
        # features (d_z). Each phase trains from scratch with its own
        # checkpoint layout, so the differing shapes are intentional
        # (resume_guard checks layout).
        self.cls_head = nn.Linear(
            (cfg.d_z + cfg.num_classes) if self.carry_belief else cfg.d_z,
            cfg.num_classes)
        if self.carry_belief:
            # Agent D's head — the ONE uncertainty representation.
            self.evidential_head = EvidentialHead(
                input_dim=cfg.d_z, num_classes=cfg.num_classes)
            # Evidential training readout (the head's own classifier readout;
            # ECE is evaluated by Agent I, never trained against).
            self.ev_readout = nn.Linear(cfg.num_classes, cfg.num_classes)
        if self.belief_dynamics:
            # Agent E's three components — the shared predictor + dynamics.
            self.predictor = ConcreteGlimpseFeaturePredictor(
                d_z=cfg.d_z, d_feat=cfg.d_z, n_tokens=16)
            self.update_net = ConcreteUpdateNet(d_z=cfg.d_z)
            self.precision = PrecisionFunction()

    # ── optimizer-group surfaces (Agent A registry; per-phase layout) ───────
    def group_params(self) -> Dict[str, List[nn.Parameter]]:
        groups: Dict[str, List[nn.Parameter]] = {
            "backbone": list(self.backbone.parameters()),
            "classifier": list(self.cls_head.parameters()),
        }
        if self.carry_belief:
            groups["evidential_head"] = (
                list(self.evidential_head.parameters())
                + list(self.ev_readout.parameters()))
        if self.belief_dynamics:
            groups["predictor"] = list(self.predictor.parameters())
            groups["update_net"] = list(self.update_net.parameters())
            groups["precision"] = list(self.precision.parameters())
        return groups

    # ── one glimpse: crop -> trunk -> (optional refinement) -> parts ────────
    def _glimpse(self, x_image: torch.Tensor, gaze: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        crop = foveal_sample(x_image, gaze, fovea_size=self.cfg.fovea_size)
        x = self.backbone._prep(crop)
        x = self.backbone._trunk_forward(x)
        if self.use_refinement:
            x = self.backbone.refinement(x, self.cfg.within_glimpse_iters)
        pooled = x[:, 0]                                   # (B, D_z)
        tokens = x[:, self.backbone.num_prefix_tokens:]    # (B, 16, D_z)
        return pooled, tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 96, 96) images -> (B, C) logits."""
        cfg = self.cfg
        B, T = x.shape[0], cfg.num_glimpses
        device = x.device

        if not self.use_recurrence:
            # ── step 1: ONE fixed center fixation, no refinement ──────────
            pooled, _ = self._glimpse(x, torch.zeros(B, 2, device=device))
            return self.cls_head(pooled)

        # ── steps 2-4: the T=4 fixed-schedule placeholder loop ─────────────
        sched = build_fixed_gaze_schedule(B, T, device)

        if not self.carry_belief:
            # ── step 2: mean-pooled glimpse features, no belief ───────────
            feats = [self._glimpse(x, sched[:, t, :])[0] for t in range(T)]
            return self.cls_head(torch.stack(feats, dim=1).mean(dim=1))

        # ── steps 3-4: belief carrier with U_t ─────────────────────────────
        zero_ev = torch.zeros(B, cfg.num_classes, device=device)
        zero_tok = torch.zeros(B, 16, cfg.d_z, device=device)
        z_t: Optional[torch.Tensor] = None
        dp_t: Optional[DirichletParams] = None
        pred_t: Optional[torch.Tensor] = None
        err: Optional[torch.Tensor] = None
        hist: List[torch.Tensor] = []

        for t in range(T):
            a_t = sched[:, t, :]
            pooled_t, tokens_t = self._glimpse(x, a_t)
            if t == 0:
                z_t = pooled_t              # observation seeds the content
                err = torch.zeros_like(tokens_t)   # LOCKED boundary E_0 := 0
            elif self.belief_dynamics:
                # z_{t+1} = z_t + Pi_t * UpdateNet(z_t, E_t); the OBSERVED
                # tokens are detached here (target convention, Part 1.A);
                # pred_t (made last step) carries the predictor's gradient.
                err = tokens_t.detach() - pred_t
                Pi_t = self.precision(dp_t)
                z_t = belief_update(z_t, err, Pi_t, self.update_net)
            else:
                z_t = pooled_t              # IDENTITY update (step 3)
                # No learned dynamics -> no prediction -> no error signal:
                # E_t is STRUCTURALLY zero (not a trained quantity here).
                err = torch.zeros_like(tokens_t)
            # Agent D's head on the OBSERVED tokens -> the ONE U-carrier.
            dp_t = self.evidential_head(tokens_t)
            # The belief OBJECT, rebuilt per step (canonical GazeState; the
            # LOCKED t=0 boundary — E_0 := 0 — holds by construction).
            belief_t = populate_belief(
                z=z_t,
                U=DirichletParams(evidence=dp_t.evidence),
                E=(zero_tok if t == 0 else err),
                A=GazeState(gaze_history=list(hist), current_glimpse_idx=t))
            if self.belief_dynamics and t + 1 < T:
                # Predict the NEXT glimpse's features from the CURRENT
                # belief (error_target = latent_next_glimpse, LOCKED).
                pred_t = self.predictor.predict_features(
                    belief_t, sched[:, t + 1, :])
            hist.append(a_t.detach())

        # Readout: content + the ONE uncertainty representation.
        feats = torch.cat([z_t, dp_t.evidence], dim=-1)
        return self.cls_head(feats) + self.ev_readout(dp_t.evidence)


def build_model(cfg: FoundationConfig, phase: str) -> FoundationModel:
    return FoundationModel(cfg, phase)


# ═══════════════════════════════════════════════════════════════════════════
# Gradient reachability — the standing rule, checked EXPLICITLY per phase
# (the single most repeated Gen-0 failure; a zero-gradient newly-active
# component is a FAILURE CONDITION, not a note).
# ═══════════════════════════════════════════════════════════════════════════
def _component_params(model: FoundationModel, name: str) -> List[nn.Parameter]:
    if name == "classifier":
        return list(model.cls_head.parameters())
    if name == "evidential_head":
        return (list(model.evidential_head.parameters())
                + list(model.ev_readout.parameters()))
    if name == "predictor":
        return list(model.predictor.parameters())
    if name == "update_net":
        return list(model.update_net.parameters())
    if name == "precision":
        return list(model.precision.parameters())
    raise ValueError(f"unknown gradient component {name!r}")


def check_gradient_reach(model: FoundationModel, x: torch.Tensor,
                         y: torch.Tensor) -> None:
    """One forward/backward; every component required by THIS phase must
    receive a nonzero gradient. Raises RuntimeError otherwise (STOP)."""
    required = GRADIENT_REQUIRED[model.phase]
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    failures = []
    for comp in required:
        params = _component_params(model, comp)
        got = [p for p in params
               if p.grad is not None and p.grad.abs().sum().item() > 0.0]
        if not got:
            failures.append(comp)
    model.zero_grad(set_to_none=True)
    if failures:
        raise RuntimeError(
            f"[{model.phase}] FAILURE CONDITION — gradient did not reach "
            f"newly-active component(s) {failures} (required: "
            f"{list(required)}). The standing non-detached-gradient rule "
            f"is violated; STOP, do not train.")


# ═══════════════════════════════════════════════════════════════════════════
# Train / eval loops
# ═══════════════════════════════════════════════════════════════════════════
def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_val(model: FoundationModel, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(total, 1)


def train_one_epoch(model, loader, optimizer, registry, device,
                    scaler) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=scaler is not None):
            loss = F.cross_entropy(model(x), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            registry.clip_grad_per_group()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            registry.clip_grad_per_group()
            optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def run_phase(phase: str, cfg: FoundationConfig, loaders: Dict[str, Any],
              device: torch.device) -> Dict[str, Any]:
    """One phase end to end: resume-or-abort -> train -> eval -> artifacts."""
    tag = f"[{phase}]"
    rolling_path = os.path.join(cfg.ckpt_dir,
                                f"foundation_{phase}_rolling.pth")
    best_path = os.path.join(cfg.ckpt_dir, f"foundation_{phase}_best.pth")
    hf_roll_name = f"foundation_{phase}_rolling.pth"
    hf_best_name = f"foundation_{phase}_best.pth"

    def up_rolling(p: str) -> None:
        if cfg.use_hf and cfg.hf_token:
            _hf_upload(p, hf_roll_name, HF_REPO_ROLLING, cfg.hf_token)

    def up_best(p: str) -> None:
        if cfg.use_hf and cfg.hf_token:
            _hf_upload(p, hf_best_name, HF_REPO, cfg.hf_token)

    # --force-fresh: an AUDIBLE cold start (never a silent restart).
    if cfg.force_fresh:
        victims = [rolling_path, best_path,
                   os.path.join(cfg.runs_dir, f"foundation_{phase}",
                                "manifest.json")]
        for p in victims:
            if os.path.exists(p):
                os.remove(p)
                print(f"{tag} [force-fresh] removed {p}", flush=True)

    # ── mandatory-resume gate (Agent A; NEVER a silent restart) ────────────
    try:
        state = resume_or_abort(
            rolling_path,
            hf_repo_id=HF_REPO_ROLLING if cfg.use_hf else None,
            hf_filename=hf_roll_name if cfg.use_hf else None,
            hf_token=cfg.hf_token,
            downloader=_hf_download if cfg.use_hf else None)
    except CheckpointResumeError as e:
        raise SystemExit(f"{tag} STOP — resume gate refused: {e}") from e

    seed_everything(cfg.seed)
    model = build_model(cfg, phase).to(device)
    registry = OptimizerGroupRegistry()
    groups = model.group_params()
    registry.register_backbone(groups["backbone"])
    for name in ("classifier", "evidential_head", "predictor", "update_net",
                 "precision"):
        if name in groups:
            registry.register(name, groups[name])
    optimizer = registry.build_optimizer(cfg.lr, cfg.momentum,
                                         cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda") if (
        cfg.amp and device.type == "cuda") else None

    start_epoch = 0
    if state is not None:
        start_epoch = int(state.get("epoch", 0))
        model.load_state_dict(state["model"])
        opt_state = state.get("optimizer")
        if opt_state is not None:
            if not registry.resume_guard(opt_state, state.get("scheduler")):
                raise SystemExit(
                    f"{tag} STOP — optimizer resume guard refused the saved "
                    f"state (group layout changed?)")
            optimizer.load_state_dict(opt_state)
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        else:
            for _ in range(start_epoch):
                scheduler.step()
        print(f"{tag} resumed from epoch {start_epoch} "
              f"(code {state.get('code_commit')!r})", flush=True)
    else:
        # Cold start: provenance manifest (Agent A; refuses silent
        # overwrite). A crash-restart with the SAME config hash is the
        # SAME experiment — keep the original manifest audibly; a
        # DIFFERENT hash under the same id is a genuine conflict: STOP.
        manifest_path = os.path.join(cfg.runs_dir, f"foundation_{phase}",
                                     "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                _old = json.load(f)
            if _old.get("config_sha256") != config_sha256(cfg.to_dict()):
                raise SystemExit(
                    f"{tag} STOP — provenance manifest for "
                    f"foundation_{phase} exists with a DIFFERENT config "
                    f"hash (existing {_old.get('config_sha256', '')[:12]}… "
                    f"vs attempted "
                    f"{config_sha256(cfg.to_dict())[:12]}…). The config "
                    f"changed; delete runs/foundation_{phase}/ explicitly "
                    "to re-run under a new manifest — never overwrite "
                    "provenance silently.")
            print(f"{tag} provenance manifest already exists with an "
                  f"IDENTICAL config hash — keeping the original "
                  f"(crash-restart of the same experiment)", flush=True)
        else:
            write_manifest(
                experiment_id=f"foundation_{phase}",
                config=cfg.to_dict(),
                seed=cfg.seed,
                dataset_version=("synthetic-smoke" if cfg.smoke
                                 else "imagenet100-structural-check-passed"),
                root_dir=cfg.runs_dir,
                optimizer_config={"base_lr": cfg.lr,
                                  "momentum": cfg.momentum,
                                  "weight_decay": cfg.weight_decay,
                                  "groups": registry.group_names},
                extra={"phase": phase,
                       "gaze_scheme": PLACEHOLDER_GAZE_LABEL,
                       "j1_steps": "1-4", "ais_v2": False})
            print(f"{tag} cold start (provenance manifest written)",
                  flush=True)

    best_acc = -1.0
    if os.path.exists(best_path):
        best_acc = float(torch.load(best_path, map_location="cpu",
                                    weights_only=False)
                         .get("metric_value", -1.0))

    # ── train ───────────────────────────────────────────────────────────────
    grad_checked = state is not None   # re-check on cold starts only
    for epoch in range(start_epoch, cfg.epochs):
        if not grad_checked:
            x0, y0 = next(iter(loaders["train"]))
            check_gradient_reach(model, x0.to(device)[:8], y0.to(device)[:8])
            print(f"{tag} gradient reach OK for "
                  f"{GRADIENT_REQUIRED[phase]}", flush=True)
            grad_checked = True
        tr_loss = train_one_epoch(model, loaders["train"], optimizer,
                                  registry, device, scaler)
        val_acc = evaluate_val(model, loaders["val"], device)
        scheduler.step()
        if (epoch + 1) % cfg.roll_every == 0 or epoch + 1 == cfg.epochs:
            save_rolling(rolling_path, epoch=epoch + 1, model=model,
                         optimizer=optimizer, scheduler=scheduler,
                         extra={"phase": phase,
                                "gaze_scheme": PLACEHOLDER_GAZE_LABEL},
                         uploader=up_rolling)
        if val_acc > best_acc:
            best_acc = val_acc
            save_best(best_path, model=model, config=cfg.to_dict(),
                      metric_value=val_acc, uploader=up_best)
        print(f"{tag} epoch {epoch + 1}/{cfg.epochs} "
              f"loss={tr_loss:.4f} val_acc={val_acc:.4f} "
              f"best={best_acc:.4f}", flush=True)

    # ── parity check BEFORE anything cites the artifacts (Agent A rule) ────
    if os.path.exists(best_path) and os.path.exists(rolling_path):
        ok, why = verify_best_rolling_parity(best_path, rolling_path)
        if not ok:
            raise SystemExit(f"{tag} STOP — best/rolling parity FAILED: {why}")
        print(f"{tag} parity: {why}", flush=True)

    # ── eval leg (Agent I's harness; consistency-asserted summary) ──────────
    def loader_factory(seed: int):
        # Fresh subset per seed (the Gen-0 convention) on REAL data.
        ds = loaders["val"].dataset
        g = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(len(ds), generator=g)[:cfg.n_eval_samples]
        return DataLoader(Subset(ds, idx.tolist()), batch_size=cfg.batch_size)

    model.eval()
    eval_out = run_clean_and_robust(
        model=model,
        loader_factory=loader_factory,
        seeds=list(cfg.eval_seeds),
        eps_list=list(cfg.eps_list),
        n_samples=cfg.n_eval_samples,
        ckpt_path=best_path if os.path.exists(best_path) else None,
        out_dir=os.path.join(cfg.report_dir, f"foundation_{phase}_eval"),
        ckpt_label=f"foundation_{phase}",
        device=str(device),
        pgd_steps=cfg.pgd_steps,
        allow_quick=cfg.smoke,
    )
    comp = compactness_report(
        model, input_size=cfg.img_size,
        out_json=os.path.join(cfg.report_dir,
                              f"foundation_{phase}_compactness.json"))

    result = {
        "phase": phase,
        "best_val_acc": best_acc,
        "eval": {"per_seed_csv": eval_out["per_seed_csv"],
                 "summary_csv": eval_out["summary_csv"]},
        "compactness": {k: comp[k] for k in
                        ("params_total", "params_trainable",
                         "est_macs_per_image")},
        "gaze_scheme": PLACEHOLDER_GAZE_LABEL,
    }
    with open(os.path.join(cfg.report_dir,
                           f"foundation_{phase}_result.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"{tag} DONE — best_val_acc={best_acc:.4f} "
          f"params={comp['params_total']:,}", flush=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════
def load_or_init_roadmap(roadmap_path: str, hf_token: Optional[str],
                         use_hf: bool) -> Dict[str, Any]:
    if os.path.exists(roadmap_path) and use_hf:
        sync_roadmap_down(roadmap_path, hf_token)
    roadmap: Dict[str, Any] = {}
    if os.path.exists(roadmap_path):
        with open(roadmap_path) as f:
            roadmap = json.load(f)
    ensure_foundation_state(roadmap)
    if not os.path.exists(roadmap_path):
        # Cold start: persist the scaffold IMMEDIATELY so the first
        # advance() finds it (the roadmap FILE is the single source of
        # truth on disk, never in-memory state).
        os.makedirs(os.path.dirname(roadmap_path) or ".", exist_ok=True)
        with open(roadmap_path, "w") as f:
            json.dump(roadmap, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return roadmap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Agent J1 — Part 2 steps 1-4 foundation trainer "
                    "(placeholder fixed gaze; AIS-v2 is J2/step 5)")
    ap.add_argument("--phase", default="all",
                    help="all | backbone_only | recurrence_only | "
                         "belief_no_f | belief_with_f")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--within-glimpse-iters", type=int, default=None,
                    choices=(2, 3))
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=None)
    ap.add_argument("--n-eval-samples", type=int, default=None)
    ap.add_argument("--pgd-steps", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda | cpu (default auto)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-hf", action="store_true",
                    help="disable HF sync (offline local runs)")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic tiny-data chain — the orchestration "
                         "proof; numbers are NOT results (no HF writes)")
    ap.add_argument("--force-fresh", action="store_true",
                    help="LOUDLY delete this run's rolling/best/manifest "
                         "artifacts before starting (audible cold start)")
    args = ap.parse_args(argv)

    cfg = FoundationConfig()
    if args.data_root:
        cfg.data_root = args.data_root
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.within_glimpse_iters:
        cfg.within_glimpse_iters = args.within_glimpse_iters
    if args.eval_seeds:
        cfg.eval_seeds = tuple(args.eval_seeds)
    if args.n_eval_samples:
        cfg.n_eval_samples = args.n_eval_samples
    if args.pgd_steps is not None:
        cfg.pgd_steps = args.pgd_steps
    cfg.amp = not args.no_amp
    cfg.force_fresh = args.force_fresh
    cfg.smoke = args.smoke
    hf_token = _resolve_hf_token(args.hf_token)
    cfg.use_hf = (not args.no_hf) and bool(hf_token)
    if cfg.smoke:
        cfg.use_hf = False          # smoke NEVER writes to HF
    cfg.hf_token = hf_token

    if args.smoke:
        cfg.epochs = min(cfg.epochs, 1)
        cfg.eval_seeds = (0,)
        cfg.n_eval_samples = 8
        cfg.pgd_steps = 1
        cfg.num_workers = 0
        cfg.batch_size = 8
        cfg.amp = False

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  smoke={cfg.smoke}  hf={cfg.use_hf}\n"
          f"gaze={PLACEHOLDER_GAZE_LABEL}", flush=True)

    roadmap_path = os.path.join(cfg.report_dir, ROADMAP_NAME)
    roadmap = load_or_init_roadmap(roadmap_path, hf_token, cfg.use_hf)

    # ── data: synthetic for smoke, VERIFIED ImageNet-100 for real runs ─────
    if cfg.smoke:
        loaders = make_synthetic_loaders(batch_size=cfg.batch_size,
                                         batches=2, seed=cfg.seed)
        print("  [SMOKE] synthetic loaders (Agent I) — numbers are NOT "
              "results", flush=True)
    else:
        # THE PRE-FLIGHT: structural check on REAL data before any launch.
        n_train = validate_imagenet100_root(cfg.data_root, split="train")
        n_val = validate_imagenet100_root(cfg.data_root, split="val")
        print(f"  ImageNet-100 verified at {cfg.data_root} "
              f"(train classes={n_train}, val classes={n_val})", flush=True)
        loaders = make_imagenet100_loaders(cfg.data_root,
                                           batch_size=cfg.batch_size,
                                           num_workers=cfg.num_workers)

    # ── which phases run ────────────────────────────────────────────────────
    if args.phase != "all":
        if args.phase not in FOUNDATION_PHASES:
            raise SystemExit(
                f"unknown phase {args.phase!r} — steps 5/6 belong to J2 "
                f"(gated on Agent F), not this trainer")
        todo = [args.phase]        # explicit single-phase (re-)entry
    else:
        fnd = roadmap["generation1_foundation"]
        todo = [p for p in FOUNDATION_PHASES
                if fnd["phases"][p]["status"] != "done"]
    print(f"  machine next action: {get_next_action(roadmap)}", flush=True)

    for phase in todo:
        print(report_state(roadmap), flush=True)
        roadmap = advance(phase, "running", roadmap_path=roadmap_path,
                          started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime()))
        if cfg.use_hf:
            sync_roadmap_up(roadmap_path, hf_token)
        result = run_phase(phase, cfg, loaders, device)
        roadmap = advance(phase, "trained", roadmap_path=roadmap_path,
                          best_val_acc=result["best_val_acc"],
                          ckpt=f"foundation_{phase}_best.pth",
                          gaze_scheme=PLACEHOLDER_GAZE_LABEL)
        roadmap = advance(phase, "eval_pending", roadmap_path=roadmap_path)
        roadmap = advance(phase, "eval_complete", roadmap_path=roadmap_path,
                          summary_csv=result["eval"]["summary_csv"])
        roadmap = advance(phase, "done", roadmap_path=roadmap_path,
                          finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime()))
        if cfg.use_hf:
            sync_roadmap_up(roadmap_path, hf_token)

    roadmap = load_or_init_roadmap(roadmap_path, hf_token, cfg.use_hf)
    print(report_state(roadmap), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
