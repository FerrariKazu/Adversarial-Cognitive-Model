"""Predictor target (EMA copy) — Agent E.

BYOL / DINO-style exponential-moving-average target encoder used for the
predictor's target signal in Generation-1 (Part 4, Adjustment 2). The
training predictor (`ConcreteGlimpseFeaturePredictor`) is the ONLY module
that receives gradients; the EMA copy is a frozen shadowing network whose
weights slide toward them each step by polyak averaging. Its output feeds
`pred_t` (the prediction AT the current gaze) in AIS-v2 and the
belief-update error, never the other way around — gradient never flows
into the EMA copy.

The EMA copy is a parameter-by-parameter `register_buffer` shadow of the
shared predictor. It is constructed to mirror the predictor's module tree,
shares no parameters with it, and is updated with a single alpha
exponential average after every training step.

Gradient rule (standing): the target path is detached; the prediction path
is the shared predictor's, which carries gradient. A pre-flight |dW|
check (OptimizerGroupRegistry "predictor_ema") is applied per the existing
Gen-0 isolation rule BEFORE any smoke test.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from noesis_vision.predictive_coding.glimpse_predictor import (
    ConcreteGlimpseFeaturePredictor,
)


class PredictorTarget(nn.Module):
    """A frozen EMA shadow of `ConcreteGlimpseFeaturePredictor`.

    Attributes:
        predictor: the training predictor (received, shared, never
            re-uploaded here).
        alpha: EMA smoothing coefficient in (0, 1); weights move a step
            toward the source predictor each update:
            `w_ema = alpha * w_ema + (1 - alpha) * w_src`.
        ema_decay: alias for alpha (kept for readability in the call site).
    """

    def __init__(
        self,
        predictor: ConcreteGlimpseFeaturePredictor,
        alpha: float = 0.996,
        ema_decay: Optional[float] = None,
    ) -> None:
        """Construct a BYOL/DINO-style EMA shadow of the shared predictor.

        ``predictor`` is intentionally NOT registered as an ``nn.Module``
        child: the shadow must expose NO learnable parameters. The
        training predictor keeps its own autograd graph; the shadow is a
        flat buffer container (frozen), updated by polyak averaging each
        step. Exposing it via ``object.__setattr__`` keeps the shadow
        structurally separate from the source's parameter tree.
        """
        super().__init__()
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        object.__setattr__(self, "predictor", predictor)
        self.alpha = float(alpha)
        if ema_decay is not None:  # backwards-compatible alias for alpha
            self.alpha = float(ema_decay)
        self._build_shadow()

    # ── construction ──────────────────────────────────────────────────────────
    def _build_shadow(self) -> None:
        """Register buffers mirroring the predictor's parameter tree.

        Each `(name, param)` pair of the source becomes a detached clone
        buffer here, so the shadow is a real `nn.Module` subgraph with
        identical topology but zero parameter sharing.  The names are
        flattened (dots replaced by underscores) so they satisfy the
        `nn.Module` buffer-name rule; the mapping shadow_name ->
        source_name is kept in ``self._shadow_names`` for lookups in
        ``update()`` and checkpoint round-trips.
        """
        self._shadow_names: dict[str, str] = {}
        for name, param in self.predictor.named_parameters():
            sh = name.replace(".", "_")          # no dots in buffer names
            buf = param.detach().clone()
            self.register_buffer(sh, buf)
            self._shadow_names[sh] = name

    def _load_shadow_to(self, module: nn.Module) -> None:
        """Copy shadow buffers into `module`'s parameters (no_grad)."""
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name in self._buffer_names():
                    buf = getattr(self, name)
                    param.copy_(buf)

    def _buffer_names(self) -> list[str]:
        return [n for n, _ in self.named_buffers()]

    # ── public API ────────────────────────────────────────────────────────────
    def update(self) -> None:
        """Polyak step: `w_ema <- alpha*w_ema + (1-alpha)*w_src`.

        Child modules are visited so that buffers introduced by nested
        submodules are carried by the state dict; each parameter pair is
        blended in-place under ``no_grad`` so the shadow stays detached.
        """
        src = self.predictor
        src_state = src.state_dict()
        with torch.no_grad():
            for sh_name in self._buffer_names():
                src_name = self._shadow_names[sh_name]
                if src_name not in src_state:
                    continue
                param = src_state[src_name]
                buf = getattr(self, sh_name)
                buf.mul_(self.alpha).add_(param, alpha=1.0 - self.alpha)

    def update_children(self) -> None:
        """Polyak step with explicit child-module recursion.

        A plain ``named_buffers`` walk can miss buffers registered on
        nested submodules.  Walk the module tree explicitly, blend the
        parameters of every child that has a shadow buffer, then blend
        that child's *own* buffers.  This makes the EMA update robust
        regardless of how the predictor is structured.
        """

        """Polyak step over the whole module tree (children + own).

        Kept as a single entry point so callers can pick either policy
        (children-first, or flat) without branching.
        """
        self.update_children()
        self.update()

    # ── prediction ────────────────────────────────────────────────────────────
    def predict_features(self, belief, gaze_location: torch.Tensor
                         ) -> torch.Tensor:
        """Predict token features through the EMA target — NEVER detached.

        This is the mechanism by which the target signal is produced:
        it is the SAME call signature as `ConcreteGlimpseFeaturePredictor`
        and is consumed identically by the belief update and the AIS-v2
        policy. Because the EMA copy is a detached shadow, nothing flows
        backward into it.
        """
        return self.predictor.predict_features(belief, gaze_location)

    def forward(self, belief, gaze_location: torch.Tensor) -> torch.Tensor:
        """Module convention — routes through the interface method."""
        return self.predict_features(belief, gaze_location)

    def score_candidates(
        self, belief, candidate_locations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score K candidates through the SAME target predictor."""
        preds, scores = [], []
        for k in range(candidate_locations.shape[1]):
            pk = self.predict_features(
                belief, candidate_locations[:, k, :]
            )
            preds.append(pk)
            scores.append(pk.mean(dim=(1, 2)))
        return torch.stack(preds, dim=1), torch.stack(scores, dim=-1)

    # ── checkpointing ─────────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        """Expose the EMA shadow state for checkpointing + resuming.

        Returns a plain dict of the shadow *buffers* (flattened names),
        so a resume can restore the EMA target without touching the
        training predictor or any other module. Only the shadow buffers
        are returned — the predictor's own buffers (e.g. the Fourier
        freqs) are NOT part of the target snapshot.
        """
        return {n: getattr(self, n) for n in self._shadow_names}

    def load_state_dict(self, state: dict) -> None:
        """Restore an EMA shadow saved by ``state_dict()``.

        The saved keys are the flattened shadow names; each is loaded
        into the matching buffer under no_grad. Raises KeyError on any
        unrecognised key so a wrong snapshot can never silently
        corrupt the target.
        """
        with torch.no_grad():
            for n, buf in state.items():
                if n in self._shadow_names:
                    getattr(self, n).copy_(buf)
                else:
                    raise KeyError(
                        f"unknown EMA shadow key {n!r}; expected "
                        f"{sorted(self._shadow_names)}"
                    )

