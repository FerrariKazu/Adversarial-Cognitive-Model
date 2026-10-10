"""Gen-2 Foundation Model Architecture — gen2_foundation/model.py
================================================================================

Implements the unified Gen-2 Foundation Model across all phases:
1. g2_gist_only (T=1): Full image resized to 56x56 -> CompactViTGen2 trunk -> cls_head.
2. recurrence_only (T=4): Gist as z_0 -> fixed schedule glimpses with tied refinement.
3. belief_no_f (T=4): Carries Dirichlet-evidence uncertainty U_t.
4. belief_with_f:
   - Arm v1: Gen-1 UpdateNet, mean-pooled prediction error, no EMA.
   - Arm v2: ObservedFeatureFusion + PositionSensitiveErrorPool (update_net_v2) +
             EMATargetEncoder (ema.py) + learned precision.
5. ais_v2_swap: Agent F AISv2GazePolicy active.
6. gen1_core: Full Gen-2 core loop.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn

from gen2_foundation.backbone import CompactViTGen2
from gen2_foundation.gist import FixedGistEncoder
from gen2_foundation.ema import EMATargetEncoder
from gen2_foundation.update_net_v2 import BeliefUpdaterV2, PositionSensitiveErrorPool
from noesis_vision.uncertainty.evidential_head import EvidentialHead
from noesis_vision.predictive_coding.glimpse_predictor import (
    ConcreteGlimpseFeaturePredictor,
)
from noesis_vision.predictive_coding.update_net import ConcreteUpdateNet
from noesis_vision.predictive_coding.precision import PrecisionFunction
from noesis_vision.gaze.ais_v2_policy import AISv2GazePolicy
from noesis_vision.models.foveation import foveal_sample, DEFAULT_FOVEA_SIZE

D_Z = 384
NUM_CLASSES = 100


class Gen2FoundationModel(nn.Module):
    """Phase-parameterized RHAN-NXA Gen-2 Foundation Model."""

    def __init__(
        self,
        phase: str = "g2_gist_only",
        arm: str = "v1",  # "v1" or "v2" for belief_with_f
        d_z: int = D_Z,
        num_classes: int = NUM_CLASSES,
        fovea_size: int = DEFAULT_FOVEA_SIZE,
        num_glimpses: int = 4,
        within_glimpse_iters: int = 2,
        use_layer_scale: bool = True,
        center_crop_mode: bool = False,
    ) -> None:
        super().__init__()
        self.phase = phase
        self.arm = arm
        self.d_z = d_z
        self.num_classes = num_classes
        self.fovea_size = fovea_size
        self.num_glimpses = num_glimpses
        self.within_glimpse_iters = within_glimpse_iters
        self.center_crop_mode = center_crop_mode

        # Backbone: DINOv2-small shaped with LayerScale
        self.backbone = CompactViTGen2(
            img_size=fovea_size,
            num_refine_iters=within_glimpse_iters,
            use_layer_scale=use_layer_scale,
        )

        # Gist encoder: reuses backbone.patch_embed (hard invariant: no second tokenizer)
        self.gist_encoder = FixedGistEncoder(
            patch_embed=self.backbone.patch_embed,
            target_size=fovea_size,
        )

        # Phase flags
        self.is_gist_only = (phase == "g2_gist_only")
        self.use_refinement = not self.is_gist_only
        self.use_recurrence = phase in ("recurrence_only", "belief_no_f", "belief_with_f", "ais_v2_swap", "gen1_core")
        self.carry_belief = phase in ("belief_no_f", "belief_with_f", "ais_v2_swap", "gen1_core")
        self.belief_dynamics = phase in ("belief_with_f", "ais_v2_swap", "gen1_core")
        self.use_ais_v2 = phase in ("ais_v2_swap", "gen1_core")

        # Readout head
        readout_dim = (d_z + num_classes) if self.carry_belief else d_z
        self.cls_head = nn.Linear(readout_dim, num_classes)

        if self.carry_belief:
            self.evidential_head = EvidentialHead(input_dim=d_z, num_classes=num_classes)
            self.ev_readout = nn.Linear(num_classes, num_classes)

        if self.belief_dynamics:
            self.predictor = ConcreteGlimpseFeaturePredictor(d_z=d_z, d_feat=d_z, n_tokens=16)
            self.precision = PrecisionFunction()

            if arm == "v2" and phase in ("belief_with_f", "ais_v2_swap", "gen1_core"):
                # Arm v2: BeliefUpdaterV2 + EMA target encoder
                self.update_net_v2 = BeliefUpdaterV2(d_z=d_z)
                self.ema_target_encoder = EMATargetEncoder(self.predictor, alpha=0.996)
                self.update_net = None
            else:
                # Arm v1: Gen-1 UpdateNet, mean-pooled error, no EMA
                self.update_net = ConcreteUpdateNet(d_z=d_z)
                self.update_net_v2 = None
                self.ema_target_encoder = None

        if self.use_ais_v2:
            self.gaze_policy = AISv2GazePolicy(
                predictor=self.predictor,
                evidential_head=self.evidential_head,
                num_candidates=6,
            )

    def group_parameters(self) -> Dict[str, List[nn.Parameter]]:
        """Return named optimizer groups for AdamW per-group LR assignment."""
        trunk_params = list(self.backbone.parameters())
        groups: Dict[str, List[nn.Parameter]] = {
            "backbone": trunk_params,
            "classifier": list(self.cls_head.parameters()),
            "gist": [self.gist_encoder.gate],
        }
        if self.carry_belief:
            groups["evidential_head"] = (
                list(self.evidential_head.parameters()) + list(self.ev_readout.parameters())
            )
        if self.belief_dynamics:
            groups["predictor"] = list(self.predictor.parameters())
            groups["precision"] = list(self.precision.parameters())
            if self.arm == "v2" and self.update_net_v2 is not None:
                groups["update_net"] = list(self.update_net_v2.parameters())
            elif self.update_net is not None:
                groups["update_net"] = list(self.update_net.parameters())
        if self.use_ais_v2:
            groups["gaze_policy"] = [self.gaze_policy.logit_scale]
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking full image x of shape (B, 3, H, W)."""
        B = x.shape[0]
        device = x.device

        # Gist representation produced from the full frame
        if self.is_gist_only:
            # S2: full image resized to 56x56 through trunk, no glimpses
            if self.center_crop_mode:
                crop = foveal_sample(x, torch.zeros(B, 2, device=device), fovea_size=self.fovea_size)
            else:
                # Gist view: full frame downscaled to 56x56
                crop = nn.functional.interpolate(x, size=(self.fovea_size, self.fovea_size), mode="bilinear", align_corners=False)
            x_prep = self.backbone._prep(crop)
            x_trunk = self.backbone._trunk_forward(x_prep)
            pooled = x_trunk[:, 0]  # CLS token (B, D_z)
            return self.cls_head(pooled)

        # Downstream multi-glimpse loop with Gist initialization
        # Extract initial gist vector
        gist_tokens = self.gist_encoder(x)  # (B, 16, D_z)
        z_t = gist_tokens.mean(dim=1)       # Initial z_0 from gist (B, D_z)

        gaze_schedule = [
            torch.zeros(B, 2, device=device),
            torch.tensor([[0.3, 0.3]], device=device).expand(B, -1),
            torch.tensor([[-0.3, 0.3]], device=device).expand(B, -1),
            torch.tensor([[0.0, -0.3]], device=device).expand(B, -1),
        ]

        if not self.carry_belief:
            # recurrence_only: T=4 fixed glimpses refining z
            for t in range(min(self.num_glimpses, len(gaze_schedule))):
                crop = foveal_sample(x, gaze_schedule[t], fovea_size=self.fovea_size)
                pooled_g, _ = self.backbone(crop)
                z_t = 0.5 * z_t + 0.5 * pooled_g
            return self.cls_head(z_t)

        # Belief phases: carry evidential uncertainty and dynamics
        dirichlet = self.evidential_head(z_t)
        current_gaze = torch.zeros(B, 2, device=device)
        prev_tokens = None
        pred_feat = None

        for t in range(self.num_glimpses):
            if self.use_ais_v2 and t > 0:
                belief_obj = SimpleNamespace(
                    z=z_t, uncertainty=dirichlet.uncertainty, evidence=dirichlet.evidence
                )
                sel = self.gaze_policy.select_next_location(
                    belief_obj,
                    observed_tokens=prev_tokens.detach() if prev_tokens is not None else torch.zeros(B, 16, self.d_z, device=device),
                    current_gaze=current_gaze,
                    predicted_tokens=pred_feat,
                    training=self.training,
                )
                current_gaze = sel.selected.clamp(-1.0, 1.0)
            elif t < len(gaze_schedule):
                current_gaze = gaze_schedule[t]
            else:
                current_gaze = torch.zeros(B, 2, device=device)

            crop = foveal_sample(x, current_gaze, fovea_size=self.fovea_size)
            pooled_g, tokens_g = self.backbone(crop)
            prev_tokens = tokens_g

            if self.belief_dynamics:
                prec = self.precision(dirichlet).unsqueeze(-1)
                belief_obj = SimpleNamespace(
                    z=z_t, uncertainty=dirichlet.uncertainty, evidence=dirichlet.evidence
                )
                pred_feat = self.predictor.predict_features(belief_obj, current_gaze)
                if self.arm == "v2" and self.update_net_v2 is not None:
                    err_map = (pred_feat - tokens_g).norm(dim=-1).view(B, 4, 4)
                    delta_z = self.update_net_v2(
                        z_t, err_map, observed=pooled_g, U=dirichlet.uncertainty
                    )
                else:
                    err = pred_feat - tokens_g
                    delta_z = self.update_net(z_t, err)
                z_t = z_t + prec * delta_z
            else:
                z_t = 0.5 * z_t + 0.5 * pooled_g

            dirichlet = self.evidential_head(z_t)

        feats = torch.cat([z_t, dirichlet.evidence], dim=-1)
        return self.cls_head(feats) + self.ev_readout(dirichlet.evidence)
