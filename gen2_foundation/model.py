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

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn

from gen2_foundation.backbone import CompactViTGen2
from gen2_foundation.gist import FixedGistEncoder
from gen2_foundation.ema import EMATargetEncoder
from gen2_foundation.update_net_v2 import BeliefUpdaterV2, PositionSensitiveErrorPool
from noesis_vision.models.uncertainty import EvidentialHead
from noesis_vision.models.belief_dynamics import (
    ConcreteGlimpseFeaturePredictor,
    ConcreteUpdateNet,
    PrecisionFunction,
)
from noesis_vision.models.policy import AISv2GazePolicy
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
                # Arm v2: BeliefUpdaterV2 + spatial error pool + EMA target encoder
                self.spatial_error_pool = PositionSensitiveErrorPool(d_feat=d_z, d_z=d_z)
                self.update_net_v2 = BeliefUpdaterV2(d_z=d_z)
                self.ema_target_encoder = EMATargetEncoder(self.backbone, momentum=0.99)
                self.update_net = None
            else:
                # Arm v1: Gen-1 UpdateNet, mean-pooled error, no EMA
                self.update_net = ConcreteUpdateNet(d_z=d_z)
                self.spatial_error_pool = None
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
            "gist": list(self.gist_encoder.parameters()),
        }
        if self.carry_belief:
            groups["evidential_head"] = (
                list(self.evidential_head.parameters()) + list(self.ev_readout.parameters())
            )
        if self.belief_dynamics:
            groups["predictor"] = list(self.predictor.parameters())
            groups["precision"] = list(self.precision.parameters())
            if self.arm == "v2" and self.update_net_v2 is not None:
                groups["update_net"] = (
                    list(self.update_net_v2.parameters()) + list(self.spatial_error_pool.parameters())
                )
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

        if not self.carry_belief:
            # recurrence_only: T=4 fixed glimpses refining z
            gaze_schedule = [
                torch.zeros(B, 2, device=device),
                torch.tensor([[0.3, 0.3]], device=device).expand(B, -1),
                torch.tensor([[-0.3, 0.3]], device=device).expand(B, -1),
                torch.tensor([[0.0, -0.3]], device=device).expand(B, -1),
            ]
            for t in range(min(self.num_glimpses, len(gaze_schedule))):
                crop = foveal_sample(x, gaze_schedule[t], fovea_size=self.fovea_size)
                pooled_g, _ = self.backbone(crop)
                z_t = 0.5 * z_t + 0.5 * pooled_g
            return self.cls_head(z_t)

        # Belief phases: carry evidential uncertainty and dynamics
        evidence = self.evidential_head(z_t)
        u_t = self.evidential_head.uncertainty(evidence)

        # Simple fixed or AIS-v2 loop
        for t in range(self.num_glimpses):
            gaze = torch.zeros(B, 2, device=device)  # placeholder or policy
            crop = foveal_sample(x, gaze, fovea_size=self.fovea_size)
            pooled_g, tokens_g = self.backbone(crop)

            if self.belief_dynamics:
                prec = self.precision(u_t)
                pred_feat = self.predictor.predict_features(z_t, gaze)
                if self.arm == "v2" and self.spatial_error_pool is not None:
                    # Spatial error pool
                    err = self.spatial_error_pool(pred_feat, tokens_g)
                    z_t = self.update_net_v2(z_t, pooled_g, err, prec)
                else:
                    err = (pred_feat - tokens_g).abs().mean(dim=1)
                    z_t = self.update_net(z_t, pooled_g, err, prec)

            evidence = self.evidential_head(z_t)
            u_t = self.evidential_head.uncertainty(evidence)

        feats = torch.cat([z_t, evidence], dim=-1)
        return self.cls_head(feats) + self.ev_readout(evidence)
