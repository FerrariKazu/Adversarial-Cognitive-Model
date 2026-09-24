"""
Candidate generation — Agent F. The Gen-0 AIS-v2 heuristic, adapted.
================================================================================

FIRST-ACTION OUTCOME (Agent F contract): the validated Gen-0
candidate-generation heuristic was located in
`rhan_core/gaze/info_gain_policy_v2.py` (InformationGainGazePolicyV2,
schema `ais_v2_smoke_gate_v1`, gate script `scripts/eval_ais_v2_gate.py`)
and is PORTED here with ONE documented adaptation: the uncertainty anchor.

Ported UNCHANGED (the validated mechanism):
  * candidate 0 is the anchor itself — the policy can always stay put;
  * candidates 1..K-1 are anchor + Gaussian noise (sigma in normalized
    image coordinates, Gen-0's validated 0.12);
  * everything clamped to +/-0.9 (Gen-0's max_abs_gaze);
  * K in the LOCKED 4-8 range (Part 1.E; schema-enforced as well).

The ONE adaptation (image-space -> patch-token granularity):
  Gen-0 anchored candidates at the argmax of the image-space
  reconstruction-error map of the foveal crop (48x48 pixels). Gen-1's
  substrate has no pixel reconstruction — the error surface IS the
  token-feature prediction error (Part 1.B). The anchor is therefore the
  argmax of the PER-TOKEN prediction-error map: ||observed_tokens -
  predicted_tokens||^2 per patch token, reshaped to the backbone's 4x4
  token grid. Same semantics ("the place the belief is most wrong about
  right now"), patch-token granularity. At t = 0 there is no prediction
  (E_0 := 0 LOCKED) and the map degenerates to the per-token feature
  magnitude ||observed_tokens||^2 — the SAME structural fallback the
  Gen-0 map has when no prediction exists; no predictor and no U_t
  participate in that branch (structurally absent, not zeroed — see
  ais_v2_policy.select_next_location's t=0 contract).

Coordinate conventions (Agent C's foveation, carried exactly):
  a point at normalized crop coords (u, v) in [-1, 1] maps to image
  coords gaze + (fovea_size / img_size) * (u, v)  (foveation.foveal_sample's
  affine_grid rows: x_in = scale * x_out + gaze_x). A patch-center offset
  of (c/2 - 0.75, r/2 - 0.75) in crop-normalized coords places candidates
  exactly on token centers. Candidate coordinates are DETACHED coordinate
  records (Part 1.A: A_t carries no gradient; differentiability lives in
  the policy's selection, which mixes SCORES, not coordinates).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

#: Gen-0 validated defaults (info_gain_policy_v2), carried unchanged.
CANDIDATE_SIGMA_DEFAULT = 0.12
MAX_ABS_GAZE_DEFAULT = 0.9
#: LOCKED candidate range (Part 1.E) — enforced here to mirror schema.py.
NUM_CANDIDATES_RANGE = (4, 8)


def _validate_num_candidates(num_candidates: int) -> None:
    if not NUM_CANDIDATES_RANGE[0] <= int(num_candidates) \
            <= NUM_CANDIDATES_RANGE[1]:
        raise ValueError(
            f"num_candidates={num_candidates} outside the LOCKED range "
            f"4-8 (Part 1.E) — same rule schema.py enforces on the config."
        )


def token_error_map(observed_tokens: torch.Tensor,
                    predicted_tokens: Optional[torch.Tensor],
                    grid_size: int = 4) -> torch.Tensor:
    """Per-token error/saliency surface, (B, grid, grid).

    predicted_tokens given  -> prediction-error map  ||obs - pred||^2
        (the t >= 1 surface: where the belief is most wrong);
    predicted_tokens is None -> per-token feature magnitude ||obs||^2
        (the t = 0 LOCKED fallback: NO predicted tensor exists in this
        branch — structurally absent, not present-and-zero).

    Args:
        observed_tokens: (B, N, D_feat) — what the encoder actually
            produced at the current fixation (detached or attached; the
            map is consumed under no_grad either way).
        predicted_tokens: (B, N, D_feat) or None — what the shared
            predictor expected at that fixation.
    """
    if observed_tokens.dim() != 3:
        got = tuple(observed_tokens.shape)
        raise ValueError(f"observed_tokens must be (B, N, D); got {got}")
    if predicted_tokens is not None:
        if predicted_tokens.shape != observed_tokens.shape:
            raise ValueError(
                f"predicted_tokens {tuple(predicted_tokens.shape)} must "
                f"match observed_tokens {tuple(observed_tokens.shape)}")
        err = (observed_tokens - predicted_tokens).pow(2).mean(dim=-1)
    else:
        # t = 0 LOCKED branch: the prediction term does not exist here.
        err = observed_tokens.pow(2).mean(dim=-1)
    side = int(round(math.sqrt(err.shape[-1])))
    if side * side != err.shape[-1] or side != grid_size:
        raise ValueError(
            f"token count {err.shape[-1]} does not form a {grid_size}x"
            f"{grid_size} grid — the token-grid geometry is fixed by the "
            f"substrate (Agent C: 56 = 4 x 14); got side {side}")
    return err.reshape(err.shape[0], grid_size, grid_size)


def token_grid_centers(gaze: torch.Tensor, fovea_size: int = 56,
                       img_size: int = 56,
                       grid_size: int = 4) -> torch.Tensor:
    """Image-space coordinates of every token center, (B, grid, grid, 2).

    Pure coordinate arithmetic (Agent C's foveation conventions):
    image_coord = gaze + (fovea_size / img_size) * crop_norm, with
    crop_norm of patch-center (r, c) = (c/2 - 0.75, r/2 - 0.75).
    Order is row-major, matching the backbone's token flattening.
    """
    if gaze.dim() != 2 or gaze.shape[1] != 2:
        got = tuple(gaze.shape)
        raise ValueError(f"gaze must be (B, 2); got {got}")
    device, dtype = gaze.device, gaze.dtype
    scale = fovea_size / float(img_size)
    offs = (torch.arange(grid_size, device=device, dtype=dtype) / 2.0) - 0.75
    # (grid, grid, 2): last dim = (x-offset, y-offset)
    gx, gy = torch.meshgrid(offs, offs, indexing="xy")
    offsets = torch.stack([gx, gy], dim=-1)              # (g, g, 2)
    return gaze[:, None, None, :] + scale * offsets[None]


def anchor_from_map(err_map: torch.Tensor, gaze: torch.Tensor,
                    fovea_size: int = 56, img_size: int = 56,
                    max_abs_gaze: float = MAX_ABS_GAZE_DEFAULT
                    ) -> torch.Tensor:
    """The argmax token center of the error map, (B, 2) image coords.

    Gen-0's anchor rule ("the argmax of the current error map — the
    place the belief is most wrong about right now"), evaluated on the
    token grid instead of the pixel grid. Detached: a coordinate record.
    """
    B, G, _ = err_map.shape
    with torch.no_grad():
        flat = err_map.detach().reshape(B, -1)
        idx = flat.argmax(dim=1)                          # (B,)
        r, c = idx // G, idx % G
        centers = token_grid_centers(gaze, fovea_size, img_size, G)
        gathered = centers.reshape(B, -1, 2).gather(
            1, idx.reshape(B, 1, 1).expand(B, 1, 2)).squeeze(1)
        # (guard the gather against future reshapes: r/c must agree)
        alt = torch.stack([
            gaze[:, 0] + (fovea_size / float(img_size)) * (c / 2.0 - 0.75),
            gaze[:, 1] + (fovea_size / float(img_size)) * (r / 2.0 - 0.75),
        ], dim=-1)
        if not torch.allclose(gathered, alt, atol=1e-6):
            raise AssertionError(
                "token-grid gather disagrees with the closed-form anchor "
                "— the row-major token layout assumption is broken")
        return torch.clamp(gathered, -max_abs_gaze, max_abs_gaze)


def saliency_at(err_map: torch.Tensor, gaze: torch.Tensor,
                candidates: torch.Tensor, fovea_size: int = 56,
                img_size: int = 56) -> torch.Tensor:
    """Heuristic per-candidate saliency score, (B, K).

    Samples the token error map (nearest token) at each candidate
    location — the t = 0 scoring surface (AIS_T0_SCORING: the fallback
    scores candidates with the SAME heuristic that generated them).
    Pure coordinate lookup under no_grad; no predictor, no U_t.
    """
    B, G, _ = err_map.shape
    K = candidates.shape[1]
    with torch.no_grad():
        crop_norm = (candidates.detach() - gaze[:, None, :]) \
            * (img_size / float(fovea_size))              # (B, K, 2) in [-1, 1]
        # normalized -> pixel -> nearest token index (row-major)
        px = (crop_norm + 1.0) * (fovea_size / 2.0)       # (B, K, 2)
        tok = (px / (fovea_size / float(G))).long().clamp(0, G - 1)
        flat_idx = tok[..., 1] * G + tok[..., 0]          # row-major (row=y)
        return err_map.detach().reshape(B, -1).gather(1, flat_idx)


class HeuristicCandidateSampler:
    """K candidates around the token-error anchor — the ported heuristic.

    Generation is deterministic given the anchor and the generator; the
    learned part of AIS-v2 lives in the POLICY's selection, never here
    (the heuristic itself stays exactly as validated in Gen-0).
    """

    def __init__(self, num_candidates: int = 4,
                 candidate_sigma: float = CANDIDATE_SIGMA_DEFAULT,
                 max_abs_gaze: float = MAX_ABS_GAZE_DEFAULT,
                 fovea_size: int = 56, img_size: int = 56,
                 grid_size: int = 4):
        _validate_num_candidates(num_candidates)
        if candidate_sigma <= 0:
            raise ValueError("candidate_sigma must be positive")
        self.num_candidates = int(num_candidates)
        self.candidate_sigma = float(candidate_sigma)
        self.max_abs_gaze = float(max_abs_gaze)
        self.fovea_size = int(fovea_size)
        self.img_size = int(img_size)
        self.grid_size = int(grid_size)

    def anchor(self, err_map: torch.Tensor, gaze: torch.Tensor
               ) -> torch.Tensor:
        return anchor_from_map(err_map, gaze, self.fovea_size,
                               self.img_size, self.max_abs_gaze)

    def saliency(self, observed_tokens: torch.Tensor,
                 predicted_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        """(B, grid, grid) error/saliency surface (see token_error_map)."""
        return token_error_map(observed_tokens, predicted_tokens,
                               self.grid_size)

    def forward(self, err_map: torch.Tensor, gaze: torch.Tensor,
                generator: Optional[torch.Generator] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, K, 2) candidates + the (B, grid, grid) map they came from.

        Candidate 0 = anchor (stay put); 1..K-1 = anchor + Gaussian
        noise (Gen-0's sampling, verbatim). Coordinates are detached.
        """
        anchor = self.anchor(err_map, gaze)               # (B, 2) detached
        B = anchor.shape[0]
        with torch.no_grad():
            noise = torch.randn(
                B, self.num_candidates, 2, device=anchor.device,
                dtype=anchor.dtype, generator=generator) * self.candidate_sigma
            cand = anchor.unsqueeze(1).expand(
                B, self.num_candidates, 2).clone()
            cand[:, 1:] = cand[:, 1:] + noise[:, 1:]
            cand = torch.clamp(cand, -self.max_abs_gaze, self.max_abs_gaze)
        return cand, err_map

    def __repr__(self) -> str:
        return (f"HeuristicCandidateSampler(K={self.num_candidates}, "
                f"sigma={self.candidate_sigma}, "
                f"max_abs_gaze={self.max_abs_gaze}, "
                f"grid={self.grid_size}x{self.grid_size}) — ported from "
                f"rhan_core.gaze.info_gain_policy_v2 at patch-token "
                f"granularity")
