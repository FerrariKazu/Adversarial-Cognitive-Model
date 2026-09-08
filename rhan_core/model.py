"""
RHANNext — the refactored, pillar-composable successor to RHAN-v12.

Design contract (enforced by tests/test_config_backward_compat.py):

  * RHANNext subclasses the FROZEN RHANv12 (phase1_training/model_rhan_v12.py
    is never modified). With the DEFAULT config (all pillars off) the state
    dict is byte-identical to RHANv12's and `forward` delegates to the exact
    v12 implementation — a v12 checkpoint loads 1:1, and the existing
    eval/pipeline keeps working unchanged.
  * New pillar components are added ONLY as new submodules behind
    RHANNextConfig toggles, so each mechanism can be isolated with an on/off
    test (project lesson #3: never add multiple mechanisms simultaneously).
  * Every new loss-bearing path (reconstruction, gaze policy, precision
    modulator, HPC stack) has an automated gradient-reachability test
    (project lesson #1).

Pillars:
  * AIS (Pillar 2, Stage 1)   — InformationGainGazePolicy + EntropyGatedHalting
                                + GlobalPrecisionModulator, gated by enable_ais.
  * HPC (Pillar 1, Stage 2)   — HierarchicalPredictiveStack (1 level), gated
                                by enable_hpc / hpc_num_levels.
  * SBR (Pillar 3) / IWM (Pillar 4) — scaffold only; NullWorldModel is always
                                wired as the safe no-op; enable_sbr/iwm
                                validate to an error.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# The frozen v12 chain inserts phase1_training on sys.path on import; we do the
# same explicitly so this package works regardless of the caller's cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_P1_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "phase1_training"))
for _p in (_P1_DIR, os.path.abspath(os.path.join(_THIS_DIR, ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_rhan_v12 import RHANv12                      # frozen backbone
from model_rhan_v10 import foveal_sample                # frozen helper

from rhan_core.beliefs.vector_belief import VectorBeliefState
from rhan_core.beliefs.structured_belief import StructuredBeliefState
from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.gaze.info_gain_policy import InformationGainGazePolicy
from rhan_core.precision.global_precision import GlobalPrecisionModulator
from rhan_core.world_model.null_world_model import NullWorldModel


class RHANNext(RHANv12):
    """
    RHAN-Next: pillar-composable successor to RHAN-v12.

    Args:
        config: RHANNextConfig — the ONLY configuration object. Defaults to
                RHANNextConfig() (v12-equivalent). Alternative: pass config
                fields as keyword arguments (RHANNext(enable_ais=True, ...)).
    """

    def __init__(self, config: Optional[RHANNextConfig] = None, **kwargs):
        self.config = config if config is not None else RHANNextConfig(**kwargs)
        self.config.validate()
        super().__init__(**self.config.v12_kwargs())
        self._build_pillars()

    # ────────────────────────────────────────────────────────────────────────
    # Pillar construction
    # ────────────────────────────────────────────────────────────────────────
    def _build_pillars(self):
        # Pillar 4 (IWM): always present as a safe no-op (zero params/buffers,
        # so the default state dict stays identical to RHANv12's).
        self.world_model = NullWorldModel()

        # ── Pillar 3 (SBR) — Stage 4-E2 (legacy) + RHAN-NX ladder ───────────
        if self.config.enable_sbr:
            # Binding mode (RHAN-NX): "legacy" keeps the E2 N=1 wiring over
            # the single pooled vector — the ONLY wiring every pre-RHAN-NX
            # SBR checkpoint carries. SBR-0..4 (gate_only .. uncertainty)
            # bind over the SPATIAL stem feature map (B, 144, 768), which is
            # what the SBR-0 gate metrics require (occupancy entropy,
            # pairwise attention-map cosine, per-slot linear probes). The
            # spatial tap is 768-dim vs slot_dim 512 — StructuredBeliefState
            # projects it via input_proj (built only in spatial mode so E2's
            # state dict loads unchanged).
            spatial_binding = self.config.sbr_stage != "legacy"
            relational = self.config.sbr_stage in ("relational", "uncertainty")
            evidence = self.config.sbr_stage in ("relational", "uncertainty")
            self.structured_belief = StructuredBeliefState(
                num_slots=self.config.sbr_num_slots,
                slot_dim=self.config.sbr_slot_dim,
                iters=self.config.sbr_slot_iters,
                max_steps=self.config.max_foraging_steps,
                num_heads=self.config.sbr_num_heads,
                input_dim=768 if spatial_binding else None,
                use_relational=relational,
                use_evidence=evidence,
                uncertainty_mode=self.config.sbr_stage == "uncertainty",
            )

        # ── Pillar 2 (AIS) — Stage 1 ─────────────────────────────────────────
        if self.config.enable_ais:
            # Wraps the shared image_precision module by plain reference (its
            # consumers are then passed to the gaze policy below).
            self.precision_modulator = GlobalPrecisionModulator(
                image_precision_module=self.image_precision,
                tau=self.config.precision_tau,
                gain=1.0)
            # Plain-reference machinery (NOT submodules) so the state dict is
            # not duplicated: the policy shares the frozen v12 components.
            machinery = {
                'foveal_sample': foveal_sample,
                'generative_prior': self.generative_prior,
                'foveal_stream': self.foveal_stream,
                'prior_predictor': self.precision_ctrl.prior_predictor,
                # Gaze-step consumer of the precision modulator (gain-scaled).
                'modulate_step_size': self.precision_modulator.modulate_step_size,
            }
            if self.config.ais_variant == "info_gain_v2":
                # AIS-v2 (RHAN-NX swap): genuine one-step-lookahead expected
                # information gain over K candidates — replaces AIS-v1's
                # relocated-Eq.-II gradient ascent on current error (the
                # mechanistic gap documented in info_gain_policy.py). The
                # module is named gaze_policy_v2 so its params carry that
                # prefix — Generation-0 group "ais_v2" claims exactly this
                # candidate-evaluation head, never AIS-v1's step_net.
                from rhan_core.gaze.info_gain_policy_v2 import (
                    InformationGainGazePolicyV2)
                self.gaze_policy_v2 = InformationGainGazePolicyV2(
                    proj_dim=self.config.proj_dim,
                    fovea_size=self.config.fovea_size,
                    halt_threshold=self.config.ais_halt_threshold,
                    halt_softness=self.config.ais_continuation_softness,
                    machinery=machinery,
                )
                self.gaze_policy = self.gaze_policy_v2
            else:
                self.gaze_policy = InformationGainGazePolicy(
                    proj_dim=self.config.proj_dim,
                    gaze_lambda=self.config.gaze_lambda,
                    fovea_size=self.config.fovea_size,
                    base_step=self.config.ais_base_step,
                    precision_step_range=self.config.ais_precision_step_range,
                    halt_threshold=self.config.ais_halt_threshold,
                    halt_softness=self.config.ais_continuation_softness,
                    machinery=machinery,
                )
            # The halt policy is owned by the gaze policy; expose it under a
            # stable name for the forward loop and tests.
            self.halt_policy = self.gaze_policy.halter

        # ── Pillar 1 (HPC) — Stage 2 (pixel target) + RHAN-NX (belief target)
        if self.config.enable_hpc and self.config.hpc_num_levels >= 1:
            if self.config.hpc_target == "belief":
                # RHAN-NX D3 swap: predict belief_{t+1} from belief_t instead
                # of the edge-map pixel target (E1's Lens finding motivated
                # this). The LevelPredictor ABC's feature_target generality
                # used for its intended purpose for the first time. Fresh-init
                # predictor — never loaded from the pixel HPC's weights.
                from rhan_core.predictive_coding.hpc_belief_level import (
                    HPCBeliefLevel)
                self.hpc_belief = HPCBeliefLevel(proj_dim=self.config.proj_dim)
                # No pixel stack exists in belief mode; the foraging loop
                # dispatches on hpc_belief vs hpc_level1.
                object.__setattr__(self, "hpc_stack", None)
            else:
                # Stage 2 wiring: exactly ONE level, tap = foveal crop.
                from rhan_core.predictive_coding.hpc_level1 import HPCLevel1
                self.hpc_level1 = HPCLevel1(
                    embed_dim=self.config.embed_dim,
                    tap_layer="foveal_crop",   # documented tap point (see class)
                    proj_dim=self.config.proj_dim,
                    fovea_size=self.config.fovea_size)
                # PLAIN-REFERENCE alias (object.__setattr__ bypasses nn.Module's
                # submodule registration — state_dict() does NOT dedup like
                # named_parameters() does, so a registered alias would duplicate
                # every hpc_weight in checkpoints; caught 2026-08-11 by
                # test_hpc_on_state_dict_has_no_duplicate_keys). State dict keys
                # live ONLY under hpc_level1.stack.*; m.hpc_stack keeps working
                # for the Stage-0-era API (same pattern as the AIS plain-reference
                # machinery).
                object.__setattr__(self, "hpc_stack", self.hpc_level1.stack)

        # ── SBR-0 freeze (narrow unlocked validate() path) ───────────────────
        # freeze_backbone_for_sbr0=True is ONLY legal under sbr_stage="gate_only"
        # (validate() enforces it). Enforced at the MODEL level, not just the
        # trainer: every parameter outside structured_belief.* is frozen, so no
        # code path (trainer, smoke test, eval) can ever silently train the D
        # backbone during the structural-convergence gate — the gate's entire
        # premise is that ONLY the slot-attention parameters move. Frozen
        # parameters still transmit gradients (the classifier path reaches the
        # slots through the frozen backbone); they simply never update.
        if self.config.freeze_backbone_for_sbr0:
            for name, p in self.named_parameters():
                if not name.startswith("structured_belief."):
                    p.requires_grad = False

    def _peripheral_pass(self, x):
        """v12 peripheral pass + the RHAN-NX spatial stem tap (SBR-0..4).

        The frozen v12 implementation returns only the CLS token; SBR-0..4's
        slot attention binds over the SPATIAL stem feature map (B, 768, 12, 12)
        = 144 positions — the input the SBR-0 gate metrics are designed for.
        We stash the stem features on a plain attribute (never a parameter or
        buffer, so the state dict is untouched) and delegate to the exact v12
        path for the return contract.
        """
        if hasattr(self, 'structured_belief') and self.config.sbr_stage != 'legacy':
            self._last_stem_features = self.stem(x)          # (B, 768, 12, 12)
        return super()._peripheral_pass(x)

    @property
    def pillars_active(self) -> bool:
        """True when any implemented pillar is enabled (non-default path)."""
        return self.config.enable_ais or (
            self.config.enable_hpc and self.config.hpc_num_levels >= 1) or \
            self.config.enable_sbr

    # ────────────────────────────────────────────────────────────────────────
    # Shared foraging loop (AIS/HPC-aware). Mirrors v12's loop exactly when
    # pillars are off; when on, adds belief/uncertainty bookkeeping, the
    # entropy-gated continuation weights, the info-gain gaze update, and the
    # HPC prediction errors.
    # ────────────────────────────────────────────────────────────────────────
    def _forage(self, x, collect_traj: bool, _step_callback=None):
        """Run the multi-step foraging loop.

        Args:
            x: (B, 3, 96, 96)
            collect_traj: build the trajectory dict (diagnostics/losses).
            _step_callback: optional callable invoked at each step with a dict
                of the current internal state. Observational only — must not
                mutate state or affect gradients. When None (default), behavior
                is identical to the original implementation.
        Returns:
            (final_belief (B, 512), trajectory dict or None)
        """
        B = x.shape[0]

        # Step 0: peripheral pass (full image, inherited).
        cls_768 = self._peripheral_pass(x)              # (B, 768)
        s = self.peripheral_proj(cls_768)                # (B, 512)
        if self.freeze_gaze:
            a = torch.zeros(B, 2, device=x.device)       # center fixation
        else:
            a = self.action_init(s)                      # (B, 2)

        # Parafoveal: computed ONCE (full-field, low-res).
        para_feat = self.parafoveal_stream(x)            # (B, 512)

        # Accumulators (constant continuation = 1 in pure v12 mode; AIS uses
        # the soft uncertainty gate below).
        weighted_belief = torch.zeros_like(s)            # (B, 512)
        weight_sum = torch.zeros(B, device=x.device)     # (B,)

        trajectory = None
        if collect_traj:
            trajectory = {
                'actions': [], 'precisions': [], 'errors': [], 'gate_alphas': [],
                'recon_errors': [], 'recon_maps': [], 'steps': 0,
                'uncertainties': [], 'continuations': [],
                'step_beliefs': [],   # per-step 512-dim belief state
            }
            hpc_on = (hasattr(self, 'hpc_belief')
                      or (hasattr(self, 'hpc_stack')
                          and self.hpc_stack is not None
                          and len(self.hpc_stack.levels) > 0))
            if hpc_on:
                trajectory['hpc_errors'] = []
                trajectory['hpc_error_maps'] = []

        history: list = []

        # SBR temporal state: slots carry forward between foraging steps
        sbr_state = None
        # RHAN-NX D3 (belief-HPC): previous step's attached belief, the
        # predictor's delayed input.
        _prev_hpc_belief = None
        # AIS-v2: the candidate head's attached prediction at the last chosen
        # candidate, paired with next step's observed surprise as TD target.
        _eig_predicted = None

        for t in range(self.max_steps):
            # Eq. II: sample foveal crop at gaze position.
            x_foveal = foveal_sample(x, a, fovea_size=self.fovea_size)
            foveal_feat = self.foveal_stream(x_foveal)   # (B, 512)

            # Tier 1.1: blend foveal + parafoveal via learned gate.
            combined_feat, alpha = self.foveal_gate(foveal_feat, para_feat, s)

            # Tier 3.1: generative prior predicts expected crop.
            predicted_crop = self.generative_prior(s)    # (B, 3, 48, 48)

            # Image-space prediction error (genuine, bounded). RAW pi_d — the
            # modulator's gain only enters via explicit consumer modulations.
            if hasattr(self, 'precision_modulator'):
                pi_d, error_mag = self.precision_modulator.precision_from_crops(
                    x_foveal, predicted_crop, s)
            else:
                pi_d, error_mag = self.image_precision(
                    x_foveal, predicted_crop, s)

            # Precision-weighted belief integration (v12 semantics).
            pi_d_unsq = pi_d.unsqueeze(-1)               # (B, 1)
            s = (1 - pi_d_unsq) * s + pi_d_unsq * combined_feat

            # SBR: slot attention → structured belief
            if hasattr(self, 'structured_belief'):
                if self.config.sbr_stage == 'legacy':
                    # E2-equivalent wiring: slots bind over the single pooled
                    # vector (N=1). Kept byte-identical to E2 so every legacy
                    # SBR checkpoint loads and runs unchanged.
                    feat_for_slots = combined_feat.unsqueeze(1)  # (B, 1, 512)
                else:
                    # SBR-0..4: slots bind over the spatial stem feature map
                    # (B, 768, 12, 12) -> (B, 144, 768) — 144 spatial positions
                    # (stashed by _peripheral_pass). Positional structure comes
                    # from the conv stem itself (12x12 receptive-field layout).
                    sf = self._last_stem_features
                    feat_for_slots = sf.flatten(2).transpose(1, 2)  # (B,144,768)
                sbr_out = self.structured_belief(feat_for_slots, sbr_state)
                sbr_state = {'slots': sbr_out['prev_slots'].detach(),
                             'pooled': sbr_out['pooled'].detach()}
                # SBR-3/4: the evidence decomposition's pooled evidence IS the
                # belief (shape/texture/spatial evidence -> combination layer).
                if sbr_out.get('evidence') is not None:
                    s = sbr_out['evidence']['pooled_evidence']
                else:
                    s = sbr_out['pooled']
                if collect_traj:
                    trajectory['sbr_entropy'] = trajectory.get('sbr_entropy', [])
                    trajectory['sbr_entropy'].append(sbr_out['entropy'].detach())
                    if sbr_out.get('relation_attn') is not None:
                        trajectory['sbr_relation_attn'] = trajectory.get(
                            'sbr_relation_attn', [])
                        trajectory['sbr_relation_attn'].append(
                            sbr_out['relation_attn'].detach())
                    if sbr_out.get('evidence') is not None:
                        trajectory['sbr_evidence'] = trajectory.get(
                            'sbr_evidence', [])
                        trajectory['sbr_evidence'].append(sbr_out['evidence'])
                    trajectory['sbr_slots'] = trajectory.get('sbr_slots', [])
                    trajectory['sbr_slots'].append(sbr_out['slots'].detach())
                    trajectory['sbr_attn'] = trajectory.get('sbr_attn', [])
                    trajectory['sbr_attn'].append(sbr_out['attn'].detach())

            # HPC prediction errors (Pillar 1) — NOT detached so the error
            # reaches the predictor's parameters through the loss. Computed
            # when trajectory is collected OR when a step callback is set
            # (live perception needs HPC data at each step).
            _hpc_pred = _hpc_err = _hpc_err_map = None
            if hasattr(self, 'hpc_level1') and (collect_traj or _step_callback):
                pred_hpc, err_hpc, err_map = self.hpc_level1(s, x_foveal)
                _hpc_pred, _hpc_err, _hpc_err_map = pred_hpc, err_hpc, err_map
                if collect_traj:
                    trajectory['hpc_errors'].append(err_hpc)     # (B,), attached
                    trajectory['hpc_error_maps'].append({
                        'min': float(err_map.min().detach()),
                        'max': float(err_map.max().detach()),
                        'std': float(err_map.std().detach()),
                    })
            elif hasattr(self, 'hpc_belief') and (collect_traj or _step_callback):
                # RHAN-NX D3: delayed belief-prediction step — predict
                # belief_t from belief_{t-1} (the predictor genuinely sees
                # belief_t and targets belief_{t+1} one step later). The
                # input is ATTACHED (prediction path backprops into it); the
                # target side is DETACHED (the bottom-up actual never
                # contributes gradients — same contract as the pixel
                # extractor's detached edge-map target).
                if t >= 1 and _prev_hpc_belief is not None:
                    pred_hpc, err_hpc, err_map = self.hpc_belief.step(
                        _prev_hpc_belief, s.detach())
                    _hpc_pred, _hpc_err, _hpc_err_map = pred_hpc, err_hpc, err_map
                    if collect_traj:
                        trajectory['hpc_errors'].append(err_hpc)  # (B,), attached
                        trajectory['hpc_error_maps'].append({
                            'min': float(err_map.min().detach()),
                            'max': float(err_map.max().detach()),
                            'std': float(err_map.std().detach()),
                        })
            if hasattr(self, 'hpc_belief'):
                # Carry the CURRENT attached belief as next step's predictor
                # input (delayed-by-one wiring). Only updated in the
                # collect/callback path to keep the no-callback path
                # byte-identical to v12.
                _prev_hpc_belief = s if (collect_traj or _step_callback) else None

            # ── Belief wrapper + policies (AIS); v12 fallback otherwise ────
            has_ais = hasattr(self, 'halt_policy') and hasattr(self, 'gaze_policy')
            if has_ais:
                if (hasattr(self, 'structured_belief')
                        and self.config.sbr_stage != 'legacy'):
                    # SBR-0..4: uncertainty is the slot-attention entropy
                    # (SBR-4: the evidence decomposition's first-class
                    # uncertainty output) — the designed SBR halting signal,
                    # replacing the flat 1 - Pi_D proxy. Legacy SBR keeps
                    # 1 - Pi_D so E2 behavior is untouched.
                    u = sbr_out.get('uncertainty', sbr_out['entropy'])
                else:
                    u = 1.0 - pi_d                         # uncertainty proxy
                belief = VectorBeliefState(s, uncertainty=u)
                ctx = {'action': a, 'image': x, 'belief_tensor': s,
                       'precision': pi_d, 'step': t}
                if hasattr(self, 'precision_modulator'):
                    ctx['halt_threshold'] = \
                        self.precision_modulator.modulate_halting_threshold(
                            pi_d, self.config.ais_halt_threshold)
                if self.config.ais_halt_enabled:
                    halt = self.halt_policy.should_halt(belief, [ctx])   # (B,) bool
                    ctx['halt'] = halt
                    history.append(ctx)
                    # Soft continuation: sigma(softness * (u - threshold)).
                    cont = self.halt_policy.continuation(belief, [ctx])   # (B,)
                else:
                    # ISOLATION A (--no-ais-halting): entropy gate forced open.
                    # cont = 1 for every sample -> belief accumulation is v12's
                    # fixed-T semantics (mean over all steps). The gaze update
                    # below still runs; only per-sample halting is disabled.
                    halt = torch.zeros(B, dtype=torch.bool, device=x.device)
                    ctx['halt'] = halt
                    history.append(ctx)
                    cont = torch.ones(B, device=x.device)
            else:
                # Exact v12 semantics: constant continuation = 1, no halt.
                halt = torch.zeros(B, dtype=torch.bool, device=x.device)
                cont = torch.ones(B, device=x.device)
                u = 1.0 - pi_d

            # Accumulate belief, weighted by continuation (AIS halting).
            weighted_belief += cont.unsqueeze(-1) * s
            weight_sum += cont

            # Record trajectory for diagnostics/losses.
            if collect_traj:
                trajectory['step_beliefs'].append(s.detach())  # (B, 512)
                trajectory['actions'].append(a.detach())
                trajectory['precisions'].append(pi_d.detach())
                trajectory['errors'].append(error_mag.detach())
                trajectory['gate_alphas'].append(alpha.detach())
                # NOTE: NOT detached — get_reconstruction_loss() must return a
                # differentiable scalar (v12 fix; v11 detached -> no-op).
                trajectory['recon_errors'].append(
                    (x_foveal - predicted_crop).pow(2).mean())
                trajectory['uncertainties'].append(u.detach())
                trajectory['continuations'].append(cont.detach())

            # ── Live perception callback (observational only) ─────────────
            # Fires at each step when _step_callback is set. Receives a
            # snapshot dict of the current internal state. Must NOT mutate
            # state or affect gradients. Detached tensors only.
            if _step_callback is not None:
                _cb_data = {
                    'step': t,
                    'max_steps': self.max_steps,
                    'gaze_x_norm': float(a[0, 0].detach()),
                    'gaze_y_norm': float(a[0, 1].detach()),
                    'foveal_crop': x_foveal[0].detach() if x_foveal is not None else None,
                    'predicted_crop': predicted_crop[0].detach() if predicted_crop is not None else None,
                    'pi_d': float(pi_d[0].detach()),
                    'error_mag': float(error_mag[0].detach()),
                    'uncertainty': float(u[0].detach()),
                    'gate_alpha': float(alpha[0].detach()),
                    'recon_error': float((x_foveal - predicted_crop).pow(2).mean().detach()),
                    'step_belief': s[0].detach(),
                }
                if _hpc_err is not None:
                    _cb_data['hpc_prediction'] = _hpc_pred[0].detach() if _hpc_pred is not None else None
                    _cb_data['hpc_error'] = float(_hpc_err[0].detach())
                    _cb_data['hpc_error_map'] = _hpc_err_map[0].detach() if _hpc_err_map is not None else None
                if has_ais:
                    _cb_data['continuation'] = float(cont[0].detach())
                    _cb_data['halted'] = bool(halt[0].detach()) if halt is not None else False
                else:
                    _cb_data['continuation'] = 1.0
                    _cb_data['halted'] = False
                try:
                    _step_callback(_cb_data)
                except Exception:
                    pass  # never let a callback error break inference

            # ── AIS-v2: one-step TD pair for the candidate-evaluation head ──
            # The head predicted the surprise at the candidate chosen at step
            # t-1; the surprise ACTUALLY observed at THIS step (the fixation
            # the policy moved to) is its detached TD target. This is the
            # ONLY gradient path into candidate_head (the candidate features
            # are detached by design — documented approximation 3), so the
            # trainer's L_eig = MSE(predicted, observed) is what makes the
            # head's predictions informative about where surprise lands.
            if (has_ais and collect_traj
                    and self.config.ais_variant == 'info_gain_v2'
                    and _eig_predicted is not None):
                prior_pred_t = self.precision_ctrl.prior_predictor(s.detach())
                obs_surprise = ((foveal_feat - prior_pred_t).norm(dim=-1)
                                / math.sqrt(foveal_feat.shape[-1]))
                # Predicted side stays ATTACHED — it is the only gradient
                # path into candidate_head (the target is detached, standard
                # TD semantics).
                trajectory.setdefault('eig_pairs', []).append(
                    (_eig_predicted, obs_surprise.detach()))

            # Eq. II v12: gaze update (info-gain policy under AIS).
            if not self.freeze_gaze and t < self.max_steps - 1:
                if hasattr(self, 'gaze_policy'):
                    a = self.gaze_policy.select_action(belief, history)
                    # AIS-v2: capture the head's ATTACHED prediction at the
                    # candidate actually chosen — next step's TD target pair.
                    if (collect_traj
                            and self.config.ais_variant == 'info_gain_v2'
                            and getattr(self.gaze_policy,
                                        'last_chosen_surprise', None)
                            is not None):
                        _eig_predicted = self.gaze_policy.last_chosen_surprise
                else:
                    g_total, recon_map = self._gaze_gradients(x, a, s)
                    if collect_traj:
                        trajectory['recon_maps'].append(recon_map.detach())
                    grad_norm = g_total.norm(dim=-1, keepdim=True) + 1e-8
                    normed_grad = g_total / grad_norm
                    step_size = 0.20 + 0.30 * pi_d.unsqueeze(-1)
                    a = torch.clamp(a + step_size * normed_grad, -0.9, 0.9)
                if collect_traj and hasattr(self, 'gaze_policy'):
                    # Diagnostics-only recon map from the policy's last step.
                    if self.gaze_policy.last_recon_map is not None:
                        trajectory['recon_maps'].append(
                            self.gaze_policy.last_recon_map)

        if collect_traj:
            trajectory['steps'] = self.max_steps

        final_belief = weighted_belief / (weight_sum.unsqueeze(-1) + 1e-8)
        return final_belief, trajectory

    # ────────────────────────────────────────────────────────────────────────
    # Public API — default config delegates EXACTLY to the frozen v12 path.
    # ────────────────────────────────────────────────────────────────────────
    def forward(self, x, return_trajectory=False, _step_callback=None):
        """v12-compatible forward. Default config: byte-for-byte v12.

        Args:
            x: input tensor.
            return_trajectory: return (logits, trajectory_dict).
            _step_callback: optional callable for live perception (observational
                only — receives a snapshot dict at each foraging step).
        """
        if not self.pillars_active:
            return super().forward(x, return_trajectory=return_trajectory)
        final_belief, trajectory = self._forage(
            x, collect_traj=return_trajectory, _step_callback=_step_callback)
        final_768 = self.belief_unproj(final_belief)     # (B, 768)
        logits = self.classifier(final_768)
        if return_trajectory:
            return logits, trajectory
        return logits

    def get_feature_vector(self, x):
        """v12-compatible 768-dim feature vector (TRADES/eval compat)."""
        if not self.pillars_active:
            return super().get_feature_vector(x)
        final_belief, _ = self._forage(x, collect_traj=False)
        return self.belief_unproj(final_belief)

    def get_hpc_loss(self, x, trajectory_logits_tuple):
        """Differentiable HPC prediction-error loss (0.0 when HPC is off).

        Args:
            x: (B, 3, 96, 96) input.
            trajectory_logits_tuple: (logits, trajectory) from forward().
        Returns:
            scalar — mean of the collected per-level prediction errors.
        """
        _, traj = trajectory_logits_tuple
        errs = traj.get('hpc_errors', []) if isinstance(traj, dict) else []
        if not errs:
            return torch.zeros((), device=x.device)
        return torch.stack(errs).mean()
