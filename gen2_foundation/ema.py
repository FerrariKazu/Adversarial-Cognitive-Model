"""G2-K7-ema — gradient-free EMA target encoder for the predictor target.

Implemented as a plain (non-`nn.Module`) class so it can NEVER be
registered to an optimizer (the gradient-free contract). Online-vs-target
agreement is logged to detect EMA lag.

Explicit EMA-on / EMA-off experiment states.
"""

from __future__ import annotations

from typing import Optional

import torch


class EMATargetEncoder:
    """Frozen EMA shadow of the shared predictor.

    Gradient-free by construction (not an nn.Module; never registered to
    an optimizer). Its output feeds `pred_t` in AIS-v2 and the belief
    update, never the other way around.

    Attributes:
        predictor: the training predictor (received, shared, never
            re-uploaded here).
        alpha: EMA smoothing coefficient in (0, 1);
            w_ema = alpha * w_ema + (1 - alpha) * w_src.
        ema_decay: alias for alpha (kept for readability in call sites).
        agreement_log: rolling buffer of online-vs-target feature
            cosine similarity, one entry per sync step.
    """

    def __init__(
        self,
        predictor: object,
        alpha: float = 0.996,
        ema_decay: Optional[float] = None,
        memory: int = 1000,
    ) -> None:
        # alpha must be in [0, 1]; alpha=1 gives 'ema-off' (no decay),
        # alpha<1 gives EMA-on (Polyak averaging).
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0,1]; got {alpha}")
        object.__setattr__(self, "predictor", predictor)
        self.alpha = float(alpha)
        if ema_decay is not None:
            self.alpha = float(ema_decay)
        self._memory = int(memory)
        self.agreement_log = []                   # one entry per sync
        # Mirror the predictor's state-dict keys as plain buffers so
        # sync() can polyak-blend them (gradient-free by construction).
        _src = predictor.state_dict()
        for sh, buf in _src.items():
            if isinstance(buf, torch.Tensor):
                object.__setattr__(self, sh, buf.detach().clone())

    # ---- sync ----

    def sync(self) -> None:
        """Polyak step: w_ema <- alpha*w_ema + (1-alpha)*w_src."""
        src_state = self.predictor.state_dict()
        with torch.no_grad():
            for sh_name in list(self.__dict__):
                if sh_name.startswith('_'):
                    continue
                if sh_name not in src_state:
                    continue
                buf = getattr(self, sh_name)
                buf.mul_(self.alpha).add_(src_state[sh_name], alpha=1.0 - self.alpha)

    # ---- agreement ----

    def record_agreement(self, online_feat: torch.Tensor,
                         target_feat: torch.Tensor) -> float:
        """Log online-vs-target feature agreement; return the value."""
        if online_feat.shape != target_feat.shape:
            raise ValueError(
                f"shape mismatch: online {tuple(online_feat.shape)} vs "
                f"target {tuple(target_feat.shape)}")
        # Flatten trailing dims (B, *rest) -> (B, -1) for cosine similarity.
        ob = online_feat.reshape(online_feat.shape[0], -1)
        tb = target_feat.reshape(target_feat.shape[0], -1)
        cos = torch.cosine_similarity(ob, tb, dim=1, eps=1e-6)
        val = float(cos.mean().item())
        self.agreement_log.append(val)
        if len(self.agreement_log) > self._memory:
            self.agreement_log.pop(0)
        return val

    def ema_lag(self) -> Optional[float]:
        """EMA lag diagnostic: negative drift => target too stale."""
        if not self.agreement_log:
            return None
        arr = torch.tensor(self.agreement_log, dtype=torch.float32)
        # Lag = forward trend of agreement. Fit a line via least squares
        # (closed form on 1-D x) and report the slope.
        if len(arr) < 2:
            return 0.0
        x = torch.arange(len(arr), dtype=torch.float32)
        x_mean = x.mean()
        y_mean = arr.float().mean()
        num = ((x - x_mean) * (arr.float() - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        slope = float(num / den) if den > 0 else 0.0
        return slope

    # ---- shadow access (plain buffers, no params) ----

    def _buffer_names(self):
        return list(self.__dict__.keys())

    @property
    def buffer_names(self):
        return list(self.__dict__.keys())

    def state_dict(self) -> dict:
        """Plain dict of shadow buffers (no optimizer state)."""
        return {n: getattr(self, n) for n in self._buffer_names()}

    def load_state_dict(self, state: dict) -> None:
        with torch.no_grad():
            for n, buf in state.items():
                if n in self._buffer_names():
                    getattr(self, n).copy_(buf)
                else:
                    raise KeyError(f"unknown EMA shadow key {n!r}")

    # ---- prediction ----

    def predict_features(self, belief, gaze_location: torch.Tensor) -> torch.Tensor:
        return self.predictor.predict_features(belief, gaze_location)

    def forward(self, belief, gaze_location: torch.Tensor) -> torch.Tensor:
        return self.predict_features(belief, gaze_location)

    def score_candidates(self, belief, candidate_locations: torch.Tensor):
        preds, scores = [], []
        for k in range(candidate_locations.shape[1]):
            pk = self.predict_features(
                belief, candidate_locations[:, k, :])
            preds.append(pk)
            scores.append(pk.mean(dim=(1, 2)))
        return torch.stack(preds, dim=1), torch.stack(scores, dim=-1)
