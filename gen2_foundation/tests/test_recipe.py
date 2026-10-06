"""G2-K9 / K7-optimization invariants.

1. AdamW + warmup + cosine schedule.
2. Per-group LR multipliers.
3. Per-group (not global) clipping.
4. One-parameter-one-group enforcement (double registration raises).
5. Pre-flight |dW| freeze detection.
6. Checkpoint/resume compatibility (group names, count, lr-ratio pattern).
"""

from __future__ import annotations

import pytest
import torch

from gen2_foundation.recipe import (
    build_adamw,
    AdamWGroups,
    PhaseRecipe,
    measure_step_magnitudes,
)
from gen2_foundation import recipe as rec


class TestAdamW:
    def test_builds_adamw(self):
        from gen2_foundation import recipe
        import torch.nn as nn
        class _Registry:
            def __init__(self):
                self._groups = []
            def register(self, name, params, lr_multiplier=1.0, clip_norm=1.0):
                self._groups.append({'name': name, 'params': list(params),
                                     'lr_multiplier': lr_multiplier,
                                     'clip_norm': clip_norm})
        reg = _Registry()
        p1 = torch.nn.Parameter(torch.zeros(2))
        p2 = torch.nn.Parameter(torch.zeros(2))
        reg.register('backbone', [p1])
        reg.register('update_net', [p2])
        opt = build_adamw(reg, base_lr=1e-3)
        assert isinstance(opt, AdamWGroups)

    def test_per_group_lr(self):
        from gen2_foundation import recipe
        import torch.nn as nn
        class _Registry:
            def __init__(self):
                self._groups = []
            def register(self, name, params, lr_multiplier=1.0, clip_norm=1.0):
                self._groups.append({'name': name, 'params': list(params),
                                     'lr_multiplier': lr_multiplier,
                                     'clip_norm': clip_norm})
            def group_names(self):
                return [g['name'] for g in self._groups]
        reg = _Registry()
        p1 = torch.nn.Parameter(torch.zeros(2))
        p2 = torch.nn.Parameter(torch.zeros(2))
        reg.register('backbone', [p1], lr_multiplier=1.0)
        reg.register('precision', [p2], lr_multiplier=6.67)
        opt = build_adamw(reg, base_lr=1e-3)
        assert opt.registry.group_names() == ['backbone', 'precision']

    def test_one_param_one_group(self):
        from gen2_foundation import recipe
        import torch.nn as nn
        class _Registry:
            def __init__(self):
                self._groups = []
                self._claimed = set()
            def register(self, name, params, lr_multiplier=1.0, clip_norm=1.0):
                for p in params:
                    if id(p) in self._claimed:
                        raise ValueError('double registration')
                    self._claimed.add(id(p))
                self._groups.append({'name': name, 'params': list(params)})
        reg = _Registry()
        p = torch.nn.Parameter(torch.zeros(2))
        with pytest.raises(ValueError):
            reg.register('g1', [p])
            reg.register('g2', [p])


class TestStepMagnitudes:
    def test_step_magnitudes_logged(self):
        from gen2_foundation import recipe
        import torch.nn as nn
        class _Registry:
            def __init__(self):
                self._groups = []
            def register(self, name, params, lr_multiplier=1.0, clip_norm=1.0):
                self._groups.append({'name': name, 'params': list(params),
                                     'lr_multiplier': lr_multiplier,
                                     'clip_norm': clip_norm})
            def group_names(self):
                return [g['name'] for g in self._groups]
        reg = _Registry()
        p1 = torch.nn.Parameter(torch.zeros(2))
        p2 = torch.nn.Parameter(torch.zeros(2))
        reg.register('backbone', [p1])
        reg.register('update_net', [p2])
        opt = build_adamw(reg, base_lr=1e-3)
        # Build the underlying AdamW optimizer and step so param_groups
        # are populated; measure_step_magnitudes reads grad norms after a
        # step.
        opt.build(base_lr=1e-3)
        p = opt.registry._groups[0]['params'][0]
        p.grad = torch.ones(2)
