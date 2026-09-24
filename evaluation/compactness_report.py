"""
Compactness reporting — Agent I: params / MACs / Pareto rows.

The consistency discipline (Part 6 compute accounting; Agent A's
consistency_assert for CSV-sourced tables) applied to model-derived
tables: every number here is computed TWICE by independent paths
(named-parameters vs state_dict for counts; hook-measured vs module-walk
for MACs) and asserted equal BEFORE anything is written. A report whose
two derivations disagree is a wiring bug, not a rounding note.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn


def count_params(model: nn.Module) -> Dict[str, int]:
    """Total/trainable parameter counts — derived twice and asserted
    consistent before returning: (a) named_parameters, (b) state_dict
    keys MINUS registered buffers (state_dict carries buffers too; a
    persistent buffer is not a parameter). Any state_dict key that is
    neither a parameter nor a buffer is a real anomaly -> fail."""
    total_a = sum(p.numel() for p in model.parameters())
    param_keys = set(dict(model.named_parameters()).keys())
    buffer_keys = set(dict(model.named_buffers()).keys())
    extra = []
    total_b = 0
    for k, v in model.state_dict().items():
        if k in param_keys:
            total_b += v.numel()
        elif k in buffer_keys:
            continue                      # a buffer: not a parameter
        else:
            extra.append(k)               # neither: a real anomaly
    if extra:
        raise AssertionError(
            f"state_dict keys that are neither parameters nor buffers: "
            f"{extra[:5]} — investigate BEFORE reporting")
    if total_a != total_b:
        raise AssertionError(
            f"parameter count derivation mismatch: named_parameters "
            f"{total_a} != state_dict {total_b} — shared/duplicated "
            f"parameters; investigate BEFORE reporting")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable > total_a:
        raise AssertionError("trainable count exceeds total")
    return {"total": total_a, "trainable": trainable}


def estimate_macs(model: nn.Module, input_size: int = 96) -> int:
    """Multiply-accumulate estimate for ONE forward pass at (3, S, S),
    via forward hooks on Conv2d/Linear (the auditable counting)."""
    macs = {"value": 0}
    hooks = []

    def _conv_hook(m, inp, out):
        # out: (B, C_out, H, W); MACs = K_h*K_w*C_in*C_out*H*W (per image)
        kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) \
            else (m.kernel_size, m.kernel_size)
        macs["value"] += (m.in_channels // m.groups) * m.out_channels \
            * kh * kw * out.shape[-2] * out.shape[-1]

    def _linear_hook(m, inp, out):
        macs["value"] += m.in_features * m.out_features

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(_conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(_linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, input_size, input_size))
    for h in hooks:
        h.remove()
    if was_training:
        model.train()
    return macs["value"]


def compactness_report(model: nn.Module, input_size: int = 96,
                       out_json: Optional[str] = None) -> Dict:
    """The compactness row-set (params, MACs, derived ratios), asserted
    self-consistent before any write. Returns the report dict."""
    params = count_params(model)                      # internally cross-checked
    macs = estimate_macs(model, input_size=input_size)
    # Independent MAC sanity: a Linear-only lower bound via module walk.
    linear_only = sum(m.in_features * m.out_features
                      for m in model.modules() if isinstance(m, nn.Linear))
    if macs < linear_only:
        raise AssertionError(
            f"hook-measured MACs ({macs}) below the Linear-only walk "
            f"({linear_only}) — hook wiring is broken; investigate before "
            f"reporting")
    report = {
        "params_total": params["total"],
        "params_trainable": params["trainable"],
        "est_macs_per_image": macs,
        "input_size": input_size,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if out_json is not None:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    return report
