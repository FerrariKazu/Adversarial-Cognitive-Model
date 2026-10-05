#!/usr/bin/env python3
"""
generate_complexity_tables.py
=============================
Measures the computational footprint of the REAL RHAN-v12 on the local GPU:

  * analytic FLOPs per module, accumulated by forward hooks over an actual
    forward pass (the T=4 loop, recurrent feedback and gaze-gradient path
    are all counted because the hooks fire per real call);
  * wall-clock latency + peak VRAM measured on the local device;
  * a comparison table vs. reference architectures (params/FLOPs from the
    literature, cited in bibliography.bib).

Emits report/tables/complexity*.csv/.tex and report/generated/complexity.json
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _report_common as C

# Literature reference table (params / FLOPs at each model's native input).
# Citations live in report/bibliography.bib.
BASELINES = [
    # name, params, gflops, input res, citation key
    ('ResNet-18',             11.69e6,  1.82, '224', 'he2016deep'),
    ('WideResNet-28-10',      36.49e6,  5.24, '32',  'zagoruyko2016wide'),
    ('EfficientNet-B0',        5.33e6,  0.39, '224', 'tan2019efficientnet'),
    ('ViT-Small',             22.05e6,  4.61, '224', 'dosovitskiy2021image'),
    ('ViT-B/16',              86.57e6, 17.56, '224', 'dosovitskiy2021image'),
    ('Swin-T',                28.29e6,  4.51, '224', 'liu2021swin'),
    ('ConvNeXt-T',            28.59e6,  4.47, '224', 'liu2022convnext'),
]

# base RHAN-Large backbone params (same stem/transformer family as v12)
from phase1_training.model_rhan_stl10_large import RHANLargeSTL10  # noqa: E402
BASE_PARAMS = sum(p.numel() for p in RHANLargeSTL10().parameters())


def flop_rows_to_tex(rows, total):
    lines = [r"\begin{longtable}{l r r}",
             r"\toprule",
             r"\textbf{Module} & \textbf{FLOPs (per image)} & "
             r"\textbf{Share} \label{tab:flops} \\",
             r"\midrule",
             r"\endhead"]
    for r in rows:
        lines.append(f"{C.tex_escape(r['module'])} & "
                     f"{r['flops']:,.0f} & "
                     f"{100.0 * r['flops'] / total:.2f}\\% \\\\")
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & {total:,.0f} & 100.00\\% \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def baselines_tex():
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\begin{tabular}{l r r l l}",
             r"\toprule",
             r"\textbf{Model} & \textbf{Params} & \textbf{GFLOPs} & "
             r"\textbf{Input} & \textbf{Source} \\",
             r"\midrule"]
    for name, params, gflops, res, cite in BASELINES:
        lines.append(f"{name} & {params:,.0f} & {gflops:.2f} & "
                     f"{res}$\\times${res} & \\cite{{{cite}}} \\\\")
    lines.append(f"RHAN-Large (backbone) & {BASE_PARAMS:,.0f} & -- & "
                 f"96$\\times$96 & \\cite{{rhanv12}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Reference architectures. FLOPs are reported at "
                 r"each model's native resolution (224 or 32); direct FLOP "
                 r"comparisons across resolutions are not apples-to-apples. "
                 r"RHAN-v12's own measured footprint appears in "
                 r"Table~\ref{tab:footprint}.}")
    lines.append(r"\label{tab:baselines}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    C.set_style()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[complexity] device = {device}", flush=True)

    model = C.build_model(load_ckpt=True, device=device)
    total_params = sum(p.numel() for p in model.parameters())

    # ---- hook-counted FLOPs on a real forward pass ----
    counter = C.FlopCounter().attach(model)
    x = torch.randn(1, 3, 96, 96, device=device)
    with torch.no_grad():
        with counter:
            model(x)
    records = counter.to_records()
    total_flops = sum(r['flops'] for r in records)

    # ---- latency + VRAM on the same device ----
    latency = C.measure_latency_memory(model, device, batch=1, reps=12)
    vram = latency.pop('peak_vram_bytes')

    # ---- top-level share ----
    top_share = {}
    for r in records:
        top = r['module'].split('.')[0]
        top_share[top] = top_share.get(top, 0.0) + r['flops']

    summary = {
        'total_params': total_params,
        'total_flops': total_flops,
        'gflops': total_flops / 1e9,
        'latency_ms_per_img': latency['ms_per_img'],
        'fps': latency['fps'],
        'peak_vram_mb': vram / 1e6,
        'top_level_flop_share': {k: round(v, 1) for k, v in
                                 sorted(top_share.items(),
                                        key=lambda kv: -kv[1])},
    }
    C.save_json(summary, 'complexity.json')
    print(f"[complexity] params={total_params:,} flops={total_flops:,.0f} "
          f"({total_flops/1e9:.2f} G) latency={latency['ms_per_img']:.1f} ms "
          f"vram={vram/1e6:.0f} MB", flush=True)

    C.save_tabular(records, 'complexity_flops.csv')
    with open(os.path.join(C.TAB_DIR, 'complexity_flops.tex'), 'w') as f:
        f.write(flop_rows_to_tex(records, total_flops))

    with open(os.path.join(C.TAB_DIR, 'footprint.tex'), 'w') as f:
        f.write(r"\begin{table}[htbp]" + "\n" + r"\centering" + "\n" +
                r"\begin{tabular}{l r}" + "\n" + r"\toprule" + "\n" +
                r"\textbf{Quantity} & \textbf{Value} \\" + "\n" + r"\midrule" + "\n" +
                f"Total parameters & {total_params:,} \\\\\n" +
                f"Total FLOPs per image (hook-counted) & {total_flops:,.0f} "
                f"({total_flops/1e9:.2f} GFLOPs) \\\\\n" +
                f"Latency per image (batch=1) & {latency['ms_per_img']:.1f} ms "
                f"({latency['fps']:.1f} FPS) \\\\\n" +
                f"Peak VRAM during forward (batch=1) & {vram/1e6:.0f} MB \\\\\n" +
                r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" +
                r"\caption{Live-measured RHAN-v12 footprint on the local "
                r"device (RTX 4060, 8 GB). FLOPs are counted by forward hooks, "
                r"so the T=4 foraging loop, the 2-step recurrent feedback and "
                r"the gaze-gradient path are included.}" + "\n" +
                r"\label{tab:footprint}" + "\n" + r"\end{table}")

    with open(os.path.join(C.TAB_DIR, 'baselines.tex'), 'w') as f:
        f.write(baselines_tex())

    print("[complexity] wrote complexity_flops.tex, footprint.tex, "
          "baselines.tex", flush=True)


if __name__ == '__main__':
    main()
