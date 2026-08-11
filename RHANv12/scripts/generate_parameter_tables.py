#!/usr/bin/env python3
"""
generate_parameter_tables.py
=============================
Introspects the real RHANv12 module tree and emits:

  report/generated/model_stats.json        -- global statistics
  report/tables/module_summary.csv/.tex     -- per-module parameter summary
  report/tables/parameter_inventory.csv/.tex-- EVERY trainable tensor
                                             (appendix A source)

The LaTeX fragments are longtable environments written directly, so the
report never carries hand-maintained numbers: rerun this script and the
appendix updates itself.

Both longtables are typeset at \\small with 4pt column separation so they fit
the text block; long identifiers (dotted module paths, camelCase class names,
init strings) receive explicit line-break opportunities so nothing runs off
the right margin.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _report_common as C


def tex_escape(s):
    return (s.replace('\\', r'\textbackslash{}')
             .replace('_', r'\_')
             .replace('&', r'\&')
             .replace('%', r'\%')
             .replace('#', r'\#')
             .replace('$', r'\$')
             .replace('{', r'\{')
             .replace('}', r'\}'))


# Short canonical labels for the (highly repetitive) init schemes. The full
# conventions are documented in the "Initialization conventions" subsection
# of Appendix A; the table column only needs a compact pointer.
_INIT_LABELS = {
    'kaiming': r'kaiming\_uniform + uniform bias',
    'bias_uniform': r'uniform\_ bias (kaiming default)',
    'norm_affine': r'ones\_/zeros\_ affine',
    'zeros': r'zeros\_',
    'ones': r'ones\_',
    'trunc_normal': r'trunc\_normal (std=0.02)',
    'freq_const': r'const 0.85/0.15',
    'default': r'PyTorch default',
}


def _short_init(desc):
    """Map the verbose init description to a compact table-cell label."""
    d = desc.lower()
    if 'trunc_normal' in d:
        return _INIT_LABELS['trunc_normal']
    if 'freq_const' in d or 'constant 0.85' in d:
        return _INIT_LABELS['freq_const']
    if 'affine' in d:
        return _INIT_LABELS['norm_affine']
    if 'ones_' in d and 'bias' not in d:
        return _INIT_LABELS['ones']
    if 'bias zeros' in d or d.startswith('affine bias'):
        return _INIT_LABELS['zeros']
    if 'uniform_ bias' in d:
        return _INIT_LABELS['bias_uniform']
    if 'kaiming' in d:
        return _INIT_LABELS['kaiming']
    return _INIT_LABELS['default']


def _breakable(s, threshold=14):
    """Insert line-break opportunities so long identifiers wrap in a p{} cell.

    Breaks are placed (a) after every escaped underscore (natural break point
    of dotted paths / snake_case names) and (b) at camelCase boundaries of
    long class names such as NonDynamicallyQuantizableLinear. The text is
    unchanged -- \\allowbreak only permits, never forces, a break.
    """
    if len(s) <= threshold:
        return s
    # after escaped underscores (natural break point of snake_case paths)
    s = s.replace(r'\_', r'\_\allowbreak ')
    # after dots (natural break point of dotted module paths)
    s = s.replace('.', r'.\allowbreak ')
    # every camelCase boundary of the (rare) very long tokens, e.g.
    # NonDynamicallyQuantizableLinear -> Non | Dynamically | Quantizable | Linear
    out = []
    prev_lower = False
    for ch in s:
        if ch.isupper() and prev_lower:
            out.append(r'\allowbreak ')
        out.append(ch)
        prev_lower = ch.islower() or ch.isdigit()
    return ''.join(out).replace('  ', ' ')


def module_summary_tex(rows):
    lines = [r"{\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{longtable}{>{\raggedright\arraybackslash}p{5.2cm} "
             r">{\raggedright\arraybackslash}p{4.2cm} r r}",
             r"\toprule",
             r"\textbf{Module} & \textbf{Class} & \textbf{Direct params} "
             r"& \textbf{Subtree params} \label{tab:modules} \\",
             r"\midrule",
             r"\endhead"]
    total = 0
    for r in rows:
        if r['module'] == '(root)':
            continue
        total += r['direct_params']
        mod = _breakable(tex_escape(r['module']))
        cls = _breakable(tex_escape(r['class']))
        lines.append(
            f"{mod} & {cls} & "
            f"{r['direct_params']:,} & {r['subtree_params']:,} \\\\")
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total (direct over modules)}} & & {total:,} & \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"}")
    return "\n".join(lines)


def inventory_tex(rows):
    lines = [r"{\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{longtable}{>{\raggedright\arraybackslash}p{4.2cm} "
             r">{\raggedright\arraybackslash}p{2.8cm} r "
             r">{\raggedright\arraybackslash}p{4.0cm}}",
             r"\toprule",
             r"\textbf{Tensor} & \textbf{Shape} & \textbf{Params} "
             r"& \textbf{Init} \label{tab:inventory} \\",
             r"\midrule",
             r"\endhead"]
    for r in rows:
        # labels are already LaTeX-safe raw strings; only add breakpoints
        init = _breakable(_short_init(r['init']))
        shp = r['shape_str'].replace('x', r'{\times}')
        shape_math = f"$({shp})$" if shp else r'\texttt{--}'
        lines.append(
            f"{_breakable(tex_escape(r['name']))} & {shape_math} & "
            f"{r['numel']:,} & {init} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"}")
    return "\n".join(lines)


def main():
    C.set_style()
    model = C.build_model(load_ckpt=False, device='cpu')
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tensors = C.enumerate_tensors(model)
    n_tensors = len(tensors)

    # top-level grouping
    groups = {}
    for t in tensors:
        top = t['module'].split('.')[0] if t['module'] else '(root)'
        groups.setdefault(top, {'numel': 0, 'tensors': 0})
        groups[top]['numel'] += t['numel']
        groups[top]['tensors'] += 1

    summary = C.summarize_modules(model)
    direct_total = sum(r['direct_params'] for r in summary)

    stats = {
        'total_params': total_params,
        'trainable_params': trainable,
        'n_tensor_tensors': n_tensors,
        'n_modules': len(list(model.modules())),
        'top_level': {k: v for k, v in sorted(groups.items(),
                                              key=lambda kv: -kv[1]['numel'])},
        'total_direct': direct_total,
    }
    C.save_json(stats, 'model_stats.json')
    print(f"[params] RHANv12 total params = {total_params:,} "
          f"({trainable:,} trainable) across {n_tensors} tensors")

    # ---- module summary ----
    C.save_tabular(summary, 'module_summary.csv')
    with open(os.path.join(C.TAB_DIR, 'module_summary.tex'), 'w') as f:
        f.write(module_summary_tex(summary))

    # ---- full tensor inventory (appendix A) ----
    C.save_tabular(tensors, 'parameter_inventory.csv')
    with open(os.path.join(C.TAB_DIR, 'parameter_inventory.tex'), 'w') as f:
        f.write(inventory_tex(tensors))

    print(f"[params] wrote module_summary.tex ({len(summary)} modules) and "
          f"parameter_inventory.tex ({n_tensors} tensors)")


if __name__ == '__main__':
    main()
