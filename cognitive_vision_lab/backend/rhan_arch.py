"""RHAN architecture data — modules, math, shapes, papers, ablations.

Drives the interactive architecture explorer (page 10) and the methodology
expanders. Pure data + networkx graph helpers (no model instantiation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


@dataclass
class ModuleSpec:
    key: str
    name: str
    purpose: str
    math: str = ""
    input_shape: str = ""
    output_shape: str = ""
    paper: str = ""
    ablation: str = ""
    children: list = field(default_factory=list)


MODULES: list[ModuleSpec] = [
    ModuleSpec(
        key="visual_cortex", name="Visual Cortex (Input)",
        purpose="96×96 normalized RGB input pipeline. Gamma-corrected photoreceptor-like intake.",
        input_shape="(B, 3, 96, 96)", output_shape="(B, 3, 96, 96)",
        paper="Hubel & Wiesel 1962 — hierarchical vision",
        ablation="Serves as the fixed entry point for all RHAN variants.",
    ),
    ModuleSpec(
        key="v1", name="V1 — Low-frequency stem",
        purpose="Low-pass Gabor-like filter bank producing an M-pathway (motion/form) stream.",
        math=r"y = \sigma(W_1 * x + b_1), \quad W_1 \sim \text{Gabor-like}",
        input_shape="(B, 3, 96, 96)", output_shape="(B, C1, 48, 48)",
        paper="Hubel & Wiesel 1962",
        ablation="Stem_low channel dropout collapses shape/texture trade-off (Finding 10).",
    ),
    ModuleSpec(
        key="v2", name="V2 — High-frequency stem",
        purpose="High-pass detail stream feeding the P-pathway (texture/local detail).",
        input_shape="(B, 3, 96, 96)", output_shape="(B, C2, 48, 48)",
        paper="Van Essen & Maunsell 1983 — parallel streams",
    ),
    ModuleSpec(
        key="freq_gate", name="Frequency Gating (M/P balance)",
        purpose="Learnable channel-wise weights steering the M-pathway dominance.",
        math=r"z = w_L \cdot z_{\mathrm{low}} + w_H \cdot z_{\mathrm{high}}, \quad w = \sigma(\theta)",
        input_shape="(B, C, 48, 48)", output_shape="(B, C, 48, 48)",
        paper="Livingstone & Hubel 1988 — M/P pathway segregation",
        ablation="wL > wH emerges spontaneously under adversarial training (Finding 7.2) — "
                "the model learns a shape-over-texture bias.",
    ),
    ModuleSpec(
        key="dorsal", name="Dorsal stream (Where)",
        purpose="Motion & spatial-location pathway; drives gaze selection and action priors.",
        input_shape="(B, 144, D)", output_shape="(B, 144, D)",
        paper="Ungerleider & Mishkin 1982 — Where pathway",
        ablation="Dorsal ablation destroys spatial foraging behaviour.",
    ),
    ModuleSpec(
        key="ventral", name="Ventral stream (What)",
        purpose="Object identity pathway; feeds the classifier and the generative prior.",
        input_shape="(B, 144, D)", output_shape="(B, 144, D)",
        paper="Ungerleider & Mishkin 1982 — What pathway",
        ablation="Ventral ablation collapses clean accuracy.",
    ),
    ModuleSpec(
        key="working_memory", name="Working Memory / Belief state",
        purpose="Recurrent belief vector updated across foraging iterations (T steps).",
        math=r"b_{t+1} = \mathrm{LN}\big(b_t + \mathrm{Attn}(b_t, z_t)\big)",
        input_shape="(B, 144, D)", output_shape="(B, 144, D)",
        paper="Baddeley 1992; Banach contraction proof (phase4_proofs §10)",
        ablation="Halting early (Fewer steps) attenuates the contraction benefit (Finding 16.1).",
    ),
    ModuleSpec(
        key="predictive_coding", name="Predictive Coding / Error units",
        purpose="Top-down prediction errors reshape the belief state (FEP-style update).",
        math=r"\varepsilon_t = z_t - g(b_t), \quad b_{t+1} = b_t + \Pi_D \, \nabla_b \log p",
        input_shape="(B, 144, D)", output_shape="(B, 144, D)",
        paper="Rao & Ballard 1999; Friston 2010 free-energy principle",
        ablation="Prediction-error magnitude grows monotonically with ε (Claim 2, validated).",
    ),
    ModuleSpec(
        key="prototype", name="Prototype Layer / Generative Prior",
        purpose="Perceptual critic enforcing reconstructibility of foveal crops.",
        math=r"\mathcal{L}_{\mathrm{FR}} = \|f(x_{\mathrm{adv}}) - g(\hat{x})\|_2^2",
        input_shape="(B, 48, 48, 3)", output_shape="(B, 768)",
        paper="VAE-style manifold constraint",
        ablation="Frozen critic from trained backbone collapses BatchNorm channels (Finding 10.1).",
    ),
    ModuleSpec(
        key="classifier", name="Classifier (cosine head)",
        purpose="Cosine-similarity head over the belief state — 10-way STL-10 logits.",
        math=r"p(y \mid x) = \mathrm{softmax}\big( \tau \cdot \cos(W, b_T) \big)",
        input_shape="(B, 768)", output_shape="(B, 10)",
        paper="Coupled head retained from Phase 0 pretraining",
        ablation="Replacing the cosine head before TRADES phases destroys features (Finding 10.5).",
    ),
]

EDGES = [
    ("visual_cortex", "v1"),
    ("visual_cortex", "v2"),
    ("v1", "freq_gate"),
    ("v2", "freq_gate"),
    ("freq_gate", "dorsal"),
    ("freq_gate", "ventral"),
    ("dorsal", "working_memory"),
    ("ventral", "working_memory"),
    ("working_memory", "predictive_coding"),
    ("predictive_coding", "prototype"),
    ("predictive_coding", "classifier"),
    ("prototype", "classifier"),
]


def module_by_key(key: str) -> ModuleSpec:
    for m in MODULES:
        if m.key == key:
            return m
    raise KeyError(key)


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for m in MODULES:
        g.add_node(m.key, name=m.name, purpose=m.purpose)
    g.add_edges_from(EDGES)
    return g


def flow_layout() -> dict:
    """Ordered (x, y) layout for a top-down flow diagram."""
    levels = {
        "visual_cortex": 0, "v1": 1, "v2": 1, "freq_gate": 2,
        "dorsal": 3, "ventral": 3, "working_memory": 4,
        "predictive_coding": 5, "prototype": 6, "classifier": 7,
    }
    pos = {}
    per_level: dict[int, int] = {}
    for key, lvl in levels.items():
        n = per_level.get(lvl, 0)
        pos[key] = (lvl, n - 0.5 if lvl in (1, 3) and n % 2 else n)
        per_level[lvl] = n + 1
    return pos
