import streamlit as st


def render():
    st.markdown("## Signal Detection Theory & εthresh")

    st.markdown(r"""
**Sensitivity Index (d′).** We frame adversarial robustness as a signal detection problem.
For each model at perturbation budget ε, we compute

$$d' = Z(\text{hit rate}) - Z(\text{false alarm rate})$$

where the hit rate is the proportion of correctly classified samples and the false alarm
rate is the proportion of incorrect classifications, both transformed through the inverse
normal (z-score) function. A perfect classifier achieves $d' \to \infty$; random guessing
gives $d' = 0$. We report two variants: *macro d'* (averaged per-class z-scores) and
*pooled d'* (pooling all predictions before computing z-scores).

**The Perceptual Collapse Threshold ($\epsilon_{\text{thresh}}$).** We define
$\epsilon_{\text{thresh}}$ as the smallest perturbation at which a model's sensitivity
drops below $d' = 1.0$. This threshold is motivated by psychophysics: a $d' < 1.0$
corresponds to ≈69% correct in a two-alternative forced-choice task — the point at which
the model has lost reliable discriminative ability. Humans maintain $d' > 1.0$ up to
$\epsilon \approx 0.30$, a ten-fold gap over every feedforward AI model tested.

**Attack Methodology.** All reported results use a domain-clamped PGD-50 attack with
per-channel normalization in pixel space $[0, 1]$. Unlike standard TRADES-style KL
divergence attacks (which can miss gradient-masked models), our cross-entropy-based
attack with explicit pixel-space domain clamping provides a stronger, more honest
evaluation. Each data point reflects $n=500$ stratified test samples per epsilon level.

**Models Evaluated.** Three RHAN variants trained through different curricula on STL-10:
ep45 (static large, 45 epochs), v10 (active inference T=2, 60M params), and v11
(tripartite active inference T=4, 75.4M params, multi-resolution foveation, generative
prior). Standard baselines (ResNet-18, EfficientNet, ViT, Swin) serve as reference
points for the broader landscape of adversarial sensitivity.
    """)

    st.markdown("---")
    st.markdown("## Benchmark Protocol")
    st.markdown(r"""
| Component | Specification |
|-----------|--------------|
| Dataset | STL-10 (10 classes, 96×96, 5000 train / 8000 test) |
| Attack | PGD-50, cross-entropy, pixel-space domain-clamped [0,1] |
| Epsilon grid | {0, 0.002, 0.004, 0.006, 0.008, 0.016, 0.024, 0.0313} |
| Samples per point | 500 (stratified, balanced across classes) |
| Metric | d' (macro and pooled), accuracy, εthresh at d'=1.0 |
| Human reference | Psychophysics experiment: n=30, 2AFC task, same stimulus set |
    """)
