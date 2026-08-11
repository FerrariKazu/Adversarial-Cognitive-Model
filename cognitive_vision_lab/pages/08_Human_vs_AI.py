"""08 — Human vs AI: interactive perceptual comparison with real human data."""
import numpy as np
import streamlit as st

from cognitive_vision_lab.backend.benchmark import sdt_systems, system_curve_sorted
from cognitive_vision_lab.components.charts import accuracy_curve_fig, dprime_curve_fig
from cognitive_vision_lab.components.layout import footer, hero, section
from cognitive_vision_lab.utils.math_helpers import EQUATIONS

MODEL_PALETTE = {
    "Human": "#16A34A", "CNN": "#DC2626", "ResNet": "#7C3AED", "Resnet": "#7C3AED",
    "ViT": "#2563EB", "EfficientNet": "#D97706",
}


def _available_systems() -> list[str]:
    systems = sdt_systems()
    order = ["Human", "CNN", "ResNet", "ViT", "EfficientNet"]
    return [s for s in order if s in systems] + [s for s in systems if s not in order]


def _curves(systems: list[str]) -> dict:
    out = {}
    for s in systems:
        eps, acc, dp = system_curve_sorted(s)
        if eps:
            out[s] = {"epsilons": eps, "accuracy": acc, "dprime": dp}
    return out


def render() -> None:
    hero("Human vs AI Vision", "The perceptual gap, visualized with real human "
                               "psychophysics and model SDT curves.")

    systems = _available_systems()
    if not systems:
        st.warning("No SDT results found (phase5_sdt/results/sdt_results.csv missing). "
                   "Run the SDT evaluation or place the CSV to populate this page.")
        footer()
        return

    curves = _curves(systems)
    selected = st.multiselect("Systems", list(curves.keys()),
                              default=[s for s in ("Human", "CNN", "ViT") if s in curves])

    if not selected:
        st.info("Select at least one system.")
        footer()
        return

    # ── Interactive epsilon slider (educational) ─────────────────────────────
    all_eps = sorted({e for c in curves.values() for e in c["epsilons"]})
    eps_choice = st.slider(
        "ε perturbation (L∞, normalized)",
        min_value=float(all_eps[0]) if all_eps else 0.0,
        max_value=float(all_eps[-1]) if all_eps else 0.3,
        value=float(all_eps[len(all_eps) // 2]) if all_eps else 0.03,
        step=0.001, format="%.3f",
        help="Increase ε and watch every system's perception change simultaneously.",
    )

    st.markdown(f"### Perceptual snapshot at ε = {eps_choice:.3f}")
    snapshot = []
    for s in selected:
        c = curves[s]
        acc = float(np.interp(eps_choice, c["epsilons"], c["accuracy"]))
        dp = float(np.interp(eps_choice, c["epsilons"], c["dprime"]))
        snapshot.append({"System": s, "Accuracy %": round(acc, 1),
                         "d′": round(dp, 3),
                         "Status": "👁 stable" if dp >= 1.0 else "💥 collapsed"})
    st.dataframe(snapshot, width="stretch", hide_index=True)

    # ── Full curves ──────────────────────────────────────────────────────────
    section("Sensitivity (d′) across ε", "Humans stay above the d′=1 collapse line; "
                                         "every model crosses it early.")
    dp_models = [{"name": s, "epsilons": curves[s]["epsilons"], "dprime": curves[s]["dprime"]}
                 for s in selected]
    st.plotly_chart(dprime_curve_fig(dp_models), width="stretch")

    section("Accuracy across ε")
    acc_models = [{"name": s, "epsilons": curves[s]["epsilons"], "accuracy": curves[s]["accuracy"]}
                  for s in selected]
    st.plotly_chart(accuracy_curve_fig(acc_models), width="stretch")

    # ── Narrative ────────────────────────────────────────────────────────────
    section("What this teaches")
    st.markdown(
        """
        - **The human visual system never crosses d′ < 1.0**, even at ε = 0.30.
        - **Every feed-forward model collapses before ε ≈ 0.03** — a ~10× gap.
        - **RHAN's recurrent/foveal architecture** pushes εthresh to 0.185 on STL-10,
          closing most of the gap without reaching human stability.
        - Collapse is **not confusion**: models remain overconfident while wrong
          (metacognitive miscalibration — Finding 5).
        """
    )
    with st.expander("📐 Sensitivity mathematics", expanded=False):
        from cognitive_vision_lab.components.equations import render_equation

        render_equation("dprime")
        st.markdown(f"**εthresh** = first ε where macro d′ < 1.0.")
    footer()


if st.runtime.exists():
    render()
