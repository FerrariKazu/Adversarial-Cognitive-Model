"""09 — RobustBench Comparison: leaderboard filters + RHAN vs SOTA."""
import streamlit as st

from cognitive_vision_lab.backend.benchmark import model_summary_table
from cognitive_vision_lab.backend.robustbench import available_threats, leaderboard_with_ours
from cognitive_vision_lab.components.charts import robust_scatter_fig
from cognitive_vision_lab.components.layout import footer, hero, section
from cognitive_vision_lab.components.tables import styled_frame


def render() -> None:
    hero("RobustBench Comparison", "Benchmark RHAN against state-of-the-art "
                                   "adversarially robust models.")

    c1, c2, c3 = st.columns(3)
    threat = c1.selectbox("Threat model", available_threats())
    dataset = c2.selectbox("Dataset", ["CIFAR-10", "All"])
    fetch = c3.toggle("Fetch live leaderboard", help="Downloads the official "
                       "RobustBench CSV from GitHub (requires internet).")

    df = leaderboard_with_ours(dataset=dataset, threat=threat, fetch_live=fetch)
    styled_frame(df[["model", "arch", "method", "dataset", "clean", "robust", "params_m"]],
                 highlight_col="robust")

    section("Robustness vs clean accuracy",
            "Two complementary views: the RobustBench standard (AutoAttack at "
            "ε = 0.031) and the project's sensitivity threshold εthresh.")
    metric = st.radio(
        "Robustness metric",
        ["AutoAttack @ ε = 0.031", "εthresh — d′ crosses 1.0"],
        horizontal=True,
        help="AutoAttack is the RobustBench standard. εthresh is the largest "
             "normalized L∞ perturbation at which sensitivity stays above chance "
             "(d′ ≥ 1.0), measured on the project's matched PGD-50 grid.",
    )

    if metric.startswith("AutoAttack"):
        rhan = df[df["model"].astype(str)
                      .str.contains("RHAN-Large", case=False, na=False)]
        ann = [(float(rhan["clean"].iloc[0]), float(rhan["robust"].iloc[0]),
                "RHAN-Large · εthresh = 0.185")] if not rhan.empty else None
        fig = robust_scatter_fig(
            df["clean"], df["robust"], names=df["model"],
            xlabel="Clean accuracy (%)",
            ylabel="Robust accuracy — AutoAttack @ ε=0.031 (%)",
            title="Clean vs robust accuracy (L∞ threat model)",
            highlight=["RHAN-Large"],
            annotations=ann,
        )
    else:
        eth = model_summary_table()
        eth = eth[eth["εthresh"].notna()].copy()
        rhan = eth[eth["Model"].astype(str)
                       .str.contains("RHAN-Large", case=False, na=False)]
        ann = None
        if not rhan.empty:
            rx, ry = float(rhan["Clean %"].iloc[0]), float(rhan["εthresh"].iloc[0])
            ann = [(rx, ry, f"RHAN-Large · εthresh = {ry:.3f}")]
        fig = robust_scatter_fig(
            eth["Clean %"], eth["εthresh"], names=eth["Model"],
            xlabel="Clean accuracy (%)",
            ylabel="εthresh (normalized L∞)",
            title="Robustness threshold vs clean accuracy",
            highlight=["RHAN-Large"],
            annotations=ann,
            height=520,
        )
        fig.add_hline(y=0.031, line_dash="dash", line_color="#94A3B8",
                      annotation_text="AA@ε=0.031 budget (RobustBench standard)",
                      annotation_position="top left")

    st.plotly_chart(fig, width="stretch")

    section("Where does RHAN stand?")
    st.markdown(
        """
        - **εthresh = 0.185** — RHAN-Large keeps d′ ≥ 1.0 up to **≈6× the standard
          AutoAttack budget** (ε = 0.031), farther than any feed-forward CIFAR-10
          entry on this page. The *εthresh* view is directly comparable: every
          marker comes from the project's matched PGD-50 grid on the same
          normalized L∞ convention.
        - In the *AutoAttack* view, RHAN's robust column (10.6%) is its own
          STL-10 AutoAttack number — a different dataset (96×96) and metric grid,
          so cross-dataset point comparisons there are only approximate.
        - Humans (εthresh = 0.30, clean ≈ 84%) remain the upper bound — RHAN is
          the closest model, at roughly 60% of the human robustness threshold.
        """
    )
    with st.expander("References", expanded=False):
        st.markdown(
            """
            - Croce, Andriushchenko et al., *RobustBench: a standardized adversarial robustness benchmark* (NeurIPS 2021).
            - Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (ICLR 2018).
            - Wang et al., *Better Diffusion Models Further Improve Adversarial Training* (ICML 2023).
            """
        )
    footer()


if st.runtime.exists():
    render()
