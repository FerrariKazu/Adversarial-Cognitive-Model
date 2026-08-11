"""10 — RHAN Architecture Explorer: interactive module diagram."""
import networkx as nx
import streamlit as st

from cognitive_vision_lab.backend.rhan_arch import EDGES, MODULES, build_graph, module_by_key
from cognitive_vision_lab.components.charts import flow_fig
from cognitive_vision_lab.components.layout import footer, hero, section


def _layout_pos():
    levels = {
        "visual_cortex": 0, "v1": 1, "v2": 1, "freq_gate": 2, "dorsal": 3,
        "ventral": 3, "working_memory": 4, "predictive_coding": 5,
        "prototype": 6, "classifier": 7,
    }
    counts: dict[int, int] = {}
    pos = {}
    for key, lvl in levels.items():
        n = counts.get(lvl, 0)
        pos[key] = (lvl, n - 0.5 if n % 2 else n)
        counts[lvl] = n + 1
    return pos


def render() -> None:
    hero("RHAN Architecture Explorer", "Click any module to inspect its purpose, "
                                       "mathematics, shapes, and ablation evidence.")

    labels = {m.key: m.name.split(" —")[0] for m in MODULES}
    pos = _layout_pos()
    fig = flow_fig(pos, labels, EDGES, title="RHAN-v11 processing flow (top-down)")
    st.plotly_chart(fig, width="stretch")

    st.caption("Hover a node for its identifier; select it below for the full spec.")

    keys = [m.key for m in MODULES]
    sel_key = st.selectbox("Module", keys, format_func=lambda k: module_by_key(k).name)

    module = module_by_key(sel_key)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"### {module.name}")
        st.write(module.purpose)
        st.markdown(f"**Paper / grounding:** {module.paper or '—'}")
        st.markdown(f"**Ablation evidence:** {module.ablation or '—'}")
    with c2:
        st.markdown("**Shapes**")
        st.metric("Input", module.input_shape or "—")
        st.metric("Output", module.output_shape or "—")

    if module.math:
        with st.expander("📐 Mathematics", expanded=True):
            st.markdown(
                f'<div class="cvl-equation">$${module.math}$$</div>',
                unsafe_allow_html=True,
            )

    section("Module graph")
    g = build_graph()
    st.caption("Directed edges follow the actual data flow. "
               f"{g.number_of_nodes()} modules, {g.number_of_edges()} connections.")
    with st.expander("Adjacency", expanded=False):
        st.json({n: list(g.successors(n)) for n in g.nodes})

    section("Validation evidence")
    st.markdown(
        """
        - **Finding 7:** recurrent feedback + dorsal/ventral separation + neural
          alignment combine to a 6.3× εthresh improvement over ResNet-18.
        - **Finding 10:** the generative prior only helps with a *freshly
          initialized* perceptual critic; frozen critics collapse BatchNorm channels.
        - **Finding 16:** the three active-inference losses (foraging, precision,
          halt) actively *oppose* the Banach contraction — zero εthresh benefit.
        - **Finding 17:** zeroing those losses reveals the architecture alone wins
          at high ε (+5.6 pp vs TRADES Large on real data).
        """
    )
    footer()


if st.runtime.exists():
    render()
