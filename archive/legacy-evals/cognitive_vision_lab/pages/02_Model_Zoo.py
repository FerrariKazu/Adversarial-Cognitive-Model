"""02 — Model Zoo: profiles for every model with stats and robustness."""
import pandas as pd
import streamlit as st

from cognitive_vision_lab.backend.benchmark import (
    curated_profiles,
    find_profile,
    model_summary_table,
)
from cognitive_vision_lab.backend.models import list_models
from cognitive_vision_lab.components.charts import bar_fig, radar_fig
from cognitive_vision_lab.components.layout import footer, hero, section


def _zoo_table() -> pd.DataFrame:
    rows = []
    for m in list_models():
        p = find_profile(m["id"]) or find_profile(m["name"]) or find_profile(m["checkpoint"])
        rows.append({
            "Model": m["name"], "Family": m["family"], "Dataset": m["dataset"],
            "Available": "" if m["available"] else "",
            "Params (M)": p.params_m if p else float("nan"),
            "Clean %": p.clean_acc if p else float("nan"),
            "εthresh": p.ethresh if p and p.ethresh else float("nan"),
            "d′": p.dprime if p and p.dprime else float("nan"),
        })
    return pd.DataFrame(rows)


def render() -> None:
    hero("Model Zoo", "Every architecture with a profile: stats, robustness, references.")

    st.caption("Availability reflects checkpoints present on this host (checkpoints/ directory).")
    df = _zoo_table()
    st.dataframe(df.style.format(precision=2), width="stretch", hide_index=True)

    section("Model profile", "Select a model to inspect its scientific profile.")
    models = list_models()
    names = [m["name"] for m in models]
    sel = st.selectbox("Model", names, index=names.index("RHAN-Large (Pseudolabel)") if
                       "RHAN-Large (Pseudolabel)" in names else 0)
    entry = next(m for m in models if m["name"] == sel)

    profile = find_profile(entry["id"]) or find_profile(sel) or find_profile(entry["checkpoint"])
    if profile is None:
        st.warning("No curated benchmark profile for this model yet. "
                   "Run it through the Experiment Manager to generate one.")
        footer()
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"### {profile.name}")
        st.caption(f"{profile.family} · {profile.dataset} · {profile.paper}")
        st.write(profile.notes)
        metric_row = [
            {"label": "Parameters", "value": f"{profile.params_m:.1f}M"},
            {"label": "Clean acc", "value": f"{profile.clean_acc:.1f}%"},
            {"label": "εthresh (d′=1)", "value": f"{profile.ethresh:.3f}" if profile.ethresh else "n/a"},
            {"label": "d′ (clean)", "value": f"{profile.dprime:.2f}" if profile.dprime else "n/a"},
        ]
        if profile.autoattack is not None:
            metric_row.append({"label": "AutoAttack", "value": f"{profile.autoattack:.1f}%"})
        cols = st.columns(len(metric_row))
        for col, m in zip(cols, metric_row):
            col.metric(m["label"], m["value"])
    with c2:
        st.markdown("**Robustness radar**")
        eps = sorted(profile.robust_at)
        if eps:
            vals = [profile.clean_acc] + [profile.robust_at[e] for e in eps]
            cats = ["clean"] + [f"ε={e:.3f}" for e in eps]
            fig = radar_fig(cats, {profile.name: vals})
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No robustness curve available.")

    if entry["source"] == "checkpoint":
        st.download_button(
            "⬇ Download checkpoint (metadata)",
            data=str(entry.get("checkpoint", "")),
            file_name=f"{profile.id}_checkpoint_ref.txt",
            help="Large files are hosted on HuggingFace; the metadata reference is downloaded here.",
        )

    section("Comparison", "Clean accuracy vs robustness across the zoo.")
    cdf = model_summary_table()
    fig = bar_fig(cdf["Model"].tolist(), cdf["Clean %"].tolist(),
                  names=cdf["Model"].tolist(), ylabel="Clean accuracy (%)",
                  title="Clean accuracy by model")
    st.plotly_chart(fig, width="stretch")
    footer()


if st.runtime.exists():
    render()
