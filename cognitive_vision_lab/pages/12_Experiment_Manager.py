"""12 — Experiment Manager: launch, track, and resume robustness sweeps.

Runs persist to cache/experiments.json; workers run in daemon threads so the
UI never blocks. Metrics are measured live when checkpoints are available,
otherwise curated profiles are reused and flagged as such.
"""
from __future__ import annotations

import streamlit as st

from cognitive_vision_lab.backend import experiments
from cognitive_vision_lab.backend.benchmark import curated_profiles
from cognitive_vision_lab.components.charts import dprime_curve_fig
from cognitive_vision_lab.components.layout import footer, hero, metric_grid, section
from cognitive_vision_lab.config import DEFAULT_EPS_GRID, DEFAULT_PGD_STEPS

ATTACKS = ["PGD", "FGSM", "CW", "DeepFool", "Square", "FAB", "APGD"]


def _model_ids() -> list[str]:
    return [p.id for p in curated_profiles() if p.robust_at]


def _render_run_detail(run) -> None:
    st.markdown(f"**{run.name}** · `{run.id}` · status **{run.status}** "
                f"· progress {run.progress:.0f}%")
    st.progress(min(run.progress / 100.0, 1.0))
    if run.logs:
        with st.expander("Run log", expanded=False):
            st.code("\n".join(run.logs[-40:]), language="text")
    models = []
    for mid, res in (run.results or {}).items():
        if "curve" in res and res["curve"]:
            models.append({"name": res.get("name", mid),
                           "epsilons": res["curve"]["epsilons"],
                           "dprime": res["curve"]["dprime"]})
    if models:
        st.plotly_chart(dprime_curve_fig(models, title=f"{run.name} — d′ curves"),
                        width="stretch")


def render() -> None:
    hero("Experiment Manager",
         "Design, launch, and track robustness experiments from the browser.")

    # ── Launcher ──────────────────────────────────────────────────────────────
    section("New experiment")
    with st.form("exp_form"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Experiment name", "PGD sweep — matched Finding-17")
        attack = c2.selectbox("Attack", ATTACKS, index=0)
        models = st.multiselect("Models", _model_ids(),
                                default=["human_stl10", "rhan_large_pseudolabel",
                                         "resnet18_cifar10"])
        c3, c4, c5 = st.columns(3)
        eps_grid = c3.text_input("ε grid (space-separated)",
                                 " ".join(str(e) for e in DEFAULT_EPS_GRID))
        steps = c4.slider("PGD steps", 5, 200, DEFAULT_PGD_STEPS, 5)
        n_samples = c5.slider("Samples per model", 10, 500, 100, 10)
        submitted = st.form_submit_button(" Launch experiment",
                                          type="primary", width="stretch")
    if submitted:
        try:
            eps = [float(x) for x in eps_grid.replace(",", " ").split() if x.strip()]
            run = experiments.create_run(name, models, "STL-10", attack,
                                         eps or list(DEFAULT_EPS_GRID), steps, n_samples)
            experiments.launch(run.id)
            st.success(f"Experiment `{run.id}` launched in the background.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not launch: {e}")

    # ── Active / history ──────────────────────────────────────────────────────
    section("Run history", "Workers run asynchronously; refresh to see progress.")
    runs = experiments.list_runs()
    if not runs:
        st.info("No experiments yet. Launch one above.")
        footer()
        return

    metric_grid([
        {"label": "Total runs", "value": len(runs)},
        {"label": "Running", "value": sum(1 for r in runs if r.status == "running")},
        {"label": "Done", "value": sum(1 for r in runs if r.status == "done")},
        {"label": "Failed", "value": sum(1 for r in runs if r.status == "failed")},
    ])

    statuses = ["all"] + sorted({r.status for r in runs})
    sel = st.segmented_control("Filter status", statuses, default="all",
                               key="exp_status_filter")
    shown = [r for r in runs if sel == "all" or r.status == sel]

    for run in reversed(shown):
        with st.container(border=True):
            st.markdown(f"### {run.name}  ·  `{run.id}`")
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            c1.caption(f"Models: {', '.join(run.model_ids)} · {run.attack} · "
                       f"ε={run.eps_grid} · {run.steps} steps")
            c2.caption(f"Status: **{run.status}**")
            c3.caption(f"Progress: {run.progress:.0f}%")
            c4.caption(f"Created: {run.created_at[:19]}")
            rc1, rc2, rc3, rc4 = st.columns([1, 1, 1, 3])
            if rc1.button("↻ Refresh", key=f"ref_{run.id}"):
                st.rerun()
            if run.status == "done":
                if rc2.button("▶ Resume", key=f"res_{run.id}"):
                    try:
                        new_id = experiments.resume(run.id)
                        st.success(f"Resumed as `{new_id}`.")
                    except KeyError:
                        st.error("Resume failed.")
            if run.status in ("queued", "failed") and rc3.button("▶ Launch",
                                                                 key=f"la_{run.id}"):
                experiments.launch(run.id)
                st.rerun()
            if run.status == "done" and rc4.button(" Show curves", key=f"cur_{run.id}"):
                _render_run_detail(run)

    with st.expander(" About measurements", expanded=False):
        st.markdown(
            "Measured runs attack real STL-10 samples (or procedural fallbacks) with "
            "PGD at each ε and report macro accuracy + d′. When a checkpoint is not "
            "available locally, the curated Finding-17 profile is reused and marked "
            "`source=curated` so results are never silently fabricated."
        )
    footer()


if st.runtime.exists():
    render()
