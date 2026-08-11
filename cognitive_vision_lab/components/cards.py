"""Reusable cards and status elements."""
from __future__ import annotations

import streamlit as st

from cognitive_vision_lab.backend.benchmark import ModelProfile


def model_profile_card(profile: ModelProfile) -> None:
    """Compact profile card for the Model Zoo."""
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**{profile.name}**")
            st.caption(f"{profile.family} · {profile.dataset}")
        with c2:
            if profile.ethresh and profile.ethresh > 0.05:
                pill = "🟢 High robustness"
            elif profile.ethresh and profile.ethresh > 0.01:
                pill = "🟡 Moderate"
            else:
                pill = "🔴 Fragile"
            st.markdown(f"<span class='cvl-pill cvl-pill-info'>{pill}</span>",
                        unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clean %", f"{profile.clean_acc:.1f}")
        m2.metric("εthresh", f"{profile.ethresh:.3f}" if profile.ethresh else "n/a")
        m3.metric("Params", f"{profile.params_m:.1f}M")
        m4.metric("d′", f"{profile.dprime:.2f}" if profile.dprime else "n/a")


def stat_grid(stats: dict[str, str]) -> None:
    cols = st.columns(len(stats))
    for col, (k, v) in zip(cols, stats.items()):
        col.metric(k, v)


def status_badge(status: str) -> str:
    kinds = {
        "done": "good", "running": "info", "queued": "warn",
        "failed": "bad", "available": "good", "missing": "bad",
    }
    kind = kinds.get(status, "info")
    return f'<span class="cvl-pill cvl-pill-{kind}">{status}</span>'
