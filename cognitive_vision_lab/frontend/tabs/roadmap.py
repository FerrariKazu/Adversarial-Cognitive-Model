import streamlit as st


def render():
    st.subheader("🗺️ Development Roadmap")
    st.warning(
        "Items below are planned for future releases. None are yet implemented. "
        "They are listed for transparency with reviewers and collaborators."
    )

    roadmap = [
        {
            "title": "⚡ Live Adversarial Attack Lab",
            "status": "Planned",
            "description": """
Upload an image, select a model, and generate PGD, FAB-T, or Square attacks 
in-browser with adjustable ε budget. View the adversarial image side-by-side 
with the clean original, and observe how different attack algorithms produce 
different failure modes.

*Requires: GPU backend, attack engine integration, real-time image display.*
            """,
        },
        {
            "title": "📤 Model Upload & Evaluation Pipeline",
            "status": "Planned",
            "description": """
Upload custom PyTorch model weights (.pth) and run the standardized d′
sensitivity sweep on-demand. Results are stored in PostgreSQL and can be
compared against the existing model zoo.

*Requires: GPU infrastructure, validated sandbox, database integration.*
            """,
        },
        {
            "title": "🧠 Human Psychophysics Comparison",
            "status": "Planned",
            "description": """
Interactive overlay of human d′ data from our psychophysics study (n=30, 2AFC)
directly on the sensitivity chart. Filter by participant demographics and
compare individual differences against model behavior.

*Requires: Human data integration from phase3_human_study/, ethics review.*
            """,
        },
        {
            "title": "📈 Per-Class Robustness Breakdown",
            "status": "Planned",
            "description": """
Expand the d′ chart to show per-class sensitivity curves instead of just
macro/pooled aggregates. Essential for diagnosing which semantic categories
drive the overall collapse curve (e.g., automobile and truck are known
to be disproportionately vulnerable).
            """,
        },
        {
            "title": "🔬 Representational Similarity Analysis (RSA)",
            "status": "Planned",
            "description": """
Upload activation matrices from model intermediate layers and compute RSA
against human fMRI data (from phase4_analysis/). Visualize the similarity
matrix as a heatmap with hierarchical clustering.

*Requires: RSA toolchain, human neuroimaging data, ethics compliance.*
            """,
        },
        {
            "title": "📊 Benchmark Leaderboard",
            "status": "Planned",
            "description": """
Persistent leaderboard of all evaluated models ranked by εthresh, clean
accuracy, and robust accuracy at ε=0.0313. Supports filtering by
architecture family, parameter count, and training method.

*Requires: PostgreSQL, benchmark pipeline, CI/CD integration.*
            """,
        },
        {
            "title": "🐳 Docker + Cloud Deployment",
            "status": "Planned",
            "description": """
Containerized deployment with docker-compose (Streamlit + FastAPI + PostgreSQL +
Redis cache). One-command deploy to any cloud provider. Kubernetes manifests
for production scaling.
            """,
        },
    ]

    for item in roadmap:
        with st.container():
            st.markdown(f"### {item['title']}")
            cols = st.columns([1, 5])
            with cols[0]:
                st.markdown(f"**`{item['status']}`**")
            with cols[1]:
                st.markdown(item["description"])
            st.button("🔒 Not Available", disabled=True, key=f"rm_{hash(item['title'])}")
            st.markdown("---")
