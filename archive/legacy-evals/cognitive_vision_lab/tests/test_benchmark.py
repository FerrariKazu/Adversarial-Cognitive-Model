"""Unit tests for cognitive_vision_lab.backend.benchmark."""
import json

import pandas as pd
import pytest

from cognitive_vision_lab.backend.benchmark import (
    curated_profiles,
    find_profile,
    isolation_sweep_profiles,
    load_isolation_sweep_json,
    model_summary_table,
    profiles_as_json,
    sdt_systems,
    sdt_system_curves,
)


class TestCuratedProfiles:
    def test_profiles_exist(self):
        profiles = curated_profiles()
        assert len(profiles) >= 8

    def test_fields_present(self):
        for p in curated_profiles():
            assert p.id and p.name and p.family
            assert 0.0 <= p.clean_acc <= 100.0
            assert p.params_m >= 0.0

    def test_human_present(self):
        assert find_profile("human_stl10") is not None

    def test_unknown_profile_returns_none(self):
        assert find_profile("does_not_exist") is None

    def test_json_serializable(self):
        parsed = json.loads(profiles_as_json())
        assert isinstance(parsed, list) and len(parsed) == len(curated_profiles())


class TestSummaryTable:
    def test_shape_and_columns(self):
        df = model_summary_table()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(curated_profiles()) + len(isolation_sweep_profiles())
        for col in ["Model", "Family", "Clean %", "ε=0.031", "ε=0.062", "ε=0.094"]:
            assert col in df.columns


class TestIsolationSweepProfiles:
    def test_three_profiles_present(self):
        profiles = isolation_sweep_profiles()
        ids = {p.id for p in profiles}
        assert ids == {"rhan_v11_isolation_norecon",
                       "rhan_v11_isolation_fixedgaze",
                       "trades_large_measured"}
        for p in profiles:
            assert p.source == "isolation_sweep"
            assert p.dataset == "STL-10"
            assert p.ethresh is not None

    def test_values_match_json(self):
        profs = {p.id: p for p in isolation_sweep_profiles()}
        run_a = profs["rhan_v11_isolation_norecon"]
        assert run_a.clean_acc == pytest.approx(50.40)
        assert run_a.robust_at[0.031] == pytest.approx(45.60)
        assert run_a.robust_at[0.062] == pytest.approx(35.20)
        assert run_a.robust_at[0.094] == pytest.approx(26.00)
        assert run_a.dprime == pytest.approx(1.4326)
        assert run_a.ethresh == pytest.approx(0.062)  # d' crosses 1.0 between 0.031 and 0.062

        run_b = profs["rhan_v11_isolation_fixedgaze"]
        assert run_b.clean_acc == pytest.approx(56.20)
        assert run_b.robust_at[0.031] == pytest.approx(47.00)
        assert run_b.robust_at[0.094] == pytest.approx(23.40)

        bsl = profs["trades_large_measured"]
        assert bsl.clean_acc == pytest.approx(55.20)
        assert bsl.robust_at[0.094] == pytest.approx(22.60)
        assert bsl.ethresh == pytest.approx(0.094)

    def test_fallback_when_json_missing(self, monkeypatch):
        monkeypatch.setattr(
            "cognitive_vision_lab.backend.benchmark.ISOLATION_SWEEP",
            type("P", (), {"exists": lambda self: False, "__str__": lambda self: "/none"})(),
        )
        assert load_isolation_sweep_json() is None
        profs = {p.id: p for p in isolation_sweep_profiles()}
        assert profs["rhan_v11_isolation_norecon"].clean_acc == pytest.approx(50.40)
        assert profs["trades_large_measured"].ethresh == pytest.approx(0.094)

    def test_loader_returns_none_on_bad_json(self, monkeypatch, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        monkeypatch.setattr("cognitive_vision_lab.backend.benchmark.ISOLATION_SWEEP", bad)
        assert load_isolation_sweep_json() is None

    def test_partial_json_falls_back_to_embedded(self, monkeypatch, tmp_path):
        partial = tmp_path / "partial.json"
        partial.write_text(json.dumps({"sweeps": {"run_a_norecon": {"points": []}}}))
        monkeypatch.setattr("cognitive_vision_lab.backend.benchmark.ISOLATION_SWEEP", partial)
        profs = {p.id: p for p in isolation_sweep_profiles()}
        assert set(profs) == {"rhan_v11_isolation_norecon",
                              "rhan_v11_isolation_fixedgaze",
                              "trades_large_measured"}
        # Values come from the embedded mirror, not the empty partial file.
        assert profs["rhan_v11_isolation_fixedgaze"].clean_acc == pytest.approx(56.20)

    def test_isolation_rows_in_summary_table(self):
        df = model_summary_table()
        names = set(df["Model"])
        assert "RHAN-v11 (Isolation A · no-recon)" in names
        assert "RHAN-v11 (Isolation B · fixed-gaze)" in names
        assert "TRADES Large (measured)" in names

    def test_sortable_numeric(self):
        df = model_summary_table().sort_values("Clean %", ascending=False)
        assert df["Clean %"].iloc[0] >= df["Clean %"].iloc[-1]


class TestSDTLoaders:
    def test_missing_file_graceful(self, monkeypatch):
        monkeypatch.setattr(
            "cognitive_vision_lab.backend.benchmark.SDT_RESULTS",
            type("P", (), {"exists": lambda self: False, "__str__": lambda self: "/none"})(),
        )
        assert sdt_systems() == []
        assert sdt_system_curves("Human") == {}


class TestRobustBenchIntegration:
    """RHAN must stay visible on the RobustBench page (regression for the
    cross-dataset reference being filtered out of the clean-vs-robust graph)."""

    def test_rhan_present_in_cifar10_filter(self):
        from cognitive_vision_lab.backend.robustbench import leaderboard

        df = leaderboard(dataset="CIFAR-10", threat="Linf", fetch_live=False)
        assert "RHAN-Large (Ours)" in set(df["model"])

    def test_rhan_present_in_all_filter(self):
        from cognitive_vision_lab.backend.robustbench import leaderboard

        df = leaderboard(dataset="All", threat="Linf", fetch_live=False)
        assert "RHAN-Large (Ours)" in set(df["model"])

    def test_leaderboard_with_ours_guarantees_rhan(self, monkeypatch):
        from cognitive_vision_lab.backend import robustbench as rb

        # Simulate a successful live fetch that contains no RHAN entry.
        monkeypatch.setattr(rb, "_fetch_live", lambda threat: [
            {"model": "Wang2023Better_WRN-70-16", "arch": "WideResNet-70-16",
             "method": "Better diffusion", "dataset": "CIFAR-10", "clean": 92.23,
             "robust": 66.95, "params_m": 266.8, "source": "RobustBench (live)"},
        ])
        df = rb.leaderboard_with_ours(dataset="CIFAR-10", threat="Linf", fetch_live=True)
        assert "RHAN-Large (Ours)" in set(df["model"])
        # The live entry must still be there.
        assert "Wang2023Better_WRN-70-16" in set(df["model"])

    def test_rhan_large_ethresh_is_0_185(self):
        prof = find_profile("rhan_large_pseudolabel")
        assert prof is not None
        assert prof.ethresh == pytest.approx(0.185)

    def test_ethresh_view_has_measured_rows(self):
        df = model_summary_table()
        eth = df[df["εthresh"].notna()]
        assert len(eth) >= 6
        assert eth["Model"].str.contains("RHAN-Large", case=False).any()

    def test_scatter_highlight_matches_substring(self):
        """The red-diamond highlight must match 'RHAN-Large' against
        display names like 'RHAN-Large (Ours)' (exact-match regression)."""
        from cognitive_vision_lab.components.charts import robust_scatter_fig

        fig = robust_scatter_fig(
            [1.0, 2.0, 3.0], [0.01, 0.185, 0.10],
            names=["EfficientNet-B0", "RHAN-Large (Pseudolabel)", "ViT-B/16"],
            highlight=["RHAN-Large"],
        )
        hl = [t for t in fig.data if t.name == "highlighted"]
        assert len(hl) == 1
        assert list(hl[0].customdata) == ["RHAN-Large (Pseudolabel)"]
