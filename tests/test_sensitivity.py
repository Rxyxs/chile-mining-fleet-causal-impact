"""Tests for `src/sensitivity/cluster_convergence.py`: the variable-N simulator must keep the
published design (equal cohorts, same true dynamic effect) and the evaluation must be
reproducible and emit the agreed JSON structure."""
import json

import numpy as np
import pytest

from src.data.simulate_staggered_did import N_MONTHS, simulate_staggered_panel
from src.sensitivity import cluster_convergence as cc


@pytest.mark.parametrize("n_sites", [16, 32, 64, 128])
def test_simulator_splits_sites_evenly_across_the_four_cohorts(n_sites):
    Y, adoption, _ = cc.simulate_site_panel(n_sites, np.random.default_rng(0))
    assert Y.shape == (n_sites, N_MONTHS)
    assert np.isnan(adoption).sum() == n_sites // 4
    for month in (12.0, 20.0, 28.0):
        assert (adoption == month).sum() == n_sites // 4


def test_simulator_rejects_site_counts_that_cannot_be_split_into_four_cohorts():
    with pytest.raises(ValueError):
        cc.simulate_site_panel(30, np.random.default_rng(0))


def test_true_att_equals_the_published_panels_value_because_cohorts_are_equal_sized():
    published = simulate_staggered_panel(seed=42)
    expected = published.loc[published["treated"] == 1, "true_effect_hours"].mean()
    for n_sites in (16, 32, 128):
        assert cc.simulate_site_panel(n_sites, np.random.default_rng(1))[2] == pytest.approx(expected)


def test_simulated_outcomes_are_positive_and_centred_like_the_original_dgp():
    Y, _, _ = cc.simulate_site_panel(128, np.random.default_rng(3))
    assert (Y > 0).all()
    published = simulate_staggered_panel(seed=42)
    assert Y.mean() == pytest.approx(published["downtime_hours"].mean(), abs=2.0)


def test_never_treated_sites_have_no_treatment_effect_in_the_outcome_level():
    # A cohort that never adopts is untouched by the program, so net of the common month shock
    # its mean outcome must sit at the site baseline (~45h). One panel's 32 never-treated sites
    # give a baseline with a standard error of ~1.4h, so average several panels.
    month_effect = 6.0 * np.sin(2 * np.pi * (np.arange(1, N_MONTHS + 1) - 3) / 12)
    levels = []
    for seed in range(10):
        Y, adoption, _ = cc.simulate_site_panel(128, np.random.default_rng(seed))
        levels.append((Y[np.isnan(adoption)].mean(axis=0) - month_effect).mean())
    assert np.mean(levels) == pytest.approx(45.0, abs=1.5)


def test_evaluate_cluster_count_is_reproducible_and_well_formed():
    a = cc.evaluate_cluster_count(16, n_panels=5, n_boot=20)
    b = cc.evaluate_cluster_count(16, n_panels=5, n_boot=20)
    assert a == b
    assert a["std_error"] > 0
    assert 0.0 <= a["coverage_95"] <= 1.0
    assert a["details"]["n_panels"] == 5


def test_error_spread_shrinks_with_more_sites():
    small = cc.evaluate_cluster_count(16, n_panels=40, n_boot=10)["std_error"]
    large = cc.evaluate_cluster_count(128, n_panels=40, n_boot=10)["std_error"]
    assert large < small


def test_run_emits_the_requested_structure():
    out = cc.run(clusters=(16, 32), n_panels=3, n_boot=10, progress=False)
    assert out["clusters_eval"] == [16, 32]
    assert set(out["metrics"]) == {"16", "32"}
    for m in out["metrics"].values():
        assert set(m) == {"std_error", "coverage_95"}
    json.dumps(out)  # serialisable


def test_main_writes_json_to_tmp_agent_b(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cc, "CLUSTER_GRID", (16,))
    monkeypatch.setattr(cc, "N_PANELS", 2)
    monkeypatch.setattr(cc, "N_BOOT", 5)
    original_run = cc.run
    monkeypatch.setattr(cc, "run", lambda: original_run(clusters=(16,), n_panels=2, n_boot=5, progress=False))
    cc.main()
    written = json.loads((tmp_path / "sensitivity_results.json").read_text(encoding="utf-8"))
    assert written["clusters_eval"] == [16]
    assert set(written["metrics"]["16"]) == {"std_error", "coverage_95"}
