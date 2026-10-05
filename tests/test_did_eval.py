"""Tests for `src/did_eval`: the matrix group-time estimator, the not-yet-treated variant,
the cluster bootstrap and the Monte Carlo summaries.

The toy panel is noise-free with common month effects and a known effect by event time, so
every estimator that respects parallel trends must return the true effect exactly.

Sites: T1, T2 adopt at month 3; L1 adopts at month 5; C1, C2 never adopt; months 1-6.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.simulate_staggered_did import simulate_staggered_panel
from src.did_eval.monte_carlo import (
    coverage_summary, distribution_summary, fit_panel, pct_error, run_monte_carlo, true_overall_att,
)
from src.did_eval.not_yet_treated import not_yet_treated_att, not_yet_treated_overall_att
from src.did_eval.panel_core import cluster_bootstrap_ci, group_time_cells, overall_from_cells, panel_to_matrix
from src.evaluation.did_estimators import group_time_att, overall_att

MONTH_EFFECT = {m: 1.5 * m for m in range(1, 7)}
SITE_BASE = {"T1": 10, "T2": 12, "L1": 11, "C1": 8, "C2": 9}
ADOPTION = {"T1": 3, "T2": 3, "L1": 5, "C1": None, "C2": None}
TRUE_EFFECT = {0: -4.0, 1: -6.0, 2: -7.0}


def _true_effect(event_time):
    if event_time is None or event_time < 0:
        return 0.0
    return TRUE_EFFECT.get(event_time, -7.0)


def _toy_panel() -> pd.DataFrame:
    rows = []
    for site, base in SITE_BASE.items():
        adoption = ADOPTION[site]
        for month in range(1, 7):
            event_time = None if adoption is None else month - adoption
            rows.append({
                "site_id": site, "adoption_month": adoption, "month": month,
                "treated": int(event_time is not None and event_time >= 0),
                "downtime_hours": base + MONTH_EFFECT[month] + _true_effect(event_time),
                "true_effect_hours": _true_effect(event_time),
            })
    return pd.DataFrame(rows)


def test_matrix_estimator_matches_reference_group_time_att():
    panel = simulate_staggered_panel(seed=42)
    reference = group_time_att(panel, baseline_window=3).sort_values(["cohort_adoption_month", "month"])
    cells = group_time_cells(*panel_to_matrix(panel), baseline_window=3).sort_values(["g", "t"])
    np.testing.assert_allclose(cells["att"].to_numpy(), reference["att"].to_numpy(), rtol=0, atol=1e-9)
    assert overall_from_cells(cells) == pytest.approx(overall_att(reference))


def test_not_yet_treated_recovers_exact_effect_on_noise_free_panel():
    result = not_yet_treated_att(_toy_panel(), baseline_window=2)
    for row in result.itertuples():
        assert row.att == pytest.approx(_true_effect(row.event_time)), (row.cohort_adoption_month, row.month)


def test_not_yet_treated_never_uses_an_already_treated_site_as_control():
    result = not_yet_treated_att(_toy_panel(), baseline_window=2).set_index(["cohort_adoption_month", "month"])
    # cohort 3 at months 3-4: L1 (adopts 5) is still untreated -> C1, C2, L1
    assert result.loc[(3, 3), "n_control_sites"] == 3
    assert result.loc[(3, 4), "n_control_sites"] == 3
    # from month 5 L1 is treated, so only the never-treated remain
    assert result.loc[(3, 5), "n_control_sites"] == 2
    assert result.loc[(3, 6), "n_control_sites"] == 2
    # cohort 5 never has a not-yet-treated cohort left
    assert (result.loc[5, "n_control_sites"] == 2).all()


def test_not_yet_treated_equals_never_treated_when_there_is_a_single_cohort():
    panel = _toy_panel()
    panel = panel[panel["site_id"] != "L1"]
    nyt = not_yet_treated_att(panel, baseline_window=2).sort_values(["cohort_adoption_month", "month"])
    never = group_time_att(panel, baseline_window=2).sort_values(["cohort_adoption_month", "month"])
    np.testing.assert_allclose(nyt["att"].to_numpy(), never["att"].to_numpy(), atol=1e-12)


def test_not_yet_treated_has_more_controls_than_never_treated_and_overall_matches_cells():
    panel = _toy_panel()
    cells = not_yet_treated_att(panel, baseline_window=2)
    assert cells["n_control_sites"].max() > 2
    assert not_yet_treated_overall_att(panel, baseline_window=2) == pytest.approx(cells["att"].mean())


def test_unknown_control_group_is_rejected():
    Y, adoption = panel_to_matrix(_toy_panel())
    with pytest.raises(ValueError):
        group_time_cells(Y, adoption, control="always")


def test_cells_without_any_comparison_site_are_dropped():
    panel = _toy_panel()
    panel = panel[~panel["site_id"].isin(["C1", "C2"])]  # no never-treated sites left
    Y, adoption = panel_to_matrix(panel)
    assert group_time_cells(Y, adoption, baseline_window=2, control="never").empty
    # not-yet-treated still has L1 as control for cohort 3 before month 5
    cells = group_time_cells(Y, adoption, baseline_window=2, control="not_yet")
    assert set(cells.loc[cells["g"] == 3, "t"]) == {3, 4}


def test_cluster_bootstrap_is_deterministic_and_brackets_the_point_estimate():
    panel = simulate_staggered_panel(seed=7)
    Y, adoption = panel_to_matrix(panel)
    a = cluster_bootstrap_ci(Y, adoption, np.random.default_rng(1), n_boot=100)
    b = cluster_bootstrap_ci(Y, adoption, np.random.default_rng(1), n_boot=100)
    assert a == b
    point = overall_from_cells(group_time_cells(Y, adoption))
    assert a["ci_low"] < point < a["ci_high"]
    assert a["se"] > 0 and a["n_boot"] == 100


def test_noise_free_bootstrap_interval_is_tight_around_the_truth():
    panel = _toy_panel()
    Y, adoption = panel_to_matrix(panel)
    ci = cluster_bootstrap_ci(Y, adoption, np.random.default_rng(0), n_boot=200, baseline_window=2, control="not_yet")
    truth = true_overall_att(panel)
    assert ci["ci_low"] <= truth + 1.0 and ci["ci_high"] >= truth - 1.0


def test_pct_error_sign_convention():
    assert pct_error(-8.5, -10.0) == pytest.approx(15.0)    # understates the magnitude
    assert pct_error(-11.0, -10.0) == pytest.approx(-10.0)  # overstates it


def test_distribution_summary_known_values():
    s = distribution_summary(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert s["mean"] == 0.0 and s["median"] == 0.0
    assert s["mean_abs"] == pytest.approx(1.2)
    assert s["rmse"] == pytest.approx(np.sqrt(2.0))
    assert (s["min"], s["max"]) == (-2.0, 2.0)


def test_coverage_summary_counts_inside_and_each_miss_direction():
    low = np.array([0.0, 0.0, 0.0, 0.0])
    high = np.array([1.0, 1.0, 1.0, 1.0])
    truth = np.array([0.5, 1.0, -0.1, 1.1])  # inside, on the edge (counts), below, above
    s = coverage_summary(low, high, truth)
    assert s["coverage"] == 0.5
    assert s["miss_below"] == 0.25 and s["miss_above"] == 0.25
    assert s["mean_width"] == 1.0 and s["n"] == 4
    assert s["mc_se"] == pytest.approx(np.sqrt(0.25 / 4))


def test_fit_panel_returns_all_estimators_and_twfe_understates_on_published_panel():
    panel = simulate_staggered_panel(seed=42)
    fit = fit_panel(panel, np.random.default_rng(0), n_boot=50)
    assert set(fit) == {"gt_never_w3", "gt_not_yet_w3", "gt_never_w1_weighted", "twfe"}
    truth = true_overall_att(panel)
    assert abs(pct_error(fit["twfe"]["att"], truth)) == pytest.approx(6.6, abs=0.1)
    assert abs(pct_error(fit["gt_never_w3"]["att"], truth)) == pytest.approx(1.5, abs=0.1)
    for r in fit.values():
        assert r["ci_low"] < r["att"] < r["ci_high"]


def test_run_monte_carlo_is_reproducible_and_well_formed():
    a = run_monte_carlo(n_panels=3, n_boot=20, progress=False)
    b = run_monte_carlo(n_panels=3, n_boot=20, progress=False)
    pd.testing.assert_frame_equal(a["per_panel"], b["per_panel"])
    assert len(a["per_panel"]) == 3 and a["per_panel"]["seed"].is_unique
    assert set(a["coverage"]) == set(a["errors"])
    assert 0.0 <= a["coverage"]["twfe"]["coverage"] <= 1.0
