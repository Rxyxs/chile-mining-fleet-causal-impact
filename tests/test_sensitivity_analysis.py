"""Sensitivity-analysis tests use small, deterministic (noise-free) toy
panels so every expected number is hand-computable -- the same style as
`test_did_estimators.py`.
"""
import pandas as pd
import pytest

from src.evaluation.sensitivity_analysis import (
    compute_placebo_pretrend_atts,
    honest_bounds,
    inject_pretrend_violation,
    run_violation_sensitivity_sweep,
)
from src.evaluation.did_estimators import group_time_att, overall_att


def _panel(pretrend_violation: float = 0.0) -> pd.DataFrame:
    """2 treated sites (T1, T2, adopt month 6) + 2 never-treated (C1, C2),
    months 1-8. Parallel trends by construction (all sites share the same
    month_effect, differing only in level) -- unless `pretrend_violation` is
    non-zero, in which case T1/T2 get an extra `pretrend_violation * month`
    term added for months before adoption (a genuine, injected violation).
    """
    month_effect = {m: m - 1 for m in range(1, 9)}
    site_base = {"T1": 10, "T2": 12, "C1": 8, "C2": 9}
    adoption_month = {"T1": 6, "T2": 6, "C1": None, "C2": None}

    rows = []
    for site, base in site_base.items():
        g = adoption_month[site]
        for month in range(1, 9):
            value = base + month_effect[month]
            if g is not None and month < g:
                value += pretrend_violation * month
            rows.append({"site_id": site, "adoption_month": g, "month": month, "downtime_hours": value})
    return pd.DataFrame(rows)


def test_placebo_atts_are_exactly_zero_under_true_parallel_trends():
    df = _panel(pretrend_violation=0.0)
    placebo = compute_placebo_pretrend_atts(df, baseline_window=3)

    assert len(placebo) == 2  # months 1 and 2 for the g=6 cohort (baseline = months 3,4,5)
    assert placebo["placebo_att"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_placebo_atts_detect_a_real_injected_violation():
    df = _panel(pretrend_violation=0.5)
    placebo = compute_placebo_pretrend_atts(df, baseline_window=3)
    by_month = placebo.set_index("month")["placebo_att"]

    # Hand-computed: see module docstring / commit description for the
    # arithmetic; both should come out non-zero and negative.
    assert by_month.loc[1] == pytest.approx(-1.5)
    assert by_month.loc[2] == pytest.approx(-1.0)


def test_honest_bounds_breakdown_value():
    m_grid = [0, 1, 2, 3, 4, 5, 6]
    bounds, breakdown_m = honest_bounds(observed_att=-10.0, max_pretrend_violation=2.0, m_grid=m_grid)

    # margin = m * 2; crosses zero once margin >= 10, i.e. m >= 5
    assert breakdown_m == pytest.approx(5.0)
    assert bounds.set_index("m").loc[4.0, "crosses_zero"] == False  # noqa: E712
    assert bounds.set_index("m").loc[5.0, "crosses_zero"] == True  # noqa: E712


def test_honest_bounds_no_breakdown_in_grid_returns_none():
    bounds, breakdown_m = honest_bounds(observed_att=-100.0, max_pretrend_violation=1.0, m_grid=[0, 1, 2])
    assert breakdown_m is None


def test_inject_pretrend_violation_shifts_group_time_att_by_exact_amount():
    """Original panel (from `test_did_estimators.py`'s toy example): 2
    treated sites adopt at month 3, true effects -4/-6/-7 at event times
    0/1/2, overall ATT = -17/3. Injecting a violation of `v` hours/month
    shifts each cohort's pre-adoption baseline mean up by 1.5v (average of
    the 2 pre-period months' multipliers 1 and 2, across 2 sites), which
    shifts every group-time ATT down by exactly 1.5v, and the overall ATT by
    the same amount.
    """
    month_effect = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    site_base = {"T1": 10, "T2": 12, "C1": 8, "C2": 9}
    true_effect_by_event_time = {0: -4, 1: -6, 2: -7}
    adoption_month = {"T1": 3, "T2": 3, "C1": None, "C2": None}

    rows = []
    for site, base in site_base.items():
        g = adoption_month[site]
        for month in range(1, 6):
            event_time = None if g is None else month - g
            treated = event_time is not None and event_time >= 0
            effect = true_effect_by_event_time.get(event_time, 0) if treated else 0
            rows.append({"site_id": site, "adoption_month": g, "month": month, "downtime_hours": base + month_effect[month] + effect})
    df = pd.DataFrame(rows)

    baseline_att = overall_att(group_time_att(df, baseline_window=3))
    assert baseline_att == pytest.approx(-17 / 3)

    corrupted = inject_pretrend_violation(df, violation_per_month=2.0)
    corrupted_att = overall_att(group_time_att(corrupted, baseline_window=3))
    assert corrupted_att == pytest.approx(-17 / 3 - 3.0)  # -26/3


def test_violation_sweep_is_monotonic_in_violation_size():
    month_effect = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    site_base = {"T1": 10, "T2": 12, "C1": 8, "C2": 9}
    adoption_month = {"T1": 3, "T2": 3, "C1": None, "C2": None}
    rows = []
    for site, base in site_base.items():
        g = adoption_month[site]
        for month in range(1, 6):
            rows.append({"site_id": site, "adoption_month": g, "month": month, "downtime_hours": base + month_effect[month]})
    df = pd.DataFrame(rows)

    sweep = run_violation_sensitivity_sweep(df, violation_grid=[-1.0, 0.0, 1.0, 2.0], baseline_window=3)
    # Each unit of violation shifts the estimated ATT by -1.5 (see the exact
    # test above), so the sweep must be monotonically decreasing.
    assert sweep["estimated_att"].is_monotonic_decreasing
