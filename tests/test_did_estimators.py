"""`group_time_att` is tested on a small, fully deterministic (noise-free)
panel where the true effect is known and hand-computable, so the estimator's
output can be checked against an exact expected number, not just "roughly
plausible." See the module docstring in `did_estimators.py` for why this
estimator (never-treated as control, event-time aggregation) exists.

Panel: sites T1, T2 (adopt at month 3) and C1, C2 (never-treated), months 1-5.
outcome = site_base_effect + month_effect + true_treatment_effect(event_time),
with no random noise, so every intermediate quantity below is exact.
"""
import pandas as pd
import pytest

from src.evaluation.did_estimators import aggregate_event_study, group_time_att, overall_att

MONTH_EFFECT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
SITE_BASE = {"T1": 10, "T2": 12, "C1": 8, "C2": 9}
TRUE_EFFECT_BY_EVENT_TIME = {0: -4, 1: -6, 2: -7}
ADOPTION_MONTH = {"T1": 3, "T2": 3, "C1": None, "C2": None}


def _toy_panel() -> pd.DataFrame:
    rows = []
    for site, base in SITE_BASE.items():
        adoption_month = ADOPTION_MONTH[site]
        for month in range(1, 6):
            event_time = None if adoption_month is None else month - adoption_month
            treated = event_time is not None and event_time >= 0
            effect = TRUE_EFFECT_BY_EVENT_TIME.get(event_time, 0) if treated else 0
            rows.append({
                "site_id": site, "adoption_month": adoption_month, "month": month,
                "treated": int(treated), "downtime_hours": SITE_BASE[site] + MONTH_EFFECT[month] + effect,
            })
    return pd.DataFrame(rows)


def test_group_time_att_recovers_exact_effect_at_each_event_time():
    df = _toy_panel()
    result = group_time_att(df, baseline_window=3)
    by_event_time = result.set_index("event_time")["att"]

    assert by_event_time.loc[0] == pytest.approx(-4.0)
    assert by_event_time.loc[1] == pytest.approx(-6.0)
    assert by_event_time.loc[2] == pytest.approx(-7.0)


def test_group_time_att_uses_only_never_treated_as_control():
    """The control side of every 2x2 comparison must come only from the
    never-treated group -- this is the entire point of the estimator versus
    naive TWFE. In the base toy panel, C1 and C2 share an identical trend
    (only their levels differ), so which never-treated sites get averaged is
    numerically invisible -- a real property of that panel, not a weak test,
    but it means we need a second panel with a genuinely divergent control
    trend to actually observe the estimator responding to *which* sites it
    uses as control.
    """
    df = _toy_panel()

    # Give C2 its own upward trend that C1 does not have, breaking the
    # perfect-parallel-trends coincidence of the base toy panel.
    df_divergent = df.copy()
    c2_extra_trend = {1: 0, 2: 0, 3: 5, 4: 10, 5: 15}
    is_c2 = df_divergent["site_id"] == "C2"
    df_divergent.loc[is_c2, "downtime_hours"] += df_divergent.loc[is_c2, "month"].map(c2_extra_trend)

    result_with_both_controls = group_time_att(df_divergent, baseline_window=3)
    result_without_c2 = group_time_att(df_divergent[df_divergent["site_id"] != "C2"], baseline_window=3)

    assert not result_with_both_controls["att"].equals(result_without_c2["att"])
    # And using only C1 (unaffected by the injected trend) as control must
    # recover the exact original, undistorted effect.
    assert result_without_c2.set_index("event_time")["att"].loc[0] == pytest.approx(-4.0)


def test_aggregate_event_study_pools_correctly():
    df = _toy_panel()
    group_time = group_time_att(df, baseline_window=3)
    event_study = aggregate_event_study(group_time)

    assert event_study.set_index("event_time")["n_cohorts"].to_dict() == {0: 1, 1: 1, 2: 1}
    assert event_study.set_index("event_time")["mean_att"].loc[0] == pytest.approx(-4.0)


def test_overall_att_is_mean_of_group_time_atts():
    df = _toy_panel()
    group_time = group_time_att(df, baseline_window=3)
    assert overall_att(group_time) == pytest.approx((-4.0 - 6.0 - 7.0) / 3)
