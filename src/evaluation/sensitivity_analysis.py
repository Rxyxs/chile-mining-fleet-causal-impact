"""Sensitivity analysis for the parallel-trends assumption behind
`did_estimators.group_time_att` (Part B).

Three complementary checks, in increasing order of how much they assume:

1. **Placebo pre-trend test** (`compute_placebo_pretrend_atts`): re-runs the
   exact same 2x2 DiD machinery entirely *within* the pre-treatment window,
   where nothing actually happened -- a non-zero "placebo effect" there is
   direct evidence the parallel-trends assumption was already broken before
   treatment, no assumption required beyond the data itself.
2. **Honest bounds / breakdown value** (`honest_bounds`), a simplified,
   hand-rolled version of Rambachan & Roth's (2023) "relative magnitudes"
   restriction: assumes an undetected post-treatment violation could be up
   to `M` times the largest pre-trend deviation actually observed, and reports
   the smallest `M` (the "breakdown value") at which the resulting bound
   would no longer rule out a zero effect. This is a bound, not a claim about
   what the violation actually is.
3. **Empirical injection sweep** (`inject_pretrend_violation` +
   `run_violation_sensitivity_sweep`): actually injects a synthetic,
   controlled parallel-trends violation of a known size into the data and
   re-estimates the group-time ATT, to measure -- not just bound -- how much
   a violation of a given size would move this project's own estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.did_estimators import group_time_att, overall_att


def compute_placebo_pretrend_atts(
    df: pd.DataFrame,
    adoption_col: str = "adoption_month",
    month_col: str = "month",
    outcome_col: str = "downtime_hours",
    baseline_window: int = 3,
) -> pd.DataFrame:
    """For each cohort, runs the same 2x2 comparison as `group_time_att`
    (cohort vs. never-treated, relative to the same pre-adoption baseline
    window) but for months strictly *before* that baseline window -- periods
    where, since nothing had happened yet, the estimated "effect" should be
    close to zero under genuine parallel trends.
    """
    never_treated = df[df[adoption_col].isna()]
    cohort_months = sorted(df.loc[df[adoption_col].notna(), adoption_col].unique())

    rows = []
    for g in cohort_months:
        g = int(g)
        baseline_months = range(max(1, g - baseline_window), g)
        cohort_sites = df[df[adoption_col] == g]
        y_g_base = cohort_sites.loc[cohort_sites[month_col].isin(baseline_months), outcome_col].mean()
        y_c_base = never_treated.loc[never_treated[month_col].isin(baseline_months), outcome_col].mean()

        placebo_end = max(1, g - baseline_window)
        for t in range(1, placebo_end):
            y_g_t = cohort_sites.loc[cohort_sites[month_col] == t, outcome_col].mean()
            y_c_t = never_treated.loc[never_treated[month_col] == t, outcome_col].mean()
            placebo_att = (y_g_t - y_g_base) - (y_c_t - y_c_base)
            rows.append({"cohort_adoption_month": g, "month": t, "event_time": t - g, "placebo_att": placebo_att})

    return pd.DataFrame(rows)


def summarize_pretrend_test(placebo_df: pd.DataFrame) -> dict:
    return {
        "mean_placebo_att": float(placebo_df["placebo_att"].mean()),
        "max_abs_placebo_att": float(placebo_df["placebo_att"].abs().max()),
        "n_placebo_estimates": int(len(placebo_df)),
    }


def honest_bounds(observed_att: float, max_pretrend_violation: float, m_grid: np.ndarray | None = None) -> tuple[pd.DataFrame, float | None]:
    """Bounds `observed_att` by +/- M * max_pretrend_violation for each M in
    `m_grid`, and returns the smallest M at which the bound first includes
    zero (the "breakdown value") -- `None` if no M in the grid breaks it.
    """
    if m_grid is None:
        m_grid = np.linspace(0, 3, 31)

    rows = []
    breakdown_m = None
    for m in m_grid:
        margin = m * max_pretrend_violation
        lower, upper = observed_att - margin, observed_att + margin
        crosses_zero = lower <= 0 <= upper
        if crosses_zero and breakdown_m is None:
            breakdown_m = float(m)
        rows.append({"m": float(m), "lower": lower, "upper": upper, "crosses_zero": crosses_zero})

    return pd.DataFrame(rows), breakdown_m


def inject_pretrend_violation(
    df: pd.DataFrame,
    violation_per_month: float,
    adoption_col: str = "adoption_month",
    month_col: str = "month",
    outcome_col: str = "downtime_hours",
) -> pd.DataFrame:
    """Returns a copy of `df` with a synthetic linear pre-trend added to every
    treated cohort's PRE-adoption downtime only (`violation_per_month` hours
    per calendar month) -- a controlled, known-size violation of parallel
    trends, so the resulting bias in `group_time_att` can be *measured*, not
    just theoretically bounded.
    """
    corrupted = df.copy()
    is_treated_cohort = corrupted[adoption_col].notna()
    pre_period = is_treated_cohort & (corrupted[month_col] < corrupted[adoption_col])
    corrupted.loc[pre_period, outcome_col] = (
        corrupted.loc[pre_period, outcome_col] + violation_per_month * corrupted.loc[pre_period, month_col]
    )
    return corrupted


def run_violation_sensitivity_sweep(df: pd.DataFrame, violation_grid: np.ndarray, baseline_window: int = 3) -> pd.DataFrame:
    """Re-estimates the overall group-time ATT after injecting each violation
    size in `violation_grid` -- an empirical sensitivity curve, not a
    theoretical bound."""
    rows = []
    for violation in violation_grid:
        corrupted = inject_pretrend_violation(df, violation_per_month=violation)
        att = overall_att(group_time_att(corrupted, baseline_window=baseline_window))
        rows.append({"violation_per_month": float(violation), "estimated_att": att})
    return pd.DataFrame(rows)
