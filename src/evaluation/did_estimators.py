"""Two difference-in-differences estimators for the staggered-adoption panel
(Part B), deliberately contrasted:

1. **Naive TWFE** (`naive_twfe_att`): a single two-way fixed-effects
   regression (`linearmodels.PanelOLS`, site + month fixed effects, one
   `treated` dummy) -- the default a team reaches for first, and forces a
   single constant treatment effect onto what is, by construction here, a
   dynamic effect.

2. **Group-time ATT** (`group_time_att`), a simplified Callaway & Sant'Anna
   (2021)-style estimator: for each adoption cohort g and each post-adoption
   period t, compares that cohort's *own* pre/post change against the
   **never-treated** group's change over the same window (not the
   "not-yet-treated" variant CS also allow -- never-treated only, for a
   simpler, still-valid comparison group under this project's parallel-trends
   design). Because it never uses an already-treated site as a control, it
   isn't exposed to the bias mechanism TWFE has under staggered timing +
   dynamic effects (Goodman-Bacon, 2021).

Both are run against data where the true effect is *known* (the simulator's
`true_effect_hours` column), so the comparison in the README is against the
real answer, not just the two estimators against each other.
"""
from __future__ import annotations

import pandas as pd
from linearmodels.panel import PanelOLS


def naive_twfe_att(df: pd.DataFrame) -> dict:
    panel = df.set_index(["site_id", "month"])
    model = PanelOLS.from_formula("downtime_hours ~ treated + EntityEffects + TimeEffects", data=panel)
    result = model.fit(cov_type="clustered", cluster_entity=True)
    return {
        "att": float(result.params["treated"]),
        "se": float(result.std_errors["treated"]),
        "ci_low": float(result.conf_int().loc["treated", "lower"]),
        "ci_high": float(result.conf_int().loc["treated", "upper"]),
    }


def group_time_att(
    df: pd.DataFrame,
    adoption_col: str = "adoption_month",
    month_col: str = "month",
    outcome_col: str = "downtime_hours",
    baseline_window: int = 3,
) -> pd.DataFrame:
    """`baseline_window` pre-treatment months (not just the single month right
    before adoption) are averaged into the "pre" side of each 2x2 comparison,
    for both the cohort and the control group -- a common practical variant of
    Callaway-Sant'Anna that trades a little robustness to a last-minute
    pre-trend for a real reduction in variance, valid here because the true
    pre-treatment effect is exactly 0 and month effects are common to every
    site (no differential pre-trend across the averaging window by
    construction of the simulator).
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

        for t in range(g, int(df[month_col].max()) + 1):
            y_g_t = cohort_sites.loc[cohort_sites[month_col] == t, outcome_col].mean()
            y_c_t = never_treated.loc[never_treated[month_col] == t, outcome_col].mean()
            att = (y_g_t - y_g_base) - (y_c_t - y_c_base)
            rows.append({"cohort_adoption_month": g, "month": t, "event_time": t - g, "att": att})

    return pd.DataFrame(rows)


def aggregate_event_study(group_time_df: pd.DataFrame) -> pd.DataFrame:
    """Pools ATT(g,t) across cohorts by event time (months since adoption) --
    the shape of this curve is what recovers the true dynamic ramp-up, which
    a single TWFE coefficient cannot represent by construction."""
    return (
        group_time_df.groupby("event_time")["att"]
        .agg(mean_att="mean", n_cohorts="count")
        .reset_index()
        .sort_values("event_time")
    )


def overall_att(group_time_df: pd.DataFrame) -> float:
    return float(group_time_df["att"].mean())
