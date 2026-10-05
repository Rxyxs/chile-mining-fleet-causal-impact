"""Group-time ATT with a "not-yet-treated" comparison group.

`src.evaluation.did_estimators.group_time_att` compares each cohort only with the
never-treated sites (8 of 32 here). Callaway & Sant'Anna (2021) also allow using every site
that has not yet adopted by period t: for the early cohort at month 15 that is the mid and
late cohorts as well as the never-treated, so the control group is up to 3x larger and the
estimate less noisy. The price is an extra identifying assumption (no anticipation, and
parallel trends that also hold between the cohorts used as controls).

A site is never used as a control in a period in which it is already treated, so the
Goodman-Bacon "forbidden comparison" that biases naive TWFE does not come back.
"""
from __future__ import annotations

import pandas as pd

from .panel_core import NOT_YET, group_time_cells, overall_from_cells, panel_to_matrix


def not_yet_treated_att(
    df: pd.DataFrame,
    adoption_col: str = "adoption_month",
    month_col: str = "month",
    outcome_col: str = "downtime_hours",
    site_col: str = "site_id",
    baseline_window: int = 3,
) -> pd.DataFrame:
    """Same layout as `group_time_att`, plus ``n_control_sites`` per cell."""
    Y, adoption = panel_to_matrix(df, site_col, month_col, adoption_col, outcome_col)
    cells = group_time_cells(Y, adoption, baseline_window=baseline_window, control=NOT_YET)
    return pd.DataFrame({
        "cohort_adoption_month": cells["g"],
        "month": cells["t"],
        "event_time": cells["t"] - cells["g"],
        "att": cells["att"],
        "n_control_sites": cells["n_control"],
    })


def not_yet_treated_overall_att(df: pd.DataFrame, weighted: bool = False, **kwargs) -> float:
    Y, adoption = panel_to_matrix(df, **{k: v for k, v in kwargs.items() if k != "baseline_window"})
    cells = group_time_cells(Y, adoption, baseline_window=kwargs.get("baseline_window", 3), control=NOT_YET)
    return overall_from_cells(cells, weighted)
