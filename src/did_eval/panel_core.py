"""Matrix implementation of the group-time ATT, fast enough to bootstrap 500 panels.

`src.evaluation.did_estimators.group_time_att` works on a long DataFrame and re-filters it
for every (cohort, month) cell. That is fine for one panel but too slow for a Monte Carlo
with a cluster bootstrap inside every replication, so this module reproduces the same
quantity on a (sites x months) matrix. `tests/test_did_eval.py` checks that, with a
never-treated control, it returns exactly the same cells as the reference estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEVER = "never"
NOT_YET = "not_yet"


def panel_to_matrix(
    df: pd.DataFrame,
    site_col: str = "site_id",
    month_col: str = "month",
    adoption_col: str = "adoption_month",
    outcome_col: str = "downtime_hours",
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced long panel -> (Y[sites, months], adoption[sites]); never-treated = NaN."""
    wide = df.pivot(index=site_col, columns=month_col, values=outcome_col).sort_index(axis=1)
    adoption = df.groupby(site_col)[adoption_col].first().reindex(wide.index).to_numpy(dtype=float)
    return wide.to_numpy(dtype=float), adoption


def group_time_cells(
    Y: np.ndarray,
    adoption: np.ndarray,
    baseline_window: int = 3,
    control: str = NEVER,
) -> pd.DataFrame:
    """ATT(g, t) for every adoption cohort g and every period t >= g.

    ``control="never"``: comparison group is the never-treated sites only.
    ``control="not_yet"``: comparison group at period t is every site not yet treated at t
    (never-treated plus cohorts adopting after t), the other Callaway-Sant'Anna option.
    The pre-period is the mean of the ``baseline_window`` months before g, taken over the
    same comparison group, so a control site is never already treated in either period.

    Returns columns: g, t, att, n_cohort (sites in cohort g), n_control (sites in the
    comparison group at t). Cells without any comparison site are dropped.
    """
    n_sites, n_months = Y.shape
    never = np.isnan(adoption)
    t_index = np.arange(1, n_months + 1)

    if control == NEVER:
        masks = np.tile(never, (n_months, 1))
    elif control == NOT_YET:
        masks = never[None, :] | (adoption[None, :] > t_index[:, None])
    else:
        raise ValueError(f"unknown control group: {control!r}")
    masks = masks.astype(float)
    n_ctrl = masks.sum(axis=1)
    safe = np.where(n_ctrl > 0, n_ctrl, np.nan)
    ctrl_at_t = (masks * Y.T).sum(axis=1) / safe

    out = []
    for g in np.unique(adoption[~never]):
        g = int(g)
        cohort = adoption == g
        base_cols = np.arange(max(1, g - baseline_window) - 1, g - 1)
        row_base = Y[:, base_cols].mean(axis=1)
        cohort_base = row_base[cohort].mean()
        ctrl_base = (masks @ row_base) / safe
        cohort_t = Y[cohort].mean(axis=0)
        for t in range(g, n_months + 1):
            if n_ctrl[t - 1] == 0:
                continue
            att = (cohort_t[t - 1] - cohort_base) - (ctrl_at_t[t - 1] - ctrl_base[t - 1])
            out.append((g, t, att, int(cohort.sum()), int(n_ctrl[t - 1])))
    return pd.DataFrame(out, columns=["g", "t", "att", "n_cohort", "n_control"])


def overall_from_cells(cells: pd.DataFrame, weighted: bool = False) -> float:
    """Mean post-treatment ATT(g, t). Unweighted matches `did_estimators.overall_att`;
    weighted by cohort size matches `weighted_overall_att`."""
    if cells.empty:
        return float("nan")
    if weighted:
        return float(np.average(cells["att"], weights=cells["n_cohort"]))
    return float(cells["att"].mean())


def cluster_bootstrap_ci(
    Y: np.ndarray,
    adoption: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 300,
    alpha: float = 0.05,
    **cell_kwargs,
) -> dict:
    """Percentile interval from resampling whole sites with replacement.

    ``weighted`` (default False) is forwarded to `overall_from_cells`; everything else in
    ``cell_kwargs`` goes to `group_time_cells`. Draws with no usable cell are skipped.
    """
    weighted = cell_kwargs.pop("weighted", False)
    n = len(Y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        est = overall_from_cells(group_time_cells(Y[idx], adoption[idx], **cell_kwargs), weighted)
        if np.isfinite(est):
            draws.append(est)
    draws = np.asarray(draws)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci_low": float(lo), "ci_high": float(hi), "se": float(draws.std(ddof=1)), "n_boot": int(len(draws))}
