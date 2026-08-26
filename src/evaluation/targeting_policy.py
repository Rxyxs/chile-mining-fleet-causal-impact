"""Budget-constrained targeting: if the maintenance program can only be
rolled out to a fraction of the fleet, which trucks should get it?

Compares three real targeting policies against the true, known counterfactual
(only computable because this is a simulation) -- this is the business
decision the whole CATE-estimation exercise is actually for: ranking by
predicted *uplift* is not the same as ranking by predicted *risk*, and the
gap between them is the concrete case for why CATE modeling earns its keep
over a simpler risk model a team might reach for by default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_targeting_policies(
    df: pd.DataFrame,
    budget_fraction: float,
    risk_score: np.ndarray,
    uplift_score: np.ndarray,
    true_cate_col: str = "true_cate_hours",
    seed: int = 42,
) -> pd.DataFrame:
    """For each policy, selects the top `budget_fraction` of the fleet by that
    policy's ranking and sums the TRUE counterfactual hours saved for the
    selected trucks -- the real-world outcome each policy would have actually
    delivered, not an in-sample metric.
    """
    n = len(df)
    budget_n = max(1, int(round(n * budget_fraction)))
    true_cate = df[true_cate_col].to_numpy()

    rng = np.random.default_rng(seed)
    random_idx = rng.choice(n, size=budget_n, replace=False)

    risk_idx = np.argsort(-risk_score)[:budget_n]
    uplift_idx = np.argsort(-uplift_score)[:budget_n]
    oracle_idx = np.argsort(-true_cate)[:budget_n]

    policies = {
        "random": random_idx,
        "highest_baseline_risk": risk_idx,
        "highest_predicted_uplift": uplift_idx,
        "oracle_true_uplift": oracle_idx,
    }

    rows = []
    for name, idx in policies.items():
        total_hours_saved = true_cate[idx].sum()
        rows.append({
            "policy": name,
            "n_trucks_targeted": budget_n,
            "total_hours_saved": round(float(total_hours_saved), 2),
            "avg_hours_saved_per_truck": round(float(total_hours_saved / budget_n), 3),
            "pct_of_oracle_achieved": round(float(total_hours_saved / true_cate[oracle_idx].sum() * 100), 1),
        })

    return pd.DataFrame(rows).sort_values("total_hours_saved", ascending=False).reset_index(drop=True)
