"""Renders every figure used in the README from real pipeline output."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"s_learner": "#4C72B0", "t_learner": "#DD8452", "x_learner": "#55A868", "causal_forest": "#C44E52"}


def plot_covariate_balance(df: pd.DataFrame, covariates: list[str], treatment_col: str, out_path) -> pd.DataFrame:
    """Standardized mean difference (treated - control) / pooled std, per
    covariate -- the standard randomization-balance check. Values within
    roughly +/-0.1 are considered well-balanced."""
    rows = []
    for col in covariates:
        x = df[col]
        if x.dtype == object or str(x.dtype) == "category":
            x = pd.get_dummies(x, drop_first=True).iloc[:, 0]
            col_label = f"{col} (one level)"
        else:
            col_label = col
        treated = x[df[treatment_col] == 1]
        control = x[df[treatment_col] == 0]
        pooled_std = np.sqrt((treated.var() + control.var()) / 2)
        smd = (treated.mean() - control.mean()) / pooled_std if pooled_std > 0 else 0.0
        rows.append({"covariate": col_label, "smd": smd})

    balance = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C44E52" if abs(v) > 0.1 else "#55A868" for v in balance["smd"]]
    ax.barh(balance["covariate"], balance["smd"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(-0.1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Standardized mean difference (treated - control)")
    ax.set_title("Covariate balance check (randomized pilot)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return balance


def plot_qini_curves(curves: dict[str, pd.DataFrame], out_path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    any_curve = next(iter(curves.values()))
    ax.plot(any_curve["k"], any_curve["random_gain"], color="black", linestyle="--", label="Random targeting")
    for name, curve in curves.items():
        ax.plot(curve["k"], curve["gain"], color=COLORS.get(name, None), label=name)
    ax.set_xlabel("Trucks targeted (top-k by predicted CATE)")
    ax.set_ylabel("Cumulative estimated hours saved")
    ax.set_title("Uplift (Qini) curves: CATE estimator comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cate_calibration(calibration: pd.DataFrame, out_path, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    lims = [
        min(calibration["mean_predicted"].min(), calibration["mean_true"].min()) - 1,
        max(calibration["mean_predicted"].max(), calibration["mean_true"].max()) + 1,
    ]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="Perfect calibration")
    ax.scatter(calibration["mean_predicted"], calibration["mean_true"], s=60, color="#4C72B0")
    ax.set_xlabel("Mean predicted CATE (hours), per decile")
    ax.set_ylabel("Mean true CATE (hours), per decile")
    ax.set_title(f"CATE calibration: {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_targeting_policy_comparison(policy_df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    order = policy_df.sort_values("total_hours_saved")
    colors = ["#8C8C8C" if p != "oracle_true_uplift" else "#C44E52" for p in order["policy"]]
    colors = [
        "#55A868" if p == "highest_predicted_uplift" else c
        for p, c in zip(order["policy"], colors)
    ]
    ax.barh(order["policy"], order["total_hours_saved"], color=colors)
    ax.set_xlabel("Total hours saved (true counterfactual) at fixed budget")
    ax.set_title("Targeting policy comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_event_study(event_study: pd.DataFrame, naive_att: float, out_path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(event_study["event_time"], -event_study["mean_att"], marker="o", color="#4C72B0", label="Group-time ATT (this project's estimator)")
    ax.axhline(-naive_att, color="#C44E52", linestyle="--", label="Naive TWFE (single constant effect)")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Months since adoption (event time)")
    ax.set_ylabel("Estimated hours saved")
    ax.set_title("Event-study: dynamic treatment effect vs. naive TWFE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
