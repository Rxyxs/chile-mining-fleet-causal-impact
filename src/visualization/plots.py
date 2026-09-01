"""Renders every figure used in the README from real pipeline output."""
from __future__ import annotations

import matplotlib

# Every figure here is saved straight to disk, never shown interactively --
# forcing the non-interactive Agg backend avoids loading Tk at all, which on
# this machine otherwise throws harmless-but-noisy `RuntimeError`s from
# Tkinter's `__del__` during interpreter shutdown once joblib's worker
# threads (spawned by cross-fitting in econml's DRLearner/CausalForestDML)
# are in play.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"s_learner": "#4C72B0", "t_learner": "#DD8452", "x_learner": "#55A868", "causal_forest": "#C44E52", "doubly_robust": "#8172B3"}


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


def plot_honest_bounds(bounds: pd.DataFrame, observed_att: float, breakdown_m: float | None, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.fill_between(bounds["m"], bounds["lower"], bounds["upper"], color="#8172B3", alpha=0.25, label="Honest bound")
    ax.plot(bounds["m"], bounds["lower"], color="#8172B3", linewidth=1)
    ax.plot(bounds["m"], bounds["upper"], color="#8172B3", linewidth=1)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", label="Zero effect")
    ax.axhline(observed_att, color="#C44E52", linewidth=1.5, linestyle=":", label="Point estimate")
    if breakdown_m is not None:
        ax.axvline(breakdown_m, color="gray", linewidth=1, linestyle="-.", label=f"Breakdown M = {breakdown_m:.2f}")
    ax.set_xlabel("M (assumed violation, x largest observed pre-trend deviation)")
    ax.set_ylabel("Bounded ATT (hours)")
    ax.set_title("Honest bounds under a hypothesized parallel-trends violation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_violation_sensitivity_sweep(sweep: pd.DataFrame, true_att: float, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(sweep["violation_per_month"], sweep["estimated_att"], marker="o", color="#4C72B0", label="Estimated ATT under injected violation")
    ax.axhline(true_att, color="#55A868", linewidth=1.5, linestyle="--", label="True ATT (no violation)")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Injected pre-trend violation (hours/month)")
    ax.set_ylabel("Estimated overall ATT (hours)")
    ax.set_title("Empirical sensitivity: estimator bias vs. injected violation size")
    ax.legend()
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


def plot_event_study_animated(event_study: pd.DataFrame, naive_att: float, out_path) -> None:
    """Racing-line animated GIF of the same event-study series plotted by
    `plot_event_study` above -- same real data, no fabricated values. Line
    progressively draws across event time, with a floating label at the
    advancing tip showing the current estimated hours saved."""
    import matplotlib.animation as animation

    ordered = event_study.sort_values("event_time").reset_index(drop=True)
    x = ordered["event_time"].to_numpy()
    y = (-ordered["mean_att"]).to_numpy()
    n_points = len(x)
    n_frames = max(2, min(60, n_points))
    # Subsample real, already-computed points if there are more than we want frames.
    frame_idx = np.unique(np.linspace(0, n_points - 1, n_frames).astype(int))

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        pad = (y.max() - y.min()) * 0.15 + 1e-6
        ax.set_ylim(min(y.min(), -naive_att) - pad, max(y.max(), -naive_att) + pad)
        ax.axhline(-naive_att, color="#C44E52", linestyle="--", linewidth=1.5, label="Naive TWFE (single constant effect)")
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Months since adoption (event time)")
        ax.set_ylabel("Estimated hours saved")
        ax.set_title("Event-study: dynamic treatment effect vs. naive TWFE")
        ax.legend(loc="upper left")

        (line,) = ax.plot([], [], marker="o", color="#7FA6E8", linewidth=2, label="Group-time ATT")
        label = ax.annotate(
            "", xy=(x[0], y[0]), xytext=(15, 15), textcoords="offset points",
            fontsize=10, color="white",
            bbox=dict(boxstyle="round,pad=0.35", fc="#4C72B0", ec="white", alpha=0.9),
        )

        def update(frame_num):
            i = frame_idx[frame_num]
            line.set_data(x[: i + 1], y[: i + 1])
            label.xy = (x[i], y[i])
            label.set_text(f"Group-time ATT: {y[i]:.2f} hrs\n(event time = {x[i]:.0f})")
            return line, label

        ani = animation.FuncAnimation(fig, update, frames=len(frame_idx), interval=200, blit=False)
        ani.save(out_path, writer="pillow")
        plt.close(fig)
