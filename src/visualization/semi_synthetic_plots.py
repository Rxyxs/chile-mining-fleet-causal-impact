"""Figures for the semi-synthetic benchmark (`src/pipeline_real_data.py`).

Uses its own categorical palette rather than `plots.COLORS`: that palette's
orange/green pair is ~4.5 Delta E apart under protanopia, too close to tell
the T- and X-learner apart for a colorblind reader. This one passes a
CVD-separation check; color is fixed per model, never per rank.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODEL_COLORS = {
    "s_learner": "#2a78d6",
    "t_learner": "#eb6834",
    "x_learner": "#1baf7a",
    "causal_forest": "#eda100",
    "doubly_robust": "#e87ba4",
}
MODEL_ORDER = list(MODEL_COLORS)
CONDITION_LABELS = {
    "synthetic": "Synthetic\n(original Part A)",
    "ai4i_x": "AI4I X\n(no real failures in Y0)",
    "ai4i_xy": "AI4I X + real failures",
    "scania_x": "Scania X\n(no real failures in Y0)",
    "scania_xy": "Scania X + real failures",
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_qini_grid(curves_by_condition: dict[str, dict[str, pd.DataFrame]], out_path) -> None:
    """One panel per condition (own y-scale: hours saved differ by dataset),
    each with the 5 estimators, the true-CATE (oracle) ranking and random."""
    conditions = list(curves_by_condition)
    fig, axes = plt.subplots(1, len(conditions), figsize=(4.2 * len(conditions), 4.2), squeeze=False)
    for ax, condition in zip(axes[0], conditions):
        curves = curves_by_condition[condition]
        oracle = curves["oracle_true_cate"]
        ax.plot(oracle["k"], oracle["random_gain"], color=MUTED, linestyle="--", linewidth=1.2, label="Random targeting")
        ax.plot(oracle["k"], oracle["gain"], color=INK, linewidth=2, label="Oracle (true CATE ranking)")
        for name in MODEL_ORDER:
            ax.plot(curves[name]["k"], curves[name]["gain"], color=MODEL_COLORS[name], linewidth=2, label=name)
        ax.set_title(CONDITION_LABELS.get(condition, condition), fontsize=9, color=INK)
        ax.set_xlabel("Units targeted (top-k by predicted CATE)", fontsize=8, color=MUTED)
        _style(ax)
    axes[0][0].set_ylabel("Cumulative estimated hours saved", fontsize=8, color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, fontsize=8)
    fig.suptitle("Qini curves on one held-out test split per condition", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_distributions(metrics: pd.DataFrame, out_path) -> None:
    """Per-seed dots plus the across-seed mean (horizontal tick), per model
    within each condition: recovery correlation, targeting value, and
    normalized RMSE (log scale, where DR instability shows up)."""
    panels = [
        ("pearson", "Correlation with true CATE (Pearson)", False),
        ("policy_value_share", "Share of oracle hours saved\n(top 30% by predicted CATE)", False),
        ("nrmse", "RMSE / std(true CATE)  (log scale)", True),
    ]
    conditions = [c for c in CONDITION_LABELS if c in set(metrics["condition"])]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 10), sharex=True)
    width = 0.8 / len(MODEL_ORDER)
    rng = np.random.default_rng(0)
    for ax, (col, ylabel, log) in zip(axes, panels):
        for i, condition in enumerate(conditions):
            for j, name in enumerate(MODEL_ORDER):
                vals = metrics.loc[(metrics.condition == condition) & (metrics.model == name), col].to_numpy()
                x = i - 0.4 + width * (j + 0.5)
                ax.scatter(x + rng.uniform(-width * 0.25, width * 0.25, len(vals)), vals, s=14,
                           color=MODEL_COLORS[name], alpha=0.75, edgecolors="none",
                           label=name if (i == 0 and col == panels[0][0]) else None)
                ax.hlines(vals.mean(), x - width * 0.4, x + width * 0.4, color=INK, linewidth=1.6)
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        _style(ax)
    axes[-1].set_xticks(range(len(conditions)))
    axes[-1].set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=8, color=INK)
    fig.legend(loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965), markerscale=1.8)
    n_seeds = metrics["seed"].nunique()
    fig.suptitle(f"CATE recovery across {n_seeds} seeds per condition (dots = seeds, tick = mean)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
