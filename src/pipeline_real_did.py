"""Part B on real data: staggered-adoption DiD checked against an independent implementation.

Runs this project's group-time ATT estimator on the ``mpdta`` county panel (Callaway & Sant'Anna's
worked example) and compares it, cell by cell, with the ``csdid`` package (a port of the authors'
reference ``did`` R package). The estimator is run with ``baseline_window=1`` so its comparison
period (the period just before adoption) is the one the reference uses.

There is no "true effect" here, so this checks *agreement with a reference* and shows how the naive
two-way fixed-effects estimate and the unweighted mean of cells differ from the weighted
aggregate, on data nobody simulated.

    python -m src.pipeline_real_did
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.real_data import FIRST_YEAR, load_mpdta_panel
from src.evaluation.did_estimators import (
    cluster_bootstrap_overall_att,
    cohort_sizes,
    group_time_att,
    naive_twfe_att,
    overall_att,
    weighted_overall_att,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TOLERANCE = 1e-3  # the reference prints ATT(g,t) rounded to 4 decimals
N_BOOT = 500

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9994"


def reference_estimates(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """ATT(g,t) cells and the simple overall ATT from the independent ``csdid`` implementation."""
    from csdid.att_gt import ATTgt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ATTgt(
            yname="lemp", gname="first.treat", idname="countyreal", tname="year", xformla="lemp~1", data=raw,
            control_group="nevertreated", est_method="reg", base_period="universal",
        ).fit(bstrap=False)
        cells = res.summ_attgt().summary2.rename(columns={"ATT(g, t)": "ref_att", "Std. Error": "ref_se"})
        agg = res.aggte(typec="simple").atte
    return cells[["Group", "Time", "ref_att", "ref_se"]], {"att": float(agg["overall_att"]), "se": float(agg["overall_se"])}


def compare_cells(mine: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    mine = mine.assign(Group=mine["cohort_adoption_month"] + (FIRST_YEAR - 1), Time=mine["month"] + (FIRST_YEAR - 1))
    merged = mine.merge(ref, on=["Group", "Time"], how="left")
    merged["diff"] = merged["att"] - merged["ref_att"]
    return merged[["Group", "Time", "att", "ref_att", "diff"]].rename(columns={"att": "mine"})


def plot_estimators(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    colors = {"twfe": ORANGE, "weighted": BLUE, "reference": INK, "unweighted": GRAY}
    for y, row in enumerate(reversed(rows)):
        c = colors[row["key"]]
        if row.get("lo") is not None:
            ax.plot([row["lo"], row["hi"]], [y, y], color=c, linewidth=2.2, solid_capstyle="round")
        ax.plot(row["att"], y, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(row["att"], y + 0.22, f"{row['att']:+.3f}", ha="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK_2, linewidth=0.9)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_yticks(range(len(rows)), [r["label"] for r in reversed(rows)], fontsize=9, color=INK)
    ax.tick_params(axis="x", colors=INK_2, labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Effect on log teen employment (95% interval where available)", color=INK_2, fontsize=9)
    ax.set_title("Overall effect of the staggered rollout, by estimator (mpdta)", loc="left", color=INK, fontsize=12, pad=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def run(n_boot: int = N_BOOT) -> dict:
    panel, raw = load_mpdta_panel()
    cells = group_time_att(panel, baseline_window=1)
    ref_cells, ref_overall = reference_estimates(raw)
    comparison = compare_cells(cells, ref_cells)
    max_diff = float(comparison["diff"].abs().max())
    if max_diff > TOLERANCE:
        raise AssertionError(f"ATT(g,t) disagrees with the reference by {max_diff:.5f} (> {TOLERANCE})")

    sizes = cohort_sizes(panel)
    weighted = weighted_overall_att(cells, sizes)
    boot = cluster_bootstrap_overall_att(panel, n_boot=n_boot, seed=0)
    twfe = naive_twfe_att(panel)
    unweighted = overall_att(cells)

    results = {
        "panel": {
            "counties": int(panel["site_id"].nunique()),
            "periods": int(panel["month"].nunique()),
            "cohort_sizes_counties": {int(k) + FIRST_YEAR - 1: int(v) for k, v in sizes.items()},
            "never_treated_counties": int(panel.loc[panel["adoption_month"].isna(), "site_id"].nunique()),
        },
        "cells_vs_reference": comparison.round(5).to_dict(orient="records"),
        "max_abs_diff_vs_reference": round(max_diff, 6),
        "overall": {
            "reference_csdid": {"att": round(ref_overall["att"], 4), "se": round(ref_overall["se"], 4)},
            "this_project_weighted": {"att": round(weighted, 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in boot.items()}},
            "naive_twfe": {k: round(v, 4) for k, v in twfe.items()},
            "unweighted_mean_of_cells": round(unweighted, 4),
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "real_did.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_estimators(
        [
            {"key": "reference", "label": "Reference (csdid)", "att": ref_overall["att"],
             "lo": ref_overall["att"] - 1.96 * ref_overall["se"], "hi": ref_overall["att"] + 1.96 * ref_overall["se"]},
            {"key": "weighted", "label": "This project, cohort-weighted\n(bootstrap interval)", "att": weighted,
             "lo": boot["ci_low"], "hi": boot["ci_high"]},
            {"key": "twfe", "label": "Naive two-way fixed effects", "att": twfe["att"], "lo": twfe["ci_low"], "hi": twfe["ci_high"]},
            {"key": "unweighted", "label": "Unweighted mean of cells", "att": unweighted},
        ],
        FIGURES_DIR / "real_did_estimators.png",
    )
    return results


if __name__ == "__main__":
    out = run()
    o = out["overall"]
    print(f"max |diff| vs reference over {len(out['cells_vs_reference'])} cells: {out['max_abs_diff_vs_reference']}")
    print(f"reference {o['reference_csdid']} | mine {o['this_project_weighted']}")
    print(f"TWFE {o['naive_twfe']['att']} | unweighted cells {o['unweighted_mean_of_cells']}")
