"""How the group-time ATT behaves as the number of sites (clusters) grows.

The published panel has 32 sites, and `src/did_eval` showed that at that size the estimator's
error is noisy (sd ~15% of the true effect) and its site-cluster bootstrap interval covers
less than the nominal 95%. This module varies the number of sites N in {16, 32, 64, 128}
(four equal cohorts: early / mid / late / never-treated, as in the published design) and
measures, over 100 simulated panels per N:

* the standard deviation of the estimation error (estimate - true overall ATT, in hours);
* the coverage of the 95% percentile interval from resampling whole sites.

The estimator is the project's headline one: group-time ATT, never-treated control,
3-month baseline, unweighted mean over post-treatment cells (`src/pipeline.py`).

`simulate_site_panel` reproduces the data-generating process of
`src.data.simulate_staggered_did` with a free number of sites; that module hard-codes 8 sites
per cohort, so it cannot be reused for N != 32. Its constants (cohort adoption months, the
true dynamic effect) are imported rather than copied.

Run: ``python -m src.sensitivity.cluster_convergence`` (writes ``tmp_agent_b/``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.data.simulate_staggered_did import COHORTS, N_MONTHS, true_effect_at_event_time
from src.did_eval.monte_carlo import coverage_summary, distribution_summary
from src.did_eval.panel_core import cluster_bootstrap_ci, group_time_cells, overall_from_cells

CLUSTER_GRID = (16, 32, 64, 128)
N_PANELS = 100
N_BOOT = 300
BASE_SEED = 5000
OUT_DIR = Path(__file__).resolve().parents[2] / "tmp_agent_b"

# 1-indexed months; never-treated sites carry NaN, the convention of `panel_core`.
_ADOPTION_BY_COHORT = [np.nan if m is None else float(m) for m in COHORTS.values()]
_TRUE_EFFECT_BY_EVENT_TIME = np.array([true_effect_at_event_time(k) for k in range(-N_MONTHS, N_MONTHS)])


def simulate_site_panel(n_sites: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """One balanced panel with ``n_sites`` sites split evenly across the four cohorts.

    Returns (Y[sites, months], adoption[sites], true overall ATT), where the true ATT is the
    mean true effect over treated site-months, as in `src/pipeline.py`.
    """
    n_cohorts = len(_ADOPTION_BY_COHORT)
    if n_sites % n_cohorts:
        raise ValueError(f"n_sites must be a multiple of {n_cohorts}, got {n_sites}")
    adoption = np.repeat(_ADOPTION_BY_COHORT, n_sites // n_cohorts)

    months = np.arange(1, N_MONTHS + 1)
    site_base = np.clip(rng.normal(45, 8, size=n_sites), 20, None)
    month_effect = 6.0 * np.sin(2 * np.pi * (months - 3) / 12)

    event_time = months[None, :] - adoption[:, None]  # NaN for never-treated
    treated = event_time >= 0  # NaN compares False
    lookup = np.where(treated, event_time, 0).astype(int) + N_MONTHS
    effect = np.where(treated, _TRUE_EFFECT_BY_EVENT_TIME[lookup], 0.0)

    mu = site_base[:, None] + month_effect[None, :] + effect
    Y = rng.gamma(shape=40.0, scale=np.maximum(mu, 5.0) / 40.0)
    return Y, adoption, float(effect[treated].mean())


def evaluate_cluster_count(n_sites: int, n_panels: int = N_PANELS, n_boot: int = N_BOOT, base_seed: int = BASE_SEED) -> dict:
    estimates, truths, lows, highs = [], [], [], []
    for i in range(n_panels):
        Y, adoption, truth = simulate_site_panel(n_sites, np.random.default_rng([base_seed, n_sites, i, 0]))
        est = overall_from_cells(group_time_cells(Y, adoption, baseline_window=3))
        ci = cluster_bootstrap_ci(Y, adoption, np.random.default_rng([base_seed, n_sites, i, 1]), n_boot=n_boot, baseline_window=3)
        estimates.append(est)
        truths.append(truth)
        lows.append(ci["ci_low"])
        highs.append(ci["ci_high"])

    estimates, truths = np.asarray(estimates), np.asarray(truths)
    err = estimates - truths
    pct = err / np.abs(truths) * 100
    cov = coverage_summary(np.asarray(lows), np.asarray(highs), truths)
    return {
        "std_error": float(err.std(ddof=1)),
        "coverage_95": cov["coverage"],
        "details": {
            "bias_hours": float(err.mean()),
            "std_error_pct_of_true_att": float(pct.std(ddof=1)),
            "std_error_times_sqrt_n": float(err.std(ddof=1) * np.sqrt(n_sites)),
            "error_hours": distribution_summary(err),
            "coverage_mc_se": cov["mc_se"],
            "coverage_miss_below": cov["miss_below"],
            "coverage_miss_above": cov["miss_above"],
            "mean_ci_width_hours": cov["mean_width"],
            "mean_true_att": float(truths.mean()),
            "n_panels": n_panels,
        },
    }


def run(clusters=CLUSTER_GRID, n_panels: int = N_PANELS, n_boot: int = N_BOOT, progress: bool = True) -> dict:
    t0 = time.time()
    per_n = {}
    for n in clusters:
        per_n[n] = evaluate_cluster_count(n, n_panels, n_boot)
        if progress:
            print(f"  N={n}: sd={per_n[n]['std_error']:.3f}h coverage={per_n[n]['coverage_95']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    return {
        "clusters_eval": list(clusters),
        "metrics": {str(n): {"std_error": r["std_error"], "coverage_95": r["coverage_95"]} for n, r in per_n.items()},
        "details": {
            "std_error": "sd over panels of (estimate - true overall ATT), in hours",
            "coverage_95": "share of panels whose 95% site-cluster percentile bootstrap interval contains the true overall ATT",
            "estimator": "group-time ATT, never-treated control, baseline_window=3, unweighted mean of post-treatment cells",
            "n_panels": n_panels, "n_boot": n_boot, "base_seed": BASE_SEED, "elapsed_s": time.time() - t0,
            "per_n": {str(n): r["details"] for n, r in per_n.items()},
        },
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Cluster convergence: N in {list(CLUSTER_GRID)}, {N_PANELS} panels each, {N_BOOT} bootstrap draws")
    out = run()
    (OUT_DIR / "sensitivity_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'sensitivity_results.json'}")


if __name__ == "__main__":
    main()
