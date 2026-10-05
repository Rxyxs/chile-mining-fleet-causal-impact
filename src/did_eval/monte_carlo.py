"""Monte Carlo over synthetic staggered-DiD panels.

Draws N independent panels from the same data-generating process as the published one
(`src.data.simulate_staggered_did`, 32 sites x 36 months, true dynamic effect known), fits
each estimator on every panel and records:

* the empirical distribution of the % error against the true overall ATT, to see whether
  the 6.6% (naive TWFE) and 1.5% (group-time ATT) reported for seed 42 are typical values
  or one lucky draw;
* the real coverage of the 95% interval each estimator reports, using a bootstrap that
  resamples whole sites for the group-time estimators and the analytic cluster-robust
  interval for TWFE.

Run: ``python -m src.did_eval.monte_carlo`` (writes ``tmp_agent_b/``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.simulate_staggered_did import SEED as PUBLISHED_SEED, simulate_staggered_panel
from src.evaluation.did_estimators import group_time_att, naive_twfe_att, overall_att

from .panel_core import NEVER, NOT_YET, cluster_bootstrap_ci, group_time_cells, overall_from_cells, panel_to_matrix

BASE_SEED = 1000  # replication i uses panel seed BASE_SEED + i; the published seed 42 is not in range
OUT_DIR = Path(__file__).resolve().parents[2] / "tmp_agent_b"

# name -> kwargs for group_time_cells / overall_from_cells
SPECS = {
    "gt_never_w3": {"control": NEVER, "baseline_window": 3, "weighted": False},  # headline estimator of src/pipeline.py
    "gt_not_yet_w3": {"control": NOT_YET, "baseline_window": 3, "weighted": False},
    # configuration of the repo's own cluster_bootstrap_overall_att (used for the mpdta interval)
    "gt_never_w1_weighted": {"control": NEVER, "baseline_window": 1, "weighted": True},
}
ESTIMATORS = [*SPECS, "twfe"]
LABELS = {
    "twfe": "Naive TWFE",
    "gt_never_w3": "Group-time, never-treated control (headline)",
    "gt_not_yet_w3": "Group-time, not-yet-treated control",
    "gt_never_w1_weighted": "Group-time, never-treated, w=1, cohort-weighted",
}


def true_overall_att(panel: pd.DataFrame) -> float:
    return float(panel.loc[panel["treated"] == 1, "true_effect_hours"].mean())


def pct_error(estimate: float, truth: float) -> float:
    """Signed % error; positive = the estimate understates the effect's magnitude."""
    return float((estimate - truth) / abs(truth) * 100)


def fit_panel(panel: pd.DataFrame, rng: np.random.Generator | None, n_boot: int) -> dict:
    """Point estimate (and 95% interval if ``rng`` is given) of every estimator on one panel."""
    Y, adoption = panel_to_matrix(panel)
    res = {}
    for name, spec in SPECS.items():
        cell_kw = {"control": spec["control"], "baseline_window": spec["baseline_window"]}
        est = overall_from_cells(group_time_cells(Y, adoption, **cell_kw), spec["weighted"])
        res[name] = {"att": est}
        if rng is not None:
            ci = cluster_bootstrap_ci(Y, adoption, rng, n_boot=n_boot, weighted=spec["weighted"], **cell_kw)
            res[name].update({k: ci[k] for k in ("ci_low", "ci_high", "se")})
    twfe = naive_twfe_att(panel)
    res["twfe"] = {k: twfe[k] for k in ("att", "ci_low", "ci_high", "se")}
    return res


def distribution_summary(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {
        "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "median": float(q[2]),
        "q05": float(q[0]), "q25": float(q[1]), "q75": float(q[3]), "q95": float(q[4]),
        "min": float(v.min()), "max": float(v.max()),
        "mean_abs": float(np.abs(v).mean()), "rmse": float(np.sqrt((v ** 2).mean())),
    }


def coverage_summary(low: np.ndarray, high: np.ndarray, truth: np.ndarray) -> dict:
    low, high, truth = (np.asarray(a, dtype=float) for a in (low, high, truth))
    covered = (low <= truth) & (truth <= high)
    p = float(covered.mean())
    return {
        "coverage": p,
        "mc_se": float(np.sqrt(p * (1 - p) / len(covered))),
        "miss_below": float((truth < low).mean()),  # truth under the interval
        "miss_above": float((truth > high).mean()),
        "mean_width": float((high - low).mean()),
        "n": int(len(covered)),
    }


def run_monte_carlo(n_panels: int = 500, n_boot: int = 300, base_seed: int = BASE_SEED, progress: bool = True) -> dict:
    rows = []
    t0 = time.time()
    for i in range(n_panels):
        panel = simulate_staggered_panel(seed=base_seed + i)
        fit = fit_panel(panel, np.random.default_rng([base_seed, i]), n_boot)
        row = {"seed": base_seed + i, "true_att": true_overall_att(panel)}
        for name, r in fit.items():
            row.update({f"{name}_{k}": v for k, v in r.items()})
        rows.append(row)
        if progress and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_panels} panels, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)

    errors, coverage = {}, {}
    for name in ESTIMATORS:
        err = ((df[f"{name}_att"] - df["true_att"]) / df["true_att"].abs() * 100).to_numpy()
        errors[name] = {"signed_pct": distribution_summary(err), "abs_pct": distribution_summary(np.abs(err))}
        coverage[name] = coverage_summary(df[f"{name}_ci_low"], df[f"{name}_ci_high"], df["true_att"])
    twfe_abs = (df["twfe_att"] - df["true_att"]).abs()
    head_to_head = {
        f"{name}_beats_twfe_share": float(((df[f"{name}_att"] - df["true_att"]).abs() < twfe_abs).mean())
        for name in SPECS
    }
    return {"errors": errors, "coverage": coverage, "head_to_head": head_to_head,
            "per_panel": df, "elapsed_s": time.time() - t0}


def published_panel_reference() -> dict:
    """The seed-42 panel behind the README's 6.6% / 1.5%, recomputed here."""
    panel = simulate_staggered_panel(seed=PUBLISHED_SEED)
    truth = true_overall_att(panel)
    ref = {"seed": PUBLISHED_SEED, "true_att": truth}
    ref["twfe_att"] = naive_twfe_att(panel)["att"]
    ref["group_time_never_w3_att"] = overall_att(group_time_att(panel))
    ref["twfe_pct_error"] = pct_error(ref["twfe_att"], truth)
    ref["group_time_never_w3_pct_error"] = pct_error(ref["group_time_never_w3_att"], truth)
    fast = overall_from_cells(group_time_cells(*panel_to_matrix(panel), baseline_window=3))
    ref["fast_implementation_matches_reference"] = bool(np.isclose(ref["group_time_never_w3_att"], fast))
    return ref


def position_of_published(per_panel: pd.DataFrame, reference: dict) -> dict:
    """Share of Monte Carlo panels whose |error| is at least as small as the published one."""
    out = {}
    for name, key in (("twfe", "twfe_pct_error"), ("gt_never_w3", "group_time_never_w3_pct_error")):
        err = ((per_panel[f"{name}_att"] - per_panel["true_att"]) / per_panel["true_att"].abs() * 100).abs()
        published = abs(reference[key])
        out[name] = {"published_abs_pct_error": published,
                     "share_of_panels_with_smaller_or_equal_error": float((err <= published).mean())}
    return out


def render_summary(results: dict, reference: dict, position: dict, n_panels: int, n_boot: int) -> str:
    e, c = results["errors"], results["coverage"]
    lines = [
        "# Monte Carlo of the staggered-DiD estimators: summary",
        "",
        f"{n_panels} synthetic panels (32 sites x 36 months, same DGP as the published panel, panel seeds "
        f"{BASE_SEED}-{BASE_SEED + n_panels - 1}). Error = (estimate - true ATT) / |true ATT|; "
        "positive means the estimate understates the effect.",
        "",
        "## Error distribution (% of true ATT)",
        "",
        "| Estimator | mean | sd | median | 5th-95th pct | mean abs | RMSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in ESTIMATORS:
        s, a = e[n]["signed_pct"], e[n]["abs_pct"]
        lines.append(f"| {LABELS[n]} | {s['mean']:+.2f} | {s['sd']:.2f} | {s['median']:+.2f} | "
                     f"{s['q05']:+.2f} to {s['q95']:+.2f} | {a['mean']:.2f} | {s['rmse']:.2f} |")
    lines += ["", "## Published seed-42 values vs. the distribution", "",
              f"Recomputed on seed {reference['seed']}: TWFE {reference['twfe_pct_error']:+.2f}%, "
              f"group-time {reference['group_time_never_w3_pct_error']:+.2f}% (true ATT {reference['true_att']:.3f}h)."]
    for n, label in (("twfe", "TWFE"), ("gt_never_w3", "group-time")):
        p = position[n]
        lines.append(f"- {label}: {p['share_of_panels_with_smaller_or_equal_error'] * 100:.1f}% of the {n_panels} panels "
                     f"have an absolute error at or below the published {p['published_abs_pct_error']:.2f}%.")
    beats = "; ".join(f"{k.replace('_beats_twfe_share', '')} {v * 100:.1f}%" for k, v in results["head_to_head"].items())
    lines += ["", f"Share of panels where each estimator is closer to the truth than naive TWFE: {beats}.", "",
              f"## Coverage of the nominal 95% interval ({n_boot} site-cluster bootstrap draws per panel; "
              "TWFE uses the analytic cluster-robust interval)", "",
              "| Estimator | coverage | MC s.e. | truth below | truth above | mean width (h) |",
              "|---|---|---|---|---|---|"]
    for n in ESTIMATORS:
        k = c[n]
        lines.append(f"| {LABELS[n]} | {k['coverage'] * 100:.1f}% | {k['mc_se'] * 100:.1f} | "
                     f"{k['miss_below'] * 100:.1f}% | {k['miss_above'] * 100:.1f}% | {k['mean_width']:.2f} |")
    lines += ["", f"Runtime {results['elapsed_s']:.0f}s. Raw numbers and per-panel values: `did_results_500_panels.json`.", ""]
    return "\n".join(lines)


def main(n_panels: int = 500, n_boot: int = 300) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Monte Carlo: {n_panels} panels, {n_boot} bootstrap draws each")
    results = run_monte_carlo(n_panels, n_boot)
    reference = published_panel_reference()
    position = position_of_published(results["per_panel"], reference)

    payload = {
        "config": {"n_panels": n_panels, "n_boot": n_boot, "base_seed": BASE_SEED, "sites": 32, "months": 36,
                   "estimand": "mean true effect over treated site-months", "specs": SPECS},
        "published_panel_reference": reference,
        "published_vs_distribution": position,
        "errors": results["errors"],
        "coverage": results["coverage"],
        "head_to_head": results["head_to_head"],
        "elapsed_s": results["elapsed_s"],
        "per_panel": results["per_panel"].round(6).to_dict(orient="list"),
    }
    (OUT_DIR / "did_results_500_panels.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    (OUT_DIR / "summary_b.md").write_text(render_summary(results, reference, position, n_panels, n_boot), encoding="utf-8")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
