"""Part A on real data: CATE rankings scored on a genuine randomized experiment.

The Hillstrom e-mail trial randomly assigned 64,000 customers to a men's campaign, a women's
campaign or no e-mail. Because assignment really was random, a held-out split can be used to
evaluate how well each estimator ranks customers by treatment effect, without a simulated truth:

* ``qini``        : area between the estimator's gain curve and random targeting (see
                    ``uplift_metrics``), with a permutation p-value against random rankings.
* ``top30_uplift``: the e-mail's effect on the visit rate among the 30% of held-out customers the
                    estimator ranks highest, estimated from the randomized arms of that slice.
                    Compare with ``all_uplift`` (the effect when targeting at random).

Each arm is compared with the no-e-mail control. The 20 seeds redraw the 60/40 split; they all
reuse the same 64,000 customers, so their spread shows sensitivity to the split and is not a
confidence interval for the population effect.

    python -m src.pipeline_real_rct
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.real_data import HILLSTROM_ARMS, HILLSTROM_FEATURES, load_hillstrom
from src.evaluation.uplift_metrics import qini_coefficient, uplift_curve
from src.models.causal_forest import CausalForestModel
from src.models.dr_learner import DoublyRobustModel
from src.models.meta_learners import SLearner, TLearner, XLearner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

SEEDS = list(range(100, 120))
TEST_SIZE = 0.4
BUDGET = 0.30
N_PERMUTATIONS = 200
MODELS = {
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "causal_forest": CausalForestModel,
    "doubly_robust": DoublyRobustModel,
}
LABELS = {"s_learner": "S-learner", "t_learner": "T-learner", "x_learner": "X-learner",
          "causal_forest": "Causal forest", "doubly_robust": "Doubly robust"}

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def top_share_uplift(cate: np.ndarray, treatment: np.ndarray, visit: np.ndarray, share: float) -> float:
    """Effect on the visit rate (treated minus control) among the top ``share`` by predicted CATE."""
    k = int(round(len(cate) * share))
    idx = np.argsort(-cate)[:k]
    t, v = treatment[idx], visit[idx]
    if t.sum() == 0 or (1 - t).sum() == 0:
        return float("nan")
    return float(v[t == 1].mean() - v[t == 0].mean())


def permutation_pvalue(cate: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, observed: float, rng) -> float:
    """Share of random rankings whose Qini is at least as large as the estimator's."""
    null = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        null[i] = qini_coefficient(uplift_curve(rng.permutation(cate), treatment, outcome))
    return float((1 + (null >= observed).sum()) / (1 + N_PERMUTATIONS))


def difference_in_means(df: pd.DataFrame) -> dict:
    """The randomized trial's own average effect on the visit rate, with a Welch 95% interval."""
    t, c = df.loc[df["treatment"] == 1, "visit"], df.loc[df["treatment"] == 0, "visit"]
    diff = float(t.mean() - c.mean())
    se = float(np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c)))
    return {"ate": diff, "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se,
            "visit_rate_treated": float(t.mean()), "visit_rate_control": float(c.mean()),
            "n_treated": int(len(t)), "n_control": int(len(c))}


def run_arm(arm: str, seeds: list[int]) -> dict:
    df = load_hillstrom(arm)
    rows = []
    for seed in seeds:
        train, test = train_test_split(df, test_size=TEST_SIZE, random_state=seed, stratify=df["treatment"])
        t_te, y_te, v_te = test["treatment"].to_numpy(), test["outcome"].to_numpy(), test["visit"].to_numpy()
        all_uplift = float(v_te[t_te == 1].mean() - v_te[t_te == 0].mean())
        rng = np.random.default_rng(seed)
        for name, cls in MODELS.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = cls().fit(train[HILLSTROM_FEATURES], train["treatment"].to_numpy(), train["outcome"].to_numpy())
                cate = model.predict_cate(test[HILLSTROM_FEATURES])
            qini = qini_coefficient(uplift_curve(cate, t_te, y_te))
            rows.append({
                "seed": seed, "model": name, "qini": qini,
                "qini_pvalue": permutation_pvalue(cate, t_te, y_te, qini, rng),
                "top30_uplift": top_share_uplift(cate, t_te, v_te, BUDGET),
                "all_uplift": all_uplift, "mean_predicted_cate": float(cate.mean()),
            })
    per_seed = pd.DataFrame(rows)
    summary = {}
    for name, g in per_seed.groupby("model"):
        summary[name] = {
            "qini_mean": float(g["qini"].mean()), "qini_sd": float(g["qini"].std(ddof=1)),
            "seeds_qini_significant": int((g["qini_pvalue"] < 0.05).sum()),
            "top30_uplift_mean": float(g["top30_uplift"].mean()), "top30_uplift_sd": float(g["top30_uplift"].std(ddof=1)),
            "all_uplift_mean": float(g["all_uplift"].mean()),
            "top30_gain_over_random": float((g["top30_uplift"] - g["all_uplift"]).mean()),
            "seeds_beating_random_at_30": int((g["top30_uplift"] > g["all_uplift"]).sum()),
            "mean_predicted_cate": float(g["mean_predicted_cate"].mean()),
        }
    return {"arm": arm, "n_customers": int(len(df)), "randomized_ate": difference_in_means(df),
            "n_seeds": len(seeds), "summary": summary, "per_seed": per_seed}


def plot_targeting(results: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.6 * len(results), 4.4), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, results):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        names = list(MODELS)
        per = res["per_seed"]
        for x, name in enumerate(names):
            g = per[per["model"] == name]["top30_uplift"]
            ax.errorbar(x, g.mean(), yerr=g.std(ddof=1), fmt="o", color=BLUE, markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=1.6, capsize=0)
            ax.text(x, g.mean() + g.std(ddof=1) + 0.003, f"{g.mean() * 100:.1f}", ha="center", fontsize=9, color=INK)
        base = per["all_uplift"].mean()
        ax.axhline(base, color=ORANGE, linewidth=1.6)
        ax.text(-0.45, base - 0.0022, f"Random targeting: {base * 100:.1f} pp", ha="left", va="top", fontsize=9, color=INK)
        ax.set_ylim(0.036, 0.108)
        ax.set_xlim(-0.55, len(names) - 0.45)
        ax.set_xticks(range(len(names)), [LABELS[n] for n in names], rotation=20, fontsize=9, color=INK_2)
        ax.tick_params(length=0, colors=INK_2, labelsize=9)
        ax.set_title(res["arm"], loc="left", color=INK, fontsize=11, pad=10)
    axes[0].set_ylabel("Effect on visit rate in the top 30% (percentage points)", color=INK_2, fontsize=9)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}")
    fig.suptitle("E-mail effect among the 30% each estimator ranks highest (mean ± sd over 20 splits)",
                 x=0.01, y=1.02, ha="left", color=INK, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def run(seeds: list[int] = SEEDS) -> dict:
    arms = [run_arm(arm, seeds) for arm in HILLSTROM_ARMS]
    out = {r["arm"]: {k: v for k, v in r.items() if k not in ("per_seed", "arm")} for r in arms}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "real_rct.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.concat([r["per_seed"].assign(arm=r["arm"]) for r in arms]).to_csv(REPORTS_DIR / "real_rct_per_seed.csv", index=False)
    plot_targeting(arms, FIGURES_DIR / "real_rct_targeting.png")
    return out


if __name__ == "__main__":
    res = run()
    for arm, r in res.items():
        ate = r["randomized_ate"]
        print(f"\n{arm}: n={r['n_customers']:,} | randomized ATE {ate['ate'] * 100:+.2f} pp [{ate['ci_low'] * 100:+.2f}, {ate['ci_high'] * 100:+.2f}]")
        for name, s in r["summary"].items():
            print(f"  {name:14s} qini {s['qini_mean']:+8.2f} ±{s['qini_sd']:6.2f} sig {s['seeds_qini_significant']:2d}/{r['n_seeds']} "
                  f"| top30 {s['top30_uplift_mean'] * 100:5.2f} vs random {s['all_uplift_mean'] * 100:5.2f} pp, beats random {s['seeds_beating_random_at_30']}/{r['n_seeds']}")
