"""Semi-synthetic benchmark: the same 5 Part A CATE estimators, re-run on real
covariates (Scania APS trucks, AI4I 2020) with a simulated block-randomized
pilot and a known heterogeneous effect (`src/data/semi_synthetic_dgp.py`),
side by side with the original fully synthetic pilot.

Five conditions, all at n=3,000 with the same 60/40 split rule, the same
models and the same preprocessing, so each contrast changes one thing:

- synthetic   : `simulate_rct` (the original Part A data).
- ai4i_x      : AI4I real covariates, smooth simulated baseline (no real failures in Y0).
- ai4i_xy     : AI4I real covariates + the dataset's real failures in Y0.
- scania_x    : Scania real covariates, smooth simulated baseline.
- scania_xy   : Scania real covariates + the dataset's real failures in Y0.

synthetic -> *_x isolates the real covariate distribution; *_x -> *_xy
isolates the real failure structure in the baseline. Every condition is
repeated over `SEEDS` (each seed redraws the real-unit sample, the assignment,
the outcome noise and the split), because a single split is too noisy to rank
estimators whose scores sit close together.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from src.data.semi_synthetic_dgp import N_UNITS, REPAIR_HOURS, create_semi_synthetic_rct, load_ai4i, load_scania
from src.data.simulate_rct import simulate_rct
from src.evaluation.uplift_metrics import cate_recovery_correlation, qini_coefficient, uplift_curve
from src.models.causal_forest import CausalForestModel
from src.models.dr_learner import DoublyRobustModel
from src.models.meta_learners import SLearner, TLearner, XLearner
from src.visualization.semi_synthetic_plots import plot_metric_distributions, plot_qini_grid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

SEEDS = list(range(100, 120))
QINI_FIGURE_SEED = SEEDS[0]
BUDGET_FRACTION = 0.30
MAX_TRAIN_MISSING_SHARE = 0.5

SYNTHETIC_FEATURES = ["site", "load_class", "truck_age_years", "utilization_pct", "cumulative_hours_1000s", "prior_90d_downtime_hours"]
MODEL_NAMES = ["s_learner", "t_learner", "x_learner", "causal_forest", "doubly_robust"]


def build_condition(condition: str, real_sources: dict, seed: int) -> tuple[pd.DataFrame, list[str]]:
    if condition == "synthetic":
        df = simulate_rct(n=N_UNITS, seed=seed)
        return df, SYNTHETIC_FEATURES
    source, variant = condition.rsplit("_", 1)
    real = real_sources[source]
    df = create_semi_synthetic_rct(real, n=N_UNITS, seed=seed, repair_hours=0.0 if variant == "x" else REPAIR_HOURS)
    return df, list(real.X.columns)


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on train only, apply to both: drop columns missing in more than
    half the TRAIN rows, median-impute the rest with TRAIN medians, and mark
    object columns as categorical. Every estimator gets this same matrix --
    LightGBM could take NaNs natively, but EconML's final stages cannot, and a
    model comparison where models see different inputs is not a comparison.
    """
    X_train, X_test = train_df[features].copy(), test_df[features].copy()
    numeric = X_train.select_dtypes(include="number").columns
    missing_share = X_train[numeric].isna().mean()
    keep_numeric = missing_share[missing_share <= MAX_TRAIN_MISSING_SHARE].index
    dropped = [c for c in numeric if c not in keep_numeric]
    X_train, X_test = X_train.drop(columns=dropped), X_test.drop(columns=dropped)

    medians = X_train[keep_numeric].median()
    X_train[keep_numeric] = X_train[keep_numeric].fillna(medians)
    X_test[keep_numeric] = X_test[keep_numeric].fillna(medians)

    for col in X_train.columns:
        if X_train[col].dtype == object or str(X_train[col].dtype) == "category":
            categories = sorted(X_train[col].astype(str).unique())
            X_train[col] = pd.Categorical(X_train[col].astype(str), categories=categories)
            X_test[col] = pd.Categorical(X_test[col].astype(str), categories=categories)
    return X_train, X_test


def fit_models(X_train: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> tuple[dict, dict]:
    constructors = {
        "s_learner": SLearner, "t_learner": TLearner, "x_learner": XLearner,
        "causal_forest": CausalForestModel, "doubly_robust": DoublyRobustModel,
    }
    models, fit_seconds = {}, {}
    for name in MODEL_NAMES:
        start = time.perf_counter()
        models[name] = constructors[name]().fit(X_train, T, Y)
        fit_seconds[name] = time.perf_counter() - start
    return models, fit_seconds


def score_predictions(pred: np.ndarray, true_cate: np.ndarray, T: np.ndarray, Y: np.ndarray, oracle_qini: float) -> dict:
    """Metrics that stay comparable across conditions whose CATE scales differ:
    correlations are scale-free, the Qini is divided by the Qini of the true
    CATE ranking on the same test units, RMSE is divided by the true CATE's
    std, and the policy value is a share of what the oracle ranking captures.
    """
    budget_n = int(round(len(pred) * BUDGET_FRACTION))
    oracle_value = np.sort(true_cate)[::-1][:budget_n].sum()
    policy_value = true_cate[np.argsort(-pred)[:budget_n]].sum()
    true_spread = np.quantile(true_cate, 0.99) - np.quantile(true_cate, 0.01)
    pred_spread = np.quantile(pred, 0.99) - np.quantile(pred, 0.01)
    return {
        "pearson": cate_recovery_correlation(pred, true_cate),
        "spearman": float(spearmanr(pred, true_cate)[0]),
        "qini_normalized": qini_coefficient(uplift_curve(pred, T, Y)) / oracle_qini,
        "nrmse": float(np.sqrt(np.mean((pred - true_cate) ** 2)) / true_cate.std()),
        "wrong_sign_pct": float((pred < 0).mean() * 100),
        "spread_ratio_p1_p99": float(pred_spread / true_spread),
        "policy_value_share": float(policy_value / oracle_value),
        "mean_pred_minus_true_ate": float(pred.mean() - true_cate.mean()),
    }


def run_one(condition: str, real_sources: dict, seed: int) -> tuple[pd.DataFrame, dict | None]:
    df, features = build_condition(condition, real_sources, seed)
    train_df, test_df = train_test_split(df, test_size=0.4, random_state=seed, stratify=df["treated"])
    X_train, X_test = preprocess(train_df, test_df, features)
    T_train, Y_train = train_df["treated"].to_numpy(), train_df["downtime_next_30d_hours"].to_numpy()
    T_test, Y_test = test_df["treated"].to_numpy(), test_df["downtime_next_30d_hours"].to_numpy()
    true_cate = test_df["true_cate_hours"].to_numpy()

    models, fit_seconds = fit_models(X_train, T_train, Y_train)
    oracle_curve = uplift_curve(true_cate, T_test, Y_test)
    oracle_qini = qini_coefficient(oracle_curve)

    rows, curves = [], {"oracle_true_cate": oracle_curve}
    for name, model in models.items():
        pred = model.predict_cate(X_test)
        curves[name] = uplift_curve(pred, T_test, Y_test)
        rows.append({
            "condition": condition, "seed": seed, "model": name,
            "n_train": len(train_df), "n_test": len(test_df), "n_features": X_train.shape[1],
            "test_real_failures": int(test_df["real_failure"].sum()) if "real_failure" in test_df else 0,
            "true_ate_test": float(true_cate.mean()), "true_cate_std_test": float(true_cate.std()),
            "fit_seconds": fit_seconds[name],
            **score_predictions(pred, true_cate, T_test, Y_test, oracle_qini),
        })
    return pd.DataFrame(rows), (curves if seed == QINI_FIGURE_SEED else None)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    agg = metrics.groupby(["condition", "model"]).agg(
        pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"),
        spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"),
        qini_norm_mean=("qini_normalized", "mean"), qini_norm_sd=("qini_normalized", "std"),
        nrmse_mean=("nrmse", "mean"), nrmse_max=("nrmse", "max"),
        wrong_sign_pct_mean=("wrong_sign_pct", "mean"), wrong_sign_pct_max=("wrong_sign_pct", "max"),
        spread_ratio_mean=("spread_ratio_p1_p99", "mean"), spread_ratio_max=("spread_ratio_p1_p99", "max"),
        policy_share_mean=("policy_value_share", "mean"), policy_share_sd=("policy_value_share", "std"),
        fit_seconds_mean=("fit_seconds", "mean"), n_seeds=("seed", "nunique"),
    ).reset_index()
    wins = (
        metrics.loc[metrics.groupby(["condition", "seed"])["pearson"].idxmax()]
        .groupby(["condition", "model"]).size().rename("seeds_best_pearson")
    )
    return agg.merge(wins, on=["condition", "model"], how="left").fillna({"seeds_best_pearson": 0})


def main(conditions: list[str] | None = None, seeds: list[int] | None = None) -> pd.DataFrame:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    conditions = conditions or ["synthetic", "ai4i_x", "ai4i_xy", "scania_x", "scania_xy"]
    seeds = seeds or SEEDS

    real_sources = {}
    if any(c.startswith("ai4i") for c in conditions):
        real_sources["ai4i"] = load_ai4i()
    if any(c.startswith("scania") for c in conditions):
        real_sources["scania"] = load_scania()

    all_metrics, qini_curves = [], {}
    for condition in conditions:
        for seed in seeds:
            start = time.perf_counter()
            metrics, curves = run_one(condition, real_sources, seed)
            all_metrics.append(metrics)
            if curves is not None:
                qini_curves[condition] = curves
            best = metrics.loc[metrics["pearson"].idxmax(), "model"]
            print(f"[{condition} seed={seed}] {time.perf_counter() - start:5.1f}s  best={best}  "
                  + "  ".join(f"{r.model}={r.pearson:.3f}" for r in metrics.itertuples()), flush=True)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(REPORTS_DIR / "semi_synthetic_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(REPORTS_DIR / "semi_synthetic_summary.csv", index=False)

    if qini_curves:
        plot_qini_grid(qini_curves, FIGURES_DIR / "semi_synthetic_qini_curves.png")
    plot_metric_distributions(metrics, FIGURES_DIR / "semi_synthetic_cate_recovery.png")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(3))
    return metrics


if __name__ == "__main__":
    main()
