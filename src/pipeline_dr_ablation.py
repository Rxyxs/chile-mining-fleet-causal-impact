"""Ablation: WHY does the DRLearner break on real covariates?

Two competing explanations, each with an arm that would fix the DRLearner if
that explanation were right and leave it broken if it were wrong:

- H1, heavy tails: the linear (OLS) final stage extrapolates on test trucks
  whose raw counters sit far outside the bulk of the training data. A signed
  log1p of every numeric feature compresses the tails. The LightGBM nuisance
  models are (near-)invariant to a monotone transform, so this arm changes
  what the final stage sees, not the pseudo-outcomes.
- H2, collinearity: ~160 features with near-collinear histogram bins make the
  OLS coefficients high-variance. A RidgeCV final stage shrinks them.

Plus the flexible LightGBM final stage the project abandoned on synthetic
data (see `dr_learner.py`): trees cannot extrapolate beyond the leaf values
seen in training, so it is the direct test of whether the earlier "switch to
linear" fix generalizes to real covariates or was specific to the synthetic
data. Causal Forest DML is fit in the same run as the reference.

Every arm shares the DGP draw, split, imputation and nuisance models within
a seed; only the final stage (or the feature transform) changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data.semi_synthetic_dgp import load_ai4i, load_scania
from src.evaluation.uplift_metrics import qini_coefficient, uplift_curve
from src.models.causal_forest import CausalForestModel
from src.models.dr_learner import DoublyRobustModel
from src.pipeline_real_data import REPORTS_DIR, SEEDS, build_condition, preprocess, score_predictions

CONDITIONS = ["synthetic", "ai4i_xy", "scania_x", "scania_xy"]


def signed_log1p(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    numeric = X.select_dtypes(include="number").columns
    X[numeric] = np.sign(X[numeric]) * np.log1p(np.abs(X[numeric]))
    return X


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15)))


def _signed_log1p_array(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


class _SkewedColumns:
    """Column selector for `ColumnTransformer`: called once at fit time on
    the final stage's TRAINING matrix, so which columns count as heavy-tailed
    is learned from train only and then frozen for predict. Constant columns
    (undefined skew) are left alone."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            col_skew = skew(np.asarray(X, dtype=float), axis=0, nan_policy="omit")
        return np.flatnonzero(np.nan_to_num(np.abs(col_skew), nan=0.0) > self.threshold)


def build_skew_aware_ridge(threshold: float = 1.0):
    """Final stage that compresses only the heavy-tailed columns (|skew| >
    `threshold` on train), leaves the rest on their original scale, then
    standardizes everything for RidgeCV. The global log1p arm hurt wherever
    features were not heavy-tailed (synthetic, AI4I); this arm tests whether
    being selective keeps the Scania gain without that cost."""
    return make_pipeline(
        ColumnTransformer(
            [("log_heavy_tails", FunctionTransformer(_signed_log1p_array), _SkewedColumns(threshold))],
            remainder="passthrough",
        ),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 5, 15)),
    )


def _lgbm():
    return LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=31, random_state=42, verbosity=-1)


ARMS = {
    # name: (model factory, apply signed log1p to X?)
    "dr_ols_raw": (lambda: DoublyRobustModel(), False),
    "dr_ols_log": (lambda: DoublyRobustModel(), True),
    "dr_ridge_raw": (lambda: DoublyRobustModel(model_final=_ridge()), False),
    "dr_ridge_log": (lambda: DoublyRobustModel(model_final=_ridge()), True),
    "dr_lgbm_raw": (lambda: DoublyRobustModel(model_final=_lgbm()), False),
    # Pre-registered threshold |skew| > 1 (the usual "highly skewed" cutoff);
    # 0.5 / 2 / 5 are a sensitivity check that the cutoff doesn't decide the result.
    "dr_ridge_skewlog": (lambda: DoublyRobustModel(model_final=build_skew_aware_ridge(1.0)), False),
    "dr_ridge_skewlog_t0.5": (lambda: DoublyRobustModel(model_final=build_skew_aware_ridge(0.5)), False),
    "dr_ridge_skewlog_t2": (lambda: DoublyRobustModel(model_final=build_skew_aware_ridge(2.0)), False),
    "dr_ridge_skewlog_t5": (lambda: DoublyRobustModel(model_final=build_skew_aware_ridge(5.0)), False),
    "causal_forest": (lambda: CausalForestModel(), False),
}


def main(conditions: list[str] | None = None, seeds: list[int] | None = None) -> pd.DataFrame:
    conditions = conditions or CONDITIONS
    seeds = seeds or SEEDS
    real_sources = {"ai4i": load_ai4i(), "scania": load_scania()}

    rows = []
    for condition in conditions:
        for seed in seeds:
            df, features = build_condition(condition, real_sources, seed)
            train_df, test_df = train_test_split(df, test_size=0.4, random_state=seed, stratify=df["treated"])
            X_train, X_test = preprocess(train_df, test_df, features)
            T_train, Y_train = train_df["treated"].to_numpy(), train_df["downtime_next_30d_hours"].to_numpy()
            T_test, Y_test = test_df["treated"].to_numpy(), test_df["downtime_next_30d_hours"].to_numpy()
            true_cate = test_df["true_cate_hours"].to_numpy()
            oracle_qini = qini_coefficient(uplift_curve(true_cate, T_test, Y_test))

            for arm, (factory, log_features) in ARMS.items():
                Xtr, Xte = (signed_log1p(X_train), signed_log1p(X_test)) if log_features else (X_train, X_test)
                pred = factory().fit(Xtr, T_train, Y_train).predict_cate(Xte)
                rows.append({"condition": condition, "seed": seed, "arm": arm,
                             **score_predictions(pred, true_cate, T_test, Y_test, oracle_qini)})
            last = pd.DataFrame(rows[-len(ARMS):])
            print(f"[{condition} seed={seed}] " + "  ".join(f"{r.arm}={r.pearson:.3f}" for r in last.itertuples()), flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(REPORTS_DIR / "semi_synthetic_dr_ablation.csv", index=False)
    summary = results.groupby(["condition", "arm"]).agg(
        pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"), pearson_min=("pearson", "min"),
        nrmse_median=("nrmse", "median"), nrmse_max=("nrmse", "max"),
        wrong_sign_pct_mean=("wrong_sign_pct", "mean"),
        policy_share_mean=("policy_value_share", "mean"),
    ).reset_index()
    summary.to_csv(REPORTS_DIR / "semi_synthetic_dr_ablation_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3))
    return results


if __name__ == "__main__":
    main()
