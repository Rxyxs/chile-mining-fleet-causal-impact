"""End-to-end orchestrator: simulates both datasets, fits all 4 CATE
estimators on the RCT (Part A), evaluates them via Qini curves / calibration
/ targeting policy, then runs the staggered DiD comparison (Part B), and
writes every figure, table, and metric this project's README reports.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from src.data.simulate_rct import simulate_rct
from src.data.simulate_staggered_did import simulate_staggered_panel
from src.evaluation.did_estimators import aggregate_event_study, group_time_att, naive_twfe_att, overall_att
from src.evaluation.sensitivity_analysis import (
    compute_placebo_pretrend_atts,
    honest_bounds,
    run_violation_sensitivity_sweep,
    summarize_pretrend_test,
)
from src.evaluation.results_db import persist_results_to_duckdb
from src.evaluation.targeting_policy import evaluate_targeting_policies
from src.evaluation.uplift_metrics import cate_calibration_bins, cate_recovery_correlation, qini_coefficient, uplift_curve
from src.models.causal_forest import CausalForestModel
from src.models.dr_learner import DoublyRobustModel
from src.models.meta_learners import SLearner, TLearner, XLearner
from src.visualization.plots import (
    plot_cate_calibration,
    plot_covariate_balance,
    plot_event_study,
    plot_event_study_animated,
    plot_honest_bounds,
    plot_qini_curves,
    plot_targeting_policy_comparison,
    plot_violation_sensitivity_sweep,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

FEATURE_COLUMNS = ["site", "load_class", "truck_age_years", "utilization_pct", "cumulative_hours_1000s", "prior_90d_downtime_hours"]
CATEGORICAL_FEATURES = ["site", "load_class"]
SEED = 42


def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def run_part_a(rct_df: pd.DataFrame) -> dict:
    train_df, test_df = train_test_split(rct_df, test_size=0.4, random_state=SEED, stratify=rct_df["treated"])

    X_train, X_test = _prep_features(train_df), _prep_features(test_df)
    T_train, T_test = train_df["treated"].to_numpy(), test_df["treated"].to_numpy()
    Y_train, Y_test = train_df["downtime_next_30d_hours"].to_numpy(), test_df["downtime_next_30d_hours"].to_numpy()
    true_cate_test = test_df["true_cate_hours"].to_numpy()

    balance = plot_covariate_balance(
        rct_df, ["truck_age_years", "utilization_pct", "cumulative_hours_1000s", "prior_90d_downtime_hours"],
        "treated", FIGURES_DIR / "covariate_balance.png",
    )

    models = {
        "s_learner": SLearner().fit(X_train, T_train, Y_train),
        "t_learner": TLearner().fit(X_train, T_train, Y_train),
        "x_learner": XLearner().fit(X_train, T_train, Y_train),
        "causal_forest": CausalForestModel().fit(X_train, T_train, Y_train),
        "doubly_robust": DoublyRobustModel().fit(X_train, T_train, Y_train),
    }

    cate_predictions = {name: model.predict_cate(X_test) for name, model in models.items()}

    curves = {name: uplift_curve(pred, T_test, Y_test) for name, pred in cate_predictions.items()}
    plot_qini_curves(curves, FIGURES_DIR / "qini_curves.png")

    qini_scores = {name: qini_coefficient(curve) for name, curve in curves.items()}
    recovery_corr = {name: cate_recovery_correlation(pred, true_cate_test) for name, pred in cate_predictions.items()}

    # Model selection uses ground-truth recovery correlation, not the Qini
    # score -- available here only because this is a simulation. In a real
    # deployment without ground truth, cross-validated Qini across multiple
    # splits (not a single train/test split, which is noisy) would be the
    # practical substitute; see the README for the honest disagreement this
    # single-split Qini ranking has with ground-truth recovery.
    best_model_name = max(recovery_corr, key=recovery_corr.get)
    calibration = cate_calibration_bins(cate_predictions[best_model_name], true_cate_test)
    plot_cate_calibration(calibration, FIGURES_DIR / "cate_calibration.png", best_model_name)

    risk_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, num_leaves=31, random_state=SEED, verbosity=-1)
    risk_model.fit(_prep_features(train_df[train_df["treated"] == 0]), train_df.loc[train_df["treated"] == 0, "downtime_next_30d_hours"])
    risk_score_test = risk_model.predict(X_test)

    policy_comparison = evaluate_targeting_policies(
        test_df.reset_index(drop=True), budget_fraction=0.30,
        risk_score=risk_score_test, uplift_score=cate_predictions[best_model_name],
    )
    plot_targeting_policy_comparison(policy_comparison, FIGURES_DIR / "targeting_policy_comparison.png")

    # Control minus treated, so positive = hours saved -- consistent with
    # `true_cate_hours` and every CATE estimator's convention in this project.
    naive_ate = float(train_df.loc[train_df.treated == 0, "downtime_next_30d_hours"].mean()
                       - train_df.loc[train_df.treated == 1, "downtime_next_30d_hours"].mean())
    true_ate_test = float(true_cate_test.mean())

    return {
        "n_train": len(train_df), "n_test": len(test_df),
        "naive_diff_in_means_ate": naive_ate,
        "true_ate_test_set": true_ate_test,
        "qini_coefficients": qini_scores,
        "cate_recovery_correlation": recovery_corr,
        "best_model": best_model_name,
        "targeting_policy_comparison": policy_comparison.to_dict(orient="records"),
        "covariate_balance": balance.to_dict(orient="records"),
    }


def run_part_b(panel_df: pd.DataFrame) -> dict:
    naive = naive_twfe_att(panel_df)
    group_time = group_time_att(panel_df)
    event_study = aggregate_event_study(group_time)
    overall_corrected_att = overall_att(group_time)

    plot_event_study(event_study, naive["att"], FIGURES_DIR / "event_study.png")
    plot_event_study_animated(event_study, naive["att"], FIGURES_DIR / "event_study_animated.gif")

    true_att = float(panel_df.loc[panel_df["treated"] == 1, "true_effect_hours"].mean())

    # Sensitivity analysis: is the group-time ATT robust to a violation of
    # parallel trends? Three checks, see src/evaluation/sensitivity_analysis.py.
    placebo_atts = compute_placebo_pretrend_atts(panel_df)
    pretrend_summary = summarize_pretrend_test(placebo_atts)

    bounds, breakdown_m = honest_bounds(overall_corrected_att, pretrend_summary["max_abs_placebo_att"])
    plot_honest_bounds(bounds, overall_corrected_att, breakdown_m, FIGURES_DIR / "honest_bounds.png")

    violation_grid = np.linspace(-2.0, 2.0, 9)
    sweep = run_violation_sensitivity_sweep(panel_df, violation_grid)
    plot_violation_sensitivity_sweep(sweep, true_att, FIGURES_DIR / "violation_sensitivity_sweep.png")

    return {
        "naive_twfe": naive,
        "group_time_overall_att": overall_corrected_att,
        "true_overall_att": true_att,
        "event_study": event_study.to_dict(orient="records"),
        "pretrend_test": pretrend_summary,
        "honest_bounds_breakdown_m": breakdown_m,
        "violation_sensitivity_sweep": sweep.to_dict(orient="records"),
    }


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Part A: RCT-based CATE estimation ===")
    rct_df = simulate_rct()
    rct_df.to_csv(RAW_DIR / "rct_fleet_pilot.csv", index=False)
    part_a_results = run_part_a(rct_df)

    print(f"Naive diff-in-means ATE: {part_a_results['naive_diff_in_means_ate']:.2f}h")
    print(f"True ATE (test set): {part_a_results['true_ate_test_set']:.2f}h")
    print("Qini coefficients:", {k: round(v, 3) for k, v in part_a_results["qini_coefficients"].items()})
    print("CATE recovery correlation (predicted vs. true):", {k: round(v, 3) for k, v in part_a_results["cate_recovery_correlation"].items()})
    print(f"Best model by ground-truth recovery: {part_a_results['best_model']}")
    print("\nTargeting policy comparison:")
    print(pd.DataFrame(part_a_results["targeting_policy_comparison"]))

    print("\n=== Part B: Staggered-adoption DiD ===")
    panel_df = simulate_staggered_panel()
    panel_df.to_csv(RAW_DIR / "staggered_site_panel.csv", index=False)
    part_b_results = run_part_b(panel_df)

    print(f"Naive TWFE ATT: {part_b_results['naive_twfe']['att']:.2f}h (se={part_b_results['naive_twfe']['se']:.2f})")
    print(f"Group-time (corrected) overall ATT: {part_b_results['group_time_overall_att']:.2f}h")
    print(f"True overall ATT: {part_b_results['true_overall_att']:.2f}h")

    pretrend = part_b_results["pretrend_test"]
    print(f"\nPlacebo pre-trend test: mean={pretrend['mean_placebo_att']:.3f}h, "
          f"max|placebo|={pretrend['max_abs_placebo_att']:.3f}h, n={pretrend['n_placebo_estimates']}")
    breakdown_m = part_b_results["honest_bounds_breakdown_m"]
    print(f"Honest-bounds breakdown M: {breakdown_m:.2f}" if breakdown_m is not None else "Honest-bounds breakdown M: none within grid")

    all_results = {"part_a_rct_uplift": part_a_results, "part_b_staggered_did": part_b_results}
    (REPORTS_DIR / "results.json").write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    db_path = persist_results_to_duckdb(all_results, REPORTS_DIR / "results.duckdb")
    print(f"\nSaved figures to {FIGURES_DIR}/, metrics to {REPORTS_DIR / 'results.json'}, and comparison tables to {db_path}")


if __name__ == "__main__":
    main()
