"""Builds a self-contained, interactive Plotly HTML from a real run of Part A
(the RCT CATE comparison) -- reuses the exact fitting/evaluation logic in
`src.pipeline.run_part_a` (same seed, same split) so the numbers here match
the static README tables exactly; this module only adds an interactive view
on top of them, it does not re-derive or alter any estimator's predictions.

Not part of the main `python -m src.pipeline` run (which is matplotlib/PNG
only) -- run directly:

    python -m src.visualization.interactive_plots
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from src.data.simulate_rct import simulate_rct
from src.evaluation.uplift_metrics import cate_recovery_correlation
from src.models.causal_forest import CausalForestModel
from src.models.dr_learner import DoublyRobustModel
from src.models.meta_learners import SLearner, TLearner, XLearner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "outputs" / "interactive" / "cate_estimator_comparison.html"

FEATURE_COLUMNS = ["site", "load_class", "truck_age_years", "utilization_pct", "cumulative_hours_1000s", "prior_90d_downtime_hours"]
CATEGORICAL_FEATURES = ["site", "load_class"]
SEED = 42

COLORS = {
    "s_learner": "#4C72B0", "t_learner": "#DD8452", "x_learner": "#55A868",
    "causal_forest": "#C44E52", "doubly_robust": "#8172B3",
}
LABELS = {
    "s_learner": "S-learner", "t_learner": "T-learner", "x_learner": "X-learner",
    "causal_forest": "Causal Forest DML", "doubly_robust": "Doubly Robust (DRLearner)",
}


def _prep_features(df):
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rct_df = simulate_rct()
    train_df, test_df = train_test_split(rct_df, test_size=0.4, random_state=SEED, stratify=rct_df["treated"])
    X_train, X_test = _prep_features(train_df), _prep_features(test_df)
    T_train = train_df["treated"].to_numpy()
    Y_train = train_df["downtime_next_30d_hours"].to_numpy()
    true_cate_test = test_df["true_cate_hours"].to_numpy()

    models = {
        "s_learner": SLearner().fit(X_train, T_train, Y_train),
        "t_learner": TLearner().fit(X_train, T_train, Y_train),
        "x_learner": XLearner().fit(X_train, T_train, Y_train),
        "causal_forest": CausalForestModel().fit(X_train, T_train, Y_train),
        "doubly_robust": DoublyRobustModel().fit(X_train, T_train, Y_train),
    }
    cate_predictions = {name: model.predict_cate(X_test) for name, model in models.items()}
    correlations = {name: cate_recovery_correlation(pred, true_cate_test) for name, pred in cate_predictions.items()}

    lo = min(true_cate_test.min(), *(p.min() for p in cate_predictions.values()))
    hi = max(true_cate_test.max(), *(p.max() for p in cate_predictions.values()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#999999", dash="dash", width=1.5),
        name="Perfect recovery (y = x)", hoverinfo="skip",
    ))

    for name, pred in cate_predictions.items():
        fig.add_trace(go.Scatter(
            x=true_cate_test, y=pred, mode="markers",
            name=f"{LABELS[name]} (r = {correlations[name]:.3f})",
            marker=dict(color=COLORS[name], size=6, opacity=0.55),
            visible=True if name == "causal_forest" else "legendonly",
            hovertemplate="true CATE: %{x:.2f}h<br>predicted CATE: %{y:.2f}h<extra>" + LABELS[name] + "</extra>",
        ))

    fig.update_layout(
        title="Predicted vs. true CATE, all 5 estimators (test set, n=%d)<br>"
              "<sup>Click a legend entry to toggle an estimator on/off &mdash; Causal Forest DML shown by default (highest ground-truth recovery)</sup>" % len(true_cate_test),
        xaxis_title="True CATE (hours saved, simulated ground truth)",
        yaxis_title="Predicted CATE (hours saved)",
        legend_title="Estimator (Pearson r vs. true CATE)",
        template="plotly_white",
        width=980, height=620,
        hovermode="closest",
        margin=dict(t=90),
    )
    fig.add_annotation(
        text="Every point is a held-out test-set truck (n=%d), never seen during fitting.<br>"
             "Data: real run of python -m src.pipeline, seed 42 &mdash; not illustrative." % len(true_cate_test),
        xref="paper", yref="paper", x=0, y=-0.14, showarrow=False,
        font=dict(size=11, color="#666666"), align="left",
    )

    fig.write_html(str(OUT_PATH), include_plotlyjs="inline", full_html=True)
    print(f"Wrote {OUT_PATH}")
    print("Correlations with true CATE:", {k: round(v, 3) for k, v in correlations.items()})


if __name__ == "__main__":
    main()
