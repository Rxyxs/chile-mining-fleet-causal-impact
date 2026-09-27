"""Ground-truth diagnostics for the semi-synthetic DGP: how hard is each
condition *by construction*, before any estimator is involved?

In the `*_xy` conditions the true CATE includes `r(x) * repair_hours * F`,
where F is the dataset's real failure label. F is not a model feature, so no
estimator can recover that term better than F can be predicted from X. This
module measures that ceiling directly: it cross-fits a LightGBM classifier
for F on the full real dataset and reports the correlation between the true
CATE and an "oracle-except-F" CATE that knows r(x) and the routine baseline
exactly but has to use the cross-fitted P(F | X) in place of F.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data.semi_synthetic_dgp import (
    BASE_ROUTINE_HOURS,
    REPAIR_HOURS,
    load_ai4i,
    load_scania,
    robust_z,
    true_relative_reduction,
)


def diagnose(real) -> dict:
    zw, zs, zl = (robust_z(s, real.heavy_tailed) for s in (real.wear, real.stress, real.load))
    routine = BASE_ROUTINE_HOURS * np.exp(0.25 * zw + 0.20 * zs + 0.15 * zl)
    r = true_relative_reduction(zw, zs)
    failure = real.real_failure.to_numpy()
    cate = r * (routine + REPAIR_HOURS * failure)

    X = real.X.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes if str(X[col].dtype) == "category" else X[col].astype("category").cat.codes
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=0, verbosity=-1)
    p_fail = cross_val_predict(clf, X, failure, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    ceiling_cate = r * (routine + REPAIR_HOURS * p_fail)

    failure_term = r * REPAIR_HOURS * failure
    numeric = real.X.select_dtypes(include="number")
    return {
        "dataset": real.name,
        "n_units": len(real.X),
        "n_features": real.X.shape[1],
        "real_failure_rate": failure.mean(),
        "share_cate_var_from_failures": failure_term.var() / cate.var(),
        "cv_auc_failure_from_X": roc_auc_score(failure, p_fail),
        "ceiling_pearson_xy": np.corrcoef(cate, ceiling_cate)[0, 1],
        "ceiling_pearson_x": 1.0,  # without the failure term the CATE is an exact function of X
        "missing_cell_share": numeric.isna().to_numpy().mean(),
        "median_abs_feature_skew": numeric.skew().abs().median(),
        "max_abs_feature_corr_excl_self": (
            numeric.corr().abs().where(~np.eye(numeric.shape[1], dtype=bool)).max().max()
        ),
    }


def main() -> pd.DataFrame:
    table = pd.DataFrame([diagnose(load_ai4i()), diagnose(load_scania())])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.T)
    return table


if __name__ == "__main__":
    main()
