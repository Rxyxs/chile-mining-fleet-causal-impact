"""Thin wrapper around econml's `CausalForestDML` -- a real, published
double-machine-learning causal forest (Athey, Tibshirani & Wager, 2019),
included as a fourth CATE estimator alongside the hand-rolled meta-learners
so the project shows both "built the mechanics by hand" and "knows the
established professional tool" rather than only one or the other.

Unlike the meta-learners, DML explicitly models and residualizes out
E[Y|X] and E[T|X] before estimating the treatment-effect function, which in
principle should make it more robust than the S/T/X-learners when the
propensity score varies with X (as it does here, slightly, by site).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from lightgbm import LGBMClassifier, LGBMRegressor


class CausalForestModel:
    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.model = CausalForestDML(
            model_y=LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=31, random_state=random_state, verbosity=-1),
            model_t=LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=31, random_state=random_state, verbosity=-1),
            discrete_treatment=True,
            n_estimators=n_estimators,
            min_samples_leaf=20,
            cv=3,
            random_state=random_state,
        )

    def _encode(self, X: pd.DataFrame, fit_columns: bool) -> np.ndarray:
        # econml's internal nuisance models (here, LightGBM) receive X as a
        # raw numpy array with no dtype metadata -- passing pandas
        # `category`-dtype columns straight through loses that information and
        # LightGBM then tries to read the literal strings as floats. One-hot
        # encoding here, with the column set fixed at fit time and reindexed
        # (zero-filled) at predict time, keeps train/predict consistent even
        # if a category is missing from a given batch.
        encoded = pd.get_dummies(X, columns=["site", "load_class"], drop_first=True)
        if fit_columns:
            self._encoded_columns = encoded.columns
        else:
            encoded = encoded.reindex(columns=self._encoded_columns, fill_value=0)
        return encoded.to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "CausalForestModel":
        X_encoded = self._encode(X, fit_columns=True)
        self.model.fit(Y=Y, T=T, X=X_encoded)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        # CausalForestDML estimates the effect of T: 0 -> 1 (an increase in
        # downtime, since T=1 is treatment here reducing downtime) -- flip the
        # sign so "positive" consistently means "hours saved," matching the
        # meta-learners' and the true-CATE convention used everywhere else in
        # this project.
        X_encoded = self._encode(X, fit_columns=False)
        return -self.model.effect(X_encoded)
