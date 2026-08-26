"""S-learner, T-learner, and X-learner CATE (conditional average treatment
effect) estimators, built directly on LightGBM regressors/classifiers --
hand-rolled rather than imported from a meta-learner library, so the
mechanics of each are auditable line by line. `causal_forest.py` provides a
fourth estimator from a real, established causal-ML library (econml) as a
point of comparison against these three.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

DEFAULT_LGBM_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "num_leaves": 31, "random_state": 42, "verbosity": -1}


class SLearner:
    """Single model f(X, T) -> Y. CATE(x) = f(x, 1) - f(x, 0).

    Simplest possible approach; prone to under-fitting the treatment effect
    if the model regularizes the (low-signal-to-noise) treatment indicator
    away in favor of the (higher-signal) covariates -- a known weakness this
    project's benchmark against the other estimators is designed to surface.
    """

    def __init__(self, params: dict | None = None):
        self.model = LGBMRegressor(**(params or DEFAULT_LGBM_PARAMS))

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "SLearner":
        X_aug = X.copy()
        X_aug["__treatment__"] = T
        self.model.fit(X_aug, Y)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        # f(x, 0) - f(x, 1): control-arm prediction minus treated-arm
        # prediction, so positive = hours saved by treatment -- the same
        # "reduction is good" convention used by every other estimator here.
        X1 = X.copy(); X1["__treatment__"] = 1
        X0 = X.copy(); X0["__treatment__"] = 0
        return self.model.predict(X0) - self.model.predict(X1)


class TLearner:
    """Two separate models, one fit on treated-only rows and one on
    control-only rows. CATE(x) = f1(x) - f0(x).

    Avoids the S-learner's risk of regularizing the treatment signal away,
    but each model only sees half the data, and the two models can disagree
    on regions of X where one arm has few observations.
    """

    def __init__(self, params: dict | None = None):
        params = params or DEFAULT_LGBM_PARAMS
        self.model_treated = LGBMRegressor(**params)
        self.model_control = LGBMRegressor(**params)

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "TLearner":
        self.model_treated.fit(X[T == 1], Y[T == 1])
        self.model_control.fit(X[T == 0], Y[T == 0])
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_control.predict(X) - self.model_treated.predict(X)


class XLearner:
    """Kunzel et al. (2019). Builds on the T-learner:
    1. Fit T-learner models f1 (treated), f0 (control).
    2. Impute individual treatment effects: for treated units, D1_i = Y_i - f0(X_i);
       for control units, D0_i = f1(X_i) - Y_i.
    3. Fit two more models, g1 on (X, D1) among treated, g0 on (X, D0) among control.
    4. Combine via the propensity score e(x): CATE(x) = e(x)*g0(x) + (1-e(x))*g1(x)
       -- weighting each arm's imputed-effect model by how *little* data the
       other arm has at that x, which is where the X-learner earns its name
       and its main advantage over the T-learner in imbalanced designs.

    Propensity is estimated from data (a logistic regression on X) rather
    than assumed to be a constant 0.5, because this pilot block-randomized
    *within site* at slightly different probabilities per site (see
    `simulate_rct.SITE_TREATMENT_PROB`) -- the true propensity is not exactly
    50/50 once site is accounted for.
    """

    def __init__(self, params: dict | None = None):
        params = params or DEFAULT_LGBM_PARAMS
        self.model_treated = LGBMRegressor(**params)
        self.model_control = LGBMRegressor(**params)
        self.model_effect_treated = LGBMRegressor(**params)
        self.model_effect_control = LGBMRegressor(**params)
        self.propensity_model = LGBMClassifier(**{**params, "n_estimators": 150})

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "XLearner":
        self.model_treated.fit(X[T == 1], Y[T == 1])
        self.model_control.fit(X[T == 0], Y[T == 0])

        # Imputed effects in the "hours saved" convention (positive = good):
        # for a treated unit, how much lower its actual outcome was than the
        # control model's counterfactual prediction; for a control unit, how
        # much lower the treated model's counterfactual prediction was than
        # its actual outcome.
        d1 = self.model_control.predict(X[T == 1]) - Y[T == 1]
        d0 = Y[T == 0] - self.model_treated.predict(X[T == 0])
        self.model_effect_treated.fit(X[T == 1], d1)
        self.model_effect_control.fit(X[T == 0], d0)

        self.propensity_model.fit(X, T)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        propensity = np.clip(self.propensity_model.predict_proba(X)[:, 1], 0.05, 0.95)
        g1 = self.model_effect_treated.predict(X)
        g0 = self.model_effect_control.predict(X)
        return propensity * g0 + (1 - propensity) * g1
