"""Meta-learners are tested on a small, fully synthetic dataset with a known,
simple linear treatment effect (not `simulate_rct`'s more complex DGP) so the
expected sign and rough magnitude of the recovered CATE is easy to reason
about directly, and so these tests run fast.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.meta_learners import SLearner, TLearner, XLearner

TINY_PARAMS = {"n_estimators": 40, "learning_rate": 0.3, "max_depth": 3, "num_leaves": 7, "random_state": 42, "verbosity": -1}


def _toy_dataset(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    X = pd.DataFrame({"x1": x1, "x2": rng.normal(0, 1, n)})
    T = rng.integers(0, 2, n)

    # True CATE (hours-saved convention) increases with x1: tau(x) = 1 + 0.5*x1
    true_cate = 1 + 0.5 * x1
    baseline = 20 + rng.normal(0, 2, n)
    Y = baseline - T * true_cate + rng.normal(0, 1, n)
    return X, T, Y, true_cate


@pytest.mark.parametrize("learner_cls", [SLearner, TLearner, XLearner])
def test_meta_learner_recovers_positive_cate_on_average(learner_cls):
    X, T, Y, true_cate = _toy_dataset()
    model = learner_cls(params=TINY_PARAMS).fit(X, T, Y)
    predicted = model.predict_cate(X)

    # "Positive = hours saved" convention: the true effect here is always
    # positive (treatment always helps), so the predicted average should be
    # positive too, not just correlated in sign with a mix of ups and downs.
    assert predicted.mean() > 0


@pytest.mark.parametrize("learner_cls", [SLearner, TLearner, XLearner])
def test_meta_learner_predicted_cate_correlates_with_true_cate(learner_cls):
    X, T, Y, true_cate = _toy_dataset()
    model = learner_cls(params=TINY_PARAMS).fit(X, T, Y)
    predicted = model.predict_cate(X)

    corr = np.corrcoef(predicted, true_cate)[0, 1]
    assert corr > 0.3  # loose threshold -- these are small/noisy models by design
