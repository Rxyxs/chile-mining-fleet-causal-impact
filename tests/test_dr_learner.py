import numpy as np
import pandas as pd

from src.models.dr_learner import DoublyRobustModel


def _toy_dataset(n=1500, seed=0):
    """Same construction as `test_meta_learners._toy_dataset`: true CATE
    (hours-saved convention) increases with x1, so a positive mean and a
    positive correlation with the true effect are the two properties any
    correctly-signed, reasonably-fit estimator should show."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    X = pd.DataFrame({
        "site": rng.choice(["A", "B", "C"], n),
        "load_class": rng.choice(["light", "heavy"], n),
        "x1": x1, "x2": rng.normal(0, 1, n),
    })
    T = rng.integers(0, 2, n)
    true_cate = 1 + 0.5 * x1
    baseline = 20 + rng.normal(0, 2, n)
    Y = baseline - T * true_cate + rng.normal(0, 1, n)
    return X, T, Y, true_cate


def test_dr_learner_fits_and_predicts_positive_mean_cate():
    X, T, Y, true_cate = _toy_dataset()
    model = DoublyRobustModel(random_state=42).fit(X, T, Y)
    predicted = model.predict_cate(X)

    assert predicted.shape == (len(X),)
    assert predicted.mean() > 0  # "positive = hours saved" convention, true effect always > 0 here


def test_dr_learner_predicted_cate_correlates_with_true_cate():
    X, T, Y, true_cate = _toy_dataset()
    model = DoublyRobustModel(random_state=42).fit(X, T, Y)
    predicted = model.predict_cate(X)

    corr = np.corrcoef(predicted, true_cate)[0, 1]
    assert corr > 0.2


def test_dr_learner_handles_categories_not_seen_at_predict_time():
    """The one-hot encoding is fit on train columns and reindexed at predict
    time -- this must not crash when a predict batch is missing a category
    level (or has one the training data didn't), which a real train/test
    split or a live deployment batch can easily produce."""
    X, T, Y, _ = _toy_dataset(n=800, seed=1)
    model = DoublyRobustModel(random_state=42).fit(X, T, Y)

    X_missing_site_c = X[X["site"] != "C"].reset_index(drop=True)
    predicted = model.predict_cate(X_missing_site_c)
    assert predicted.shape == (len(X_missing_site_c),)
    assert np.isfinite(predicted).all()
