import numpy as np
import pytest

from src.evaluation.uplift_metrics import cate_recovery_correlation, qini_coefficient, uplift_curve


def test_uplift_curve_matches_hand_computed_example():
    """4 units, already sorted by predicted CATE descending (3,2,1,0).
    treatment=[1,0,1,0], outcome=[5,10,6,10] (lower is better).
    Worked by hand in the PR/commit description; see comments below."""
    predicted_cate = np.array([3, 2, 1, 0])
    treatment = np.array([1, 0, 1, 0])
    outcome = np.array([5, 10, 6, 10])

    curve = uplift_curve(predicted_cate, treatment, outcome)

    assert curve["k"].tolist() == [1, 2, 3, 4]
    assert curve["gain"].to_numpy() == pytest.approx([-5.0, 10.0, 13.5, 18.0])
    assert curve["random_gain"].to_numpy() == pytest.approx([4.5, 9.0, 13.5, 18.0])


def test_qini_coefficient_matches_hand_computed_example():
    predicted_cate = np.array([3, 2, 1, 0])
    treatment = np.array([1, 0, 1, 0])
    outcome = np.array([5, 10, 6, 10])

    curve = uplift_curve(predicted_cate, treatment, outcome)
    assert qini_coefficient(curve) == pytest.approx(-0.9375)


def test_uplift_curve_gain_at_full_population_equals_total_ate():
    """At k = n, `gain` must equal (control mean - treated mean) * n exactly,
    regardless of the ranking used to sort -- a basic internal-consistency
    check independent of the hand-computed example above."""
    rng = np.random.default_rng(0)
    n = 200
    treatment = rng.integers(0, 2, n)
    outcome = rng.normal(10, 2, n) - treatment * 3
    predicted_cate = rng.normal(0, 1, n)  # arbitrary ranking

    curve = uplift_curve(predicted_cate, treatment, outcome)
    expected_total_gain = (outcome[treatment == 0].mean() - outcome[treatment == 1].mean()) * n
    assert curve["gain"].iloc[-1] == pytest.approx(expected_total_gain)


def test_perfect_ranking_beats_random_by_a_positive_qini():
    """Ranking by the *true* CATE should always score a positive Qini
    coefficient when the treatment effect is genuinely heterogeneous."""
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.uniform(0, 10, n)
    true_cate = x  # monotonic in x, genuinely heterogeneous
    treatment = rng.integers(0, 2, n)
    outcome = 20 - treatment * true_cate + rng.normal(0, 1, n)

    curve = uplift_curve(true_cate, treatment, outcome)
    assert qini_coefficient(curve) > 0


def test_cate_recovery_correlation_perfect_and_inverted():
    true_cate = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cate_recovery_correlation(true_cate, true_cate) == pytest.approx(1.0)
    assert cate_recovery_correlation(-true_cate, true_cate) == pytest.approx(-1.0)
