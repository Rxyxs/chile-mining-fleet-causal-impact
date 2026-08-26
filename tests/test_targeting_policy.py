import numpy as np
import pandas as pd
import pytest

from src.evaluation.targeting_policy import evaluate_targeting_policies


@pytest.fixture
def toy_df():
    return pd.DataFrame({"true_cate_hours": [2, 1, 1, 2, 1, 9]})


def test_targeting_policy_selects_correct_units_and_sums(toy_df):
    """6 units, budget_fraction=0.5 -> top 3.
    risk_score=[1,5,3,2,4,0]   -> top3 idx [1,4,2] -> true_cate sum = 1+1+1 = 3
    uplift_score=[10,9,1,2,3,0] -> top3 idx [0,1,4] -> true_cate sum = 2+1+1 = 4
    oracle (true_cate itself)   -> top3 idx [5,0,3] -> true_cate sum = 9+2+2 = 13
    """
    risk_score = np.array([1, 5, 3, 2, 4, 0])
    uplift_score = np.array([10, 9, 1, 2, 3, 0])

    result = evaluate_targeting_policies(toy_df, budget_fraction=0.5, risk_score=risk_score, uplift_score=uplift_score, seed=42)
    by_policy = result.set_index("policy")

    assert by_policy.loc["highest_baseline_risk", "total_hours_saved"] == pytest.approx(3.0)
    assert by_policy.loc["highest_predicted_uplift", "total_hours_saved"] == pytest.approx(4.0)
    assert by_policy.loc["oracle_true_uplift", "total_hours_saved"] == pytest.approx(13.0)
    assert by_policy.loc["oracle_true_uplift", "pct_of_oracle_achieved"] == pytest.approx(100.0)
    # The implementation rounds this column to 1 decimal place -- match that
    # rounding here rather than comparing to the unrounded fraction.
    assert by_policy.loc["highest_baseline_risk", "pct_of_oracle_achieved"] == pytest.approx(round(3 / 13 * 100, 1))
    assert by_policy.loc["highest_predicted_uplift", "pct_of_oracle_achieved"] == pytest.approx(round(4 / 13 * 100, 1))


def test_targeting_policy_random_matches_seeded_choice(toy_df):
    risk_score = np.array([1, 5, 3, 2, 4, 0])
    uplift_score = np.array([10, 9, 1, 2, 3, 0])

    result = evaluate_targeting_policies(toy_df, budget_fraction=0.5, risk_score=risk_score, uplift_score=uplift_score, seed=42)

    expected_idx = np.random.default_rng(42).choice(6, size=3, replace=False)
    expected_sum = toy_df["true_cate_hours"].to_numpy()[expected_idx].sum()

    random_row = result.set_index("policy").loc["random"]
    assert random_row["total_hours_saved"] == pytest.approx(float(expected_sum))


def test_targeting_policy_budget_size_rounds_correctly(toy_df):
    risk_score = np.zeros(6)
    uplift_score = np.zeros(6)
    result = evaluate_targeting_policies(toy_df, budget_fraction=0.34, risk_score=risk_score, uplift_score=uplift_score, seed=1)
    assert (result["n_trucks_targeted"] == round(6 * 0.34)).all()


def test_oracle_always_achieves_100_percent(toy_df):
    risk_score = np.random.default_rng(3).normal(size=6)
    uplift_score = np.random.default_rng(4).normal(size=6)
    result = evaluate_targeting_policies(toy_df, budget_fraction=0.5, risk_score=risk_score, uplift_score=uplift_score, seed=2)
    oracle_row = result.set_index("policy").loc["oracle_true_uplift"]
    assert oracle_row["pct_of_oracle_achieved"] == pytest.approx(100.0)
