import numpy as np
import pandas as pd

from src.data.semi_synthetic_dgp import (
    BLOCK_TREATMENT_PROB,
    RealCovariates,
    assign_block_randomized_treatment,
    create_semi_synthetic_rct,
)


def _fake_real(n=4000, seed=0, heavy_tailed=False):
    """A stand-in for a loaded real dataset, so these tests never need the
    UCI downloads: two numeric drivers, one baseline-only column, a block
    label, and a rare failure flag tied to the wear driver."""
    rng = np.random.default_rng(seed)
    wear = pd.Series(rng.lognormal(3, 1.5, n) if heavy_tailed else rng.uniform(0, 250, n), name="wear")
    stress = pd.Series(rng.normal(310, 1.5, n), name="stress")
    load = pd.Series(rng.normal(40, 10, n), name="load")
    X = pd.DataFrame({"wear": wear, "stress": stress, "load": load})
    X.loc[rng.random(n) < 0.05, "stress"] = np.nan  # real data has holes; the DGP must cope
    failure = pd.Series((rng.random(n) < 0.02 + 0.08 * (wear.rank(pct=True) > 0.9)).astype(int))
    return RealCovariates(
        name="fake", X=X, block=pd.Series(rng.choice(["a", "b", "c"], n)),
        wear=X["wear"], stress=X["stress"], load=X["load"], real_failure=failure,
        heavy_tailed=heavy_tailed, driver_names={"wear": "wear", "stress": "stress", "load": "load"},
    )


def test_block_randomization_treats_the_exact_share_within_each_block():
    rng = np.random.default_rng(1)
    block = pd.Series(rng.choice(["x", "y", "z"], 3000))
    treated = assign_block_randomized_treatment(block, rng)
    for i, label in enumerate(sorted(block.unique())):
        in_block = (block == label).to_numpy()
        assert treated[in_block].sum() == round(BLOCK_TREATMENT_PROB[i] * in_block.sum())


def test_true_cate_is_the_gap_between_arm_means_and_is_positive():
    df = create_semi_synthetic_rct(_fake_real(), n=2000, seed=3)
    assert (df["true_cate_hours"] > 0).all()
    np.testing.assert_allclose(df["true_cate_hours"], df["baseline_mean_hours"] * df["true_relative_reduction"])
    assert df["true_cate_hours"].std() > 0


def test_outcome_is_strictly_positive_without_clipping():
    df = create_semi_synthetic_rct(_fake_real(heavy_tailed=True), n=2000, seed=4)
    assert (df["downtime_next_30d_hours"] > 0).all()


def test_treatment_reduces_mean_outcome_by_roughly_the_true_ate():
    df = create_semi_synthetic_rct(_fake_real(n=20000), n=20000, seed=5)
    naive = df.loc[df.treated == 0, "downtime_next_30d_hours"].mean() - df.loc[df.treated == 1, "downtime_next_30d_hours"].mean()
    assert abs(naive - df["true_cate_hours"].mean()) < 0.15 * df["true_cate_hours"].mean()


def test_repair_hours_zero_removes_the_real_failure_term_from_the_truth():
    real = _fake_real()
    with_failures = create_semi_synthetic_rct(real, n=2000, seed=6)
    without = create_semi_synthetic_rct(real, n=2000, seed=6, repair_hours=0.0)
    failed = without["real_failure"] == 1
    assert failed.any()
    assert (with_failures.loc[failed, "true_cate_hours"] > without.loc[failed, "true_cate_hours"]).all()
    np.testing.assert_allclose(with_failures.loc[~failed, "true_cate_hours"], without.loc[~failed, "true_cate_hours"])


def test_real_features_reach_the_output_untouched_including_missing_values():
    real = _fake_real()
    df = create_semi_synthetic_rct(real, n=1000, seed=7)
    assert df["stress"].isna().any()
    assert set(real.X.columns) <= set(df.columns)


def test_same_seed_reproduces_the_same_pilot():
    real = _fake_real()
    a = create_semi_synthetic_rct(real, n=1000, seed=8)
    b = create_semi_synthetic_rct(real, n=1000, seed=8)
    pd.testing.assert_frame_equal(a, b)
