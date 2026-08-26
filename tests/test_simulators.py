from src.data.simulate_rct import simulate_rct
from src.data.simulate_staggered_did import simulate_staggered_panel


def test_rct_downtime_is_physically_plausible():
    """Downtime over a 30-day (720h) window should be a small fraction of the
    window, not a majority of it -- catches the kind of DGP double-counting
    bug that once made this generator average ~400h/30d (an impossible >50%
    downtime rate) before the fix in `simulate_rct._baseline_log_mu`."""
    df = simulate_rct(n=500, seed=1)
    assert df["downtime_next_30d_hours"].mean() < 100
    assert (df["downtime_next_30d_hours"] >= 0).all()


def test_rct_treatment_reduces_mean_downtime():
    df = simulate_rct(n=3000, seed=42)
    treated_mean = df.loc[df.treated == 1, "downtime_next_30d_hours"].mean()
    control_mean = df.loc[df.treated == 0, "downtime_next_30d_hours"].mean()
    assert treated_mean < control_mean


def test_rct_true_cate_is_positive_and_heterogeneous():
    df = simulate_rct(n=3000, seed=42)
    assert (df["true_cate_hours"] > 0).all()
    assert df["true_cate_hours"].std() > 0  # genuinely heterogeneous, not a constant


def test_rct_treatment_probability_is_roughly_balanced_per_site():
    df = simulate_rct(n=5000, seed=42)
    site_rates = df.groupby("site")["treated"].mean()
    assert site_rates.between(0.35, 0.65).all()


def test_staggered_panel_shape():
    df = simulate_staggered_panel(seed=1)
    assert df["month"].max() == 36
    assert set(df["cohort"].unique()) == {"early", "mid", "late", "never"}


def test_staggered_panel_never_treated_cohort_is_never_treated():
    df = simulate_staggered_panel(seed=1)
    never = df[df["cohort"] == "never"]
    assert (never["treated"] == 0).all()
    assert never["adoption_month"].isna().all()


def test_staggered_panel_pre_treatment_effect_is_zero():
    df = simulate_staggered_panel(seed=1)
    pre_period = df[df["treated"] == 0]
    assert (pre_period["true_effect_hours"] == 0).all()
