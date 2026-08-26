"""Part A: simulates a randomized pilot of a proactive maintenance program on a
CAEX haul-truck fleet.

Design: trucks are block-randomized to treatment *within site* (each site gets
its own coin flip probability, close to but not exactly 0.5, to mimic how a
real pilot's local site managers apply the assignment protocol with slightly
different actual compliance) -- this is what makes the covariate-balance check
in the pipeline a genuine check rather than a formality, and why propensity is
estimated from data rather than hardcoded to 0.5 downstream.

The true treatment effect is built to be heterogeneous on purpose: older,
more heavily utilized trucks get a *larger* proportional reduction in downtime
from the same proactive-maintenance action, because more incipient wear means
more failures an inspection can actually catch before they become unplanned
downtime. This is the ground truth every CATE estimator downstream is scored
against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_TRUCKS = 3000
SITES = ["Crucero", "Cardones", "Quillota", "AltoJahuel", "Charrua"]
SITE_TREATMENT_PROB = {"Crucero": 0.52, "Cardones": 0.48, "Quillota": 0.55, "AltoJahuel": 0.45, "Charrua": 0.50}
LOAD_CLASSES = ["light", "medium", "heavy"]


def _simulate_covariates(n: int, rng: np.random.Generator) -> pd.DataFrame:
    site = rng.choice(SITES, size=n)
    load_class = rng.choice(LOAD_CLASSES, size=n, p=[0.25, 0.45, 0.30])
    load_factor = pd.Series(load_class).map({"light": 0.8, "medium": 1.0, "heavy": 1.25}).to_numpy()

    truck_age_years = np.clip(rng.gamma(shape=4.0, scale=2.2, size=n), 0.5, 20)
    utilization_pct = np.clip(rng.normal(65 + 8 * load_factor, 12, size=n), 15, 100)
    cumulative_hours_1000s = np.clip(
        truck_age_years * (utilization_pct / 100) * 2.1 + rng.normal(0, 1.5, size=n), 0.5, None
    )

    covariates = pd.DataFrame({
        "truck_id": [f"T{i:05d}" for i in range(n)],
        "site": site,
        "load_class": load_class,
        "truck_age_years": truck_age_years.round(2),
        "utilization_pct": utilization_pct.round(1),
        "cumulative_hours_1000s": cumulative_hours_1000s.round(2),
    })

    # Prior-period downtime is a noisy proxy for the same latent "wear" state
    # as the covariates above -- drawn from its own Gamma with its own noise,
    # NOT computed by plugging age/hours/utilization into the current-period
    # mean formula again. Feeding an already-derived quantity back into the
    # generator that will also see its own inputs directly would double-count
    # the same wear signal and blow up the current-period mean nonlinearly
    # (caught empirically: an earlier version of this generator produced a
    # ~400h average downtime in a 30-day/720h window, a physically impossible
    # >50% downtime rate).
    prior_log_mu = _baseline_log_mu(covariates) + np.log(3.0)  # 90d approx. = 3x the 30d mean
    prior_90d_downtime_hours = rng.gamma(shape=2.2, scale=np.exp(prior_log_mu) / 2.2)

    covariates["prior_90d_downtime_hours"] = prior_90d_downtime_hours.round(2)
    return covariates


def _assign_treatment(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    site_prob = df["site"].map(SITE_TREATMENT_PROB).to_numpy()
    return (rng.random(len(df)) < site_prob).astype(int)


def _baseline_log_mu(df: pd.DataFrame) -> np.ndarray:
    """log E[downtime_30d | X, control], as a function of the *primitive*
    covariates only (age, cumulative hours, utilization, load class) -- never
    of `prior_90d_downtime_hours`, which is itself derived from this same
    function (see `_simulate_covariates`) and would double-count the signal.
    Calibrated so a truck at the sample-average covariates gets ~30h/30d
    (~4% downtime), and a truck at the high end (age 20y, fully utilized,
    heavy load) gets on the order of 100-150h/30d -- a badly-run but not
    physically impossible unit.
    """
    load_factor = df["load_class"].map({"light": 0.85, "medium": 1.0, "heavy": 1.2}).to_numpy()
    return (
        2.7
        + 0.022 * df["truck_age_years"].to_numpy()
        + 0.018 * df["cumulative_hours_1000s"].to_numpy()
        + 0.006 * df["utilization_pct"].to_numpy()
        + np.log(load_factor)
    )


def _baseline_mean_downtime(df: pd.DataFrame) -> np.ndarray:
    """E[downtime | X, control] -- the counterfactual mean absent treatment."""
    return np.exp(_baseline_log_mu(df))


def true_relative_reduction(df: pd.DataFrame) -> np.ndarray:
    """The ground-truth proportional downtime reduction from treatment, as a
    function of covariates -- exported so the pipeline can score CATE
    estimators against the real answer, which only exists because this is a
    simulation.

    Centered near the sample means (age ~8.8y, utilization ~73%) so the
    "average" truck gets close to an 18% reduction, older/harder-used trucks get more.
    """
    age_term = 0.012 * (df["truck_age_years"].to_numpy() - 8.8)
    utilization_term = 0.003 * (df["utilization_pct"].to_numpy() - 73.0)
    reduction = 0.18 + age_term + utilization_term
    return np.clip(reduction, 0.03, 0.55)


def true_cate_hours(df: pd.DataFrame) -> np.ndarray:
    """The ground-truth CATE in absolute hours: E[Y|X,control] - E[Y|X,treated]."""
    return _baseline_mean_downtime(df) * true_relative_reduction(df)


def simulate_rct(n: int = N_TRUCKS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _simulate_covariates(n, rng)
    df["treated"] = _assign_treatment(df, rng)

    baseline_mu = _baseline_mean_downtime(df)
    relative_reduction = true_relative_reduction(df)
    mu = np.where(df["treated"] == 1, baseline_mu * (1 - relative_reduction), baseline_mu)

    # Gamma-distributed outcome: strictly positive, right-skewed, a realistic
    # shape for downtime hours (most trucks have modest downtime, a long tail
    # of bad-luck failures) -- shape fixed, scale = mu / shape to hit the target mean.
    shape = 2.2
    downtime_next_30d_hours = rng.gamma(shape=shape, scale=mu / shape)

    df["downtime_next_30d_hours"] = downtime_next_30d_hours.round(2)
    df["true_cate_hours"] = true_cate_hours(df).round(3)
    return df


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parents[2] / "data" / "raw" / "rct_fleet_pilot.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = simulate_rct()
    df.to_csv(out, index=False)

    print(f"Simulated {len(df):,} trucks across {df['site'].nunique()} sites.")
    print(f"Treated: {df['treated'].sum():,} ({df['treated'].mean():.1%})")
    print(f"Mean downtime -- control: {df.loc[df.treated == 0, 'downtime_next_30d_hours'].mean():.2f}h, "
          f"treated: {df.loc[df.treated == 1, 'downtime_next_30d_hours'].mean():.2f}h")
    print(f"Mean true CATE (hours saved): {df['true_cate_hours'].mean():.2f}")
    print(f"Saved to {out}")
