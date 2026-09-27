"""Semi-synthetic version of the Part A pilot: REAL covariates from a public
predictive-maintenance dataset, SIMULATED treatment assignment and treatment
effect, so the CATE estimators face real sensor distributions (heavy tails,
missingness, collinear histogram bins, real failure structure) while the true
effect every estimate is scored against stays known by construction.

What comes from the real data, and what does not:

- X (every model feature) is the real record, untouched -- including its NaNs.
- The real failure label enters the untreated outcome Y0 as a large repair
  term (`repair_hours`), so part of the baseline is the dataset's own failure
  mechanism rather than a smooth function written here. Setting
  `repair_hours=0` removes it, which is the ablation that separates "real
  covariate distribution" from "real failure structure" in the benchmark.
- Treatment T is never read from the data (neither dataset has one): it is a
  simulated pilot, completely randomized within blocks, like Part A's sites.
- The effect is simulated: a proportional downtime reduction r(x) that grows
  with two real columns (a wear/usage driver and a stress driver), applied to
  the unit's baseline mean. The true CATE is `r(x) * mu0` in hours saved.

The outcome is a Gamma draw around the arm's mean, as in `simulate_rct.py`.
It is never clipped: a `max(0, Y0 - tau)` floor would silently change each
unit's effect and the recorded `true_cate_hours` would stop being true. The
multiplicative form keeps Y strictly positive with no clipping.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

N_UNITS = 3000  # same pilot size as `simulate_rct.N_TRUCKS`, so the comparison changes the data, not n
BLOCK_TREATMENT_PROB = (0.45, 0.50, 0.55)  # per block, in sorted block-label order
BASE_ROUTINE_HOURS = 20.0  # routine downtime per 30d for a unit at the median of every driver
REPAIR_HOURS = 60.0  # extra expected downtime per 30d for a unit carrying a real failure
GAMMA_SHAPE = 2.2  # same outcome shape as `simulate_rct.py`


@dataclass(frozen=True)
class RealCovariates:
    """A real dataset reduced to what the DGP needs. `X` is what the models
    see; the drivers are real columns of X the simulated effect depends on;
    `real_failure` is the dataset's own failure label, used only inside Y0."""

    name: str
    X: pd.DataFrame
    block: pd.Series
    wear: pd.Series
    stress: pd.Series
    load: pd.Series
    real_failure: pd.Series
    heavy_tailed: bool
    driver_names: dict


def load_ai4i(raw_dir: Path = RAW_DIR) -> RealCovariates:
    df = pd.read_csv(raw_dir / "ai4i2020.csv")
    X = pd.DataFrame({
        "machine_type": df["Type"].astype("category"),
        "air_temp_k": df["Air temperature [K]"],
        "process_temp_k": df["Process temperature [K]"],
        "rotational_speed_rpm": df["Rotational speed [rpm]"],
        "torque_nm": df["Torque [Nm]"],
        "tool_wear_min": df["Tool wear [min]"],
    })
    return RealCovariates(
        name="ai4i_2020",
        X=X,
        block=df["Type"].astype(str),
        wear=X["tool_wear_min"],
        stress=X["process_temp_k"],
        load=X["torque_nm"],
        real_failure=df["Machine failure"].astype(int),
        heavy_tailed=False,
        driver_names={"wear": "tool_wear_min", "stress": "process_temp_k", "load": "torque_nm"},
    )


def load_scania(raw_dir: Path = RAW_DIR) -> RealCovariates:
    """The 60,000-truck training file; the separate test file is not needed
    since every split here is drawn from this one population.

    Drivers are three low-missingness single-value counters (not histogram
    bins). Scania anonymized every column name, so they are treated as usage
    (aa_000, strongly tied to the total of the ag_* histogram), a second
    operational load counter (ci_000), and a baseline-only counter (bj_000):
    the effect's *shape* is simulated, but the values it is computed from are
    real. Blocks are usage tertiles of aa_000, i.e. the pilot is stratified by
    how hard the truck has been used, as a fleet manager would do.
    """
    df = pd.read_csv(raw_dir / "aps_failure_training_set.csv", skiprows=20, na_values="na")
    X = df.drop(columns="class").astype(float)
    block = pd.qcut(X["aa_000"].rank(method="first"), q=3, labels=["usage_low", "usage_mid", "usage_high"]).astype(str)
    return RealCovariates(
        name="scania_aps",
        X=X,
        block=block,
        wear=X["aa_000"],
        stress=X["ci_000"],
        load=X["bj_000"],
        real_failure=(df["class"] == "pos").astype(int),
        heavy_tailed=True,
        driver_names={"wear": "aa_000", "stress": "ci_000", "load": "bj_000"},
    )


def robust_z(x: pd.Series, log_scale: bool) -> np.ndarray:
    """Median/IQR standardization, clipped to +-3, computed on the FULL real
    population so r(x) is one fixed function across every resampled pilot.
    Heavy-tailed counters go through log1p first; without it, a handful of
    extreme trucks would own the whole effect. Missing driver values sit at
    the median (z = 0) in the DGP only -- the models still see the NaN.
    """
    v = np.log1p(x.clip(lower=0)) if log_scale else x.astype(float)
    median = v.median()
    scale = (v.quantile(0.75) - v.quantile(0.25)) / 1.349
    z = ((v - median) / scale).fillna(0.0)
    return np.clip(z.to_numpy(), -3.0, 3.0)


def assign_block_randomized_treatment(
    block: pd.Series, rng: np.random.Generator, probs: tuple[float, ...] = BLOCK_TREATMENT_PROB
) -> np.ndarray:
    """Complete randomization within each block: exactly round(p_b * n_b)
    units treated per block, chosen uniformly without replacement. Blocks get
    slightly different probabilities (as Part A's sites do), so the true
    propensity varies with X and has to be estimated, not assumed to be 0.5.
    """
    treated = np.zeros(len(block), dtype=int)
    block_values = block.to_numpy()
    for i, label in enumerate(sorted(pd.unique(block_values))):
        idx = np.flatnonzero(block_values == label)
        n_treated = int(round(probs[i % len(probs)] * len(idx)))
        treated[rng.choice(idx, size=n_treated, replace=False)] = 1
    return treated


def true_relative_reduction(z_wear: np.ndarray, z_stress: np.ndarray) -> np.ndarray:
    """Proportional downtime reduction: 18% at the median unit, more for worn
    and stressed units -- the same story as `simulate_rct.true_relative_reduction`,
    now driven by real columns."""
    return np.clip(0.18 + 0.06 * z_wear + 0.05 * z_stress, 0.03, 0.55)


def create_semi_synthetic_rct(
    real: RealCovariates,
    n: int = N_UNITS,
    seed: int = 42,
    repair_hours: float = REPAIR_HOURS,
) -> pd.DataFrame:
    """One simulated pilot on `n` real units sampled without replacement.
    Each seed redraws the sample, the assignment, and the outcome noise.

    Returns the real features plus `block`, `treated`,
    `downtime_next_30d_hours` (observed Y), `true_cate_hours`,
    `true_relative_reduction`, `baseline_mean_hours` and `real_failure`.
    Only the real feature columns are model inputs; the rest is design or
    ground truth for evaluation.
    """
    rng = np.random.default_rng(seed)

    z_wear = robust_z(real.wear, real.heavy_tailed)
    z_stress = robust_z(real.stress, real.heavy_tailed)
    z_load = robust_z(real.load, real.heavy_tailed)

    idx = rng.choice(len(real.X), size=n, replace=False)
    df = real.X.iloc[idx].reset_index(drop=True)
    block = real.block.iloc[idx].reset_index(drop=True)
    failure = real.real_failure.iloc[idx].to_numpy()
    zw, zs, zl = z_wear[idx], z_stress[idx], z_load[idx]

    routine_mean = BASE_ROUTINE_HOURS * np.exp(0.25 * zw + 0.20 * zs + 0.15 * zl)
    baseline_mean = routine_mean + repair_hours * failure
    reduction = true_relative_reduction(zw, zs)

    treated = assign_block_randomized_treatment(block, rng)
    arm_mean = np.where(treated == 1, baseline_mean * (1 - reduction), baseline_mean)
    outcome = rng.gamma(shape=GAMMA_SHAPE, scale=arm_mean / GAMMA_SHAPE)

    df["block"] = block.to_numpy()
    df["treated"] = treated
    df["downtime_next_30d_hours"] = outcome
    df["true_cate_hours"] = baseline_mean * reduction
    df["true_relative_reduction"] = reduction
    df["baseline_mean_hours"] = baseline_mean
    df["real_failure"] = failure
    return df


LOADERS = {"ai4i_2020": load_ai4i, "scania_aps": load_scania}


if __name__ == "__main__":
    for loader in LOADERS.values():
        real = loader()
        sim = create_semi_synthetic_rct(real)
        print(f"== {real.name}: {len(real.X):,} real units, {real.X.shape[1]} features, "
              f"{real.real_failure.mean():.2%} real failure rate")
        print(f"   pilot n={len(sim):,}, treated={sim['treated'].mean():.1%}, "
              f"mean Y control={sim.loc[sim.treated == 0, 'downtime_next_30d_hours'].mean():.1f}h, "
              f"treated={sim.loc[sim.treated == 1, 'downtime_next_30d_hours'].mean():.1f}h")
        print(f"   true CATE: mean={sim['true_cate_hours'].mean():.2f}h, "
              f"p5={sim['true_cate_hours'].quantile(0.05):.2f}h, p95={sim['true_cate_hours'].quantile(0.95):.2f}h, "
              f"max={sim['true_cate_hours'].max():.2f}h")
        print("   treated share by block:", sim.groupby("block")["treated"].mean().round(3).to_dict())
