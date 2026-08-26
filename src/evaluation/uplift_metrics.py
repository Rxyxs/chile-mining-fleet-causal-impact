"""Uplift/Qini curve evaluation for CATE estimators, and a direct
predicted-vs-true CATE recovery check.

The standard Qini/uplift curve formulation is built for binary conversion
outcomes; this project's outcome is continuous (downtime hours, where LOWER
is better), so the curve below is the continuous-outcome generalization
(Radcliffe, 2007-style): sort units by predicted CATE descending, and at
each cutoff k compute the *hours saved* if the top-k population had been
assigned by that ranking, using each arm's realized within-cutoff mean as
the counterfactual estimate for that arm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def uplift_curve(cate_pred: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> pd.DataFrame:
    """Returns a DataFrame with columns `k` (units targeted), `gain` (the
    model's cumulative estimated hours saved among the top-k), and
    `random_gain` (the same quantity under random targeting) -- lower
    `outcome` is better, so `gain` is (mean_control - mean_treated) * k
    within the top-k, i.e. positive gain means the ranking is finding units
    where treatment helps.
    """
    order = np.argsort(-cate_pred)
    t_sorted = treatment[order].astype(float)
    y_sorted = outcome[order].astype(float)

    cum_n_t = np.cumsum(t_sorted)
    cum_n_c = np.cumsum(1 - t_sorted)
    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - t_sorted))

    mean_t = np.divide(cum_y_t, cum_n_t, out=np.zeros_like(cum_y_t), where=cum_n_t > 0)
    mean_c = np.divide(cum_y_c, cum_n_c, out=np.zeros_like(cum_y_c), where=cum_n_c > 0)

    k = np.arange(1, len(t_sorted) + 1)
    gain = (mean_c - mean_t) * k
    random_gain = gain[-1] * (k / k[-1])

    return pd.DataFrame({"k": k, "gain": gain, "random_gain": random_gain})


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the model's gain curve and the random-targeting line,
    normalized by population size -- positive means the ranking beats random
    targeting on average; the higher, the better the CATE ranking."""
    diff = (curve["gain"] - curve["random_gain"]).to_numpy()
    area = np.trapezoid(diff, curve["k"].to_numpy())
    return float(area / curve["k"].iloc[-1])


def cate_recovery_correlation(predicted_cate: np.ndarray, true_cate: np.ndarray) -> float:
    """Pearson correlation between a model's predicted CATE and the DGP's
    known true CATE -- only computable because this is a simulation with a
    known ground truth, used purely to validate the estimators, never as a
    feature."""
    return float(np.corrcoef(predicted_cate, true_cate)[0, 1])


def cate_calibration_bins(predicted_cate: np.ndarray, true_cate: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bins units by predicted-CATE decile and compares the average predicted
    vs. average true CATE within each bin -- a calibration check: a
    well-calibrated model's points should fall near the y=x line."""
    df = pd.DataFrame({"predicted_cate": predicted_cate, "true_cate": true_cate})
    df["bin"] = pd.qcut(df["predicted_cate"], q=n_bins, labels=False, duplicates="drop")
    return (
        df.groupby("bin")
        .agg(mean_predicted=("predicted_cate", "mean"), mean_true=("true_cate", "mean"), n=("predicted_cate", "size"))
        .reset_index()
    )
