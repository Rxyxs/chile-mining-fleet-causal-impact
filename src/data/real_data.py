"""Two real public datasets used to check the estimators outside a simulator.

* ``mpdta`` -- county-level panel (500 US counties, 2003-2007) with a staggered rollout of
  minimum-wage increases; the standard worked example of Callaway & Sant'Anna (2021).
  Used to check the staggered-adoption DiD estimators of Part B against an independent
  reference implementation.
* Hillstrom e-mail experiment -- 64,000 customers randomly assigned to a men's e-mail
  campaign, a women's e-mail campaign or no e-mail. A real randomized trial, used to
  check the CATE rankings of Part A with a Qini curve on held-out randomized data.

Neither dataset is about mining. They are here because a causal estimator cannot be scored
against the true effect on real data: what can be checked is agreement with an independent
implementation (DiD) and the ability to rank units in a genuine randomized experiment (CATE).
"""
from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MPDTA_URL = "https://raw.githubusercontent.com/bcallaway11/did/master/data/mpdta.rda"
HILLSTROM_URL = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"

FIRST_YEAR = 2003  # month = year - 2002, so the panel starts at month 1


def _fetch(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size == 0:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        target.write_bytes(resp.content)
    return target


def download_mpdta(raw_dir: Path = RAW_DIR) -> Path:
    return _fetch(MPDTA_URL, raw_dir / "mpdta.rda")


def download_hillstrom(raw_dir: Path = RAW_DIR) -> Path:
    gz = _fetch(HILLSTROM_URL, raw_dir / "hillstrom.csv.gz")
    csv = raw_dir / "hillstrom.csv"
    if not csv.exists():
        with gzip.open(gz, "rb") as src, csv.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return csv


def mpdta_to_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename ``mpdta`` columns to the schema the DiD estimators expect.

    ``treated`` is the *post-adoption* indicator (1 once the county's rollout has started),
    which is not the same as the dataset's own ``treat`` column (ever-treated group).
    Never-treated counties have ``adoption_month`` missing.
    """
    first = raw["first.treat"].where(raw["first.treat"] > 0)
    return pd.DataFrame(
        {
            "site_id": raw["countyreal"].astype(int),
            "month": (raw["year"] - (FIRST_YEAR - 1)).astype(int),
            "adoption_month": first - (FIRST_YEAR - 1),
            "downtime_hours": raw["lemp"].astype(float),  # outcome column name fixed by the estimators
            "treated": ((raw["first.treat"] > 0) & (raw["year"] >= raw["first.treat"])).astype(int),
        }
    )


def load_mpdta_panel(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (panel in estimator schema, raw mpdta for the reference implementation)."""
    import pyreadr

    raw = pyreadr.read_r(str(download_mpdta(raw_dir)))["mpdta"]
    return mpdta_to_panel(raw), raw


HILLSTROM_FEATURES = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
HILLSTROM_ARMS = {"Mens E-Mail": "mens_email", "Womens E-Mail": "womens_email"}


def load_hillstrom(arm: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One treated arm against the no-e-mail control, in the estimators' convention.

    The estimators treat LOWER outcomes as better, so the outcome column is ``-visit``:
    a positive CATE then means the e-mail raises the probability of a site visit.
    """
    if arm not in HILLSTROM_ARMS:
        raise ValueError(f"arm must be one of {list(HILLSTROM_ARMS)}")
    df = pd.read_csv(download_hillstrom(raw_dir))
    df = df[df["segment"].isin([arm, "No E-Mail"])].reset_index(drop=True)
    out = df[HILLSTROM_FEATURES].copy()
    for col in ("zip_code", "channel"):
        out[col] = out[col].astype("category")
    out["treatment"] = (df["segment"] == arm).astype(int)
    out["outcome"] = -df["visit"].astype(float)
    out["visit"] = df["visit"].astype(int)
    return out
