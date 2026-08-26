"""Part B: simulates a site-level, non-randomized rollout of the same
maintenance program modeled in `simulate_rct.py`, once the pilot (Part A) had
already shown it worked. Budget cycles, not randomization, decide which mine
adopts when -- some sites adopt early, some late, some never within the
observation window (never-treated).

The true effect is deliberately **dynamic**: it ramps up over the first few
months after adoption (crews take time to build the new inspection routine
into their workflow) and then plateaus. This dynamic, staggered-timing
combination is exactly the condition under which a naive two-way fixed
effects (TWFE) regression is known to be biased (Goodman-Bacon 2021,
Callaway & Sant'Anna 2021) -- already-treated sites get used as an implicit
"control" for later-adopting sites, and because their own effect is still
growing, that comparison silently nets out part of the true effect. That
bias is the empirical point this dataset exists to demonstrate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_MONTHS = 36
# Adoption month per cohort (1-indexed month of first treatment), plus a
# never-treated cohort. `None` = never adopts within the 36-month window.
COHORTS = {"early": 12, "mid": 20, "late": 28, "never": None}
# 8 sites/cohort (32 sites total): each site-month observation is itself a
# fleet-wide aggregate over many trucks (not one truck), so its month-to-month
# noise is naturally tighter than a single truck's -- reflected below via a
# higher Gamma shape parameter, not just a sample-size choice.
COHORT_SITE_COUNTS = {"early": 8, "mid": 8, "late": 8, "never": 8}

# True dynamic effect by "months since adoption" (event time 0 = adoption
# month). Ramps from -2h to a -10h plateau by month 3 post-adoption.
TRUE_DYNAMIC_EFFECT = {0: -2.0, 1: -5.0, 2: -8.0}
TRUE_PLATEAU_EFFECT = -10.0


def true_effect_at_event_time(event_time: int) -> float:
    if event_time < 0:
        return 0.0
    return TRUE_DYNAMIC_EFFECT.get(event_time, TRUE_PLATEAU_EFFECT)


def simulate_staggered_panel(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    site_rows = []
    site_id = 0
    for cohort, adoption_month in COHORTS.items():
        for _ in range(COHORT_SITE_COUNTS[cohort]):
            site_rows.append({"site_id": f"S{site_id:02d}", "cohort": cohort, "adoption_month": adoption_month})
            site_id += 1
    sites = pd.DataFrame(site_rows)

    # Site fixed effect: some mines just run a structurally hotter/cooler
    # fleet (older equipment, harsher terrain) independent of the program.
    site_base_downtime = rng.normal(45, 8, size=len(sites))
    sites["site_base_downtime"] = np.clip(site_base_downtime, 20, None)

    # Common month fixed effect: a shared seasonal/macro shock (e.g. a rainy
    # season raising downtime system-wide for a few months), identical for
    # every site in a given month -- the "parallel trends" absent treatment.
    months = np.arange(1, N_MONTHS + 1)
    month_effect = 6.0 * np.sin(2 * np.pi * (months - 3) / 12)

    rows = []
    for _, site in sites.iterrows():
        for month in months:
            event_time = None if pd.isna(site["adoption_month"]) else int(month - site["adoption_month"])
            treated_now = event_time is not None and event_time >= 0
            effect = true_effect_at_event_time(event_time) if treated_now else 0.0

            mu = site["site_base_downtime"] + month_effect[month - 1] + effect
            # shape=40 (vs. the individual-truck simulator's shape=2.2): a
            # site-month figure aggregates many trucks, so by the CLT it is
            # naturally far less noisy relative to its mean than one truck's
            # downtime -- this is a realism choice, not a fit-for-clarity one.
            downtime = rng.gamma(shape=40.0, scale=max(mu, 5.0) / 40.0)

            rows.append({
                "site_id": site["site_id"],
                "cohort": site["cohort"],
                "adoption_month": site["adoption_month"],
                "month": int(month),
                "event_time": event_time,
                "treated": int(treated_now),
                "downtime_hours": round(float(downtime), 2),
                "true_effect_hours": round(effect, 3),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parents[2] / "data" / "raw" / "staggered_site_panel.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = simulate_staggered_panel()
    df.to_csv(out, index=False)

    print(f"Simulated {df['site_id'].nunique()} sites x {df['month'].nunique()} months = {len(df):,} site-months.")
    print(f"Cohorts: {df.groupby('cohort')['site_id'].nunique().to_dict()}")
    treated_rows = df[df["treated"] == 1]
    print(f"Treated site-months: {len(treated_rows):,}, mean true effect: {treated_rows['true_effect_hours'].mean():.2f}h")
    print(f"Saved to {out}")
