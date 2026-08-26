[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# 1. Project Title

## Causal Impact of a Fleet Maintenance Program: RCT Uplift Modeling and Staggered-Adoption DiD

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2.x-0193B0?style=flat)
![EconML](https://img.shields.io/badge/EconML-CausalForestDML-6A5ACD?style=flat)
![linearmodels](https://img.shields.io/badge/linearmodels-PanelOLS-337AB7?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Pytest](https://img.shields.io/badge/tests-26%20passing-brightgreen?style=flat&logo=pytest&logoColor=white)
![Status](https://img.shields.io/badge/status-real%20pipeline%20run-lightgrey?style=flat)

This project answers two different causal questions about the same intervention — a proactive maintenance program for a CAEX haul-truck fleet — depending on how it was rolled out:

1. **When the intervention was randomized** (a pilot, Part A): which trucks benefit most, so a maintenance budget can be targeted at the highest-value units? Answered with 4 CATE (conditional average treatment effect) estimators — S-learner, T-learner, X-learner, and EconML's `CausalForestDML` — evaluated with uplift (Qini) curves and, because this is a simulation with a known ground truth, checked directly against the real individual-level effect.
2. **When the intervention was rolled out to whole sites on a staggered, non-random schedule** (Part B): what is the aggregate causal effect, when a naive before/after comparison risks confusing the treatment effect with time trends, or — as the modern difference-in-differences literature shows — with the bias a constant-effect regression introduces when adoption timing varies and the true effect is dynamic? Answered by contrasting a naive two-way fixed-effects (TWFE) regression against a group-time ATT estimator, against the known true effect.

Every number in §7 comes from an actual run of `python -m src.pipeline` (seed 42) on synthetic data built with a known, deliberately heterogeneous (Part A) and dynamic (Part B) true effect — the only reason any of these estimators can be validated against a real answer at all.

---

# 2. Motivation

A mining operation weighing a proactive-maintenance program for its haul-truck fleet cannot answer "does it work, and for whom?" from a raw before/after comparison, for the same reason no observational comparison of treated vs. untreated units ever can: whatever confounds the assignment (older trucks might get flagged for maintenance *because* they're already failing more; sites might adopt the program precisely when demand is highest) also confounds the outcome, and the truck- or site-level counterfactual — what would have happened without treatment — is never observed. This is the fundamental problem causal inference exists to address, and different data-collection designs call for genuinely different tools:

- **A randomized pilot** removes the confounding-by-design problem — treatment assignment no longer depends on the outcome. What it does *not* automatically give you is *which* units benefit most; a program with a real average benefit can still be worth withholding from units where it does nothing, if there's a fleet-wide budget constraint. That is a heterogeneous-treatment-effect (CATE) question, not an average-treatment-effect one.
- **A staggered, budget-driven rollout across sites** is not randomized — some sites adopt sooner because of when their budget cycle allows it, not because of anything about the outcome, which still supports a difference-in-differences design, but a constant-effect two-way fixed-effects regression (the default a team reaches for) is only valid under assumptions that stop holding once treatment effects are *dynamic* and adoption is *staggered* — a well-documented problem in the recent econometrics literature (Goodman-Bacon, 2021; Callaway & Sant'Anna, 2021) that this project reproduces and corrects for directly, rather than citing abstractly.

Both datasets here are synthetic — no free, public dataset exists that pairs individual-level randomized maintenance assignment with a staggered site-level rollout of the same program — but each is built with a **known, deliberately non-trivial true effect** (heterogeneous by truck age/utilization in Part A; dynamic, ramping up over the months after adoption in Part B) specifically so this project's estimators can be checked against the real answer, not just against each other. That check is only possible in a simulation; it is the entire point of building one.

---

# 3. Theoretical Framework

## 3.1 CATE estimation from a randomized pilot

- **S-learner**: a single model `f(X, T) -> Y`; `CATE(x) = f(x, control) - f(x, treated)`. Simplest, but a strong learner can regularize away a weak, single binary feature (the treatment indicator) in favor of the many higher-signal covariates — this project's results (§7.1) show this failure mode concretely, not just in theory.
- **T-learner**: two separate models, one per arm. Avoids the S-learner's regularization risk, at the cost of each model only seeing half the data.
- **X-learner** (Kunzel et al., 2019): imputes an individual treatment effect per unit using the *other* arm's model as a counterfactual, fits a second pair of models on those imputed effects, then combines them weighted by the propensity score — designed to do better than the T-learner specifically when the two arms are imbalanced in size or in covariate distribution.
- **Causal Forest DML** (Athey, Tibshirani & Wager, 2019; via EconML's `CausalForestDML`): explicitly residualizes out `E[Y|X]` and `E[T|X]` before estimating the treatment-effect function (double machine learning), which in principle makes it more robust than the meta-learners when the propensity score genuinely varies with covariates, as it does here (site-level block randomization at slightly different probabilities per site, see §5).

## 3.2 Evaluating CATE estimators without knowing the truth: Qini curves

The real-world way to evaluate a CATE ranking (when the true individual effect is unobservable, as it always is outside a simulation) is a **Qini/uplift curve**: sort units by predicted CATE, and at each cutoff compute the cumulative benefit that ranking would have delivered, compared against random targeting. The area between the model's curve and the random-targeting line (normalized by population size) is the **Qini coefficient**. This project computes the continuous-outcome generalization of this curve (most literature examples are binary-conversion outcomes) directly from realized downtime hours.

## 3.3 Targeting under a budget constraint: risk is not uplift

A team without a CATE model will typically target an intervention at the **highest-risk** units (highest predicted downtime absent treatment) — a reasonable-sounding heuristic that is not the same question as **highest-uplift** (who benefits most *from the intervention*, which is not necessarily the same as who is worst off to begin with). §7.1 quantifies the real gap between these two targeting policies on this project's own data, using the known true effect to score each policy fairly.

## 3.4 Staggered-adoption DiD and the two-way fixed-effects bias

A regression of the outcome on unit fixed effects, time fixed effects, and a single treatment dummy (naive TWFE) estimates a treatment effect as a weighted average of *all* possible 2x2 (treated-vs-control, before-vs-after) comparisons the data supports. When adoption is staggered, some of those comparisons implicitly use **already-treated units as the control group** for later-adopting units. If the true effect is constant over time, this is harmless. If it is **dynamic** — as it realistically is here, ramping up over the months following adoption — those comparisons subtract out part of an effect that hadn't stopped growing yet, biasing the single TWFE coefficient (Goodman-Bacon, 2021). This project's `group_time_att` (a simplified Callaway & Sant'Anna, 2021-style estimator) avoids this by comparing each adoption cohort only against the **never-treated** group, never against another treated cohort, and reports the effect broken out by event time (months since adoption) rather than forcing it to a single number.

---

# 4. Explanation

## Pipeline architecture

```mermaid
flowchart TB
    subgraph A["Part A: RCT / individual-level CATE"]
        A1["simulate_rct.py<br/>3,000 trucks, block-randomized by site<br/>known heterogeneous true CATE"] --> A2["train/test split (60/40)"]
        A2 --> A3["meta_learners.py<br/>S-learner / T-learner / X-learner"]
        A2 --> A4["causal_forest.py<br/>EconML CausalForestDML"]
        A3 --> A5["uplift_metrics.py<br/>Qini curves, recovery correlation, calibration"]
        A4 --> A5
        A5 --> A6["targeting_policy.py<br/>random vs. risk vs. uplift vs. oracle"]
    end

    subgraph B["Part B: staggered rollout / aggregate ATT"]
        B1["simulate_staggered_did.py<br/>32 sites x 36 months, staggered adoption<br/>known dynamic true effect"] --> B2["did_estimators.py<br/>naive TWFE (linearmodels)"]
        B1 --> B3["did_estimators.py<br/>group-time ATT (never-treated control)"]
        B3 --> B4["event-study aggregation"]
    end

    A6 --> P["pipeline.py<br/>orchestrator"]
    B2 --> P
    B4 --> P
    P --> O["outputs/figures/, outputs/reports/results.json"]
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| [`src/data/simulate_rct.py`](src/data/simulate_rct.py) | Simulates the randomized pilot: covariates, site-block-randomized treatment, Gamma-distributed downtime with a known heterogeneous true CATE. |
| [`src/data/simulate_staggered_did.py`](src/data/simulate_staggered_did.py) | Simulates the staggered site-level rollout: 4 adoption cohorts (including never-treated), a known dynamic (ramp-then-plateau) true effect. |
| [`src/models/meta_learners.py`](src/models/meta_learners.py) | Hand-rolled S-learner, T-learner, X-learner on LightGBM. |
| [`src/models/causal_forest.py`](src/models/causal_forest.py) | Thin wrapper around EconML's `CausalForestDML`, with the one-hot encoding its internal LightGBM nuisance models require. |
| [`src/evaluation/uplift_metrics.py`](src/evaluation/uplift_metrics.py) | Uplift/Qini curve construction, the Qini coefficient, and ground-truth CATE recovery correlation/calibration. |
| [`src/evaluation/targeting_policy.py`](src/evaluation/targeting_policy.py) | Budget-constrained targeting comparison: random vs. risk vs. uplift vs. oracle. |
| [`src/evaluation/did_estimators.py`](src/evaluation/did_estimators.py) | Naive TWFE (`linearmodels.PanelOLS`) and the hand-rolled group-time ATT / event-study estimator. |
| [`src/visualization/plots.py`](src/visualization/plots.py) | Renders every figure in this README from real pipeline output. |
| [`src/pipeline.py`](src/pipeline.py) | End-to-end orchestrator for both parts. |

---

# 5. Methodology

- **No leakage from ground truth into any estimator.** `true_cate_hours` (Part A) and `true_effect_hours` (Part B) are used exclusively for evaluation and are never available as a feature to any model — they exist only because this is a simulation.
- **Part A evaluation is entirely out-of-sample.** All 4 CATE estimators are fit on a 1,800-truck training split and evaluated (Qini, recovery correlation, calibration, targeting) on a held-out 1,200-truck test split they never saw.
- **Model selection for the targeting decision uses ground-truth recovery correlation, not the single-split Qini score.** §7.1 reports both, and they disagree here — the model selected for the calibration plot and the targeting-policy comparison is the one that best recovers the true CATE, which is only checkable because the data is synthetic. In a real deployment without ground truth, cross-validated Qini across multiple splits (not a single split, which is noisy) would be the practical substitute; this is flagged as a limitation, not smoothed over.
- **The group-time ATT estimator uses only never-treated sites as the control group** (not the "not-yet-treated" variant Callaway & Sant'Anna also allow), and averages the last 3 pre-adoption months into each cohort's baseline (rather than a single month) to reduce variance — both are disclosed, deliberate simplifications, not the full published estimator.
- **Randomization balance is checked directly, not assumed.** §7.1 reports the standardized mean difference for every covariate in the pilot.

---

# 6. Development

## Installation and setup

```powershell
git clone https://github.com/Rxyxs/chile-mining-fleet-causal-impact.git
cd chile-mining-fleet-causal-impact
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Full pipeline (one command)

```powershell
python -m src.pipeline
```

Simulates both datasets, fits all 4 CATE estimators, runs both DiD estimators, and writes every figure and number in §7 below to `outputs/`.

## Individual stages (for debugging)

```powershell
python -m src.data.simulate_rct
python -m src.data.simulate_staggered_did
```

## Tests

```powershell
pytest -v
```

26 tests: feature-level correctness of the uplift curve and Qini coefficient against a hand-computed example, the group-time ATT against an exact hand-computed effect on a noise-free toy panel, meta-learner sign-convention and ground-truth-correlation checks, targeting-policy selection logic, and DGP sanity checks (physical plausibility, balance, zero pre-treatment effect).

## Project structure

```
chile-mining-fleet-causal-impact/
├── src/
│   ├── data/
│   │   ├── simulate_rct.py
│   │   └── simulate_staggered_did.py
│   ├── models/
│   │   ├── meta_learners.py
│   │   └── causal_forest.py
│   ├── evaluation/
│   │   ├── uplift_metrics.py
│   │   ├── targeting_policy.py
│   │   └── did_estimators.py
│   ├── visualization/
│   │   └── plots.py
│   └── pipeline.py
├── outputs/
│   ├── figures/     # result figures (png, version-controlled)
│   └── reports/     # results.json (generated)
├── tests/           # 26 tests, pytest
├── requirements.txt
├── README.md
└── README.es.md
```

---

# 7. Results

Every number and figure below comes from an actual run of `python -m src.pipeline` (seed 42) — nothing here is estimated.

## 7.1 Part A: RCT-based CATE estimation

**Sample**: 3,000 trucks across 5 sites, split 1,800 train / 1,200 test.

**Randomization balance** (standardized mean difference, treated − control; all well inside the conventional ±0.1 threshold):

| Covariate | SMD |
|---|---:|
| truck_age_years | −0.032 |
| utilization_pct | +0.043 |
| cumulative_hours_1000s | −0.009 |
| prior_90d_downtime_hours | +0.023 |

![Covariate balance](outputs/figures/covariate_balance.png)

**Average effect**: naive diff-in-means ATE (training set) = **10.69h saved**; true ATE on the test set = **7.17h saved** — the naive estimate overstates the true average, a reminder that even a randomized pilot's simple difference in means is a noisy single-sample estimate of the true average effect, not the true average effect itself.

**CATE estimator comparison** — Qini coefficient (the metric available without ground truth) vs. correlation with the true CATE (available only in this simulation):

| Estimator | Qini coefficient | Correlation with true CATE |
|---|---:|---:|
| S-learner | **1233.98** | 0.569 |
| T-learner | 742.25 | 0.349 |
| X-learner | 993.57 | 0.478 |
| Causal Forest DML | 973.96 | **0.888** |

**Honest finding, not smoothed over**: the S-learner has the *highest* single-split Qini score, yet the Causal Forest DML recovers the *true* individual-level effect far better (0.888 vs. 0.569 correlation) — the model that would look best by the one metric available in a real deployment is not the model that is actually closest to correct. This project selects the Causal Forest DML for the downstream calibration and targeting analysis below precisely because ground-truth recovery is checkable here; the real lesson for a deployment without ground truth is that a single train/test split's Qini score is noisy enough that it can rank estimators differently from how they'd rank on the true effect, and cross-validated Qini across several splits is the practical mitigation.

![Qini curves](outputs/figures/qini_curves.png)
![CATE calibration](outputs/figures/cate_calibration.png)

## 7.2 Targeting policy comparison (30% fleet budget, 360 trucks)

| Policy | Total hours saved (true counterfactual) | % of achievable |
|---|---:|---:|
| Oracle (true uplift) | 4,690.87 | 100.0% |
| **Predicted uplift (Causal Forest DML)** | **4,582.00** | **97.7%** |
| Highest baseline risk | 4,208.65 | 89.7% |
| Random | 2,570.41 | 54.8% |

![Targeting policy comparison](outputs/figures/targeting_policy_comparison.png)

Targeting by predicted uplift captures 97.7% of the achievable benefit at this budget — a real, measured 8-point improvement over the "target the riskiest trucks" heuristic a team without a CATE model would likely default to, and more than 40 points over random assignment.

## 7.3 Part B: staggered-adoption difference-in-differences

**Sample**: 32 sites (8 per cohort: early/mid/late adopters + never-treated), 36 months.

| Estimator | Estimated effect | vs. true effect (−9.12h) |
|---|---:|---:|
| Naive TWFE (`linearmodels.PanelOLS`) | −8.51h (se 0.62) | 6.7% error |
| **Group-time ATT (this project's estimator)** | **−9.25h** | **1.4% error** |
| True overall ATT | −9.12h | — |

The naive constant-effect regression understates the true effect's magnitude — consistent with the Goodman-Bacon mechanism (§3.4): some of its implicit 2x2 comparisons use already-treated, still-improving sites as controls for later adopters, netting out part of a real, still-growing effect. The group-time estimator, which never makes that comparison, lands within 1.4% of the truth.

![Event study](outputs/figures/event_study.png)

The event-study curve shows the effect starting near zero at adoption and growing toward the plateau over the following months, with visibly increasing noise at later event-times — an honest, structural feature of a staggered design: only the earliest-adopting cohort has data that far past its own adoption date, so later event-time points are estimated from far fewer sites, not from a worse method.

---

# 8. Conclusion

- **Two genuinely different causal-inference designs, applied to the same intervention, both validated against a real known answer**: individual-level heterogeneity from a randomized pilot (§7.1-7.2), and an aggregate effect from a staggered, non-randomized rollout (§7.3) — the two situations a data scientist most commonly has to tell apart before reaching for a method.
- **The best-performing model by the metric you'd actually have in production (Qini) was not the model closest to the truth** (§7.1) — reported honestly rather than picking whichever ranking made the narrative cleaner, and used as the basis for a concrete recommendation (cross-validated Qini, not a single split) rather than left as an unresolved caveat.
- **Uplift-based targeting delivered a real, quantified improvement over a risk-based heuristic** (97.7% vs. 89.7% of the achievable benefit at a fixed budget, §7.2) — the concrete business case for building a CATE model at all, rather than defaulting to "target whoever looks riskiest."
- **The naive TWFE regression's bias under staggered adoption is not a textbook abstraction here** — it produced an estimate 6.7% off the true effect, on this project's own simulated data, for the specific mechanism (already-treated units as invalid controls under a dynamic effect) the recent DiD literature describes, and the corrected estimator's 1.4% error is the direct, measured payoff of accounting for it.

## Future work

- **Not-yet-treated as the comparison group** (the other Callaway & Sant'Anna variant), to check how much the never-treated-only choice here affects the result.
- **Doubly-robust CATE estimation** (e.g. EconML's `DRLearner`), to see whether it closes the gap between the S-learner's Qini performance and the Causal Forest's ground-truth recovery.
- **Sensitivity analysis for the parallel-trends assumption** in Part B (e.g. Rambachan & Roth, 2023's honest-DiD bounds), rather than assuming it holds outright.
- **A cost-aware targeting policy** that weighs each truck's maintenance cost against its predicted uplift, instead of ranking by uplift alone.

---

# 9. Author

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)

## Data source & license

Both datasets are **synthetically simulated** (`src/data/simulate_rct.py`, `src/data/simulate_staggered_did.py`) with a fixed seed (42) — there is no external data dependency. Each simulator is built with a known true treatment effect specifically so this project's estimators can be validated against a real answer, which is not observable in any real-world causal inference problem.

Code: MIT — see [LICENSE](LICENSE).
