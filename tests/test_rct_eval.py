"""Agregacion Monte Carlo del RCT: se verifica contra tablas pequenas calculadas a mano."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.rct_eval import metrics

ROOT = Path(__file__).resolve().parents[1]


def row(seed, oracle, risk, est, qini, recovery=None, random_expected=0.0, total=0.0):
    return {
        "seed": seed, "qini": qini, "recovery": recovery or qini,
        "budgets": {"0.5": {"n_selected": 3, "oracle": oracle, "total_effect": total, "risk": risk,
                            "random_expected": random_expected, "estimators": est}},
    }


# ------------------------------------------------------------------------ valor de politica
def test_budget_size_rounds_like_the_main_pipeline():
    assert metrics.budget_size(1200, 0.3) == 360 and metrics.budget_size(1200, 0.1) == 120
    assert metrics.budget_size(5, 0.01) == 1  # nunca 0 camiones


def test_policy_value_is_the_sum_of_the_true_effect_of_the_top_ranked():
    true = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    assert metrics.policy_value(true, true, 3) == 12.0                       # oraculo: 5+4+3
    assert metrics.policy_value(true, -true, 3) == 3.0                       # peor orden: 2+1+0
    assert metrics.policy_value(true, np.array([0, 9, 0, 9, 0, 0.0]), 2) == 6.0  # camiones 1 y 3: 4+2


def test_seed_budget_row_values_by_hand():
    true = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    r = metrics.seed_budget_row(true, risk_score=np.array([0, 9, 0, 9, 8, 0.0]), cate_scores={"a": true, "b": -true}, fraction=0.5)
    assert r["n_selected"] == 3 and r["oracle"] == 12.0
    assert r["risk"] == 4.0 + 2.0 + 1.0                                      # camiones 1, 3, 4
    assert r["estimators"] == {"a": 12.0, "b": 3.0}
    assert r["random_expected"] == pytest.approx(15.0 * 3 / 6)               # 7,5: la mitad del efecto total


def test_a_constant_true_effect_leaves_nothing_to_gain_from_targeting():
    """Sin heterogeneidad, cualquier orden captura lo mismo que el azar en esperanza y que el oraculo."""
    true = np.full(100, 2.0)
    scores = np.random.default_rng(0).normal(size=100)
    r = metrics.seed_budget_row(true, scores, {"a": scores}, 0.3)
    assert r["estimators"]["a"] == r["oracle"] == pytest.approx(r["random_expected"])


# ------------------------------------------------------------------------------ rankings
def test_ranking_orders_by_value_with_a_deterministic_tiebreak():
    assert metrics.ranking({"b": 1.0, "a": 1.0, "c": 3.0}) == ["c", "a", "b"]


def test_identical_rankings_are_fully_conserved():
    s = metrics.ranking_stability([["a", "b", "c"]] * 4)
    assert s["kendall_w"] == pytest.approx(1.0) and s["exact_match_rate"] == 1.0
    assert s["kendall_tau_vs_consensus"]["mean"] == pytest.approx(1.0)
    assert s["consensus"] == ["a", "b", "c"] and s["top1_frequency"] == {"a": 4}
    assert all(v == 1.0 for v in s["pairwise_agreement"].values())


def test_exactly_opposite_rankings_have_zero_concordance():
    s = metrics.ranking_stability([["a", "b", "c"], ["c", "b", "a"]])
    assert s["kendall_w"] == pytest.approx(0.0)                              # rangos medios iguales: sin consenso
    assert s["exact_match_rate"] == 0.5 and s["top1_frequency"] == {"a": 1, "c": 1}


def test_one_swapped_pair_by_hand():
    """3 semillas con orden a>b>c y una con a>c>b: acuerdo del par b>c = 3/4, W = 12*S/(n^2*(k^3-k))."""
    s = metrics.ranking_stability([["a", "b", "c"]] * 3 + [["a", "c", "b"]])
    assert s["pairwise_agreement"]["b>c"] == pytest.approx(0.75) and s["pairwise_agreement"]["a>b"] == 1.0
    assert s["mean_rank"] == {"a": 1.0, "b": 2.25, "c": 2.75}
    # rangos: sumas a=4, b=9, c=11; media 8; S = 16+1+9 = 26; W = 12*26 / (16*24)
    assert s["kendall_w"] == pytest.approx(12 * 26 / (16 * 24))
    assert s["exact_match_rate"] == 0.75


# ------------------------------------------------------------------------------ agregados
def two_seeds():
    return [
        row(1, oracle=10.0, risk=6.0, est={"A": 9.0, "B": 5.0}, qini={"A": 2.0, "B": 1.0}, random_expected=5.0),
        row(2, oracle=20.0, risk=12.0, est={"A": 10.0, "B": 16.0}, qini={"A": 3.0, "B": 1.0}, random_expected=10.0),
    ]


def test_aggregate_by_hand():
    b = metrics.aggregate(two_seeds(), [0.5])["budgets"]["0.5"]
    assert b["pct_of_oracle"]["A"]["mean"] == pytest.approx(70.0)           # (90 + 50) / 2
    assert b["pct_of_oracle"]["B"]["mean"] == pytest.approx(65.0)           # (50 + 80) / 2
    assert b["risk_rule_pct"]["mean"] == pytest.approx(60.0)
    assert b["random_expected_pct"]["mean"] == pytest.approx(50.0)
    assert b["seeds_estimator_beats_risk"] == {"A": 1, "B": 1}
    q = b["qini_selection"]
    assert q["pick_frequency"] == {"A": 2} and q["truth_pick_frequency"] == {"A": 1, "B": 1}
    assert q["qini_agrees_with_truth_pick"] == 1
    assert q["relative_loss_pct"]["mean"] == pytest.approx((0.0 + 100 * (16 - 10) / 16) / 2)   # 18,75
    assert q["qini_pick_pct_of_oracle"]["mean"] == pytest.approx(70.0)
    assert q["best_pct_of_oracle"]["mean"] == pytest.approx((90 + 80) / 2)
    assert q["mean_qini_pick_minus_risk_pp"] == pytest.approx(((90 - 60) + (50 - 60)) / 2)       # 10
    assert q["seeds_qini_pick_beats_risk"] == 1


def test_the_loss_of_picking_by_qini_is_zero_when_qini_and_truth_agree():
    seeds = [row(i, 10.0, 6.0, {"A": 9.0, "B": 5.0}, {"A": 2.0, "B": 1.0}) for i in range(5)]
    q = metrics.aggregate(seeds, [0.5])["budgets"]["0.5"]["qini_selection"]
    assert q["relative_loss_pct"]["max"] == 0.0 and q["qini_agrees_with_truth_pick"] == 5


def test_bootstrap_interval_is_reproducible_and_contains_the_mean_of_a_constant():
    assert metrics._bootstrap_mean_ci([2.0] * 10) == [2.0, 2.0]
    v = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert metrics._bootstrap_mean_ci(v) == metrics._bootstrap_mean_ci(v)
    lo, hi = metrics._bootstrap_mean_ci(v)
    assert lo <= np.mean(v) <= hi


def test_the_aggregate_is_strict_json():
    json.dumps(metrics.aggregate(two_seeds(), [0.5]), allow_nan=False)


# ------------------------------------------------------------------ coherencia con el pipeline
def test_seed_42_reproduces_the_main_pipeline_figures():
    """Control de coherencia: la semilla del pipeline principal da el 97,7% / 89,7% del README."""
    from src.rct_eval.run import run_seed

    ref = json.loads((ROOT / "outputs" / "reports" / "results.json").read_text(encoding="utf-8"))["part_a_rct_uplift"]
    r = run_seed(42)
    b = r["budgets"]["0.3"]
    assert b["n_selected"] == 360
    assert 100 * b["estimators"]["causal_forest"] / b["oracle"] == pytest.approx(97.7, abs=0.05)
    assert 100 * b["risk"] / b["oracle"] == pytest.approx(89.7, abs=0.05)
    for name, q in ref["qini_coefficients"].items():
        assert r["qini"][name] == pytest.approx(q, rel=1e-9)


def test_the_exported_file_is_consistent_with_its_own_per_seed_rows():
    path = ROOT / "tmp_agent_a" / "rct_results_30_seeds.json"
    if not path.exists():
        pytest.skip("el archivo se genera con `python -m src.rct_eval.run`")
    data = json.loads(path.read_text(encoding="utf-8"))
    again = metrics.aggregate(data["per_seed"], [0.1, 0.3, 0.6])
    assert data["n_seeds"] == 30 and len(set(data["seeds"])) == 30 and 42 in data["seeds"]
    assert again["budgets"] == data["budgets"] and again["qini"] == data["qini"]
