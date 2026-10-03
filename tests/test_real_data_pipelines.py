"""Tests de la validacion con datos reales. Los que descargan datasets se saltan si no hay red."""
import numpy as np
import pandas as pd
import pytest

from src.data import real_data
from src.evaluation import did_estimators as D
from src.pipeline_real_rct import difference_in_means, permutation_pvalue, top_share_uplift


# ----------------------------------------------------------------- agregacion ponderada
def _gt(rows):
    return pd.DataFrame(rows, columns=["cohort_adoption_month", "att"])


def test_weighted_overall_att_by_hand():
    # cohorte 2: 20 sitios con celdas -0.1 y -0.3 ; cohorte 4: 40 sitios con celda -0.2
    gt = _gt([(2, -0.1), (2, -0.3), (4, -0.2)])
    sizes = pd.Series({2: 20.0, 4: 40.0})
    expected = (20 * -0.1 + 20 * -0.3 + 40 * -0.2) / (20 + 20 + 40)
    assert D.weighted_overall_att(gt, sizes) == pytest.approx(expected)


def test_weighted_equals_unweighted_when_cohorts_are_equal_size():
    gt = _gt([(2, -0.1), (2, -0.3), (4, -0.2)])
    sizes = pd.Series({2: 10.0, 4: 10.0})
    assert D.weighted_overall_att(gt, sizes) == pytest.approx(D.overall_att(gt.assign(att=gt["att"])))


def test_weighted_differs_from_unweighted_when_cohorts_differ():
    gt = _gt([(2, -0.4), (2, -0.4), (4, -0.05)])
    sizes = pd.Series({2: 5.0, 4: 95.0})
    assert abs(D.weighted_overall_att(gt, sizes) - D.overall_att(gt)) > 0.1


def test_cohort_sizes_counts_sites_not_rows():
    df = pd.DataFrame({"site_id": [1, 1, 1, 2, 2, 2, 3, 3, 3], "adoption_month": [2.0] * 6 + [np.nan] * 3})
    sizes = D.cohort_sizes(df)
    assert sizes.to_dict() == {2.0: 2.0}


def _staggered_panel(n_sites=30, n_months=6, effect=-1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sites):
        adoption = [np.nan, 3, 5][s % 3]
        site_effect = rng.normal(0, 0.3)
        for m in range(1, n_months + 1):
            treated = int(not np.isnan(adoption) and m >= adoption)
            rows.append({"site_id": s, "month": m, "adoption_month": adoption, "treated": treated,
                         "downtime_hours": 10 + site_effect + 0.2 * m + effect * treated + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_group_time_att_recovers_a_constant_effect():
    gt = D.group_time_att(_staggered_panel(effect=-1.0), baseline_window=1)
    assert D.weighted_overall_att(gt, D.cohort_sizes(_staggered_panel())) == pytest.approx(-1.0, abs=0.1)


def test_cluster_bootstrap_is_reproducible_and_brackets_the_estimate():
    panel = _staggered_panel()
    gt = D.group_time_att(panel, baseline_window=1)
    point = D.weighted_overall_att(gt, D.cohort_sizes(panel))
    a = D.cluster_bootstrap_overall_att(panel, n_boot=60, seed=3)
    b = D.cluster_bootstrap_overall_att(panel, n_boot=60, seed=3)
    assert a == b
    assert a["ci_low"] <= point <= a["ci_high"]
    assert a["se"] > 0 and a["n_boot"] == 60


# ----------------------------------------------------------------------- adaptadores
def test_mpdta_adapter_builds_the_post_adoption_dummy():
    raw = pd.DataFrame({
        "countyreal": [1, 1, 1, 2, 2, 2], "year": [2003, 2004, 2005] * 2,
        "first.treat": [2004.0, 2004.0, 2004.0, 0.0, 0.0, 0.0],
        "lemp": [1.0, 1.1, 1.2, 2.0, 2.0, 2.0], "treat": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    })
    panel = real_data.mpdta_to_panel(raw)
    assert panel["month"].tolist() == [1, 2, 3, 1, 2, 3]
    assert panel["treated"].tolist() == [0, 1, 1, 0, 0, 0]  # NO es la columna `treat` del dataset
    assert panel.loc[panel["site_id"] == 1, "adoption_month"].eq(2).all()
    assert panel.loc[panel["site_id"] == 2, "adoption_month"].isna().all()


def test_hillstrom_rejects_unknown_arm():
    with pytest.raises(ValueError):
        real_data.load_hillstrom("Kids E-Mail")


# ------------------------------------------------------------------ metricas de focalizacion
def test_top_share_uplift_finds_the_responsive_customers():
    n = 4000
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, n)
    responsive = np.arange(n) < n // 2  # solo la mitad responde al tratamiento
    visit = (rng.random(n) < np.where(responsive & (t == 1), 0.6, 0.1)).astype(int)
    perfect_cate = responsive.astype(float)
    random_cate = rng.random(n)
    assert top_share_uplift(perfect_cate, t, visit, 0.4) > 0.4
    assert top_share_uplift(perfect_cate, t, visit, 0.4) > top_share_uplift(random_cate, t, visit, 0.4) + 0.2


def test_top_share_uplift_is_nan_without_both_arms():
    t = np.ones(100, dtype=int)
    assert np.isnan(top_share_uplift(np.arange(100.0), t, np.ones(100), 0.3))


def test_permutation_pvalue_is_small_for_a_good_ranking_and_large_for_a_random_one():
    n = 3000
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, n)
    responsive = rng.random(n) < 0.5
    visit = (rng.random(n) < np.where(responsive & (t == 1), 0.7, 0.1)).astype(int)
    outcome = -visit.astype(float)
    from src.evaluation.uplift_metrics import qini_coefficient, uplift_curve

    good = responsive.astype(float) + rng.normal(0, 0.01, n)
    q_good = qini_coefficient(uplift_curve(good, t, outcome))
    assert permutation_pvalue(good, t, outcome, q_good, np.random.default_rng(0)) < 0.05
    junk = rng.random(n)
    q_junk = qini_coefficient(uplift_curve(junk, t, outcome))
    assert permutation_pvalue(junk, t, outcome, q_junk, np.random.default_rng(0)) > 0.05


def test_difference_in_means():
    df = pd.DataFrame({"treatment": [1] * 4 + [0] * 4, "visit": [1, 1, 0, 0, 1, 0, 0, 0]})
    out = difference_in_means(df)
    assert out["ate"] == pytest.approx(0.5 - 0.25)
    assert out["ci_low"] < out["ate"] < out["ci_high"]
    assert out["n_treated"] == 4 and out["n_control"] == 4


# ------------------------------------------- contra una implementacion de referencia (red)
def _download_or_skip(fn):
    try:
        return fn()
    except Exception as exc:  # sin red o sitio caido
        pytest.skip(f"sin acceso al dataset: {exc}")


def test_mpdta_matches_the_reference_implementation(tmp_path):
    pytest.importorskip("pyreadr")
    panel, _ = _download_or_skip(lambda: real_data.load_mpdta_panel(tmp_path))
    gt = D.group_time_att(panel, baseline_window=1).set_index(["cohort_adoption_month", "month"])["att"]
    # ATT(g, t) calculados con `csdid` (port del paquete `did` de los autores), sin covariables,
    # grupo de control never-treated y periodo base g-1. Son valores de referencia, no un resultado
    # publicado que yo haya verificado contra el articulo.
    published = {(2, 2): -0.0105, (2, 3): -0.0704, (2, 4): -0.1373, (2, 5): -0.1008, (4, 4): -0.0046, (4, 5): -0.0412}
    for cell, value in published.items():
        assert gt[cell] == pytest.approx(value, abs=6e-4)
    sizes = D.cohort_sizes(panel)
    assert D.weighted_overall_att(gt.reset_index(), sizes) == pytest.approx(-0.04, abs=2e-3)
    assert D.naive_twfe_att(panel)["att"] == pytest.approx(-0.0365, abs=1e-3)
