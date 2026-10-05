"""La pagina de GitHub Pages se genera desde los resultados del repo. Sin red ni navegador."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import site

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = re.compile(r'<script type="application/json" id="data">(.*?)</script>', re.S)
REPORTS = ROOT / "outputs" / "reports"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    docs = tmp_path_factory.mktemp("docs")
    out = site.build_site(docs_dir=docs)
    html = out.read_text(encoding="utf-8")
    return docs, html, json.loads(PAYLOAD.search(html).group(1).replace("<\\/", "</"))


@pytest.fixture(scope="module")
def results():
    return json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def units():
    return pd.read_csv(REPORTS / "targeting_units.csv")


def policy_value(true, score, m):
    """Suma del efecto verdadero de los ``m`` camiones con mayor ``score``."""
    return float(true[np.argsort(-score, kind="stable")[:m]].sum())


# ----------------------------------------------------------------------- archivos
def test_the_page_is_written_with_its_assets(built):
    docs, html, _ = built
    assert (docs / "index.html").exists() and (docs / ".nojekyll").exists()
    assert {p.name for p in (docs / "figures").glob("*.png")} == set(site.FIGURES_USED)
    assert "__DATA__" not in html


def test_embedded_data_is_strict_json(built):
    _, html, _ = built
    raw = PAYLOAD.search(html).group(1)
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw.replace("<\\/", "</"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_a_closing_script_tag_inside_the_data_cannot_break_the_page():
    html = site.render({"x": "</script><script>alert(1)</script>"})
    payload = PAYLOAD.search(html).group(1)
    assert "</" not in payload and "<\\/script>" in payload


def test_the_page_has_no_external_scripts_or_styles():
    html = site.TEMPLATE.read_text(encoding="utf-8")
    assert not re.search(r'<script[^>]+src="https?://', html)
    assert not re.search(r'<link[^>]+href="https?://', html)


def test_the_reports_the_page_depends_on_are_versioned():
    """Estan en .gitignore por defecto (outputs/reports/*): sin la excepcion, la pagina no se podria regenerar en CI."""
    import subprocess

    needed = {f"outputs/reports/{n}" for n in ("results.json", "targeting_units.csv", "real_did.json", "real_rct.json")}
    # lista lo versionado y lo versionable (no rastreado pero no ignorado); `check-ignore` no sirve: tambien
    # devuelve 0 cuando coincide una regla de excepcion (`!archivo`)
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *sorted(needed)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert needed <= set(out.stdout.split()), f"estan ignorados: {needed - set(out.stdout.split())}"


# ------------------------------------------------------------------- datos incrustados
def test_every_series_has_one_value_per_truck(built):
    _, _, data = built
    n = data["n_units"]
    assert n == 1200
    assert len(data["units"]["true"]) == len(data["units"]["risk"]) == n
    assert set(data["units"]["cate"]) == set(site.ESTIMATORS)
    assert all(len(v) == n for v in data["units"]["cate"].values())


def test_the_policy_values_reproduce_the_readme_table_at_30_percent(built, results, units):
    """Los 4.690,87 / 4.582,00 / 4.208,65 horas del README salen de estos mismos datos por camion."""
    _, _, data = built
    true = np.array(data["units"]["true"])
    m = 360
    reported = {p["policy"]: p["total_hours_saved"] for p in results["part_a_rct_uplift"]["targeting_policy_comparison"]}
    assert policy_value(true, true, m) == pytest.approx(reported["oracle_true_uplift"], abs=0.05)
    assert policy_value(true, np.array(data["units"]["risk"]), m) == pytest.approx(reported["highest_baseline_risk"], abs=0.05)
    best = results["part_a_rct_uplift"]["best_model"]
    assert policy_value(true, np.array(data["units"]["cate"][best]), m) == pytest.approx(reported["highest_predicted_uplift"], abs=0.05)


def test_the_expected_random_value_is_close_to_the_seeded_draw_in_the_readme(built):
    _, _, data = built
    true = np.array(data["units"]["true"])
    oracle = policy_value(true, true, 360)
    expected_pct = 100 * true.sum() * 0.3 / oracle
    assert expected_pct == pytest.approx(data["seeded_random_pct"], abs=1.5)


def test_the_estimator_chosen_with_the_truth_is_not_the_one_the_practical_metric_picks(built):
    _, _, data = built
    assert data["best_by_truth"] == "causal_forest" and data["best_by_qini"] == "doubly_robust"


def test_most_estimators_do_worse_than_the_simple_risk_rule(built):
    """El hallazgo central: elegir por efecto predicho no siempre le gana a mirar solo el riesgo."""
    _, _, data = built
    true = np.array(data["units"]["true"])
    oracle = policy_value(true, true, 360)
    risk = policy_value(true, np.array(data["units"]["risk"]), 360) / oracle
    below = [k for k, v in data["units"]["cate"].items() if policy_value(true, np.array(v), 360) / oracle < risk]
    assert sorted(below) == ["s_learner", "t_learner", "x_learner"]


def test_a_better_ranking_never_captures_less_than_the_oracle_allows(built):
    _, _, data = built
    true = np.array(data["units"]["true"])
    for m in (12, 120, 600, 1200):
        oracle = policy_value(true, true, m)
        for name, scores in data["units"]["cate"].items():
            assert policy_value(true, np.array(scores), m) <= oracle + 1e-6, (name, m)


def test_at_full_budget_every_policy_captures_everything(built):
    _, _, data = built
    true = np.array(data["units"]["true"])
    for scores in [data["units"]["risk"], *data["units"]["cate"].values()]:
        assert policy_value(true, np.array(scores), 1200) == pytest.approx(true.sum())


# ------------------------------------------------------------------ DiD y validacion real
def test_did_errors_are_computed_from_the_results_not_from_rounded_values(built, results):
    _, _, data = built
    b = results["part_b_staggered_did"]
    truth = b["true_overall_att"]
    assert data["part_b"]["twfe"]["error_pct"] == pytest.approx(100 * abs(b["naive_twfe"]["att"] - truth) / abs(truth), abs=0.05)
    assert data["part_b"]["group_time"]["error_pct"] == pytest.approx(100 * abs(b["group_time_overall_att"] - truth) / abs(truth), abs=0.05)
    assert data["part_b"]["twfe"]["error_pct"] > 4 * data["part_b"]["group_time"]["error_pct"]


def test_real_data_numbers_match_their_report_files(built):
    _, _, data = built
    did = json.loads((REPORTS / "real_did.json").read_text(encoding="utf-8"))
    assert data["real_did"]["max_diff"] == did["max_abs_diff_vs_reference"]
    assert data["real_did"]["mine"]["att"] == did["overall"]["this_project_weighted"]["att"]
    rct = json.loads((REPORTS / "real_rct.json").read_text(encoding="utf-8"))
    assert set(data["real_rct"]) == set(rct)
    for arm, v in rct.items():
        assert data["real_rct"][arm]["n"] == v["n_customers"]
        assert set(data["real_rct"][arm]["models"]) == set(site.ESTIMATORS)


# ------------------------------------------------------------------- coherencia con el README
def test_readme_reports_the_corrected_did_errors(built):
    _, _, data = built
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{data['part_b']['twfe']['error_pct']:.1f}%" in readme
    assert f"{data['part_b']['group_time']['error_pct']:.1f}%" in readme


def test_readme_states_which_estimator_the_practical_metric_picks(built):
    _, _, data = built
    true = np.array(data["units"]["true"])
    oracle = policy_value(true, true, 360)
    picked = policy_value(true, np.array(data["units"]["cate"][data["best_by_qini"]]), 360) / oracle * 100
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{picked:.1f}%" in readme


# --------------------------------------------------------------------------------- textos
def _i18n_keys(template, lang):
    start = template.index(f"  {lang}: {{")
    end = template.index("\n  }", start)
    return set(re.findall(r'(?:^\s{4}|,\s)([a-z0-9_]+):\s*["\[]', template[start:end], re.M))


def test_both_languages_define_the_same_keys():
    template = site.TEMPLATE.read_text(encoding="utf-8")
    es, en = _i18n_keys(template, "es"), _i18n_keys(template, "en")
    assert es == en, (es ^ en)


def test_every_translated_element_has_a_key_in_both_languages():
    template = site.TEMPLATE.read_text(encoding="utf-8")
    used = set(re.findall(r'data-i18n="([a-z0-9_]+)"', template))
    assert used <= _i18n_keys(template, "es") and used <= _i18n_keys(template, "en")


def test_the_page_is_accessible_by_keyboard():
    template = site.TEMPLATE.read_text(encoding="utf-8")
    assert "tabindex" in template and "ArrowLeft" in template and "aria-pressed" in template and "aria-describedby" in template
