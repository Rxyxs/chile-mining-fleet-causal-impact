"""Genera la pagina de GitHub Pages (``docs/index.html``) de este proyecto.

Todo lo que muestra sale de archivos del repo (``results.json``, ``targeting_units.csv``, ``real_did.json``,
``real_rct.json``), asi que no puede desincronizarse del README. El valor de cada politica de seleccion para
cualquier presupuesto se calcula en el navegador con los datos por camion del conjunto de test del piloto.

    python -m src.site
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("site_template.html")
REPORTS = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"
DOCS_DIR = ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"  # graficas del Monte Carlo (scripts/build_plots.py); la plantilla las referencia
FIGURES_USED = ["event_study.png", "targeting_policy_comparison.png", "real_did_estimators.png", "real_rct_targeting.png"]
ESTIMATORS = ["s_learner", "t_learner", "x_learner", "causal_forest", "doubly_robust"]


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def build_data(results: dict, units: pd.DataFrame, real_did: dict, real_rct: dict) -> dict:
    a, b = results["part_a_rct_uplift"], results["part_b_staggered_did"]
    qini, recovery = a["qini_coefficients"], a["cate_recovery_correlation"]
    truth = b["true_overall_att"]

    def pct_error(estimate: float) -> float:
        return 100.0 * abs(estimate - truth) / abs(truth)

    return {
        "n_units": len(units),
        "units": {
            "true": [_r(v) for v in units["true_cate_hours"]],
            "risk": [_r(v) for v in units["risk_score"]],
            "cate": {name: [_r(v) for v in units[f"cate_{name}"]] for name in ESTIMATORS},
        },
        "estimators": [
            {"key": name, "qini": _r(qini[name], 1), "recovery": _r(recovery[name], 3)} for name in ESTIMATORS
        ],
        "best_by_truth": a["best_model"],
        "best_by_qini": max(qini, key=qini.get),
        "seeded_random_pct": next(p["pct_of_oracle_achieved"] for p in a["targeting_policy_comparison"] if p["policy"] == "random"),
        "part_b": {
            "twfe": {"att": _r(b["naive_twfe"]["att"], 2), "se": _r(b["naive_twfe"]["se"], 2), "error_pct": _r(pct_error(b["naive_twfe"]["att"]), 1)},
            "group_time": {"att": _r(b["group_time_overall_att"], 2), "error_pct": _r(pct_error(b["group_time_overall_att"]), 1)},
            "truth": _r(truth, 2),
        },
        "real_did": {
            "reference": real_did["overall"]["reference_csdid"],
            "mine": real_did["overall"]["this_project_weighted"],
            "twfe": real_did["overall"]["naive_twfe"],
            "unweighted": real_did["overall"]["unweighted_mean_of_cells"],
            "max_diff": real_did["max_abs_diff_vs_reference"],
            "cohorts": real_did["panel"]["cohort_sizes_counties"],
            "never": real_did["panel"]["never_treated_counties"],
        },
        "real_rct": {
            arm: {
                "n": v["n_customers"], "ate": _r(v["randomized_ate"]["ate"] * 100, 1),
                "ci": [_r(v["randomized_ate"]["ci_low"] * 100, 1), _r(v["randomized_ate"]["ci_high"] * 100, 1)],
                "n_seeds": v["n_seeds"],
                "models": {
                    m: {"top30": _r(s["top30_uplift_mean"] * 100, 1), "random": _r(s["all_uplift_mean"] * 100, 1),
                        "sig": s["seeds_qini_significant"]}
                    for m, s in v["summary"].items()
                },
            }
            for arm, v in real_rct.items()
        },
    }


def render(data: dict) -> str:
    payload = json.dumps(data, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")  # un "</script>" dentro de un dato cerraria la etiqueta
    return TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)


def build_site(docs_dir: Path = DOCS_DIR, reports: Path = REPORTS, figures_dir: Path = FIGURES_DIR) -> Path:
    results = json.loads((reports / "results.json").read_text(encoding="utf-8"))
    units = pd.read_csv(reports / "targeting_units.csv")
    real_did = json.loads((reports / "real_did.json").read_text(encoding="utf-8"))
    real_rct = json.loads((reports / "real_rct.json").read_text(encoding="utf-8"))
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "index.html"
    out.write_text(render(build_data(results, units, real_did, real_rct)), encoding="utf-8")
    (docs_dir / ".nojekyll").write_text("", encoding="utf-8")
    target = docs_dir / "figures"
    target.mkdir(exist_ok=True)
    for name in FIGURES_USED:
        shutil.copy2(figures_dir / name, target / name)
    assets_dst = docs_dir / "assets"
    if ASSETS_DIR.exists() and ASSETS_DIR.resolve() != assets_dst.resolve():  # al generar en docs/ ya estan en su lugar
        shutil.copytree(ASSETS_DIR, assets_dst, dirs_exist_ok=True)
    return out


if __name__ == "__main__":
    path = build_site()
    print(f"{path} ({path.stat().st_size / 1024:.0f} KB)")
