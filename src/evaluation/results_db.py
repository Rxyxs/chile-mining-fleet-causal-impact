"""Persists this project's comparative estimator metrics (Part A CATE
estimators, Part B DiD estimators) to a local DuckDB database so they can be
queried with SQL alongside the JSON/figure outputs the pipeline already
produces. Purely additive: it reads the same `results` dict `pipeline.py`
already writes to `results.json` and does not change any estimator or plot.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def _part_a_estimator_comparison(part_a: dict[str, Any]) -> pd.DataFrame:
    qini = part_a["qini_coefficients"]
    recovery = part_a["cate_recovery_correlation"]
    estimators = sorted(set(qini) | set(recovery))
    return pd.DataFrame(
        {
            "estimator": estimators,
            "qini_coefficient": [qini.get(e) for e in estimators],
            "cate_recovery_correlation": [recovery.get(e) for e in estimators],
            "is_best_by_ground_truth": [e == part_a.get("best_model") for e in estimators],
        }
    )


def _part_b_estimator_comparison(part_b: dict[str, Any]) -> pd.DataFrame:
    true_att = part_b["true_overall_att"]
    naive_att = part_b["naive_twfe"]["att"]
    corrected_att = part_b["group_time_overall_att"]
    return pd.DataFrame(
        [
            {
                "estimator": "naive_twfe",
                "att": naive_att,
                "true_att": true_att,
                "abs_pct_error": abs(naive_att - true_att) / abs(true_att) * 100.0,
            },
            {
                "estimator": "group_time_att",
                "att": corrected_att,
                "true_att": true_att,
                "abs_pct_error": abs(corrected_att - true_att) / abs(true_att) * 100.0,
            },
        ]
    )


def persist_results_to_duckdb(results: dict[str, Any], db_path: str | Path) -> Path:
    """Writes 4 comparison tables into a DuckDB file, replacing them each run:

    - part_a_cate_estimator_comparison: Qini + ground-truth recovery per CATE learner.
    - part_a_targeting_policy_comparison: hours saved per targeting policy.
    - part_b_did_estimator_comparison: naive TWFE vs. group-time ATT vs. true effect.
    - part_b_event_study: the dynamic ATT by event time (months since adoption).

    Returns the resolved path to the database file.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    part_a = results["part_a_rct_uplift"]
    part_b = results["part_b_staggered_did"]

    cate_comparison = _part_a_estimator_comparison(part_a)
    targeting_comparison = pd.DataFrame(part_a["targeting_policy_comparison"])
    did_comparison = _part_b_estimator_comparison(part_b)
    event_study = pd.DataFrame(part_b["event_study"])

    con = duckdb.connect(str(db_path))
    try:
        con.execute("DROP TABLE IF EXISTS part_a_cate_estimator_comparison")
        con.execute("DROP TABLE IF EXISTS part_a_targeting_policy_comparison")
        con.execute("DROP TABLE IF EXISTS part_b_did_estimator_comparison")
        con.execute("DROP TABLE IF EXISTS part_b_event_study")
        con.execute(
            "CREATE TABLE part_a_cate_estimator_comparison AS SELECT * FROM cate_comparison"
        )
        con.execute(
            "CREATE TABLE part_a_targeting_policy_comparison AS SELECT * FROM targeting_comparison"
        )
        con.execute(
            "CREATE TABLE part_b_did_estimator_comparison AS SELECT * FROM did_comparison"
        )
        con.execute("CREATE TABLE part_b_event_study AS SELECT * FROM event_study")
    finally:
        con.close()

    return db_path
