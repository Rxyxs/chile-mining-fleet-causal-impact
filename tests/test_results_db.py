"""Tests for the DuckDB persistence of Part A / Part B comparison metrics."""
from __future__ import annotations

import duckdb
import pytest

from src.evaluation.results_db import persist_results_to_duckdb

SAMPLE_RESULTS = {
    "part_a_rct_uplift": {
        "best_model": "causal_forest",
        "qini_coefficients": {
            "s_learner": 1233.98,
            "t_learner": 742.25,
            "x_learner": 993.57,
            "causal_forest": 973.96,
            "doubly_robust": 1254.65,
        },
        "cate_recovery_correlation": {
            "s_learner": 0.5686,
            "t_learner": 0.3493,
            "x_learner": 0.4783,
            "causal_forest": 0.8876,
            "doubly_robust": 0.7993,
        },
        "targeting_policy_comparison": [
            {"policy": "oracle_true_uplift", "n_trucks_targeted": 360, "total_hours_saved": 4690.87, "pct_of_oracle_achieved": 100.0},
            {"policy": "highest_predicted_uplift", "n_trucks_targeted": 360, "total_hours_saved": 4582.0, "pct_of_oracle_achieved": 97.7},
            {"policy": "random", "n_trucks_targeted": 360, "total_hours_saved": 2570.41, "pct_of_oracle_achieved": 54.8},
        ],
    },
    "part_b_staggered_did": {
        "naive_twfe": {"att": -8.514, "se": 0.622},
        "group_time_overall_att": -9.251,
        "true_overall_att": -9.1176,
        "event_study": [
            {"event_time": 0, "mean_att": -2.31, "n_cohorts": 3},
            {"event_time": 1, "mean_att": -2.79, "n_cohorts": 3},
        ],
    },
}


@pytest.fixture
def db_path(tmp_path):
    return persist_results_to_duckdb(SAMPLE_RESULTS, tmp_path / "results.duckdb")


def test_persist_creates_db_file(db_path):
    assert db_path.exists()


def test_part_a_cate_estimator_comparison_table(db_path):
    con = duckdb.connect(str(db_path))
    df = con.execute("SELECT * FROM part_a_cate_estimator_comparison ORDER BY estimator").df()
    con.close()
    assert set(df["estimator"]) == {"s_learner", "t_learner", "x_learner", "causal_forest", "doubly_robust"}
    best_row = df[df["estimator"] == "causal_forest"].iloc[0]
    assert best_row["is_best_by_ground_truth"]
    assert not df[df["estimator"] == "s_learner"].iloc[0]["is_best_by_ground_truth"]


def test_part_a_targeting_policy_table(db_path):
    con = duckdb.connect(str(db_path))
    df = con.execute(
        "SELECT policy, pct_of_oracle_achieved FROM part_a_targeting_policy_comparison ORDER BY pct_of_oracle_achieved DESC"
    ).df()
    con.close()
    assert df.iloc[0]["policy"] == "oracle_true_uplift"
    assert df.iloc[-1]["policy"] == "random"


def test_part_b_did_estimator_comparison_accuracy(db_path):
    con = duckdb.connect(str(db_path))
    df = con.execute("SELECT * FROM part_b_did_estimator_comparison").df()
    con.close()
    naive = df[df["estimator"] == "naive_twfe"].iloc[0]
    corrected = df[df["estimator"] == "group_time_att"].iloc[0]
    # The group-time estimator should recover the true effect more closely
    # than the naive two-way fixed-effects regression, matching §7 of the README.
    assert corrected["abs_pct_error"] < naive["abs_pct_error"]


def test_part_b_event_study_table_row_count(db_path):
    con = duckdb.connect(str(db_path))
    n = con.execute("SELECT COUNT(*) FROM part_b_event_study").fetchone()[0]
    con.close()
    assert n == len(SAMPLE_RESULTS["part_b_staggered_did"]["event_study"])


def test_rerun_replaces_tables_instead_of_appending(tmp_path):
    path = tmp_path / "results.duckdb"
    persist_results_to_duckdb(SAMPLE_RESULTS, path)
    persist_results_to_duckdb(SAMPLE_RESULTS, path)
    con = duckdb.connect(str(path))
    n = con.execute("SELECT COUNT(*) FROM part_a_cate_estimator_comparison").fetchone()[0]
    con.close()
    assert n == 5
