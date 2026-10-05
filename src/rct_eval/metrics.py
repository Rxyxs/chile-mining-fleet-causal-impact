"""Agregacion Monte Carlo del valor de politicas de targeting sobre muchas realizaciones del RCT simulado.

Funciones puras (sin entrenar modelos): reciben los valores por semilla y devuelven los agregados, asi que
se pueden verificar contra tablas pequenas calculadas a mano.

Fila por semilla (``per_seed``)::

    {"seed": int,
     "qini": {estimador: float}, "recovery": {estimador: float},
     "budgets": {"0.3": {"oracle": float, "total_effect": float, "risk": float,
                          "random_expected": float, "estimators": {estimador: float}}}}

Todos los valores de politica son horas ahorradas verdaderas (suma del CATE verdadero de los camiones elegidos).
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
from scipy.stats import kendalltau


def budget_size(n_units: int, fraction: float) -> int:
    """Mismo redondeo que ``evaluate_targeting_policies``."""
    return max(1, int(round(n_units * fraction)))


def policy_value(true_effect: np.ndarray, score: np.ndarray, m: int) -> float:
    """Suma del efecto verdadero de los ``m`` camiones con mayor ``score`` (desempate estable)."""
    return float(true_effect[np.argsort(-score, kind="stable")[:m]].sum())


def seed_budget_row(true_effect, risk_score, cate_scores: dict, fraction: float) -> dict:
    true_effect = np.asarray(true_effect, dtype=float)
    m = budget_size(len(true_effect), fraction)
    return {
        "n_selected": m,
        "oracle": policy_value(true_effect, true_effect, m),
        "total_effect": float(true_effect.sum()),
        "risk": policy_value(true_effect, np.asarray(risk_score, dtype=float), m),
        "random_expected": float(true_effect.sum()) * m / len(true_effect),
        "estimators": {k: policy_value(true_effect, np.asarray(v, dtype=float), m) for k, v in cate_scores.items()},
    }


def _pct(value: float, oracle: float) -> float:
    return 100.0 * value / oracle


def _summary(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "median": float(np.median(v)),
        "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)), "min": float(v.min()), "max": float(v.max()),
    }


def _bootstrap_mean_ci(values, n_boot: int = 10_000, seed: int = 0) -> list[float]:
    """IC 95% percentil de la media, remuestreando semillas."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def ranking(values: dict) -> list[str]:
    """Nombres ordenados de mayor a menor valor (desempate alfabetico, determinista)."""
    return [k for k, _ in sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))]


def ranking_stability(rankings: list[list[str]]) -> dict:
    """Cuanto se conserva el orden de estimadores entre semillas.

    - ``consensus``: orden por rango medio.
    - ``kendall_tau_vs_consensus``: tau de cada semilla contra el consenso.
    - ``kendall_w``: concordancia de Kendall (1 = todas las semillas ordenan igual, 0 = sin acuerdo).
    - ``exact_match_rate``: fraccion de semillas cuyo orden completo coincide con el consenso.
    - ``top1_frequency``: cuantas veces cada estimador queda primero.
    - ``pairwise_agreement``: fraccion de semillas donde cada par queda en el mismo orden que el consenso.
    """
    names = sorted(rankings[0])
    n_seeds, k = len(rankings), len(names)
    rank_of = np.array([[r.index(n) + 1 for n in names] for r in rankings], dtype=float)
    mean_rank = rank_of.mean(axis=0)
    consensus = [n for _, n in sorted(zip(mean_rank, names), key=lambda t: (t[0], t[1]))]
    consensus_pos = {n: i for i, n in enumerate(consensus)}
    taus = []
    for r in rankings:
        a = [consensus_pos[n] for n in r]
        taus.append(float(kendalltau(a, list(range(k))).statistic))
    s = ((rank_of.sum(axis=0) - rank_of.sum() / k) ** 2).sum()
    w = float(12 * s / (n_seeds**2 * (k**3 - k)))
    pairs = {}
    for a, b in combinations(consensus, 2):
        agree = sum(r.index(a) < r.index(b) for r in rankings)
        pairs[f"{a}>{b}"] = agree / n_seeds
    return {
        "consensus": consensus,
        "mean_rank": {n: float(m) for n, m in zip(names, mean_rank)},
        "kendall_tau_vs_consensus": _summary(taus),
        "kendall_w": w,
        "exact_match_rate": float(np.mean([r == consensus for r in rankings])),
        "top1_frequency": dict(Counter(r[0] for r in rankings)),
        "pairwise_agreement": pairs,
    }


def aggregate(per_seed: list[dict], fractions: list[float]) -> dict:
    """Agregados sobre semillas. Todos los porcentajes se calculan desde valores sin redondear."""
    estimators = sorted(per_seed[0]["qini"])
    out = {"n_seeds": len(per_seed), "seeds": [r["seed"] for r in per_seed], "budgets": {}}
    for f in fractions:
        key = str(f)
        rows = [r["budgets"][key] for r in per_seed]
        pct = {e: [_pct(r["estimators"][e], r["oracle"]) for r in rows] for e in estimators}
        risk_pct = [_pct(r["risk"], r["oracle"]) for r in rows]
        random_pct = [_pct(r["random_expected"], r["oracle"]) for r in rows]

        # Estimador que elegiria Qini y estimador que elegiria la verdad (mejor valor de politica realizado)
        qini_pick = [max(r["qini"], key=lambda e: (r["qini"][e], e)) for r in per_seed]
        truth_pick = [max(row["estimators"], key=lambda e: (row["estimators"][e], e)) for row in rows]
        corr_pick = [max(r["recovery"], key=lambda e: (r["recovery"][e], e)) for r in per_seed]
        qini_pct = [_pct(row["estimators"][p], row["oracle"]) for row, p in zip(rows, qini_pick)]
        best_pct = [_pct(row["estimators"][p], row["oracle"]) for row, p in zip(rows, truth_pick)]
        corr_pct = [_pct(row["estimators"][p], row["oracle"]) for row, p in zip(rows, corr_pick)]
        # perdida relativa: fraccion del valor del mejor estimador que se pierde al elegir por Qini
        rel_loss = [100.0 * (row["estimators"][t] - row["estimators"][q]) / row["estimators"][t]
                    for row, q, t in zip(rows, qini_pick, truth_pick)]
        qini_minus_risk = [q - r for q, r in zip(qini_pct, risk_pct)]

        out["budgets"][key] = {
            "n_selected": rows[0]["n_selected"],
            "pct_of_oracle": {e: _summary(pct[e]) for e in estimators},
            "risk_rule_pct": _summary(risk_pct),
            "random_expected_pct": _summary(random_pct),
            "seeds_estimator_beats_risk": {e: int(sum(p > r for p, r in zip(pct[e], risk_pct))) for e in estimators},
            "ranking_by_policy_value": ranking_stability([ranking({e: pct[e][i] for e in estimators}) for i in range(len(rows))]),
            "qini_selection": {
                "pick_frequency": dict(Counter(qini_pick)),
                "truth_pick_frequency": dict(Counter(truth_pick)),
                "qini_agrees_with_truth_pick": int(sum(q == t for q, t in zip(qini_pick, truth_pick))),
                "qini_pick_pct_of_oracle": _summary(qini_pct),
                "best_pct_of_oracle": _summary(best_pct),
                "relative_loss_pct": _summary(rel_loss),
                "mean_qini_pick_minus_risk_pp": float(np.mean(qini_minus_risk)),
                "mean_qini_pick_minus_risk_ci95_pp": _bootstrap_mean_ci(qini_minus_risk),
                "seeds_qini_pick_beats_risk": int(sum(d > 0 for d in qini_minus_risk)),
                "recovery_corr_pick_frequency": dict(Counter(corr_pick)),
                "recovery_corr_pick_pct_of_oracle": _summary(corr_pct),
            },
        }
    out["qini"] = {e: _summary([r["qini"][e] for r in per_seed]) for e in estimators}
    out["recovery_correlation"] = {e: _summary([r["recovery"][e] for r in per_seed]) for e in estimators}
    return out
