"""Monte Carlo del RCT simulado: 30 semillas fijas x 5 estimadores de CATE x presupuestos 10/30/60%.

    python -m src.rct_eval.run            # reanuda si ya hay semillas calculadas
    python -m src.rct_eval.run --seeds 3  # prueba rapida (no escribe el resultado final)

Cada semilla simula un RCT nuevo y usa el mismo split 60/40 y la misma configuracion de modelos que
``src.pipeline.run_part_a``. La semilla 42 es la del pipeline principal: es el control de coherencia.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from src.data.simulate_rct import simulate_rct
from src.evaluation.uplift_metrics import cate_recovery_correlation, qini_coefficient, uplift_curve
from src.models.causal_forest import CausalForestModel
from src.models.dr_learner import DoublyRobustModel
from src.models.meta_learners import SLearner, TLearner, XLearner
from src.pipeline import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.rct_eval.metrics import aggregate, seed_budget_row

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tmp_agent_a"
SEEDS = [42] + list(range(1, 30))
FRACTIONS = [0.1, 0.3, 0.6]
MODELS = {
    "s_learner": SLearner, "t_learner": TLearner, "x_learner": XLearner,
    "causal_forest": CausalForestModel, "doubly_robust": DoublyRobustModel,
}


def _features(df):
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def run_seed(seed: int) -> dict:
    df = simulate_rct(seed=seed)
    train, test = train_test_split(df, test_size=0.4, random_state=seed, stratify=df["treated"])
    X_train, X_test = _features(train), _features(test)
    T_train, T_test = train["treated"].to_numpy(), test["treated"].to_numpy()
    Y_train, Y_test = train["downtime_next_30d_hours"].to_numpy(), test["downtime_next_30d_hours"].to_numpy()
    true = test["true_cate_hours"].to_numpy()

    cate = {name: cls().fit(X_train, T_train, Y_train).predict_cate(X_test) for name, cls in MODELS.items()}
    qini = {n: qini_coefficient(uplift_curve(p, T_test, Y_test)) for n, p in cate.items()}
    recovery = {n: cate_recovery_correlation(p, true) for n, p in cate.items()}

    controls = train["treated"] == 0
    risk = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, num_leaves=31, random_state=42, verbosity=-1)
    risk.fit(_features(train[controls]), train.loc[controls, "downtime_next_30d_hours"])
    risk_score = risk.predict(X_test)

    return {
        "seed": seed, "qini": qini, "recovery": recovery,
        "budgets": {str(f): seed_budget_row(true, risk_score, cate, f) for f in FRACTIONS},
    }


def _fmt(x: float) -> str:
    return f"{x:.1f}"


def summary_markdown(result: dict, control: dict | None) -> str:
    """Resumen legible; cada cifra sale de ``result`` (nada escrito a mano)."""
    est = sorted(result["qini"])
    lines = [f"# Monte Carlo del RCT: {result['n_seeds']} semillas fijas", "",
             f"Semillas: {result['seeds']}. Valores = % del beneficio alcanzable por el oraculo (efecto verdadero), "
             "media [p05, p95] entre semillas.", ""]
    for key, b in result["budgets"].items():
        lines += [f"## Presupuesto {float(key):.0%} ({b['n_selected']} camiones)", "",
                  "| Estimador | % alcanzable | Semillas que superan la regla de riesgo | rango medio |", "|---|---|---:|---:|"]
        rk = b["ranking_by_policy_value"]
        for e in est:
            s = b["pct_of_oracle"][e]
            lines.append(f"| {e} | {_fmt(s['mean'])} [{_fmt(s['p05'])}, {_fmt(s['p95'])}] | "
                         f"{b['seeds_estimator_beats_risk'][e]} / {result['n_seeds']} | {rk['mean_rank'][e]:.2f} |")
        r, rnd = b["risk_rule_pct"], b["random_expected_pct"]
        lines.append(f"| *regla de mayor riesgo* | {_fmt(r['mean'])} [{_fmt(r['p05'])}, {_fmt(r['p95'])}] | | |")
        lines.append(f"| *aleatorio (esperado)* | {_fmt(rnd['mean'])} | | |")
        q = b["qini_selection"]
        lo, hi = q["mean_qini_pick_minus_risk_ci95_pp"]
        lines += ["",
                  f"- **Conservacion del ranking**: consenso {' > '.join(rk['consensus'])}; W de Kendall {rk['kendall_w']:.3f}; "
                  f"orden completo identico al consenso en {rk['exact_match_rate']:.0%} de las semillas; "
                  f"tau medio contra el consenso {rk['kendall_tau_vs_consensus']['mean']:.3f}.",
                  f"- **Primero en el ranking**: {rk['top1_frequency']}.",
                  f"- **Elige Qini**: {q['pick_frequency']}; coincide con el mejor realizado en {q['qini_agrees_with_truth_pick']} / {result['n_seeds']} semillas.",
                  f"- **Perdida relativa de elegir por Qini** (frente al mejor estimador realizado): media {q['relative_loss_pct']['mean']:.2f}% "
                  f"[p05 {q['relative_loss_pct']['p05']:.2f}%, p95 {q['relative_loss_pct']['p95']:.2f}%]; el elegido captura "
                  f"{_fmt(q['qini_pick_pct_of_oracle']['mean'])}% vs {_fmt(q['best_pct_of_oracle']['mean'])}% del mejor.",
                  f"- **Estimador elegido por Qini menos regla de riesgo**: {q['mean_qini_pick_minus_risk_pp']:+.2f} pp "
                  f"(IC95% bootstrap sobre semillas [{lo:+.2f}, {hi:+.2f}]); le gana a la regla en {q['seeds_qini_pick_beats_risk']} / {result['n_seeds']} semillas.", ""]
    if control:
        lines += ["## Control de coherencia (semilla 42 vs `outputs/reports/results.json`)", "", control["text"], ""]
    return "\n".join(lines)


def control_check(per_seed: list[dict]) -> dict | None:
    """Compara la semilla 42 con el pipeline principal (si results.json existe)."""
    path = ROOT / "outputs" / "reports" / "results.json"
    if not path.exists():
        return None
    ref = json.loads(path.read_text(encoding="utf-8"))["part_a_rct_uplift"]
    row = next(r for r in per_seed if r["seed"] == 42)
    b = row["budgets"]["0.3"]
    ours = {e: 100 * v / b["oracle"] for e, v in b["estimators"].items()}
    ref_qini = ref["qini_coefficients"]
    diff_qini = max(abs(row["qini"][e] - ref_qini[e]) for e in ref_qini)
    cf_ref = next(p["pct_of_oracle_achieved"] for p in ref["targeting_policy_comparison"] if p["policy"] == "highest_predicted_uplift")
    risk_ref = next(p["pct_of_oracle_achieved"] for p in ref["targeting_policy_comparison"] if p["policy"] == "highest_baseline_risk")
    ok = abs(ours["causal_forest"] - cf_ref) < 0.1 and abs(100 * b["risk"] / b["oracle"] - risk_ref) < 0.1 and diff_qini < 1e-6
    text = (f"Causal Forest {ours['causal_forest']:.2f}% (results.json: {cf_ref}%), regla de riesgo {100 * b['risk'] / b['oracle']:.2f}% "
            f"(results.json: {risk_ref}%), diferencia maxima de Qini {diff_qini:.2e}. Coherente: {'si' if ok else 'NO'}.")
    return {"ok": bool(ok), "text": text, "max_qini_diff": diff_qini}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    args = parser.parse_args()
    seeds = SEEDS[: args.seeds]
    OUT_DIR.mkdir(exist_ok=True)
    cache = OUT_DIR / "per_seed.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["seed"]] = row
    for s in seeds:
        if s in done:
            continue
        t0 = time.time()
        done[s] = run_seed(s)
        with cache.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(done[s], allow_nan=False) + "\n")
        print(f"seed {s}: {time.time() - t0:.0f}s ({len(done)}/{len(seeds)})", flush=True)
    if args.seeds < len(SEEDS):
        return
    per_seed = [done[s] for s in seeds]
    result = aggregate(per_seed, FRACTIONS)
    control = control_check(per_seed)
    result["control_seed_42"] = control
    result["per_seed"] = per_seed
    (OUT_DIR / "rct_results_30_seeds.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (OUT_DIR / "summary_a.md").write_text(summary_markdown(result, control), encoding="utf-8")
    print("escrito", OUT_DIR / "rct_results_30_seeds.json")


if __name__ == "__main__":
    np.seterr(all="warn")
    main()
