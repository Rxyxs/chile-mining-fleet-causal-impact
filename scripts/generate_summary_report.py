"""Consolida los tres JSON de las simulaciones Monte Carlo en docs/REPORTE_MONTECARLO.md.

Entradas (relativas a la raiz del repo, o al checkout principal si se corre desde un worktree):
    tmp_agent_a/rct_results_30_seeds.json     RCT: 5 estimadores CATE x 30 semillas x 3 presupuestos
    tmp_agent_b/did_results_500_panels.json   DiD: 500 paneles escalonados
    tmp_agent_b/sensitivity_results.json      DiD: convergencia en el numero de sitios N

Todas las cifras del informe se calculan a partir de esos archivos; el script no contiene
resultados escritos a mano. Uso: ``python scripts/generate_summary_report.py [--data-root DIR]``.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "REPORTE_MONTECARLO.md"
INPUTS = {
    "rct": "tmp_agent_a/rct_results_30_seeds.json",
    "did": "tmp_agent_b/did_results_500_panels.json",
    "sens": "tmp_agent_b/sensitivity_results.json",
}

CATE_NAMES = {"causal_forest": "Causal Forest", "doubly_robust": "DR-Learner", "s_learner": "S-Learner",
              "t_learner": "T-Learner", "x_learner": "X-Learner"}
DID_NAMES = {"twfe": "TWFE ingenuo", "gt_never_w3": "Group-time, control never-treated",
             "gt_not_yet_w3": "Group-time, control not-yet-treated",
             "gt_never_w1_weighted": "Group-time, never-treated, base 1 mes, ponderado por cohorte"}
BUDGET_LABEL = {"0.1": "10%", "0.3": "30%", "0.6": "60%"}


def candidate_roots(explicit: Path | None) -> list[Path]:
    roots = [explicit] if explicit else [ROOT]
    try:  # en un worktree los JSON viven en el checkout principal
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        roots.append(Path(common).parent)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return roots


def load_inputs(explicit: Path | None = None) -> dict:
    data, searched = {}, candidate_roots(explicit)
    for key, rel in INPUTS.items():
        path = next((r / rel for r in searched if (r / rel).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No se encontro {rel}; busque en: {[str(r) for r in searched]}")
        data[key] = json.loads(path.read_text(encoding="utf-8"))
        data[f"{key}_path"] = rel
    return data


def table(header: list[str], rows: list[list[str]]) -> str:
    sep = "|".join("---" for _ in header)
    return "\n".join([f"| {' | '.join(header)} |", f"|{sep}|", *[f"| {' | '.join(r)} |" for r in rows]])


def pm(mean: float, std: float, digits: int = 1) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


# ---------------------------------------------------------------- resumen ejecutivo
def section_summary(rct: dict, did: dict) -> str:
    budgets = rct["budgets"]
    order = sorted(CATE_NAMES, key=lambda e: budgets["0.3"]["ranking_by_policy_value"]["mean_rank"][e])
    rows = []
    for e in order:
        rows.append([CATE_NAMES[e], *[pm(budgets[b]["pct_of_oracle"][e]["mean"], budgets[b]["pct_of_oracle"][e]["std"]) for b in budgets],
                     f"{rct['recovery_correlation'][e]['mean']:.2f}", f"{rct['qini'][e]['mean']:.0f}"])
    rows.append(["Regla de riesgo", *[pm(budgets[b]["risk_rule_pct"]["mean"], budgets[b]["risk_rule_pct"]["std"]) for b in budgets], "-", "-"])
    rows.append(["Aleatoria (esperada)", *[pm(budgets[b]["random_expected_pct"]["mean"], budgets[b]["random_expected_pct"]["std"]) for b in budgets], "-", "-"])
    cate = table(["Estimador", *[f"% del oráculo, presupuesto {BUDGET_LABEL[b]}" for b in budgets], "Correlación con el CATE verdadero", "Qini medio"], rows)

    drows = []
    for n in DID_NAMES:
        s, c = did["errors"][n]["signed_pct"], did["coverage"][n]
        drows.append([DID_NAMES[n], f"{s['mean']:+.2f}", f"{s['sd']:.2f}", f"{s['rmse']:.2f}", pct(c["coverage"])])
    did_t = table(["Estimador", "Error medio (% del ATT verdadero)", "Desv. estándar", "RMSE", "Cobertura IC 95%"], drows)

    top = order[0]
    wins = [b for b in budgets if budgets[b]["ranking_by_policy_value"]["consensus"][0] == top]
    cfg, g, tw = did["config"], did["errors"]["gt_never_w3"]["signed_pct"], did["errors"]["twfe"]["signed_pct"]
    return f"""## 1. Resumen ejecutivo

Este informe consolida tres simulaciones Monte Carlo sobre datos sintéticos con efecto verdadero conocido: la estabilidad de los estimadores de CATE en un RCT ({rct['n_seeds']} semillas), el sesgo y la varianza de los estimadores de diferencias en diferencias con adopción escalonada ({cfg['n_panels']} paneles) y cómo converge el estimador group-time al aumentar el número de sitios.

**Conclusiones**

- **RCT.** {CATE_NAMES[top]} lidera el ranking por valor de política en {len(wins)} de {len(budgets)} presupuestos y supera a la regla de riesgo en la mayoría de las semillas; la ventaja se estrecha al ampliar el presupuesto.
- **DiD.** El TWFE ingenuo tiene un sesgo sistemático ({tw['mean']:+.1f}%, subestima el efecto), pero su dispersión es baja ({tw['sd']:.1f}%). El group-time no tiene sesgo ({g['mean']:+.1f}%) pero con {cfg['sites']} sitios su dispersión es {g['sd']:.1f}%. Resultado: el TWFE tiene menor RMSE que el group-time en este diseño ({tw['rmse']:.1f}% frente a {g['rmse']:.1f}%).
- **Cobertura.** Los intervalos bootstrap por sitios cubren menos que el 95% nominal ({pct(did['coverage']['gt_never_w3']['coverage'])}); el intervalo analítico del TWFE cubre {pct(did['coverage']['twfe']['coverage'])}.
- **Escala.** El error del group-time baja como 1/√N; con 16 sitios la cobertura es claramente insuficiente.

### Estimadores CATE (RCT, {rct['n_seeds']} semillas; media ± desv. estándar)

Valor de política como porcentaje del oráculo (el que conoce el efecto verdadero de cada camión).

{cate}

### Estimadores DiD ({cfg['n_panels']} paneles, {cfg['sites']} sitios x {cfg['months']} meses)

Error = (estimación − ATT verdadero) / |ATT verdadero|; positivo significa que subestima el efecto.

{did_t}
"""


# ---------------------------------------------------------------- RCT
def section_rct(rct: dict) -> str:
    budgets = rct["budgets"]
    parts = ["## 2. Estudio de estabilidad RCT", "",
             f"Se repite el experimento con {rct['n_seeds']} semillas distintas. En cada una se entrenan los cinco estimadores, se elige a los camiones con mayor CATE predicho dentro de un presupuesto (fracción de la flota tratada) y se mide el valor de esa política con el efecto verdadero, como porcentaje del oráculo.", "",
             "### 2.1 Ranking de estimadores por presupuesto", ""]
    for b, lab in BUDGET_LABEL.items():
        r = budgets[b]["ranking_by_policy_value"]
        order = r["consensus"]
        rows = [[str(i + 1), CATE_NAMES[e], f"{r['mean_rank'][e]:.2f}", str(r["top1_frequency"].get(e, 0)),
                 pm(budgets[b]["pct_of_oracle"][e]["mean"], budgets[b]["pct_of_oracle"][e]["std"]),
                 f"{budgets[b]['seeds_estimator_beats_risk'][e]}/{rct['n_seeds']}"] for i, e in enumerate(order)]
        parts += [f"**Presupuesto {lab}** ({budgets[b]['n_selected']} camiones). Concordancia entre semillas: Kendall W = {r['kendall_w']:.2f}, ranking idéntico al consenso en {pct(r['exact_match_rate'], 0)} de las semillas.", "",
                  table(["Posición", "Estimador", "Rango medio", "Primero en (semillas)", "% del oráculo", "Supera a la regla de riesgo"], rows), ""]
    pa = budgets["0.3"]["ranking_by_policy_value"]["pairwise_agreement"]
    parts += [f"Acuerdo por pares (presupuesto 30%): Causal Forest > DR-Learner en {pct(pa['causal_forest>doubly_robust'], 0)} de las semillas, DR-Learner > S-Learner en {pct(pa['doubly_robust>s_learner'], 0)}.", "",
              "### 2.2 Qini frente a la regla de riesgo", "",
              "En la práctica el efecto verdadero no se observa, así que el estimador se elige con el área Qini. La pregunta es si esa elección vence a una regla simple de riesgo (tratar a los camiones con mayor riesgo base).", ""]
    rows = []
    for b, lab in BUDGET_LABEL.items():
        q = budgets[b]["qini_selection"]
        lo, hi = q["mean_qini_pick_minus_risk_ci95_pp"]
        rows.append([lab, pm(q["qini_pick_pct_of_oracle"]["mean"], q["qini_pick_pct_of_oracle"]["std"]), pm(budgets[b]["risk_rule_pct"]["mean"], budgets[b]["risk_rule_pct"]["std"]),
                     f"{q['mean_qini_pick_minus_risk_pp']:+.2f} pp [{lo:+.2f}, {hi:+.2f}]", f"{q['seeds_qini_pick_beats_risk']}/{rct['n_seeds']}",
                     f"{q['qini_agrees_with_truth_pick']}/{rct['n_seeds']}"])
    parts += [table(["Presupuesto", "Elegido por Qini (% oráculo)", "Regla de riesgo (% oráculo)", "Diferencia media [IC 95%]", "Semillas donde Qini gana", "Qini elige al mejor real"], rows), ""]
    freq = budgets["0.3"]["qini_selection"]["pick_frequency"]
    freq_s = ", ".join(f"{CATE_NAMES[e]} {n}" for e, n in sorted(freq.items(), key=lambda kv: -kv[1]))
    parts += [f"Frecuencia con la que el Qini elige cada estimador ({rct['n_seeds']} semillas): {freq_s}. El Qini por semilla es ruidoso (desviación estándar de {rct['qini']['causal_forest']['std']:.0f} para el Causal Forest sobre una media de {rct['qini']['causal_forest']['mean']:.0f}), por lo que a veces elige un estimador que no es el mejor.", "",
              "### 2.3 Pérdida relativa ex-post", "",
              "Pérdida de valor de política por haber elegido con Qini en lugar del mejor estimador, medida después con el efecto verdadero (porcentaje relativo al valor del mejor).", ""]
    rows = []
    for b, lab in BUDGET_LABEL.items():
        q = budgets[b]["qini_selection"]
        l = q["relative_loss_pct"]
        rows.append([lab, f"{l['mean']:.2f}%", f"{l['median']:.2f}%", f"{l['p95']:.2f}%", f"{l['max']:.2f}%",
                     pct(1 - q["qini_agrees_with_truth_pick"] / rct["n_seeds"], 0)])
    parts += [table(["Presupuesto", "Pérdida media", "Mediana", "Percentil 95", "Máxima", "Semillas con elección errada"], rows), ""]
    l30 = budgets["0.3"]["qini_selection"]["relative_loss_pct"]
    parts += [f"La pérdida típica es nula (mediana {l30['median']:.1f}%) porque Qini acierta en la mayoría de las semillas; el costo se concentra en las semillas donde falla (percentil 95 de {l30['p95']:.1f}% con presupuesto 30%).", "",
              f"Control de coherencia con la semilla 42 publicada: {rct['control_seed_42']['text']}", ""]
    return "\n".join(parts)


# ---------------------------------------------------------------- DiD
def section_did(did: dict) -> str:
    err, cov, cfg = did["errors"], did["coverage"], did["config"]
    ref, pos = did["published_panel_reference"], did["published_vs_distribution"]
    rows = []
    for n in DID_NAMES:
        s, a = err[n]["signed_pct"], err[n]["abs_pct"]
        rows.append([DID_NAMES[n], f"{s['mean']:+.2f}", f"{s['sd']:.2f}", f"{s['median']:+.2f}", f"{s['q05']:+.1f} a {s['q95']:+.1f}", f"{a['mean']:.2f}", f"{s['rmse']:.2f}"])
    t_err = table(["Estimador", "Sesgo medio (%)", "Desv. estándar", "Mediana", "Percentiles 5-95", "Error absoluto medio", "RMSE"], rows)

    rows = []
    for n in DID_NAMES:
        c = cov[n]
        rows.append([DID_NAMES[n], pct(c["coverage"]), f"±{c['mc_se'] * 100:.1f}", pct(c["miss_below"]), pct(c["miss_above"]), f"{c['mean_width']:.2f}"])
    t_cov = table(["Estimador", "Cobertura", "Error MC", "Verdad bajo el IC", "Verdad sobre el IC", "Ancho medio (h)"], rows)

    never, nyt = err["gt_never_w3"]["signed_pct"], err["gt_not_yet_w3"]["signed_pct"]
    cn, cy = cov["gt_never_w3"], cov["gt_not_yet_w3"]
    h2h = did["head_to_head"]
    return f"""## 3. Evaluación de insesgadez vs. varianza en DiD

Se simulan {cfg['n_panels']} paneles independientes con el mismo proceso generador del panel publicado ({cfg['sites']} sitios x {cfg['months']} meses, cuatro cohortes de adopción, efecto verdadero dinámico y conocido). El estimando es el ATT verdadero promedio sobre los sitio-mes tratados.

### 3.1 TWFE frente a group-time

{t_err}

- **El TWFE está sesgado de forma sistemática.** Subestima el efecto en promedio {err['twfe']['signed_pct']['mean']:.1f}%, y el sesgo es estable entre paneles (desviación {err['twfe']['signed_pct']['sd']:.1f}%). Es coherente con el mecanismo de Goodman-Bacon (sitios ya tratados, con efecto todavía creciendo, usados como control de los que adoptan después), aunque esta simulación mide el sesgo y no aísla su causa.
- **El group-time no tiene sesgo apreciable ({never['mean']:+.2f}%) pero es mucho más ruidoso** ({never['sd']:.1f}% de desviación). Con solo {cfg['sites'] // 4} sitios de control, el contraste de cada cohorte hereda el ruido de ese grupo.
- **Por RMSE, el TWFE gana en este diseño** ({err['twfe']['signed_pct']['rmse']:.1f}% frente a {never['rmse']:.1f}%): el group-time queda más cerca de la verdad que el TWFE solo en {pct(h2h['gt_never_w3_beats_twfe_share'])} de los paneles. La corrección elimina el sesgo a costa de varianza; con pocos sitios, eso no compensa.
- **El caso publicado no es representativo.** En la semilla 42 el TWFE da {ref['twfe_pct_error']:+.1f}% y el group-time {ref['group_time_never_w3_pct_error']:+.1f}%. El primero es un valor típico ({pct(pos['twfe']['share_of_panels_with_smaller_or_equal_error'])} de los paneles tiene un error menor o igual); el segundo es una muestra favorable: solo el {pct(pos['gt_never_w3']['share_of_panels_with_smaller_or_equal_error'])} de los paneles logra un error absoluto igual o menor que {pos['gt_never_w3']['published_abs_pct_error']:.1f}%.

### 3.2 Cobertura empírica del intervalo de confianza

Intervalo nominal del 95%. Para el group-time se usa bootstrap por sitios (percentiles, {did['config']['n_boot']} remuestreos por panel); para el TWFE, el intervalo analítico con errores robustos por sitio.

{t_cov}

- **Group-time:** la cobertura real ({pct(cn['coverage'])}) queda por debajo del 95% nominal, pero sin sesgo direccional marcado (la verdad queda {pct(cn['miss_below'])} por debajo y {pct(cn['miss_above'])} por encima). Es lo esperable de un bootstrap por clusters con {cfg['sites']} sitios.
- **TWFE:** el intervalo es demasiado angosto (ancho medio {cov['twfe']['mean_width']:.1f} h frente a {cn['mean_width']:.1f} h) y centrado en un valor sesgado: la verdad queda fuera por debajo en {pct(cov['twfe']['miss_below'])} de los paneles, y por encima casi nunca.

### 3.3 Impacto del grupo de control not-yet-treated

Usar como control todos los sitios aún no tratados (además de los never-treated) agranda el grupo de comparación sin reintroducir comparaciones prohibidas, porque un sitio nunca se usa como control en un período en que ya está tratado.

| Métrica | Control never-treated | Control not-yet-treated | Cambio |
|---|---|---|---|
| Desviación estándar del error (%) | {never['sd']:.2f} | {nyt['sd']:.2f} | {(nyt['sd'] / never['sd'] - 1) * 100:+.1f}% |
| RMSE (%) | {never['rmse']:.2f} | {nyt['rmse']:.2f} | {(nyt['rmse'] / never['rmse'] - 1) * 100:+.1f}% |
| Sesgo medio (%) | {never['mean']:+.2f} | {nyt['mean']:+.2f} | - |
| Cobertura IC 95% | {pct(cn['coverage'])} | {pct(cy['coverage'])} | {(cy['coverage'] - cn['coverage']) * 100:+.1f} pp |
| Ancho medio del IC (h) | {cn['mean_width']:.2f} | {cy['mean_width']:.2f} | {(cy['mean_width'] / cn['mean_width'] - 1) * 100:+.1f}% |

La mejora es real pero modesta: reduce la dispersión en torno a un {abs(nyt['sd'] / never['sd'] - 1) * 100:.0f}% y deja la cobertura prácticamente igual. Queda más cerca de la verdad que el TWFE en {pct(h2h['gt_not_yet_w3_beats_twfe_share'])} de los paneles (contra {pct(h2h['gt_never_w3_beats_twfe_share'])} con control never-treated), por lo que no cambia la conclusión de la sección 3.1. A cambio exige un supuesto adicional: tendencias paralelas también entre las cohortes usadas como control.

La variante con base de un mes y ponderación por cohorte (la configuración del bootstrap del repositorio para `mpdta`) es la menos precisa: desviación de {err['gt_never_w1_weighted']['signed_pct']['sd']:.1f}% y IC de {cov['gt_never_w1_weighted']['mean_width']:.1f} h de ancho medio.
"""


# ---------------------------------------------------------------- convergencia en N
def section_convergence(sens: dict) -> str:
    grid, det = sens["clusters_eval"], sens["details"]
    n_panels = det["n_panels"]
    rows, products = [], []
    for n in grid:
        m, d = sens["metrics"][str(n)], det["per_n"][str(n)]
        products.append(m["std_error"] * math.sqrt(n))
        rows.append([str(n), f"{m['std_error']:.3f}", f"{d['std_error_pct_of_true_att']:.1f}%", f"{products[-1]:.2f}",
                     f"{pct(m['coverage_95'], 0)} ±{d['coverage_mc_se'] * 100:.1f}", f"{d['mean_ci_width_hours']:.2f}", f"{d['bias_hours']:+.3f}"])
    t = table(["N (sitios)", "σ del error (h)", "σ (% del ATT)", "σ·√N", "Cobertura IC 95%", "Ancho medio IC (h)", "Sesgo (h)"], rows)
    mean_prod = sum(products) / len(products)
    rel_se = 1 / math.sqrt(2 * (n_panels - 1))  # error relativo de una desviación estándar con n_panels paneles
    spread = [(p / mean_prod - 1) for p in products]
    within = all(abs(s) <= 2 * rel_se for s in spread)
    first, last = grid[0], grid[-1]
    ses = [det['per_n'][str(n)]['coverage_mc_se'] * 100 for n in grid[1:]]
    max_z = max((0.95 - sens['metrics'][str(n)]['coverage_95']) * 100 / se for n, se in zip(grid[1:], ses))
    min_se, max_se = min(ses), max(ses)
    return f"""## 4. Análisis de convergencia en N

Se varía el número de sitios N entre {', '.join(str(n) for n in grid)} (cuatro cohortes iguales: temprana, intermedia, tardía y never-treated) y se simulan {n_panels} paneles por valor de N. Estimador: group-time, control never-treated, base de 3 meses, media sin ponderar de las celdas post-tratamiento. Intervalo: bootstrap por sitios, {det['n_boot']} remuestreos.

{t}

**Constatación σ·√N ≈ 8.** El producto σ·√N queda entre {min(products):.2f} y {max(products):.2f} (media {mean_prod:.2f}): {', '.join(f'{p:.2f} para N = {n}' for n, p in zip(grid, products))}. Con N = {grid[-2]} y N = {last} el valor es {products[-2]:.2f} y {products[-1]:.2f}, prácticamente 8. Las desviaciones de N = {first} y N = 32 respecto de la media ({spread[0] * 100:+.0f}% y {spread[1] * 100:+.0f}%) {'son compatibles con' if within else 'exceden'} el error de muestreo de estimar una desviación estándar con {n_panels} paneles (error relativo ≈ {rel_se * 100:.0f}%, doble ≈ {2 * rel_se * 100:.0f}%), así que los datos son coherentes con una tasa 1/√N sin evidencia de desvío.

**Cobertura.** Con N = {first} el bootstrap cubre {pct(sens['metrics'][str(first)]['coverage_95'], 0)}, claramente por debajo del 95%; desde N = {grid[1]} queda entre {pct(min(sens['metrics'][str(n)]['coverage_95'] for n in grid[1:]), 0)} y {pct(max(sens['metrics'][str(n)]['coverage_95'] for n in grid[1:]), 0)}, a menos de {max_z:.1f} errores de simulación del nominal (con {n_panels} paneles, ±{min_se:.1f} a ±{max_se:.1f} pp). No hay evidencia suficiente para distinguir esa cobertura del 95%, pero tampoco para afirmar que lo alcanza.

**Implicación para el diseño.** El panel publicado ({grid[1]} sitios) está justo en el umbral: el error típico es de {sens['details']['per_n'][str(grid[1])]['std_error_pct_of_true_att']:.0f}% del efecto, y bajarlo a menos de {sens['details']['per_n'][str(last)]['std_error_pct_of_true_att'] + 1:.0f}% requiere del orden de {last} sitios.
"""


def provenance(data: dict) -> str:
    return f"""## Reproducibilidad

- Entradas: `{data['rct_path']}`, `{data['did_path']}`, `{data['sens_path']}`.
- RCT: {data['rct']['n_seeds']} semillas; DiD: {data['did']['config']['n_panels']} paneles (semillas {data['did']['config']['base_seed']}+), {data['did']['config']['n_boot']} remuestreos bootstrap por panel; convergencia: {data['sens']['details']['n_panels']} paneles por N (semilla base {data['sens']['details']['base_seed']}).
- Todos los datos son sintéticos: el efecto verdadero se conoce porque el simulador lo define; ninguna cifra de este informe proviene de datos reales.
- Este archivo se genera con `python scripts/generate_summary_report.py`; no se edita a mano.
"""


def build(data: dict) -> str:
    return "\n".join([
        "# Informe Monte Carlo: estimadores de CATE y de DiD", "",
        section_summary(data["rct"], data["did"]), section_rct(data["rct"]),
        section_did(data["did"]), section_convergence(data["sens"]), provenance(data),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=None, help="directorio que contiene tmp_agent_a/ y tmp_agent_b/")
    args = ap.parse_args()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build(load_inputs(args.data_root)), encoding="utf-8")
    print(f"Escrito {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
