"""Genera las dos graficas del Monte Carlo en ``docs/assets/``.

    python scripts/build_plots.py

Lee solo ``tmp_agent_a/rct_results_30_seeds.json`` y ``tmp_agent_b/did_results_500_panels.json``. Toda cifra anotada
en las graficas se calcula desde esos archivos (nada escrito a mano).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RCT_JSON = ROOT / "tmp_agent_a" / "rct_results_30_seeds.json"
DID_JSON = ROOT / "tmp_agent_b" / "did_results_500_panels.json"
OUT = ROOT / "docs" / "assets"

BUDGET = "0.3"
# estimador -> (etiqueta, color); paleta Okabe-Ito, distinguible con daltonismo
RCT_ESTIMATORS = {
    "causal_forest": ("Causal Forest", "#0072B2"),
    "doubly_robust": ("Doblemente robusto", "#009E73"),
    "s_learner": ("S-learner", "#E69F00"),
    "x_learner": ("X-learner", "#CC79A7"),
    "t_learner": ("T-learner", "#56B4E9"),
}
TWFE_COLOR, GROUPTIME_COLOR, INK, MUTED = "#D55E00", "#0072B2", "#222222", "#666666"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Falta {path.relative_to(ROOT)}: generalo antes (ver el modulo que lo produce).")
    return json.loads(path.read_text(encoding="utf-8"))


def rct_capture_distribution(data: dict, out: Path) -> dict:
    per_seed = data["per_seed"]
    names = list(RCT_ESTIMATORS)
    pct = {n: np.array([100 * r["budgets"][BUDGET]["estimators"][n] / r["budgets"][BUDGET]["oracle"] for r in per_seed]) for n in names}
    risk = np.array([100 * r["budgets"][BUDGET]["risk"] / r["budgets"][BUDGET]["oracle"] for r in per_seed])
    risk_mean = float(risk.mean())

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=150)
    rng = np.random.default_rng(0)  # solo desplaza horizontalmente los puntos para que no se tapen
    for i, n in enumerate(names):
        label, color = RCT_ESTIMATORS[n]
        ax.boxplot(pct[n], positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                   boxprops={"facecolor": color, "alpha": 0.25, "edgecolor": color},
                   medianprops={"color": color, "linewidth": 2}, whiskerprops={"color": color}, capprops={"color": color})
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(pct[n])), pct[n], s=16, color=color, alpha=0.85, zorder=3, edgecolor="white", linewidth=0.4)
        ax.text(i, 101.5, f"mediana {np.median(pct[n]):.1f}%", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.axhline(risk_mean, color=INK, linestyle="--", linewidth=1.4, zorder=2)
    ax.text(len(names) - 0.55, risk_mean + 0.7, f"Regla de mayor riesgo: {risk_mean:.1f}% (media de las {len(risk)} semillas)",
            ha="right", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(names)), [RCT_ESTIMATORS[n][0] for n in names])
    ax.set_ylabel("Beneficio alcanzable capturado (%)")
    ax.set_ylim(min(60, min(v.min() for v in pct.values()) - 3), 107)
    ax.set_title(f"Beneficio capturado por estimador en {len(per_seed)} semillas del RCT simulado\n"
                 f"(presupuesto {float(BUDGET):.0%} de la flota; cada punto es una semilla)", fontsize=12, loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {"risk_mean": risk_mean, "medians": {n: float(np.median(pct[n])) for n in names}}


def did_error_distribution(data: dict, out: Path) -> dict:
    p = data["per_panel"]
    truth = np.array(p["true_att"])
    err = {  # error con signo, % del efecto verdadero; positivo = subestima la magnitud
        "twfe": (np.array(p["twfe_att"]) - truth) / np.abs(truth) * 100,
        "gt": (np.array(p["gt_never_w3_att"]) - truth) / np.abs(truth) * 100,
    }
    stats = {k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))} for k, v in err.items()}
    lo, hi = min(v.min() for v in err.values()), max(v.max() for v in err.values())
    bins = np.linspace(np.floor(lo / 5) * 5, np.ceil(hi / 5) * 5, 41)

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=150)
    specs = [("twfe", "TWFE ingenuo", TWFE_COLOR), ("gt", "Group-time ATT (controles nunca tratados)", GROUPTIME_COLOR)]
    for key, label, color in specs:
        s = stats[key]
        ax.hist(err[key], bins=bins, color=color, alpha=0.5, edgecolor="white", linewidth=0.5,
                label=f"{label}: media {s['mean']:+.1f}%, desv. est. {s['sd']:.1f}%")
        ax.axvline(s["mean"], color=color, linewidth=1.8)
    ax.axvline(0, color=INK, linestyle="--", linewidth=1.2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)  # espacio arriba para leyenda y etiqueta
    ax.text(0.8, ax.get_ylim()[1] * 0.93, "sin error", ha="left", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("Error con signo respecto al efecto verdadero (% del efecto; positivo = subestima la magnitud)")
    ax.set_ylabel("Paneles simulados (cantidad)")
    ax.set_title(f"Distribución del error de dos estimadores DiD en {len(truth)} paneles simulados", fontsize=12, loc="left", pad=46)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return stats


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    a = rct_capture_distribution(_load(RCT_JSON), OUT / "rct_capture_distribution.png")
    b = did_error_distribution(_load(DID_JSON), OUT / "did_error_distribution.png")
    print("RCT: regla de riesgo", round(a["risk_mean"], 2), "| medianas", {k: round(v, 2) for k, v in a["medians"].items()})
    print("DiD:", {k: {m: round(x, 2) for m, x in v.items()} for k, v in b.items()})
    for f in sorted(OUT.glob("*.png")):
        print(f.relative_to(ROOT), f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
