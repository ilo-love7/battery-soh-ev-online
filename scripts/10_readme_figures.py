# -*- coding: utf-8 -*-
"""生成 README 用的英文标注图（docs/figures/）。

只依赖 results/metrics/ 下的 OOF 预测与指标 JSON，不重跑模型。
中文标注的完整 17 张图由 scripts/03–09 生成在 results/figures/。

用法：
    python scripts/10_readme_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_PURPLE, C_RED  # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures")
MET = os.path.join(ROOT, "results", "metrics")

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#ececec",
    "grid.linewidth": 0.8,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
})


def _load(name: str) -> dict:
    with open(os.path.join(MET, name), encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}  ({os.path.getsize(path) / 1024:.0f} KB)")


# ---------------------------------------------------------------- 1. hero
def fig_hero() -> None:
    """Predicted vs. reference capacity, P2 (unseen cars)."""
    df = pd.read_parquet(os.path.join(MET, "oof_predictions.parquet"),
                         columns=["dataset", "capacity", "pred_xgb"])
    tr = _load("train_results.json")["protocols"]["P2_group_by_car"]["xgb"]

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([30, 48], [30, 48], color=C_GRAY, lw=1.2, ls="--", label="y = x")
    colors = {"battery_dataset1": C_MAIN, "battery_dataset2": C_ACC, "battery_dataset3": C_RED}
    for ds, sub in df.groupby("dataset"):
        ax.scatter(sub["capacity"], sub["pred_xgb"], s=3, alpha=0.18,
                   color=colors.get(ds, C_GRAY), linewidths=0, label=f"{ds} (n={len(sub):,})")
    ax.set_xlabel("Reference capacity (Ah)")
    ax.set_ylabel("Predicted capacity (Ah)")
    ax.set_title("Leave-cars-out (P2): XGBoost on daily charging snippets", pad=10)
    ax.set_xlim(30, 48)
    ax.set_ylim(30, 48)
    ax.set_aspect("equal")
    ax.text(0.035, 0.955,
            f"RMSE {tr['rmse']:.3f} Ah\nMAE  {tr['mae']:.3f} Ah\n"
            f"R²   {tr['r2']:.3f}\nSOH  {tr['soh_rmse_pp']:.2f} pp",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#dddddd"))
    ax.legend(loc="lower right", markerscale=4, fontsize=8)
    _save(fig, "hero_pred_vs_true.png")


# ---------------------------------------------------------------- 2. ablation
def fig_ablation() -> None:
    ab = _load("train_results.json")["ablation_xgb_P2"]
    order = [("full", "All 114 features"), ("coulomb_only", "Coulomb-count only"),
             ("curve_only", "IC curve shape only"), ("no_coulomb", "All but coulomb-count")]
    labels = [b for _, b in order]
    vals = [ab[k]["rmse"] for k, _ in order]
    cols = [C_MAIN, C_ACC, C_PURPLE, C_PURPLE]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.barh(range(len(vals)), vals, color=cols, height=0.6)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE on unseen cars (Ah)")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title("Both feature families are needed (P2, XGBoost)", pad=8)
    for b, v, k in zip(bars, vals, [k for k, _ in order]):
        ax.text(v + max(vals) * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:.3f}  ({ab[k]['n_features']} feats)", va="center", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    _save(fig, "ablation.png")


# ---------------------------------------------------------------- 3. intervals
def fig_interval() -> None:
    u = _load("uncertainty.json")
    rows = [("Quantile\nregression", u["quantile_interval"]),
            ("CV+", u["cvplus_interval"]),
            ("Normalized\nconformal", u["normalized_conformal"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    names = [r[0] for r in rows]
    cov = [r[1]["coverage"] * 100 for r in rows]
    wid = [r[1]["width_mean"] for r in rows]
    cols = [C_RED, C_MAIN, C_MAIN]

    axes[0].axhline(90, color=C_GRAY, ls="--", lw=1.2, label="nominal 90%")
    bars = axes[0].bar(names, cov, color=cols, width=0.55)
    axes[0].set_ylim(70, 100)
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].set_title("Coverage")
    for b, v in zip(bars, cov):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%",
                     ha="center", fontsize=9)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="x", visible=False)

    bars = axes[1].bar(names, wid, color=cols, width=0.55)
    axes[1].set_ylabel("Mean interval width (Ah)")
    axes[1].set_title("Width")
    axes[1].set_ylim(0, max(wid) * 1.2)
    for b, v in zip(bars, wid):
        axes[1].text(b.get_x() + b.get_width() / 2, v + max(wid) * 0.02, f"{v:.2f}",
                     ha="center", fontsize=9)
    axes[1].grid(axis="x", visible=False)

    fig.suptitle("90% prediction intervals: raw quantiles under-cover, calibration fixes it",
                 fontsize=10.5, y=1.02)
    _save(fig, "intervals.png")


# ---------------------------------------------------------------- 4. cold start
def fig_coldstart() -> None:
    c = _load("coldstart.json")["curves"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    spec = [("random_bias", "Same-distribution split (potential)", C_MAIN, "o"),
            ("chrono_bias", "Chronological split (deployable)", C_ACC, "s")]
    for key, label, col, mk in spec:
        xs = [r["k"] for r in c[key]]
        base = c[key][0]["car_mae_mean"]
        ys = [(base - r["car_mae_mean"]) / base * 100 for r in c[key]]
        ax.plot(xs, ys, marker=mk, ms=4.5, lw=1.6, color=col, label=label)
        ax.annotate(f"{ys[-1]:+.1f}%", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=9, color=col, va="center")
    ax.axhline(0, color=C_GRAY, lw=1, ls="--")
    ax.set_xlabel("Labelled sessions used per car for calibration (k)")
    ax.set_ylabel("MAE reduction vs. zero-shot (%)")
    ax.set_title("Per-car bias correction: gain is real but decays when time moves forward", pad=8)
    ax.set_xlim(-0.8, 22)
    ax.legend(loc="lower right")
    _save(fig, "coldstart.png")


# ---------------------------------------------------------------- 5. error vs span
def fig_error_vs_span() -> None:
    e = _load("error_analysis.json")
    rows = e["by_soc_span"]
    mid = [np.mean([float(x) for x in r["label"].replace("–", "-").split("-")]) for r in rows]
    mae = [r["mae"] for r in rows]
    tgt = e["min_span_for_target"]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(mid, mae, marker="o", ms=5, lw=1.8, color=C_MAIN, label="MAE (Ah)")
    for x, y, r in zip(mid, mae, rows):
        if r["label"] in ("20–25", "80–100"):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(-10, 10), ha="center", fontsize=8.5, color=C_MAIN)
    ah_per_pp = e["overall"]["mae"] / e["overall_mae_pp"]  # ≈0.425 Ah per SOH point
    for pp, col, lab in ((2.0, C_PURPLE, "2.0 pp"), (1.5, C_ACC, "1.5 pp")):
        ah = pp * ah_per_pp
        ax.axhline(ah, color=col, ls="--", lw=1.2, label=f"{lab} target  ({ah:.2f} Ah)")
    ax.set_xlabel("SOC span covered by the snippet (%)")
    ax.set_ylabel("MAE (Ah)  ·  1 Ah ≈ 2.3 pp SOH")
    ax.set_title("Error falls monotonically with snippet length", pad=8)
    ax.set_xlim(18, 92)
    ax.set_ylim(0, 1.15)
    note = (f"≥ {tgt['1.5pp']:.0f}% SOC span → MAE < 1.5 pp\n"
            f"≥ {tgt['2.0pp']:.0f}% SOC span → MAE < 2.0 pp")
    ax.text(0.97, 0.95, note, transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#dddddd"))
    ax.legend(loc="lower left")
    _save(fig, "error_vs_span.png")


def main() -> None:
    print("writing README figures ->", os.path.relpath(OUT, ROOT))
    fig_hero()
    fig_ablation()
    fig_interval()
    fig_coldstart()
    fig_error_vs_span()
    print("done")


if __name__ == "__main__":
    main()
