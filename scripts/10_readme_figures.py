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
    ax.grid(visible=False)
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


# ---------------------------------------------------------------- 6. LOVO + decomposition
def fig_lovo() -> None:
    """严格留一车：逐车误差分布 + 误差由什么解释。"""
    d = _load("lovo_decomposition.json")
    pred = pd.read_parquet(os.path.join(ROOT, "results", "cache", "lovo_predictions.parquet"),
                           columns=["car", "capacity", "pred_lovo"])
    pred["sq"] = (pred["pred_lovo"] - pred["capacity"]) ** 2
    per = pred.groupby("car")["sq"].mean().pow(0.5)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.hist(per, bins=28, color=C_MAIN, alpha=0.85, edgecolor="white")
    ax.axvline(per.median(), color=C_RED, ls="--", lw=1.4,
               label=f"median {per.median():.2f} Ah")
    ax.axvline(per.max(), color=C_ACC, ls=":", lw=1.4, label=f"worst {per.max():.2f} Ah")
    ax.set_xlabel("Per-car RMSE (Ah), leave-one-car-out")
    ax.set_ylabel("Number of cars")
    ax.set_title(f"191 folds: a long tail of hard cars", pad=8)
    ax.legend(loc="upper right")

    ax = axes[1]
    dc = d["decomposition"]
    blocks = [("Snippet\nattributes", dc["r2_attr"], C_MAIN),
              ("Car\nidentity", dc["r2_car"], C_ACC),
              ("Both", dc["r2_both"], C_PURPLE)]
    xs = np.arange(len(blocks))
    bars = ax.bar(xs, [b[1] * 100 for b in blocks], color=[b[2] for b in blocks], width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in blocks], fontsize=8.5)
    ax.set_ylabel("Variance of per-car RMSE (%)")
    ax.set_ylim(0, 100)
    ax.set_title("What drives cross-car error", pad=8)
    for b, (_l, v, _c) in zip(bars, blocks):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 2, f"{v * 100:.0f}%",
                ha="center", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    _save(fig, "lovo_decomposition.png")


# ---------------------------------------------------------------- 7. CQR coverage
def fig_cqr() -> None:
    """条件覆盖：整体都是 90%，分箱后差别才出来。"""
    d = _load("cqr.json")["methods"]
    keys = [("CVplus", "CV+", C_ACC), ("NC", "Normalized conformal", C_GRAY),
            ("CQR", "CQR", C_MAIN)]
    span_order = ["(20, 30]", "(30, 40]", "(40, 50]", "(50, 60]", "(60, 70]", "(70, 100]"]
    labels = [s.strip("()[]").replace(", ", "–").replace("100", "100") for s in span_order]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for key, label, col in keys:
        by = d[key]["by_span"]["bins"]
        ys = [by[s]["coverage"] * 100 for s in span_order]
        ax.plot(range(len(ys)), ys, marker="o", ms=5, lw=1.7, color=col, label=label)
    ax.axhline(90, color=C_RED, ls="--", lw=1.2, label="nominal 90%")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel("SOC span of the snippet (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Overall coverage is identical; conditional coverage is not", pad=8)
    ax.set_ylim(70, 96)
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, "cqr_coverage.png")


# ---------------------------------------------------------------- 8. personalization
def fig_personalize() -> None:
    """个性化：参数越少越稳，选样比校正形式更重要。"""
    pers = _load("personalize_v2.json")["curves"]
    gate = _load("quality_gate.json")["rows"]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    styles = [("chrono_affine", "affine a·p+b", C_ACC, "s"),
              ("chrono_ridge", "per-car ridge head", C_PURPLE, "^"),
              ("chrono_bias", "bias only", C_GRAY, "o"),
              ("chrono_active", "bias + pick best k", C_MAIN, "D")]
    lo, hi = -12.0, 12.0
    for key, label, col, mk in styles:
        c = pers[key]
        base = c[0]["car_mae_mean"]
        ks = np.array([r["k"] for r in c], dtype=float)
        ys = np.array([(base - r["car_mae_mean"]) / base * 100 for r in c])
        vis = ys >= lo
        axes[0].plot(ks[vis], ys[vis], marker=mk, ms=4.5, lw=1.6, color=col, label=label)
        if (~vis).any():
            axes[0].plot(ks[~vis][:1], [lo + 0.7], marker=mk, ms=4.5, color=col,
                         ls="none", mfc="white")
        axes[0].annotate(f"{ys[-1]:+.1f}%", (ks[-1], ys[-1]), textcoords="offset points",
                         xytext=(6, 0), fontsize=8, color=col, va="center")
    axes[0].axhline(0, color=C_GRAY, ls="--", lw=1)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xlim(-0.8, 23.5)
    axes[0].set_xlabel("Labelled sessions used per car (k)")
    axes[0].set_ylabel("MAE reduction vs zero-shot (%)")
    axes[0].set_title("(a) More parameters need more data", pad=8)
    axes[0].legend(loc="upper left", fontsize=7.6, ncol=2, columnspacing=1.0,
                   handlelength=1.6, borderpad=0.4)
    axes[0].text(0.98, 0.04, "affine / ridge reach −45% / −47% at k=1 (off-scale)",
                 transform=axes[0].transAxes, fontsize=7.2, color=C_GRAY,
                 style="italic", ha="right")

    xs = [r["retained_frac"] * 100 for r in gate]
    ys = [r["mae"] for r in gate]
    axes[1].plot(xs, ys, marker="o", ms=5, lw=1.7, color=C_MAIN)
    for i, (x, y_, r) in enumerate(zip(xs, ys, gate)):
        if i in (1, 2):
            continue
        txt = "no gate" if r["threshold"] == 0.0 else f"≥{r['threshold']:.2f}"
        axes[1].annotate(txt, (x, y_), textcoords="offset points", xytext=(-4, 9),
                         ha="center", fontsize=8, color=C_GRAY)
    axes[1].set_xlabel("Sessions retained (%)")
    axes[1].set_ylabel("MAE (Ah)")
    axes[1].set_title("(b) Rejecting low-quality snippets", pad=8)
    axes[1].set_ylim(min(ys) - 0.012, max(ys) + 0.018)
    axes[1].invert_xaxis()
    _save(fig, "personalization.png")


# ---------------------------------------------------------------- 9. condition norm
def fig_condition_norm() -> None:
    """工况不变性：同架构下开/关 GRL，收益集中在最冷的箱。"""
    d = _load("condition_norm.json")["methods"]
    bins = d["baseline"]["by_temp"]["bins"]
    order = [b for b in ["<10", "10-18", "18-24", "24-30", "30-36", ">36"] if b in bins]
    series = [("baseline", "XGBoost", C_GRAY), ("mlp_plain", "MLP, no GRL", C_ACC),
              ("target_invar", "MLP + GRL", C_MAIN)]

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    w = 0.26
    xs = np.arange(len(order))
    for k, (key, label, col) in enumerate(series):
        vals = [d[key]["by_temp"]["bins"][b]["mae"] for b in order]
        ax.bar(xs + (k - 1) * w, vals, width=w, color=col, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b}°C" for b in order], fontsize=8.5)
    ax.set_xlabel("Mean temperature of the snippet")
    ax.set_ylabel("MAE (Ah)")
    ax.set_title("Temperature invariance is bought in the representation,\n"
                 "and it pays off where it matters most", pad=8, fontsize=9.5)
    ax.set_ylim(0, max(d["baseline"]["by_temp"]["bins"][b]["mae"] for b in order) * 1.35)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95, ncol=2)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    _save(fig, "condition_norm.png")


# ---------------------------------------------------------------- 10. sequence vs manual
def fig_sequence() -> None:
    """表示 vs 模型：原始曲线喂 CNN 明显更差。"""
    d = _load("sequence_vs_manual.json")["methods"]
    order = [("xgb_manual", "XGBoost\n+ manual feats", C_GRAY),
             ("mlp_manual", "MLP\n+ manual feats", C_ACC),
             ("seq_cnn", "1D CNN\n+ raw curves", C_MAIN)]
    base = d["xgb_manual"]["rmse"]

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ys = np.arange(len(order))
    vals = [d[k]["rmse"] for k, _l, _c in order]
    cols = [c for _k, _l, c in order]
    ax.hlines(ys, base, vals, color=cols, lw=2.4, alpha=0.75)
    for i, (v, c) in enumerate(zip(vals, cols)):
        ax.plot([v], [i], marker="o", ms=8, color=c, zorder=3)
        d_ = "" if i == 0 else f"  ({(v / base - 1) * 100:+.1f}%)"
        ax.text(v, i + 0.36, f"{v:.4f}{d_}", ha="center", va="top", fontsize=8.4, color=c)
    ax.axvline(base, color=C_GRAY, ls="--", lw=1.1, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([l for _k, l, _c in order], fontsize=8.4)
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.30, -0.45)
    ax.set_xlim(base - 0.07, max(vals) + 0.08)
    ax.set_xlabel("RMSE (Ah), 5-fold by car on 11,966 sessions")
    ax.set_title("Raw curves do not beat hand-crafted features", pad=8)
    ax.grid(visible=False)
    _save(fig, "sequence_vs_manual.png")


# ---------------------------------------------------------------- 11. interval frontier
def fig_interval_frontier() -> None:
    """宽度-覆盖率前沿：CQR 落在噪声下限上。"""
    d = _load("ensemble_intervals.json")
    m = d["methods"]
    diag = d["diagnostics"]
    meta = [("quantile", "Quantile regression", C_GRAY, "o", (8, 5), "left"),
            ("cqr", "CQR", C_MAIN, "D", (0, -15), "center"),
            ("ens", "Deep ensemble (7)", C_ACC, "s", (8, 5), "left"),
            ("ens_cal", "Ensemble + conformal", C_PURPLE, "^", (-2, 9), "center")]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, label, col, mk, off, ha in meta:
        x, yv = m[key]["width_mean"], m[key]["coverage"] * 100
        ax.scatter([x], [yv], s=70, color=col, marker=mk, zorder=3, label=label)
        ax.annotate(f"{yv:.1f}%", (x, yv), textcoords="offset points", xytext=off,
                    fontsize=8.2, color=col, ha=ha)
    floor = 2.0 * diag["abs_residual_p90"]
    ax.axvline(floor, color=C_RED, ls=":", lw=1.3)
    ax.text(floor - 0.05, 30, f"noise floor\n2×P90(|r|) = {floor:.2f} Ah",
            fontsize=7.6, color=C_RED, va="bottom", ha="right")
    ax.axhline(90, color=C_RED, ls="--", lw=1.1)
    ax.set_xlabel("Mean interval width (Ah)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_xlim(0.0, 2.95)
    ax.set_ylim(15, 98)
    ax.set_title("CQR sits on the noise floor: wider is the only way to cover more",
                 pad=8, fontsize=9.5)
    ax.legend(loc="center right", fontsize=8, framealpha=0.92)
    _save(fig, "interval_frontier.png")


def fig_model_frontier() -> None:
    """Model family x feature set: which axis actually moves the error."""
    d = _load("model_frontier.json")
    fams = d["families"]
    sets = ["full", "no_coulomb", "coulomb_only", "no_context"]
    set_lab = ["full 114", "no Coulomb", "Coulomb only", "no context"]
    fam_lab = ["ElasticNet", "Random forest", "HistGB", "XGBoost", "MLP"]
    base = d["grid"]["full"]["xgb"]["rmse"]
    M = np.array([[d["grid"][fs][f]["rmse"] for fs in sets] for f in fams])
    D = (M / base - 1.0) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    # The linear model's "no Coulomb" cell is +82.5% and would wash out the
    # informative -5..+20 band, so the colour scale is clipped and the printed
    # number still carries the true value.
    vmax = 25.0
    im = axes[0].imshow(D, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            col = "white" if abs(D[i, j]) > vmax * 0.55 else "#222222"
            axes[0].text(j, i, f"{M[i, j]:.3f}\n{D[i, j]:+.1f}%", ha="center",
                         va="center", fontsize=7.6, color=col)
    axes[0].set_xticks(range(len(sets)))
    axes[0].set_xticklabels(set_lab, fontsize=7.8)
    axes[0].set_yticks(range(len(fams)))
    axes[0].set_yticklabels(fam_lab, fontsize=7.8)
    axes[0].grid(visible=False)
    axes[0].add_patch(plt.Rectangle((-0.5, fams.index("xgb") - 0.5), 1, 1,
                                    fill=False, ec=C_MAIN, lw=2.0))
    axes[0].set_title("RMSE (Ah) / change vs. main model", fontsize=9.5, pad=8)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.038, pad=0.02)
    cb.set_label("delta (%), clipped at ±25", fontsize=7.6)
    cb.ax.tick_params(labelsize=7)

    ax_r = d["axes"]
    items = [("Model family\n(full features)", ax_r["model_axis_full"]["range_pct"], C_MAIN),
             ("Feature set\n(XGBoost, loss only)", ax_r["feature_axis_xgb"]["range_pct"], C_ACC),
             ("XGB tuning", ax_r["tuning_axis"]["xgb"]["range_pct"], C_PURPLE),
             ("MLP tuning", ax_r["tuning_axis"]["mlp"]["range_pct"], C_PURPLE),
             ("HistGB tuning", ax_r["tuning_axis"]["hgb"]["range_pct"], C_PURPLE)]
    bars = axes[1].barh(range(len(items)), [v for _l, v, _c in items],
                        color=[c for _l, _v, c in items], height=0.58)
    axes[1].set_yticks(range(len(items)))
    axes[1].set_yticklabels([l for l, _v, _c in items], fontsize=7.8)
    axes[1].invert_yaxis()
    for b, (_l, v, _c) in zip(bars, items):
        axes[1].text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                     va="center", fontsize=7.8)
    axes[1].set_xlabel("Worst-to-best RMSE spread (%)")
    axes[1].set_title("Which axis is worth the effort?", fontsize=9.5, pad=8)
    axes[1].set_xlim(0, max(v for _l, v, _c in items) * 1.22)
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "model_frontier.png")


def fig_interval_floor() -> None:
    """Averaging sessions per car: how far the error and the interval can shrink."""
    d = _load("interval_floor.json")
    ks = [int(k) for k in d["aggregate"]]
    rmse = [d["aggregate"][str(k)]["rmse"] for k in ks]
    width = [d["aggregate"][str(k)]["width"] for k in ks]
    theo = [d["variance"]["pred_rmse_k"][str(k)] for k in ks]
    flat = d["variance"]["sigma_between"]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
    axes[0].plot(ks, rmse, marker="o", ms=5, lw=1.8, color=C_MAIN, label="measured")
    axes[0].plot(ks, theo, ls="--", lw=1.3, color=C_GRAY, label=r"$\sqrt{\sigma_b^2+\sigma_w^2/k}$")
    axes[0].axhline(flat, color=C_RED, ls=":", lw=1.2,
                    label=f"car-bias floor {flat:.2f} Ah")
    axes[0].set_xscale("log")
    axes[0].set_xticks(ks)
    axes[0].set_xticklabels([str(k) for k in ks])
    axes[0].set_xlabel("Sessions averaged per car (k)")
    axes[0].set_ylabel("RMSE (Ah)")
    axes[0].set_title("(a) Averaging beats tuning", fontsize=9.5, pad=8)
    axes[0].legend(fontsize=7.6, loc="upper right")
    axes[0].annotate(f"{rmse[0]:.2f}", (ks[0], rmse[0]), textcoords="offset points",
                     xytext=(6, 6), fontsize=7.8)
    axes[0].annotate(f"{rmse[-1]:.2f}", (ks[-1], rmse[-1]), textcoords="offset points",
                     xytext=(-4, 8), fontsize=7.8)

    axes[1].plot(ks, width, marker="s", ms=5, lw=1.8, color=C_ACC)
    axes[1].set_xscale("log")
    axes[1].set_xticks(ks)
    axes[1].set_xticklabels([str(k) for k in ks])
    axes[1].set_xlabel("Sessions averaged per car (k)")
    axes[1].set_ylabel("90% interval width (Ah)")
    axes[1].set_title("(b) The interval shrinks too (coverage stays 88-89%)",
                      fontsize=9.5, pad=8)
    for k, v in zip(ks, width):
        if k in (ks[0], 8):
            axes[1].annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                             xytext=(0, 9), ha="center", fontsize=7.8, color=C_ACC)
    axes[1].set_ylim(min(width) * 0.85, max(width) * 1.12)
    fig.tight_layout()
    _save(fig, "interval_floor.png")


def main() -> None:
    print("writing README figures ->", os.path.relpath(OUT, ROOT))
    fig_hero()
    fig_ablation()
    fig_interval()
    fig_coldstart()
    fig_error_vs_span()
    fig_lovo()
    fig_cqr()
    fig_personalize()
    fig_condition_norm()
    fig_sequence()
    fig_interval_frontier()
    fig_model_frontier()
    fig_interval_floor()
    print("done")


if __name__ == "__main__":
    main()
