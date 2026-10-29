# -*- coding: utf-8 -*-
"""07 不确定性量化：分位数回归 90% 区间 + 三种区间方案对比 + 校准检验。

方案
  Q   分位数回归直接给 [P5, P95]（区间宽度随样本变化）
  CV+ 折外残差分位数给常数半宽（覆盖率稳、宽度不随样本变化）
  NC  归一化一致性（normalized conformal）：用 Q 的预测半宽做尺度，再按折外得分校准

指标：覆盖率（目标 0.90）、平均/中位宽度、越界方向、PIT 直方图（理想为均匀）。

输出：results/metrics/uncertainty.json + fig12 / fig13
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import (group_kfold_by_car, interval_metrics, pit_histogram,  # noqa: E402
                            regression_metrics)
from evsoh.features import active_features, feature_columns  # noqa: E402
from evsoh.models import build_model  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_RED, plt, savefig  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402


def main() -> None:
    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, _ = add_soh_labels(ses, cfg)
    feats = active_features(df, feature_columns())
    df = df.dropna(subset=["capacity"]).reset_index(drop=True)
    X = df[feats].astype(float)
    y = df["capacity"].values.astype(float)
    cars = df["car"].values
    ref = df["bol_ref"].values.astype(float)

    alphas = cfg["uncertainty"]["alphas"]
    folds = list(group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                    seed=cfg["splits"]["random_state"]))

    pred = {a: np.full(len(df), np.nan) for a in alphas}
    for fi, (tr, te) in enumerate(folds):
        for a in alphas:
            mdl = build_model("quantile", alpha=a)
            mdl.fit(X.iloc[tr], y[tr])
            pred[a][te] = mdl.predict(X.iloc[te])
        print(f"  fold {fi+1}/{len(folds)} 完成", flush=True)

    lo, med, hi = pred[0.05], pred[0.5], pred[0.95]
    m_point = regression_metrics(y, med)

    # ---- 方案 Q：直接分位数区间 ----
    q_int = interval_metrics(y, lo, hi)

    # ---- 方案 CV+：折外残差分位数 ----
    resid = y - med
    half_cv = np.full(len(df), np.nan)
    for fi, (_tr, te) in enumerate(folds):
        mask = np.ones(len(df), bool)
        mask[te] = False
        half_cv[te] = np.percentile(np.abs(resid[mask]), 90)
    cv_int = interval_metrics(y, med - half_cv, med + half_cv)

    # ---- 方案 NC：归一化一致性（用 Q 半宽做尺度）----
    scale = np.maximum((hi - lo) / 2.0, 1e-6)
    score = np.abs(resid) / scale
    q_nc = np.full(len(df), np.nan)
    for fi, (_tr, te) in enumerate(folds):
        mask = np.ones(len(df), bool)
        mask[te] = False
        q_nc[te] = np.percentile(score[mask], 90)
    half_nc = q_nc * scale
    nc_int = interval_metrics(y, med - half_nc, med + half_nc)

    hist_q, _ = pit_histogram(y, lo, med, hi)
    hist_nc, _ = pit_histogram(y, med - half_nc, med, med + half_nc)

    res = {
        "point_rmse_Ah": m_point["rmse"],
        "point_soh_rmse_pp": m_point["rmse"] / float(ref.mean()) * 100,
        "nominal_coverage": cfg["uncertainty"]["nominal_coverage"],
        "quantile_interval": q_int,
        "cvplus_interval": cv_int,
        "normalized_conformal": nc_int,
        "pit_quantile": hist_q.tolist(),
        "pit_normalized_conformal": hist_nc.tolist(),
        "pit_uniform_dev": float(np.abs(hist_q - 1.0 / len(hist_q)).sum() / 2),
    }
    for k in ("quantile_interval", "cvplus_interval", "normalized_conformal"):
        r = res[k]
        print(f"{k:22s} 覆盖率={r['coverage']:.3f}  宽度均值={r['width_mean']:.2f} Ah "
              f"→ {r['width_mean']/float(ref.mean())*100:.2f} pp")

    # ---- 图 12：区间宽度与覆盖率随 SOC 跨度 ----
    bins = [20, 30, 40, 50, 60, 70, 100]
    df["_bin"] = pd.cut(df["d_soc"].abs(), bins=bins)
    labels = [f"{b.left:.0f}-{b.right:.0f}" for b in df["_bin"].cat.categories]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    for name, hw, color in [("分位数回归 Q", (hi - lo) / 2, C_MAIN),
                            ("归一化一致性 NC", half_nc, C_ACC),
                            ("CV+ 常数宽", half_cv, C_GRAY)]:
        g = pd.DataFrame({"w": hw, "b": df["_bin"]}).groupby("b", observed=True)["w"].median()
        axes[0].plot(range(len(g)), g.values, "o-", color=color, label=name)
    axes[0].set_xticks(range(len(labels)), labels)
    axes[0].set_xlabel("SOC 跨度区间（%）")
    axes[0].set_ylabel("90% 区间半宽中位数（Ah）")
    axes[0].set_title("(a) 区间宽度随片段信息量变化")
    axes[0].legend()

    for name, l, h, color in [("分位数回归 Q", lo, hi, C_MAIN),
                              ("归一化一致性 NC", med - half_nc, med + half_nc, C_ACC)]:
        cov = pd.DataFrame({"c": ((y >= l) & (y <= h)).astype(float), "b": df["_bin"]}
                           ).groupby("b", observed=True)["c"].mean()
        axes[1].plot(range(len(cov)), cov.values, "o-", color=color, label=name)
    axes[1].axhline(0.9, color=C_RED, ls="--", lw=1, label="目标 0.90")
    axes[1].set_xticks(range(len(labels)), labels)
    axes[1].set_xlabel("SOC 跨度区间（%）")
    axes[1].set_ylabel("实际覆盖率")
    axes[1].set_title("(b) 覆盖率随片段信息量变化")
    axes[1].legend()
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig12_区间宽度与覆盖率.png"))

    # ---- 图 13：PIT 校准 + 实例区间 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(np.arange(len(hist_q)) + 0.5, hist_q, width=0.42, color=C_MAIN,
                alpha=0.85, label="分位数回归 Q")
    axes[0].bar(np.arange(len(hist_nc)) + 0.92, hist_nc, width=0.42, color=C_ACC,
                alpha=0.85, label="归一化一致性 NC")
    axes[0].axhline(1.0 / len(hist_q), color=C_RED, ls="--", lw=1, label="理想均匀")
    axes[0].set_xlabel("PIT（0=落在区间下界，1=落在上界）")
    axes[0].set_ylabel("样本占比")
    axes[0].set_title("(a) PIT 校准直方图")
    axes[0].legend(fontsize=8)

    order = np.argsort(med)[:60]
    ax = axes[1]
    ax.errorbar(np.arange(len(order)), med[order],
                yerr=[med[order] - (med - half_nc)[order], (med + half_nc)[order] - med[order]],
                fmt="o", ms=3, lw=0.8, color=C_ACC, ecolor="#d9c3a5", capsize=0,
                label="NC 90% 区间")
    ax.plot(np.arange(len(order)), y[order], "x", ms=4, color=C_RED, label="真值")
    ax.set_xlabel("样本（按预测容量排序，取最低 60 条）")
    ax.set_ylabel("容量（Ah）")
    ax.set_title("(b) 区间实例（覆盖情况一目了然）")
    ax.legend(fontsize=8)
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig13_PIT与区间实例.png"))

    out = df[["dataset", "car", "segment", "capacity", "bol_ref", "d_soc", "quality"]].copy()
    out["pred_med"] = med
    out["pi_lo_q"] = lo
    out["pi_hi_q"] = hi
    out["pi_lo_nc"] = med - half_nc
    out["pi_hi_nc"] = med + half_nc
    out.to_parquet(os.path.join(cfg["paths"]["metrics_dir"], "uncertainty_oof.parquet"), index=False)

    p = save_json(res, "uncertainty")
    print(f"\n写出 {p}")


if __name__ == "__main__":
    main()
