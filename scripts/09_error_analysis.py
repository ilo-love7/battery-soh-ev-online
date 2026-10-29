# -*- coding: utf-8 -*-
"""09 误差分析：找出"什么样的片段估不准"，给出工程可用性结论。

分层维度：SOC 跨度、片段质量分、里程、温度、电流、厂商分块
核心产出：**"最少需要多长的充电片段"**——MAE 达到给定阈值所需的最小 SOC 跨度。

输出：results/metrics/error_analysis.json + fig16 / fig17
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import regression_metrics  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_RED, PALETTE, plt, savefig  # noqa: E402


def binned(df: pd.DataFrame, col: str, bins, metric_col: str = "abs_err",
           fmt: str = "{:.0f}") -> pd.DataFrame:
    d = df.copy()
    d["_b"] = pd.cut(d[col], bins=bins)
    g = d.groupby("_b", observed=True).agg(
        n=(metric_col, "size"),
        mae=(metric_col, "mean"),
        p90=(metric_col, lambda s: float(np.percentile(s, 90))),
        med_x=(col, "median"),
    ).reset_index()
    g["label"] = [f"{fmt.format(iv.left)}–{fmt.format(iv.right)}" for iv in g["_b"]]
    return g


def main() -> None:
    cfg = load_config()
    oof = pd.read_parquet(os.path.join(cfg["paths"]["metrics_dir"], "oof_predictions.parquet"))
    oof = oof.dropna(subset=["pred_xgb", "capacity"]).reset_index(drop=True)
    ref_mean = float(oof["bol_ref"].mean())
    oof["err"] = oof["pred_xgb"] - oof["capacity"]
    oof["abs_err"] = oof["err"].abs()
    oof["abs_err_pp"] = oof["abs_err"] / oof["bol_ref"] * 100
    oof["d_soc_abs"] = oof["d_soc"].abs()
    print(f"样本 {len(oof)} / {oof['car'].nunique()} 车；整体 MAE={oof['abs_err'].mean():.3f} Ah")

    res = {"n_sessions": int(len(oof)), "n_cars": int(oof["car"].nunique()),
           "overall": regression_metrics(oof["capacity"], oof["pred_xgb"]),
           "overall_mae_pp": float(oof["abs_err_pp"].mean())}

    # ---------- 1. SOC 跨度 → 最小可用片段 ----------
    soc_bins = [20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100]
    g_soc = binned(oof, "d_soc_abs", soc_bins)
    res["by_soc_span"] = g_soc[["label", "n", "mae", "p90"]].to_dict(orient="records")
    print("\n按 SOC 跨度：")
    print(g_soc[["label", "n", "mae", "p90"]].round(3).to_string(index=False))

    # 达到目标精度的最小跨度（用分箱中位跨度做插值）
    targets = [1.0, 1.5, 2.0]  # SOH 百分点
    min_span = {}
    x = g_soc["med_x"].values
    y_pp = g_soc["mae"].values / ref_mean * 100
    for t in targets:
        ok = np.where(y_pp <= t)[0]
        if ok.size:
            i = ok[0]
            if i == 0:
                min_span[f"{t}pp"] = float(x[0])
            else:
                # 线性插值到阈值交点
                x0, x1 = x[i - 1], x[i]
                y0, y1 = y_pp[i - 1], y_pp[i]
                min_span[f"{t}pp"] = float(x0 + (t - y0) * (x1 - x0) / (y1 - y0))
        else:
            min_span[f"{t}pp"] = None
    res["min_span_for_target"] = min_span
    print(f"\n达到目标精度所需最小 SOC 跨度: {min_span}")

    # ---------- 2. 片段质量 ----------
    q_bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    g_q = binned(oof, "quality", q_bins, fmt="{:.2f}")
    res["by_quality"] = g_q[["label", "n", "mae", "p90"]].to_dict(orient="records")

    # ---------- 3. 里程 ----------
    ms_bins = [0, 2e4, 5e4, 8e4, 1.2e5, 2e5, 3e5]
    g_ms = binned(oof, "mileage", ms_bins)
    res["by_mileage"] = g_ms[["label", "n", "mae", "p90"]].to_dict(orient="records")

    # ---------- 4. 温度 ----------
    t_bins = [-20, 0, 10, 20, 30, 40, 60]
    g_t = binned(oof, "t_mean", t_bins)
    res["by_temperature"] = g_t[["label", "n", "mae", "p90"]].to_dict(orient="records")

    # ---------- 5. 厂商分块 ----------
    g_ds = oof.groupby("dataset").agg(
        n=("abs_err", "size"), mae=("abs_err", "mean"),
        mae_pp=("abs_err_pp", "mean"), d_soc_med=("d_soc_abs", "median"),
        cars=("car", "nunique")).reset_index()
    res["by_dataset"] = g_ds.round(4).to_dict(orient="records")
    print("\n按分块：")
    print(g_ds.round(3).to_string(index=False))

    # ---------- 6. 最差车辆 ----------
    pc = oof.groupby("car").agg(n=("abs_err", "size"), mae=("abs_err", "mean"),
                                mae_pp=("abs_err_pp", "mean"),
                                d_soc_med=("d_soc_abs", "median"),
                                ms_max=("mileage", "max")).reset_index()
    pc = pc.sort_values("mae_pp", ascending=False)
    res["worst_cars"] = pc.head(10).round(3).to_dict(orient="records")
    print("\n最差 10 辆车（按 SOH 误差）：")
    print(pc.head(10).round(2).to_string(index=False))

    # ---------- 图 16：分层误差 ----------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))
    ax = axes[0, 0]
    ax.plot(range(len(g_soc)), g_soc["mae"] / ref_mean * 100, "o-", color=C_MAIN)
    ax.set_xticks(range(len(g_soc)), g_soc["label"], rotation=45, fontsize=8)
    ax.set_xlabel("SOC 跨度（%）")
    ax.set_ylabel("MAE（SOH 百分点）")
    ax.set_title("(a) 片段越长越准")
    for t, c in zip([1.0, 2.0], ["#2e8b57", C_ACC]):
        ax.axhline(t, color=c, ls=":", lw=1)
        ax.text(0.1, t + 0.03, f"{t} pp 目标", fontsize=7.5, color=c)

    ax = axes[0, 1]
    ax.plot(range(len(g_q)), g_q["mae"] / ref_mean * 100, "o-", color=C_ACC)
    ax.set_xticks(range(len(g_q)), g_q["label"], rotation=45, fontsize=8)
    ax.set_xlabel("片段质量分")
    ax.set_ylabel("MAE（SOH 百分点）")
    ax.set_title("(b) 质量分越高质量越好")

    ax = axes[1, 0]
    ax.plot(range(len(g_ms)), g_ms["mae"] / ref_mean * 100, "o-", color=C_RED)
    ax.set_xticks(range(len(g_ms)), [f"{v/1e4:.0f}" for v in g_ms["med_x"]], fontsize=8)
    ax.set_xlabel("里程（万 km，取分箱中位数）")
    ax.set_ylabel("MAE（SOH 百分点）")
    ax.set_title("(c) 高里程段误差是否变大")

    ax = axes[1, 1]
    ax.plot(range(len(g_t)), g_t["mae"] / ref_mean * 100, "o-", color=C_MAIN)
    ax.set_xticks(range(len(g_t)), g_t["label"], fontsize=8)
    ax.set_xlabel("平均温度（℃）")
    ax.set_ylabel("MAE（SOH 百分点）")
    ax.set_title("(d) 温度分层")
    fig.suptitle("图 16　误差分层分析（跨车外推协议 OOF）", fontsize=12)
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig16_误差分层.png"))

    # ---------- 图 17：误差-跨度散点 + 理论量化下限 ----------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter(oof["d_soc_abs"], oof["abs_err_pp"], s=4, alpha=0.15, color=C_MAIN,
               edgecolors="none", label="会话")
    ax.plot(g_soc["med_x"], g_soc["mae"] / ref_mean * 100, "o-", color=C_RED, lw=1.6,
            label="分箱 MAE")
    xs = np.linspace(20, 100, 100)
    ax.plot(xs, 100.0 / xs, "--", color=C_GRAY, lw=1.2,
            label="朴素量化参考 1/ΔSOC（未计与标签共源的抵消）")
    ax.set_xlabel("SOC 跨度（%）")
    ax.set_ylabel("SOH 误差（百分点）")
    ax.set_ylim(0, min(12, np.percentile(oof["abs_err_pp"], 99.5)))
    ax.set_title("图 17　误差随片段长度单调下降（参考线仅为量级参照）")
    ax.legend()
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig17_误差-跨度关系.png"))

    p = save_json(res, "error_analysis")
    print(f"\n写出 {p}")


if __name__ == "__main__":
    main()
