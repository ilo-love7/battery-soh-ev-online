# -*- coding: utf-8 -*-
"""03 数据体检与 EDA：规模、标签覆盖、容量-里程轨迹、SOH 分布、库仑计标度诊断。

输出：results/figures/fig01..fig05 + results/metrics/eda_summary.json
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.plotting import (C_ACC, C_GRAY, C_GREEN, C_MAIN, C_RED, PALETTE, plt,  # noqa: E402
                            savefig)
from evsoh.target import add_soh_labels  # noqa: E402


def main() -> None:
    cfg = load_config()
    fig_dir = cfg["paths"]["figures_dir"]
    win = pd.read_parquet(cfg["paths"]["window_index"])
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    print(f"窗口 {win.shape} / 会话 {ses.shape}")

    # ---------- 概况 ----------
    per_ds = []
    for ds, g in win.groupby("dataset"):
        s = ses[ses["dataset"] == ds]
        per_ds.append({
            "dataset": ds,
            "windows": int(len(g)),
            "sessions_all": int(g.groupby(["car", "segment"]).ngroups),
            "cars_all": int(g["car"].nunique()),
            "sessions_labeled": int(len(s)),
            "cars_labeled": int(s["car"].nunique()),
            "cap_cover": float(len(s) / max(g.groupby(["car", "segment"]).ngroups, 1)),
        })
    per_ds = pd.DataFrame(per_ds)
    print(per_ds.to_string(index=False))

    # ---------- SOH 标签 ----------
    ses_lab, lab_stats = add_soh_labels(ses, cfg)
    print("\nSOH 标签:", lab_stats)

    # ---------- 库仑计标度诊断 ----------
    ok = ses[(ses["capacity"] > 0) & (ses["d_soc"].abs() > 20) & (ses["q_per_soc"] > 0)].copy()
    ok["ratio"] = ok["q_per_soc"] / ok["capacity"]
    ratio_med = float(ok["ratio"].median())
    ratio_iqr = float(ok["ratio"].quantile(0.75) - ok["ratio"].quantile(0.25))

    # ---------- 图 1：规模与标签覆盖 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(per_ds))
    w = 0.38
    axes[0].bar(x - w / 2, per_ds["sessions_all"] / 1000, w, label="全部会话", color="#c9d6ea")
    axes[0].bar(x + w / 2, per_ds["sessions_labeled"] / 1000, w, label="有容量标签", color=C_MAIN)
    axes[0].set_xticks(x, [d.replace("battery_dataset", "分块") for d in per_ds["dataset"]])
    axes[0].set_ylabel("充电会话数（千）")
    axes[0].set_title("(a) 会话规模与容量标签覆盖")
    for i, r in per_ds.iterrows():
        axes[0].text(i + w / 2, r["sessions_labeled"] / 1000, f"{r['cap_cover']*100:.0f}%",
                     ha="center", va="bottom", fontsize=8, color=C_MAIN)
    axes[0].legend()

    axes[1].bar(x - w / 2, per_ds["cars_all"], w, label="全部车辆", color="#c9d6ea")
    axes[1].bar(x + w / 2, per_ds["cars_labeled"], w, label="有容量标签", color=C_ACC)
    axes[1].set_xticks(x, [d.replace("battery_dataset", "分块") for d in per_ds["dataset"]])
    axes[1].set_ylabel("车辆数")
    axes[1].set_title("(b) 车辆规模与标签覆盖")
    axes[1].legend()
    savefig(fig, os.path.join(fig_dir, "fig01_数据规模与标签覆盖.png"))

    # ---------- 图 2：容量-里程轨迹（按车容量标签覆盖度挑代表车） ----------
    cnt = ses[ses["capacity"] > 0].groupby("car").size().sort_values(ascending=False)
    picks = cnt.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=False)
    for ax, c in zip(axes.ravel(), picks):
        g = ses[(ses["car"] == c) & (ses["capacity"] > 0)].sort_values("mileage")
        ax.plot(g["mileage"] / 1e4, g["capacity"], "o-", ms=2.5, lw=0.9, color=C_MAIN)
        if len(g) >= 3 and g["mileage"].nunique() > 1:
            k, b = np.polyfit(g["mileage"], g["capacity"], 1)
            r = np.corrcoef(g["mileage"], g["capacity"])[0, 1]
            ax.plot(g["mileage"] / 1e4, k * g["mileage"] + b, "--", color=C_RED, lw=1)
            ax.set_title(f"车 {c}（n={len(g)}，r={r:.2f}）", fontsize=10)
        else:
            ax.set_title(f"车 {c}（n={len(g)}）", fontsize=10)
        ax.set_xlabel("里程（万 km）")
        ax.set_ylabel("容量（Ah）")
    fig.suptitle("图 2　容量标签随里程的衰减轨迹（代表性车辆）", fontsize=12)
    savefig(fig, os.path.join(fig_dir, "fig02_容量-里程轨迹.png"))

    # ---------- 图 3：SOH 分布 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(ses_lab["soh"], bins=50, color=C_MAIN, alpha=0.85)
    axes[0].axvline(1.0, color=C_RED, ls="--", lw=1, label="BOL 参考 = 1.0")
    axes[0].set_xlabel("SOH（相对每车 BOL 参考）")
    axes[0].set_ylabel("会话数")
    axes[0].set_title(f"(a) SOH 分布（{len(ses_lab):,} 条会话 / {ses_lab['car'].nunique()} 辆车）")
    axes[0].legend()

    for i, (ds, g) in enumerate(ses_lab.groupby("dataset")):
        axes[1].hist(g["soh"], bins=40, histtype="step", lw=1.6,
                     color=PALETTE[i], label=f"{ds.replace('battery_dataset','分块')}（n={len(g)}）")
    axes[1].set_xlabel("SOH")
    axes[1].set_ylabel("会话数")
    axes[1].set_title("(b) 按厂商分块")
    axes[1].legend()
    savefig(fig, os.path.join(fig_dir, "fig03_SOH分布.png"))

    # ---------- 图 4：单会话充电轨迹 ----------
    sample = ses_lab.sort_values("quality", ascending=False).iloc[0]
    wi = win[(win["dataset"] == sample["dataset"]) & (win["car"] == sample["car"]) &
             (win["segment"] == sample["segment"])].sort_values("t_start")
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.2))
    ax = axes[0, 0]
    ax.plot(np.arange(len(wi)), wi["soc_end"], "o-", ms=3, color=C_MAIN)
    ax.set_xlabel("窗口序号（每个 128 步 ≈ 1270 s）")
    ax.set_ylabel("窗口末端 SOC（%）")
    ax.set_title("(a) SOC 轨迹")
    ax = axes[0, 1]
    ax.plot(wi["soc_end"], wi["v_end"], "o-", ms=3, color=C_ACC)
    ax.set_xlabel("SOC（%）")
    ax.set_ylabel("平均单体电压（V）")
    ax.set_title("(b) 充电电压曲线")
    ax = axes[1, 0]
    ax.plot(wi["soc_end"], wi["i_abs_mean"], "o-", ms=3, color=C_GREEN)
    ax.set_xlabel("SOC（%）")
    ax.set_ylabel("平均充电电流（A）")
    ax.set_title("(c) 电流剖面")
    ax = axes[1, 1]
    ax.plot(wi["soc_end"], wi["t_max"], "o-", ms=3, color=C_RED)
    ax.set_xlabel("SOC（%）")
    ax.set_ylabel("最高温度（℃）")
    ax.set_title("(d) 温度剖面")
    fig.suptitle(f"图 4　一次完整充电会话的轨迹（车 {sample['car']} / 片段 {sample['segment']}，"
                 f"ΔSOC={sample['d_soc']:.1f}%）", fontsize=11)
    savefig(fig, os.path.join(fig_dir, "fig04_单会话轨迹.png"))

    # ---------- 图 5：库仑计标度诊断 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.scatter(ok["capacity"], ok["q_per_soc"], s=5, alpha=0.25, color=C_MAIN, edgecolors="none")
    lim = [ok["capacity"].min(), ok["capacity"].max()]
    ax.plot(lim, lim, "--", color=C_GRAY, lw=1, label="1:1")
    ax.plot(lim, [ratio_med * v for v in lim], "-", color=C_RED, lw=1.2,
            label=f"标度 {ratio_med:.2f}×")
    ax.set_xlabel("数据集容量标签（Ah）")
    ax.set_ylabel("原始库仑计 ΔQ/ΔSOC（Ah）")
    ax.set_title("(a) 原始库仑计与标签的系统性标度差")
    ax.legend()

    ax = axes[1]
    ax.hist(ok["ratio"], bins=60, color=C_ACC, alpha=0.85)
    ax.axvline(ratio_med, color=C_RED, ls="--", lw=1.2, label=f"中位数 {ratio_med:.2f}")
    ax.set_xlabel("原始库仑计 / 容量标签")
    ax.set_ylabel("会话数")
    ax.set_title(f"(b) 标度因子分布（IQR {ratio_iqr:.2f}）")
    ax.legend()
    savefig(fig, os.path.join(fig_dir, "fig05_库仑计标度诊断.png"))

    # ---------- 汇总 ----------
    summary = {
        "windows_total": int(len(win)),
        "windows_valid_rate": float(win["valid"].mean()),
        "sessions_all": int(win.groupby(["dataset", "car", "segment"]).ngroups),
        "cars_all": int(win["car"].nunique()),
        "sessions_labeled": int((ses["capacity"] > 0).sum()),
        "cars_labeled": int(ses.loc[ses["capacity"] > 0, "car"].nunique()),
        "per_dataset": per_ds.to_dict(orient="records"),
        "soh_label": lab_stats,
        "coulomb_scale_median": ratio_med,
        "coulomb_scale_iqr": ratio_iqr,
        "n_sessions_scale_diag": int(len(ok)),
        "d_soc_median": float(ses["d_soc"].median()),
        "quality_median": float(ses["quality"].median()),
    }
    p = save_json(summary, "eda_summary")
    print(f"\n写出 {p}")
    print(f"库仑计标度因子中位数 = {ratio_med:.3f}（IQR {ratio_iqr:.3f}），n={len(ok)}")


if __name__ == "__main__":
    main()
