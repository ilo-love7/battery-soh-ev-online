# -*- coding: utf-8 -*-
"""04 物理基线：库仑计容量估计（原始标度 / 训练集标度校准）与常数基线。

这一步的作用是把"不用机器学习能做到什么程度"钉死，后续所有模型都要与之对比。
标度因子 k 只在训练折上拟合（k = median(capacity / q_per_soc)），测试折完全不见标签。

输出：results/metrics/physics_baseline.json + fig06_物理基线.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import group_kfold_by_car, group_metrics, regression_metrics  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_RED, plt, savefig  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402


def main() -> None:
    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, lab = add_soh_labels(ses, cfg)
    df = df[df["q_per_soc"] > 0].copy()
    print(f"建模样本: {len(df)} 条会话 / {df['car'].nunique()} 辆车")

    cars = df["car"].values
    y_cap = df["capacity"].values
    y_soh = df["soh"].values
    q = df["q_per_soc"].values

    # 逐折：训练折拟合标度，测试折预测
    pred_raw = np.full(len(df), np.nan)
    pred_cal = np.full(len(df), np.nan)
    pred_const = np.full(len(df), np.nan)
    ks = []
    for tr, te in group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                     seed=cfg["splits"]["random_state"]):
        k = float(np.median(y_cap[tr] / q[tr]))
        ks.append(k)
        pred_raw[te] = q[te]
        pred_cal[te] = k * q[te]
        pred_const[te] = float(np.mean(y_cap[tr]))

    df["pred_raw"] = pred_raw
    df["pred_cal"] = pred_cal
    df["pred_const"] = pred_const
    # 容量 → SOH（用该车自己的 BOL 参考）
    for c in ("pred_raw", "pred_cal", "pred_const"):
        df[f"{c}_soh"] = df[c] / df["bol_ref"]

    out = {"n_sessions": int(len(df)), "n_cars": int(df["car"].nunique()),
           "scale_k_per_fold": ks, "scale_k_median": float(np.median(ks))}
    rows = []
    for name, cap_col, soh_col in [
        ("常数基线（训练集均值）", "pred_const", "pred_const_soh"),
        ("库仑计·原始标度", "pred_raw", "pred_raw_soh"),
        ("库仑计·训练集标度校准", "pred_cal", "pred_cal_soh"),
    ]:
        m_cap = regression_metrics(y_cap, df[cap_col])
        m_soh = regression_metrics(y_soh, df[soh_col])
        g_cap = group_metrics(df, "capacity", cap_col)
        rows.append({
            "model": name,
            "cap_rmse": m_cap["rmse"], "cap_mae": m_cap["mae"], "cap_bias": m_cap["bias"],
            "cap_r2": m_cap["r2"], "cap_mape": m_cap["mape_pct"],
            "soh_rmse_pp": m_soh["rmse"] * 100, "soh_mae_pp": m_soh["mae"] * 100,
            "soh_r2": m_soh["r2"],
            "car_rmse_mean": g_cap["car_rmse_mean"],
        })
    res = pd.DataFrame(rows)
    print("\n" + res.round(3).to_string(index=False))
    out["results"] = rows

    # 理论下限：SOC 量化误差引起的容量误差（1% SOC 量化 → 相对误差 1/ΔSOC）
    ds = df["d_soc"].abs().values
    out["quantization_floor"] = {
        "assumed_soc_resolution_pct": 1.0,
        "median_d_soc": float(np.median(ds)),
        "relative_err_at_median_span": float(1.0 / np.median(ds)),
        "relative_err_at_20pct_span": float(1.0 / 20.0),
        "relative_err_at_60pct_span": float(1.0 / 60.0),
    }

    # 图：预测 vs 真值
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    for ax, (name, col, color) in zip(axes, [
        ("常数基线", "pred_const", C_GRAY),
        ("库仑计·原始标度", "pred_raw", C_ACC),
        ("库仑计·训练集标度校准", "pred_cal", C_MAIN),
    ]):
        ax.scatter(y_cap, df[col], s=4, alpha=0.2, color=color, edgecolors="none")
        lim = [min(y_cap.min(), df[col].min()), max(y_cap.max(), df[col].max())]
        ax.plot(lim, lim, "--", color=C_RED, lw=1)
        m = regression_metrics(y_cap, df[col])
        ax.set_title(f"{name}\nRMSE={m['rmse']:.2f} Ah，MAPE={m['mape_pct']:.1f}%", fontsize=10)
        ax.set_xlabel("容量标签（Ah）")
        ax.set_ylabel("预测（Ah）")
    fig.suptitle("图 6　物理基线：库仑计容量估计（5 折按车分组）", fontsize=12)
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig06_物理基线.png"))

    p = save_json(out, "physics_baseline")
    print(f"\n写出 {p}")


if __name__ == "__main__":
    main()
