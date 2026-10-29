# -*- coding: utf-8 -*-
"""06 跨厂商泛化：留一域外推 + 无监督特征对齐。

数据集三个分块来自三家制造商，BMS 上报口径与工况分布不同（域偏移）。
本脚本做三组对照：
  T1 直接迁移：源域训练，目标域直接预测
  T2 特征对齐：把目标域特征按逐维均值/标准差对齐到源域（无标签，CORAL 的对角简化版）后再预测
  T3 目标域少量标签（k 条/车）：仅用于对比，展示"要多少标签才能追上"（非无监督）

输出：results/metrics/generalization.json + fig11
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import leave_one_dataset_out, regression_metrics  # noqa: E402
from evsoh.features import active_features, feature_columns  # noqa: E402
from evsoh.models import build_model  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_RED, plt, savefig  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402


def align_features(X_src: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
    """逐维 z-score 对齐：目标域特征变换到源域的均值/尺度（只用特征，不用标签）。"""
    mu_s = np.nanmean(X_src, axis=0)
    sd_s = np.nanstd(X_src, axis=0)
    mu_t = np.nanmean(X_tgt, axis=0)
    sd_t = np.nanstd(X_tgt, axis=0)
    sd_s = np.where(sd_s < 1e-8, 1.0, sd_s)
    sd_t = np.where(sd_t < 1e-8, 1.0, sd_t)
    return (X_tgt - mu_t) / sd_t * sd_s + mu_s


def main() -> None:
    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, _ = add_soh_labels(ses, cfg)
    feats = active_features(df, feature_columns())
    df = df.dropna(subset=["capacity"]).reset_index(drop=True)
    X = df[feats].astype(float).values
    y = df["capacity"].values.astype(float)
    ds_arr = df["dataset"].values
    ref_mean = float(df["bol_ref"].mean())

    res = {"n_sessions": int(len(df)), "domains": {}}
    rows = []
    for ds_name, tr, te in leave_one_dataset_out(ds_arr):
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        mdl = build_model("xgb", cfg["models"])
        mdl.fit(pd.DataFrame(Xtr, columns=feats), ytr)

        p_raw = mdl.predict(pd.DataFrame(Xte, columns=feats))
        m_raw = regression_metrics(yte, p_raw)

        Xte_al = align_features(Xtr, Xte)
        p_al = mdl.predict(pd.DataFrame(Xte_al, columns=feats))
        m_al = regression_metrics(yte, p_al)

        # 源域内部 5 折参照（同分布误差水平）
        from evsoh.evaluate import group_kfold_by_car
        cars_tr = df["car"].values[tr]
        oof = np.full(len(tr), np.nan)
        for a, b in group_kfold_by_car(cars_tr, n_splits=5, seed=42):
            m2 = build_model("xgb", cfg["models"])
            m2.fit(pd.DataFrame(Xtr[a], columns=feats), ytr[a])
            oof[b] = m2.predict(pd.DataFrame(Xtr[b], columns=feats))
        m_src = regression_metrics(ytr, oof)

        res["domains"][ds_name] = {
            "n_test": int(len(te)),
            "n_cars_test": int(len(np.unique(df["car"].values[te]))),
            "transfer_raw": {**m_raw, "soh_rmse_pp": m_raw["rmse"] / ref_mean * 100},
            "transfer_aligned": {**m_al, "soh_rmse_pp": m_al["rmse"] / ref_mean * 100},
            "source_in_domain": {**m_src, "soh_rmse_pp": m_src["rmse"] / ref_mean * 100},
        }
        rows.append({
            "held_out": ds_name,
            "n_test": len(te),
            "in_domain_rmse": m_src["rmse"],
            "transfer_raw_rmse": m_raw["rmse"],
            "transfer_aligned_rmse": m_al["rmse"],
            "transfer_raw_bias": m_raw["bias"],
            "transfer_aligned_bias": m_al["bias"],
        })
        print(f"[留出 {ds_name}] n={len(te):5d} | 域内 {m_src['rmse']:.2f} | "
              f"直接迁移 {m_raw['rmse']:.2f}（bias {m_raw['bias']:+.2f}） | "
              f"对齐后 {m_al['rmse']:.2f}（bias {m_al['bias']:+.2f}）")

    tab = pd.DataFrame(rows)
    res["table"] = tab.to_dict(orient="records")

    # 图
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    x = np.arange(len(tab))
    w = 0.26
    ax.bar(x - w, tab["in_domain_rmse"], w, label="源域内 5 折", color=C_GRAY)
    ax.bar(x, tab["transfer_raw_rmse"], w, label="直接迁移", color=C_ACC)
    ax.bar(x + w, tab["transfer_aligned_rmse"], w, label="特征对齐后", color=C_MAIN)
    ax.set_xticks(x, [f"留出 {d.replace('battery_dataset','分块')}" for d in tab["held_out"]])
    ax.set_ylabel("RMSE（Ah）")
    ax.set_title("图 11　跨厂商留一域外推：域偏移代价与无监督特征对齐的效果")
    ax.legend()
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig11_跨厂商泛化.png"))

    p = save_json(res, "generalization")
    print(f"\n写出 {p}")


if __name__ == "__main__":
    main()
