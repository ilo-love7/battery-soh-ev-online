# -*- coding: utf-8 -*-
"""05 SOH/容量回归建模与两档评测协议。

协议
  P1 同车随机划分：每车内部随机 20% 做测试 —— 乐观上界（该车历史已见过）
  P2 按车分组 5 折：同一辆车只出现在训练或测试一侧 —— 跨车外推（主结论）

模型递进：弹性网络 → 随机森林 → XGBoost，另加 4 组特征消融。

输出
  results/metrics/train_results.json
  results/metrics/oof_predictions.parquet   （P2 的 OOF 预测，供 06/07/08/09 复用）
  results/figures/fig07..fig10
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import (group_kfold_by_car, group_metrics, random_split_within_car,  # noqa: E402
                            regression_metrics)
from evsoh.features import ABLATIONS, active_features, feature_columns  # noqa: E402
from evsoh.models import build_model, feature_importance  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_RED, PALETTE, plt, savefig  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402


def to_soh_pp(rmse_ah: float, ref_ah: float) -> float:
    """容量误差（Ah）换算成 SOH 百分点。"""
    return rmse_ah / ref_ah * 100.0


def main() -> None:
    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, lab = add_soh_labels(ses, cfg)
    feats = active_features(df, feature_columns())
    df = df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)
    print(f"建模样本 {len(df)} 会话 / {df['car'].nunique()} 车；特征 {len(feats)} 个")

    X = df[feats].astype(float)
    y = df["capacity"].values.astype(float)
    cars = df["car"].values
    ref_mean = float(df["bol_ref"].mean())

    results = {"n_sessions": int(len(df)), "n_cars": int(df["car"].nunique()),
               "n_features": len(feats), "label": lab, "protocols": {}}

    # ---------------- P2：按车分组 5 折（主协议） ----------------
    model_names = ["elasticnet", "rf", "xgb"]
    oof = {m: np.full(len(df), np.nan) for m in model_names}
    t0 = time.time()
    folds = list(group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                    seed=cfg["splits"]["random_state"]))
    for mi, name in enumerate(model_names):
        for fi, (tr, te) in enumerate(folds):
            mdl = build_model(name, cfg["models"])
            mdl.fit(X.iloc[tr], y[tr])
            oof[name][te] = mdl.predict(X.iloc[te])
        m = regression_metrics(y, oof[name])
        g = group_metrics(pd.DataFrame({"capacity": y, "p": oof[name], "car": cars}), "capacity", "p")
        results["protocols"].setdefault("P2_group_by_car", {})[name] = {
            **m, **g, "soh_rmse_pp": to_soh_pp(m["rmse"], ref_mean)}
        print(f"[P2] {name:11s} RMSE={m['rmse']:.3f} Ah  MAE={m['mae']:.3f}  "
              f"R2={m['r2']:.4f}  → SOH RMSE={to_soh_pp(m['rmse'], ref_mean):.2f} pp  "
              f"({time.time()-t0:.0f}s)")

    # ---------------- P1：同车随机（乐观上界） ----------------
    tr, te = random_split_within_car(cars, test_frac=cfg["splits"]["test_frac"],
                                     seed=cfg["splits"]["random_state"])
    p1 = {}
    for name in model_names:
        mdl = build_model(name, cfg["models"])
        mdl.fit(X.iloc[tr], y[tr])
        pred = mdl.predict(X.iloc[te])
        m = regression_metrics(y[te], pred)
        p1[name] = {**m, "soh_rmse_pp": to_soh_pp(m["rmse"], ref_mean)}
        print(f"[P1] {name:11s} RMSE={m['rmse']:.3f} Ah  MAE={m['mae']:.3f}  R2={m['r2']:.4f}")
    results["protocols"]["P1_random_within_car"] = p1

    # ---------------- 消融（XGBoost，P2 协议） ----------------
    abl = {}
    for aname, cols in ABLATIONS.items():
        cols = [c for c in cols if c in feats]
        pred = np.full(len(df), np.nan)
        for tr2, te2 in folds:
            mdl = build_model("xgb", cfg["models"])
            mdl.fit(X[cols].iloc[tr2], y[tr2])
            pred[te2] = mdl.predict(X[cols].iloc[te2])
        m = regression_metrics(y, pred)
        abl[aname] = {**m, "n_features": len(cols), "soh_rmse_pp": to_soh_pp(m["rmse"], ref_mean)}
        print(f"[消融] {aname:13s} ({len(cols):3d} 特征) RMSE={m['rmse']:.3f} Ah  R2={m['r2']:.4f}")
    results["ablation_xgb_P2"] = abl

    # ---------------- 落盘 OOF ----------------
    out = df[["dataset", "car", "segment", "mileage", "capacity", "bol_ref", "soh",
              "d_soc", "quality", "q_span", "q_current", "q_temp", "i_abs_mean", "t_mean"]].copy()
    for m in model_names:
        out[f"pred_{m}"] = oof[m]
    # 物理基线（标度校准）也放进 OOF，便于统一对比
    k_all = []
    phys = np.full(len(df), np.nan)
    for tr2, te2 in folds:
        k = float(np.median(y[tr2] / df["q_per_soc"].values[tr2]))
        k_all.append(k)
        phys[te2] = k * df["q_per_soc"].values[te2]
    out["pred_physics"] = phys
    out["scale_k"] = float(np.median(k_all))
    out.to_parquet(os.path.join(cfg["paths"]["metrics_dir"], "oof_predictions.parquet"), index=False)
    results["physics_scale_k"] = float(np.median(k_all))

    # ---------------- 图 7：预测 vs 真值（P2） ----------------
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    for ax, name in zip(axes, model_names):
        m = results["protocols"]["P2_group_by_car"][name]
        ax.scatter(y, oof[name], s=4, alpha=0.2, color=C_MAIN, edgecolors="none")
        lim = [min(y.min(), np.nanmin(oof[name])), max(y.max(), np.nanmax(oof[name]))]
        ax.plot(lim, lim, "--", color=C_RED, lw=1)
        ax.set_title(f"{name}\nRMSE={m['rmse']:.2f} Ah / {m['soh_rmse_pp']:.2f} pp，R²={m['r2']:.3f}",
                     fontsize=10)
        ax.set_xlabel("容量标签（Ah）")
        ax.set_ylabel("预测（Ah）")
    fig.suptitle("图 7　跨车外推协议（按车分组 5 折）下的预测-真值对照", fontsize=12)
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig07_预测真值对照.png"))

    # ---------------- 图 8：误差分布 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, name in enumerate(model_names):
        err = oof[name] - y
        axes[0].hist(err, bins=60, histtype="step", lw=1.6, color=PALETTE[i],
                     label=f"{name}（σ={np.nanstd(err):.2f} Ah）")
    axes[0].set_xlabel("残差（预测 − 真值，Ah）")
    axes[0].set_ylabel("会话数")
    axes[0].set_title("(a) 残差分布")
    axes[0].legend()

    # 误差 vs SOC 跨度
    bins = [20, 30, 40, 50, 60, 70, 100]
    df["_bin"] = pd.cut(df["d_soc"].abs(), bins=bins)
    for i, name in enumerate(model_names):
        g = pd.DataFrame({"e": np.abs(oof[name] - y), "b": df["_bin"]}).groupby("b", observed=True)["e"]
        axes[1].plot([str(c) for c in g.median().index], g.median().values, "o-",
                     color=PALETTE[i], label=name)
    axes[1].set_xlabel("SOC 跨度区间（%）")
    axes[1].set_ylabel("|误差| 中位数（Ah）")
    axes[1].set_title("(b) 误差随 SOC 跨度变化")
    axes[1].legend()
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig08_残差与跨度.png"))

    # ---------------- 图 9：特征重要性 ----------------
    mdl = build_model("xgb", cfg["models"])
    mdl.fit(X, y)
    imp = feature_importance(mdl, feats)
    order = np.argsort(imp)[::-1][:20]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(range(len(order))[::-1], imp[order], color=C_ACC)
    ax.set_yticks(range(len(order))[::-1], [feats[i] for i in order], fontsize=8)
    ax.set_xlabel("XGBoost 重要性（增益归一化）")
    ax.set_title("图 9　Top-20 特征重要性（全量数据拟合，仅作解释用）")
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig09_特征重要性.png"))
    results["top_features"] = [{"feature": feats[i], "importance": float(imp[i])} for i in order]

    # ---------------- 图 10：消融 ----------------
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = list(abl.keys())
    vals = [abl[n]["rmse"] for n in names]
    bars = ax.bar(names, vals, color=[C_GRAY, C_ACC, C_MAIN, C_RED])
    for b, v, n in zip(bars, vals, names):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}\n({abl[n]['n_features']}维)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("RMSE（Ah）")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("图 10　特征族消融（XGBoost，按车分组 5 折）")
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig10_特征消融.png"))

    p = save_json(results, "train_results")
    print(f"\n写出 {p}")

    # ---------------- 保存可部署模型（全量数据拟合） ----------------
    import joblib

    model_dir = os.path.join(cfg["paths"]["results_dir"], "models")
    os.makedirs(model_dir, exist_ok=True)
    bundle = {
        "model": mdl,
        "features": feats,
        "target": "capacity_Ah",
        "bol_ref_mean": ref_mean,
        "physics_scale_k": results["physics_scale_k"],
        "train_metrics": results["protocols"]["P2_group_by_car"]["xgb"],
        "n_train": int(len(df)),
        "n_cars": int(df["car"].nunique()),
    }
    mpath = os.path.join(model_dir, "xgb_capacity.joblib")
    joblib.dump(bundle, mpath)
    print(f"模型已保存 {mpath}")


if __name__ == "__main__":
    main()
