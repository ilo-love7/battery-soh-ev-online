# -*- coding: utf-8 -*-
"""19 模型族的天花板：换模型和换表示，到底哪个轴更值钱。

第三轮留了两条互相矛盾的线索（报告 §16.4）：
  - "换模型比换表示划算"：同一套手工特征，MLP 比 XGBoost 低 0.3%，
    而 XGB+MLP 等权融合低 3.8%；反过来，把表示从手工特征换成原始序列（CNN）差 26.8%。
  - 但这两条都是"两两对比"，没有把两个轴放在同一张网格上量过。
    于是"3.8%"到底是模型红利，还是仅仅一次幸运的方差抵消，说不清。

本脚本把两个轴铺成一张网格，在**同一套按车分组 5 折**里跑完：

  轴 1（模型族，5 档）：ElasticNet（线性）/ 随机森林（bagging）/
                        HistGradientBoosting（sklearn 提升）/ XGBoost（现主模型）/ MLP（torch）
  轴 2（表示，4 档）：full（114 维）/ no_coulomb / coulomb_only / curve_only

再补一条第三轴（**族内调参**，6 个变体，只在 full 上跑）：
  xgb_deep / xgb_shallow / xgb_reg / hgb_deep / mlp_wide / mlp_narrow

三轴一起才能回答"该往哪投人力"：
  如果族内调参的极差 < 跨族极差，说明**超参搜索的边际收益低于换族**；
  如果跨族极差 ≈ 跨表示极差，说明两者等价，选便宜的那个；
  如果最好的组合仍比现主模型好不到 1%，说明**这个任务已经饱和**，
  继续在模型上投入不如去改标签（§16.3 已指向标签噪声）。

输出
  results/metrics/model_frontier.json
  results/cache/model_frontier_oof.parquet
  results/figures/fig33_模型族与特征集网格.png
  results/figures/fig34_两个轴的收益对比.png
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import (group_kfold_by_car, group_metrics,  # noqa: E402
                            regression_metrics)
from evsoh.features import (ABLATIONS, FEATURE_GROUPS, active_features,  # noqa: E402
                            feature_columns)
from evsoh.models import build_model  # noqa: E402
from evsoh.nn import fit_torch, make_mlp, prep_tabular  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_PURPLE, C_RED, plt, savefig  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402

# 轴 2：四个特征集（顺序即图 33 的列顺序，从"全量"到"只剩曲线形状"）
# 注意：ABLATIONS 里的 "no_coulomb" 与 "curve_only" 在分组定义下是**同一个集合**
# （全量 − 库仑计 = 形状 + SOC 时长 + 工况上下文），实测两者 RMSE 逐位相同，
# 所以这里用 "no_context"（全量 − 27 个工况上下文特征）作为第四档，避免一列白占。
FEATURE_SETS = ["full", "no_coulomb", "coulomb_only", "no_context"]
FS_LABELS = {
    "full": "全量 114 维",
    "no_coulomb": "去库仑计",
    "coulomb_only": "只库仑计",
    "no_context": "去工况上下文",
}
# 轴 1：五个模型族（顺序即图 33 的行顺序，从"最弱"到"最强"）
FAMILIES = ["elasticnet", "rf", "hgb", "xgb", "mlp"]
FAM_LABELS = {
    "elasticnet": "ElasticNet（线性）",
    "rf": "随机森林（bagging）",
    "hgb": "HistGB（sklearn 提升）",
    "xgb": "XGBoost（现主模型）",
    "mlp": "MLP（torch）",
}
# 轴 3：族内调参变体（只在 full 上跑，用来量"调参"这个轴的幅度）
TUNING = ["xgb_deep", "xgb_shallow", "xgb_reg", "hgb_deep", "mlp_wide", "mlp_narrow"]
TUNING_LABELS = {
    "xgb_deep": "XGB 更深（d=9, 1200 树, lr=.03）",
    "xgb_shallow": "XGB 更浅（d=3, 600 树, lr=.06）",
    "xgb_reg": "XGB 强正则（λ=20, 子采样 .7）",
    "hgb_deep": "HistGB 更多轮（700）",
    "mlp_wide": "MLP 更宽（256）",
    "mlp_narrow": "MLP 更窄（64）",
}
MLP_WIDTH = {"mlp": 128, "mlp_wide": 256, "mlp_narrow": 64}
MLP_SEEDS = 2          # MLP 的种子噪声约 ±0.004 Ah，2 个种子平均后不再主导比较
MLP_EPOCHS = 50


def _cols_of(fs: str) -> list:
    """特征集 → 列名。no_context 不在 ABLATIONS 里，单独算。"""
    if fs == "no_context":
        return [c for c in feature_columns() if c not in FEATURE_GROUPS["context"]]
    return ABLATIONS[fs]


def _derive(res: dict, fs_list: list) -> dict:
    """从 grid/tuning_res 推出"三条轴"的极差与最优组合（主流程与合并模式共用）。"""
    base = res["grid"]["full"]["xgb"]["rmse"]
    for fs in fs_list:
        for fam in FAMILIES:
            res["grid"][fs][fam]["delta_vs_base_pct"] = (
                res["grid"][fs][fam]["rmse"] / base - 1.0) * 100.0
    for t in TUNING:
        res["tuning_res"][t]["delta_vs_base_pct"] = (
            res["tuning_res"][t]["rmse"] / base - 1.0) * 100.0

    def _range(vals: dict) -> dict:
        v = np.array(list(vals.values()), dtype=float)
        return {"values": {k: float(x) for k, x in vals.items()},
                "best": float(v.min()), "worst": float(v.max()),
                "range": float(v.max() - v.min()),
                "range_pct": float((v.max() / v.min() - 1.0) * 100.0)}

    model_axis = _range({f: res["grid"]["full"][f]["rmse"] for f in FAMILIES})
    feat_axis = _range({fs: res["grid"][fs]["xgb"]["rmse"] for fs in fs_list})
    tune_axis = {
        "xgb": _range({"xgb": res["grid"]["full"]["xgb"]["rmse"],
                       **{t: res["tuning_res"][t]["rmse"]
                          for t in ("xgb_deep", "xgb_shallow", "xgb_reg")}}),
        "hgb": _range({"hgb": res["grid"]["full"]["hgb"]["rmse"],
                       "hgb_deep": res["tuning_res"]["hgb_deep"]["rmse"]}),
        "mlp": _range({"mlp": res["grid"]["full"]["mlp"]["rmse"],
                       **{t: res["tuning_res"][t]["rmse"]
                          for t in ("mlp_wide", "mlp_narrow")}}),
    }
    all_cells = {f"{fs}|{f}": res["grid"][fs][f]["rmse"]
                 for fs in fs_list for f in FAMILIES}
    all_cells.update({f"full|{t}": res["tuning_res"][t]["rmse"] for t in TUNING})
    best_key = min(all_cells, key=all_cells.get)
    res["axes"] = {"model_axis_full": model_axis, "feature_axis_xgb": feat_axis,
                   "tuning_axis": tune_axis,
                   "all_cells": {k: float(v) for k, v in all_cells.items()},
                   "best": {"key": best_key, "rmse": float(all_cells[best_key]),
                            "delta_vs_base_pct": float((all_cells[best_key] / base - 1) * 100)}}
    return res


def _mlp_pred(Xtr, ytr, Xte, width: int, seed: int) -> np.ndarray:
    """MLP 训练：与脚本 16/18 同一套预处理与早停，只是层宽可调。"""
    A, B = prep_tabular(Xtr.to_numpy(dtype=float), Xte.to_numpy(dtype=float))
    ps = [fit_torch(lambda: make_mlp(A.shape[1], width), A, ytr, B,
                    seed=seed + s, epochs=MLP_EPOCHS, patience=8)
          for s in range(MLP_SEEDS)]
    return np.mean(ps, axis=0)


def fit_config(key: str, Xtr, ytr, Xte, cfg: dict, seed: int = 42):
    """跑一个配置，返回 (测试集预测, 训练+预测耗时秒)。"""
    t0 = time.time()

    if key == "elasticnet":
        m = build_model("elasticnet", cfg["models"], alpha=0.01, l1_ratio=0.3)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)

    elif key == "rf":
        # 400 棵/depth16 单折要 66 s，300 棵/depth12 只要 42 s，RMSE 差 0.6%
        # （0.809 → 0.803，fold 1 实测），网格里用后者
        m = build_model("rf", cfg["models"], n_estimators=300, max_depth=12,
                        min_samples_leaf=5)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)

    elif key == "hgb":
        m = build_model("hgb", None, max_iter=400, learning_rate=0.06,
                        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)

    elif key == "hgb_deep":
        m = build_model("hgb", None, max_iter=700, learning_rate=0.03,
                        max_leaf_nodes=63, min_samples_leaf=10, l2_regularization=2.0)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)

    elif key == "xgb":
        m = build_model("xgb", cfg["models"])
        m.fit(Xtr, ytr)
        p = m.predict(Xte)

    elif key == "xgb_deep":
        m = build_model("xgb", cfg["models"], n_estimators=1200, max_depth=9,
                        learning_rate=0.03, min_child_weight=8)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)

    elif key == "xgb_shallow":
        m = build_model("xgb", cfg["models"], n_estimators=600, max_depth=3,
                        learning_rate=0.06)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)

    elif key == "xgb_reg":
        m = build_model("xgb", cfg["models"], reg_lambda=20.0, subsample=0.70,
                        colsample_bytree=0.60)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)

    elif key in MLP_WIDTH:
        p = _mlp_pred(Xtr, ytr, Xte, MLP_WIDTH[key], seed)

    else:
        raise ValueError(f"未知配置: {key}")

    return np.asarray(p, dtype=float), time.time() - t0


def main() -> None:
    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, lab = add_soh_labels(ses, cfg)
    df = df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)
    print(f"样本 {len(df)} 会话 / {df['car'].nunique()} 车", flush=True)

    # 每个特征集各自剔除"全缺失列"，否则插补器会静默跳过、模型维度对不上
    fs_cols = {}
    for fs in FEATURE_SETS:
        cols = active_features(df, _cols_of(fs))
        fs_cols[fs] = cols
        print(f"  特征集 {fs:13s} {len(cols):3d} 维", flush=True)

    y = df["capacity"].to_numpy(dtype=float)
    cars = df["car"].to_numpy()
    ref_mean = float(df["bol_ref"].mean())
    d_soc = df["d_soc"].to_numpy(dtype=float)

    folds = list(group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                    seed=cfg["splits"]["random_state"]))
    if os.environ.get("SMOKE"):
        folds = folds[:1]
        print("SMOKE 模式：只跑第 1 折", flush=True)

    # 待跑配置：网格 5×4 + 调参 6（只在 full 上）。
    # ONLY_SET=<set> 时只跑该列（用于"补跑一列再合并"，见 --assemble）。
    only = os.environ.get("ONLY_SET")
    run_sets = [only] if only else FEATURE_SETS
    if only:
        print(f"ONLY_SET={only}：只跑该特征集的 {len(FAMILIES)} 个模型", flush=True)
    jobs = [(fs, fam) for fs in run_sets for fam in FAMILIES]
    if not only:
        jobs += [("full", t) for t in TUNING]
    n_jobs_total = len(jobs) * len(folds)
    print(f"共 {len(jobs)} 个配置 × {len(folds)} 折 = {n_jobs_total} 次拟合", flush=True)

    preds = {f"{fs}|{k}": np.full(len(df), np.nan) for fs, k in jobs}
    secs = {f"{fs}|{k}": [] for fs, k in jobs}
    t0 = time.time()

    for fi, (tr, te) in enumerate(folds):
        print(f"[折 {fi + 1}/{len(folds)}] 训练 {len(tr)} / 测试 {len(te)}", flush=True)
        for fs, key in jobs:
            cols = fs_cols[fs]
            X = df[cols].astype(float)
            p, sec = fit_config(key, X.iloc[tr], y[tr], X.iloc[te], cfg,
                                seed=42 + fi * 10)
            preds[f"{fs}|{key}"][te] = p
            secs[f"{fs}|{key}"].append(sec)
            print(f"    {fs:12s} {key:12s} {sec:5.1f}s", flush=True)
        print(f"    折 {fi + 1} 完成，累计 {time.time() - t0:.0f}s", flush=True)

    # ---------------- 汇总 ----------------
    res = {
        "n_sessions": int(len(df)), "n_cars": int(df["car"].nunique()),
        "label": lab, "feature_sets": {k: len(v) for k, v in fs_cols.items()},
        "families": FAMILIES, "tuning": TUNING,
        "grid": {}, "tuning_res": {},
        "note": ("全部配置在 P2 按车分组 5 折内完成，折划分与主实验一致；"
                 "MLP 用 2 个种子平均（种子噪声 ±0.004 Ah），其余单种子；"
                 "耗时含训练+预测，单位秒，仅用于相对比较（同一台机器、同一折）。"),
    }

    def _metrics(p):
        mm = regression_metrics(y, p)
        gm = group_metrics(pd.DataFrame({"capacity": y, "p": p, "car": cars}),
                           "capacity", "p")
        return {**mm, **gm, "soh_rmse_pp": mm["rmse"] / ref_mean * 100.0}

    for fs in run_sets:
        res["grid"][fs] = {}
        for fam in FAMILIES:
            k = f"{fs}|{fam}"
            m = _metrics(preds[k])
            m["sec"] = float(np.mean(secs[k]))
            res["grid"][fs][fam] = m
    if not only:                  # 调参变体只在主流程里跑，补跑模式没有这些 key
        for t in TUNING:
            k = f"full|{t}"
            m = _metrics(preds[k])
            m["sec"] = float(np.mean(secs[k]))
            res["tuning_res"][t] = m

    if only:                      # 补跑模式：只落该列的原始结果，交给 --assemble 合并
        out = os.path.join(cfg["paths"]["cache_dir"], f"_frontier_{only}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res["grid"], f, ensure_ascii=False, indent=2, default=float)
        pd.DataFrame({"car": cars, "d_soc": d_soc, "capacity": y,
                      **{f"pred_{k.replace('|', '_')}": preds[k] for k in preds}}).to_parquet(
            os.path.join(cfg["paths"]["cache_dir"],
                         f"model_frontier_oof_{only}.parquet"), index=False)
        print(f"\n补跑完成，已写入 {out}（用 --assemble 合并）", flush=True)
        return

    res = _derive(res, FEATURE_SETS)
    base = res["grid"]["full"]["xgb"]["rmse"]
    model_axis = res["axes"]["model_axis_full"]
    feat_axis = res["axes"]["feature_axis_xgb"]
    tune_axis = res["axes"]["tuning_axis"]
    all_cells = res["axes"]["all_cells"]
    best_key = res["axes"]["best"]["key"]

    print(f"\n参照（full + XGB）：RMSE {base:.4f} Ah", flush=True)
    print(f"  跨模型族极差（full）：{model_axis['range']:.4f} Ah "
          f"({model_axis['range_pct']:.1f}%)", flush=True)
    print(f"  跨特征集极差（XGB） ：{feat_axis['range']:.4f} Ah "
          f"({feat_axis['range_pct']:.1f}%)", flush=True)
    for f, d in tune_axis.items():
        print(f"  族内调参极差（{f:4s}）：{d['range']:.4f} Ah ({d['range_pct']:.1f}%)",
              flush=True)
    print(f"  最优组合：{best_key}  RMSE {all_cells[best_key]:.4f} Ah "
          f"({(all_cells[best_key] / base - 1) * 100:+.1f}%)", flush=True)

    save_json(res, "model_frontier")
    pd.DataFrame({"car": cars, "d_soc": d_soc, "capacity": y,
                  **{f"pred_{k.replace('|', '_')}": preds[k] for k in preds}}).to_parquet(
        os.path.join(cfg["paths"]["cache_dir"], "model_frontier_oof.parquet"), index=False)

    fig_grid(res, os.path.join(cfg["paths"]["figures_dir"], "fig33_模型族与特征集网格.png"))
    fig_axes(res, os.path.join(cfg["paths"]["figures_dir"], "fig34_两个轴的收益对比.png"))
    print(f"\n完成，用时 {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 图 33
def fig_grid(res: dict, path: str) -> None:
    """图 33：RMSE 热力图（行=模型族，列=特征集），带与主模型的相对变化。"""
    base = res["grid"]["full"]["xgb"]["rmse"]
    M = np.array([[res["grid"][fs][f]["rmse"] for fs in FEATURE_SETS] for f in FAMILIES])
    D = (M / base - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    vmax = 25.0        # 截断：no_coulomb 的 +82% 会饱和，其余格子才有对比度
    im = ax.imshow(D, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = f"{M[i, j]:.3f}\n{D[i, j]:+.1f}%"
            col = "white" if abs(D[i, j]) > vmax * 0.55 else "#222222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.4, color=col)

    ax.set_xticks(range(len(FEATURE_SETS)))
    ax.set_xticklabels([FS_LABELS[f] for f in FEATURE_SETS], fontsize=8.8)
    ax.set_yticks(range(len(FAMILIES)))
    ax.set_yticklabels([FAM_LABELS[f] for f in FAMILIES], fontsize=8.8)
    ax.set_xlabel("Feature set (representation axis)")
    ax.set_title("RMSE (Ah) and change vs. main model (full features + XGBoost)",
                 fontsize=10.5, pad=10)
    ax.grid(visible=False)
    # 主模型格子描边，便于定位参照点
    ax.add_patch(plt.Rectangle((0 - 0.5, FAMILIES.index("xgb") - 0.5), 1, 1,
                               fill=False, ec=C_MAIN, lw=2.2))
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label("Δ RMSE vs. main model (%), clipped at ±25", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    savefig(fig, path)


# ---------------------------------------------------------------- 图 34
def fig_axes(res: dict, path: str) -> None:
    """图 34：三条轴的极差对比 + 精度-耗时权衡。

    两个细节是必需的，否则图会失去信息量：
      - 跨特征集的范围被 no_coulomb（+82%）拉爆，x 轴必须截断并把出界点标出来，
        否则 0.81–1.0 这段真正要看的东西会被压成一条线；
      - 耗时是跨 5 个数量级的，必须用对数轴并显式给刻度。
    """
    ax_r = res["axes"]
    base = res["grid"]["full"]["xgb"]["rmse"]
    XMAX = 1.12

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- (a) 每个族的 4 个特征集结果连成一段；调参变体单独一组 ----
    rows, labels = [], []
    ypos = 0
    for fam in FAMILIES:
        vals = [res["grid"][fs][fam]["rmse"] for fs in FEATURE_SETS]
        rows.append((ypos, vals, C_MAIN if fam == "xgb" else C_GRAY))
        labels.append(FAM_LABELS[fam])
        ypos += 1
    sep_y = ypos - 0.5
    for t in ("xgb_deep", "xgb_shallow", "xgb_reg"):
        rows.append((ypos, [res["tuning_res"][t]["rmse"]], C_PURPLE))
        labels.append(TUNING_LABELS[t])
        ypos += 1

    for y_, vals, col in rows:
        lo, hi = min(vals), max(vals)
        axes[0].plot([min(lo, XMAX), min(hi, XMAX)], [y_, y_], color=col, lw=5,
                     alpha=0.32, solid_capstyle="round", zorder=2)
        inb = [v for v in vals if v <= XMAX]
        axes[0].scatter(inb, [y_] * len(inb), s=26, color=col, zorder=3,
                        edgecolor="white", linewidth=0.6)
        if hi > XMAX:      # 出界点：画三角形 + 真实数值，避免把轴拉到 1.5
            axes[0].scatter([XMAX], [y_], marker=">", s=34, color=col, zorder=4)
            axes[0].text(XMAX + 0.008, y_, f"{hi:.2f}", va="center", fontsize=7.4,
                         color=col)

    axes[0].axvline(base, color=C_RED, ls="--", lw=1.2, zorder=1)
    axes[0].text(base + 0.007, -0.55, f"main model {base:.3f}", color=C_RED,
                 fontsize=7.6, va="center")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(labels, fontsize=8.0)
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(rows) - 0.42, -0.95)      # 给顶部 "main model" 标注留位置
    axes[0].set_xlim(0.79, XMAX + 0.045)
    axes[0].set_xlabel("RMSE (Ah)")
    axes[0].set_title("(a) Spread across the representation axis", fontsize=10, pad=8)
    axes[0].grid(axis="y", visible=False)
    axes[0].axhline(sep_y, color="#cccccc", lw=0.9, ls=":")
    axes[0].legend(handles=[
        plt.Line2D([], [], color=C_GRAY, lw=5, alpha=0.32, label="spread over 4 feature sets"),
        plt.Line2D([], [], color=C_MAIN, lw=5, alpha=0.32, label="XGBoost (main model)"),
        plt.Line2D([], [], color=C_PURPLE, lw=5, alpha=0.32,
                   label="XGB tuning variants (full)")],
        loc="lower right", fontsize=7.2, handlelength=1.6, borderpad=0.35)

    # ---- (b) 精度 vs 耗时 ----
    fam_col = {"elasticnet": C_GRAY, "rf": C_ACC, "hgb": C_PURPLE,
               "xgb": C_MAIN, "mlp": C_RED}
    YMIN, YMAX = 0.79, 1.05
    off = []
    for fs in FEATURE_SETS:
        for fam in FAMILIES:
            d = res["grid"][fs][fam]
            if d["rmse"] > YMAX:
                off.append((d["sec"], d["rmse"], fam))
                continue
            axes[1].scatter(d["sec"], d["rmse"], s=30, color=fam_col[fam],
                            alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
    for t in TUNING:
        d = res["tuning_res"][t]
        fam = "xgb" if t.startswith("xgb") else ("mlp" if t.startswith("mlp") else "hgb")
        if d["rmse"] <= YMAX:
            axes[1].scatter(d["sec"], d["rmse"], s=30, color=fam_col[fam], alpha=0.85,
                            edgecolor="white", linewidth=0.6, zorder=3, marker="D")
    for sec, rmse, fam in off:      # 出界点标在顶边
        axes[1].scatter(sec, YMAX, marker="^", s=34, color=fam_col[fam], zorder=4)
        axes[1].text(sec, YMAX + 0.006, f"{rmse:.2f}", ha="center", fontsize=7.0,
                     color=fam_col[fam])
    axes[1].axhline(base, color=C_RED, ls="--", lw=1.1)
    axes[1].set_xscale("log")
    axes[1].set_xticks([3, 5, 10, 20, 50])
    axes[1].set_xticklabels(["3", "5", "10", "20", "50"])
    axes[1].xaxis.set_minor_formatter(plt.NullFormatter())
    axes[1].set_ylim(YMIN, YMAX + 0.035)
    axes[1].set_xlabel("Fit time per fold (s, log scale)")
    axes[1].set_ylabel("RMSE (Ah)")
    axes[1].set_title("(b) Accuracy vs. cost", fontsize=10, pad=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c,
                          label=FAM_LABELS[f].split("（")[0]) for f, c in fam_col.items()]
    axes[1].legend(handles=handles, loc="center left", fontsize=7.4, ncol=1,
                   handletextpad=0.4, borderpad=0.35, framealpha=0.95)
    fig.tight_layout()
    savefig(fig, path)


if __name__ == "__main__":
    if "--assemble" in sys.argv:
        assemble()
    elif "--replot" in sys.argv:
        _cfg = load_config()
        with open(os.path.join(_cfg["paths"]["metrics_dir"], "model_frontier.json"),
                  encoding="utf-8") as f:
            _res = json.load(f)
        fig_grid(_res, os.path.join(_cfg["paths"]["figures_dir"], "fig33_模型族与特征集网格.png"))
        fig_axes(_res, os.path.join(_cfg["paths"]["figures_dir"], "fig34_两个轴的收益对比.png"))
        print("重绘完成")
    else:
        main()
