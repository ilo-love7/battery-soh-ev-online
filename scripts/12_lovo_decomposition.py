# -*- coding: utf-8 -*-
"""12 D2 深化：严格留一车（LOVO）+ 误差分解 + CORAL 域对齐。

上一版用"按车分组 5 折"当主协议，每辆车只被留出一次，但同折内 38 辆车一起评估，
掩盖了两件事：
  ① 逐车误差的长尾有多长（最差的车差到什么程度）；
  ② 跨车误差到底由什么决定 —— 是车本身（化学体系/容量/里程，不可干预），
     还是片段属性（SOC 跨度/质量/温度，可以靠"等一段更好的充电"来改善）。

本脚本三件事：
  1. 191 折留一车（LOVO，最严格协议），得到逐车 RMSE 分布与最差车辆清单；
  2. 以逐车 RMSE 为因变量做误差分解：片段属性块 vs 车身份块各解释多少方差；
  3. 补做 CORAL（二阶统计对齐）作为域自适应的第二次尝试 —— 上一版的逐维 z-score
     对齐把绝对电量尺度抹掉了（0.88 → 2.34 Ah），CORAL 保协方差结构，值得单独验证。

输出
    results/metrics/lovo_decomposition.json
    results/cache/lovo_predictions.parquet
    results/figures/fig20_留一车误差分布与分解.png
    results/figures/fig21_域对齐三种方式.png
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
from evsoh.evaluate import regression_metrics  # noqa: E402
from evsoh.features import active_features, feature_columns  # noqa: E402
from evsoh.models import build_model  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_GREEN, C_MAIN, C_PURPLE, C_RED, plt, savefig  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402

N_WORKERS = 8


# ---------------------------------------------------------------- LOVO
_G: dict = {}


def _init_worker(X, y, cars, cfg):
    _G["X"], _G["y"], _G["cars"], _G["cfg"] = X, y, cars, cfg


def _fit_one(car: int):
    X, y, cars, cfg = _G["X"], _G["y"], _G["cars"], _G["cfg"]
    te = np.where(cars == car)[0]
    tr = np.where(cars != car)[0]
    mdl = build_model("xgb", cfg["models"], n_jobs=1)
    mdl.fit(X.iloc[tr], y[tr])
    return te, mdl.predict(X.iloc[te])


def lovo_fit(X: pd.DataFrame, y: np.ndarray, cars: np.ndarray, cfg: dict,
             workers: int = N_WORKERS) -> np.ndarray:
    """逐车留一：191 次训练。

    用**进程池**而不是线程池：XGBoost 的 QuantileDMatrix 在同一进程内并发构造会偶发
    崩溃（实测 160/191 折时抛 "Data must has at least 1 column"），进程隔离最稳。
    """
    import multiprocessing as mp

    pred = np.full(len(y), np.nan)
    uniq = np.unique(cars).astype(int)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    done = 0
    with ctx.Pool(workers, initializer=_init_worker, initargs=(X, y, cars, cfg)) as pool:
        for te, p in pool.imap_unordered(_fit_one, uniq.tolist()):
            pred[te] = p
            done += 1
            if done % 40 == 0:
                print(f"    LOVO {done}/{len(uniq)} 车  ({time.time() - t0:.0f}s)", flush=True)
    print(f"    LOVO 完成：{len(uniq)} 折 / {time.time() - t0:.0f}s", flush=True)
    return pred


# ---------------------------------------------------------------- 误差分解
ATTR_COLS = ["d_soc_med", "quality_med", "t_mean_med", "i_abs_med", "n_sess"]
CAR_COLS = ["mileage_med", "bol_ref", "soh_spread", "is_ds3", "is_ds2"]


def car_table(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """每辆车一行：逐车 RMSE + 片段属性 + 车身份属性。"""
    d = df.copy()
    d["pred"] = pred
    d["sq_err"] = (d["pred"] - d["capacity"]) ** 2
    rows = []
    for car, g in d.groupby("car"):
        rmse = float(np.sqrt(g["sq_err"].mean()))
        rows.append({
            "car": car,
            "n_sess": len(g),
            "rmse": rmse,
            "mae": float(np.mean(np.abs(g["pred"] - g["capacity"]))),
            "bias": float(np.mean(g["pred"] - g["capacity"])),
            "d_soc_med": float(g["d_soc"].median()),
            "quality_med": float(g["quality"].median()),
            "t_mean_med": float(g["t_mean"].median()),
            "i_abs_med": float(g["i_abs_mean"].median()),
            "mileage_med": float(g["mileage"].median()),
            "bol_ref": float(g["bol_ref"].iloc[0]),
            "soh_spread": float(g["soh"].max() - g["soh"].min()),
            "is_ds2": float(g["dataset"].iloc[0] == "battery_dataset2"),
            "is_ds3": float(g["dataset"].iloc[0] == "battery_dataset3"),
        })
    return pd.DataFrame(rows)


def block_r2(t: pd.DataFrame, cols: list) -> float:
    """某个属性块对逐车 RMSE 的解释力（OLS 的 R²）。

    **必须标准化后再解**。原始尺度下 mileage_med(~1e5) 与 n_sess(~1e2) 相差三个量级，
    设计矩阵条件数约 2.7e6，sklearn 的 LinearRegression 在这种病态条件下退化成次优解
    （实测合并块 SSR=13.22，而真正的最小二乘 SSR=6.71，R² 从 0.67 被压到 0.36）。
    这里统一做 z-score + lstsq，既稳定又让系数大小可比。
    """
    X = t[cols].to_numpy(dtype=float)
    y = t["rmse"].to_numpy(dtype=float)
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if m.sum() < len(cols) + 3:
        return float("nan")
    X, y = X[m], y[m]
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - X.mean(0)) / sd])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    res = y - Z @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - float((res ** 2).sum()) / sst)


# ---------------------------------------------------------------- CORAL
def _coral_src_to_tgt(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """CORAL：把源域特征白化后按目标域协方差重着色（二阶统计对齐，无需目标标签）。"""
    ms, mt = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])

    def _isqrt(C):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-12, None)
        return V @ np.diag(w ** -0.5) @ V.T

    def _sqrt(C):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-12, None)
        return V @ np.diag(w ** 0.5) @ V.T

    A = _isqrt(Cs) @ _sqrt(Ct)
    return (Xs - ms) @ A + mt


def coral_transfer(df: pd.DataFrame, feats: list, y: np.ndarray, cfg: dict) -> dict:
    """留一厂商：raw / z-score 分布对齐 / CORAL 三种方式对比。

    注意：CORAL 作用在协方差矩阵上，**必须先插补**——带 NaN 的矩阵求协方差会全 NaN，
    变换后整份特征变成 NaN，模型直接报 "0 features"。

    这里**不用** SimpleImputer：它会静默丢弃全缺失列（如 ic_raw_02，充电起始电压普遍
    高于 3.15 V），transform 出来的列数比 feats 少，构造 DataFrame 时列数对不上直接报
    "Shape of passed values is (X, N), indices imply (X, M)"。手动按训练集中位数插补，
    全缺失列退化为常数 0，列数始终等于 feats。
    """
    out = {}
    for dom in sorted(df["dataset"].unique()):
        te = (df["dataset"] == dom).to_numpy()
        tr = ~te
        Xtr = df.loc[tr, feats].astype(float)
        Xte = df.loc[te, feats].astype(float)

        med = Xtr.median().fillna(0.0)          # 全缺失列 → 0，保留列结构
        A_tr = Xtr.fillna(med)
        A_te = Xte.fillna(med)

        # raw
        mdl = build_model("xgb", cfg["models"])
        mdl.fit(A_tr, y[tr])
        p_raw = mdl.predict(A_te)

        # z-score 分布对齐：把目标域的每一维搬到源域的均值/尺度上（只用特征，不用标签）。
        # 注意这与"共享标准化"不同——共享标准化对树模型是恒等变换（分裂只看分位点），
        # 真正会改变预测的是"把测试集分布搬到训练集分布"这件事，§7 做的就是后者。
        mu_s, sd_s = A_tr.mean(axis=0), A_tr.std(axis=0).replace(0, 1.0)
        mu_t, sd_t = A_te.mean(axis=0), A_te.std(axis=0).replace(0, 1.0)
        A_te_al = (A_te - mu_t) / sd_t * sd_s + mu_s
        mdl_z = build_model("xgb", cfg["models"])
        mdl_z.fit(A_tr, y[tr])
        p_z = mdl_z.predict(A_te_al)

        # CORAL：把源域特征搬到目标域的二阶统计上（无监督，用目标域无标签特征）
        A_tr_c = _coral_src_to_tgt(A_tr.to_numpy(), A_te.to_numpy())
        A_te_c = _coral_src_to_tgt(A_te.to_numpy(), A_te.to_numpy())
        mdl_c = build_model("xgb", cfg["models"])
        mdl_c.fit(pd.DataFrame(A_tr_c, columns=feats), y[tr])
        p_c = mdl_c.predict(pd.DataFrame(A_te_c, columns=feats))

        out[dom] = {
            "n_test": int(te.sum()), "n_cars_test": int(df.loc[te, "car"].nunique()),
            "raw": regression_metrics(y[te], p_raw),
            "zscore": regression_metrics(y[te], p_z),
            "coral": regression_metrics(y[te], p_c),
        }
        print(f"  [{dom}] raw {out[dom]['raw']['rmse']:.3f} | z-score "
              f"{out[dom]['zscore']['rmse']:.3f} | CORAL {out[dom]['coral']['rmse']:.3f} Ah", flush=True)
    return out


# ---------------------------------------------------------------- 图
def fig_lovo(t: pd.DataFrame, decomp: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    ax.hist(t["rmse"], bins=28, color=C_MAIN, alpha=0.85, edgecolor="white")
    med = t["rmse"].median()
    ax.axvline(med, color=C_RED, ls="--", lw=1.4, label=f"median {med:.3f} Ah")
    ax.axvline(t["rmse"].max(), color=C_ACC, ls=":", lw=1.4,
               label=f"worst {t['rmse'].max():.2f} Ah")
    ax.set_xlabel("Per-car RMSE (Ah), leave-one-car-out")
    ax.set_ylabel("Number of cars")
    ax.set_title(f"Per-car error has a long tail (n={len(t)} cars)", pad=8)
    ax.legend(loc="upper right", fontsize=8.5)

    ax = axes[1]
    blocks = [("Snippet attrs\n(SOC span, Q, T, C-rate)", decomp["r2_attr"], C_MAIN),
              ("Car identity\n(mileage, BOL, chem.)", decomp["r2_car"], C_ACC),
              ("Both", decomp["r2_both"], C_PURPLE)]
    xs = np.arange(len(blocks))
    bars = ax.bar(xs, [b[1] * 100 for b in blocks], color=[b[2] for b in blocks], width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in blocks], fontsize=7.6)
    ax.set_ylabel("Variance of per-car RMSE explained (%)")
    ax.set_ylim(0, 100)
    ax.set_title("What drives cross-car error", pad=8)
    for b, (_l, v, _c) in zip(bars, blocks):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 2, f"{v * 100:.0f}%",
                ha="center", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    fig.suptitle("Strict leave-one-vehicle-out: error distribution and its drivers",
                 fontsize=10.5, y=1.03)
    savefig(fig, path)


def fig_domain(coral: dict, path: str) -> None:
    doms = list(coral.keys())
    methods = [("raw", "No alignment", C_GRAY), ("zscore", "z-score align (test→train)", C_ACC),
               ("coral", "CORAL (2nd order)", C_MAIN)]
    w = 0.26
    xs = np.arange(len(doms))

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for i, (key, label, col) in enumerate(methods):
        vals = [coral[d][key]["rmse"] for d in doms]
        bars = ax.bar(xs + (i - 1) * w, vals, width=w, color=col, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([d.replace("battery_dataset", "dataset ") for d in doms])
    ax.set_ylabel("RMSE on held-out split (Ah)")
    ax.set_title("Distribution matching erases the absolute charge scale:\n"
                 "both alignments hurt, CORAL least so",
                 pad=8, fontsize=9.5)
    ax.set_ylim(0, max(c[k]["rmse"] for c in coral.values() for k, _l, _c in methods) * 1.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8.2,
              frameon=False)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    savefig(fig, path)


def main() -> None:
    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, _lab = add_soh_labels(ses, cfg)
    df = df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)
    feats = active_features(df, feature_columns())
    X = df[feats].astype(float)
    y = df["capacity"].values.astype(float)
    cars = df["car"].values
    ref_mean = float(df["bol_ref"].mean())
    print(f"样本 {len(df)} 会话 / {df['car'].nunique()} 车 / {len(feats)} 特征", flush=True)

    print("[1/3] 严格留一车（191 折）…", flush=True)
    cache_lovo = os.path.join(cfg["paths"]["cache_dir"], "lovo_predictions.parquet")
    if os.environ.get("REUSE_LOVO") == "1" and os.path.exists(cache_lovo):
        cached = pd.read_parquet(cache_lovo)
        if len(cached) == len(df) and np.isfinite(cached["pred_lovo"]).all():
            pred = cached["pred_lovo"].to_numpy(float)
            print(f"  复用已有 LOVO 预测（{len(pred)} 条，跳过 191 折重训）", flush=True)
        else:
            pred = lovo_fit(X, y, cars, cfg)
    else:
        pred = lovo_fit(X, y, cars, cfg)
    m = regression_metrics(y, pred)
    m["soh_rmse_pp"] = m["rmse"] / ref_mean * 100.0
    print(f"  LOVO 总体：RMSE={m['rmse']:.4f} Ah  MAE={m['mae']:.4f}  R2={m['r2']:.4f}  "
          f"→ {m['soh_rmse_pp']:.2f} pp", flush=True)

    pd.DataFrame({"car": df["car"], "segment": df["segment"], "capacity": y,
                  "pred_lovo": pred, "dataset": df["dataset"]}).to_parquet(cache_lovo, index=False)

    t = car_table(df, pred)
    worst = t.nlargest(8, "rmse")[["car", "n_sess", "rmse", "d_soc_med", "mileage_med", "is_ds3"]]
    print("  最差 8 辆车：")
    for _, r in worst.iterrows():
        print(f"    car {int(r['car']):4d}  n={int(r['n_sess']):3d}  RMSE={r['rmse']:.2f}  "
              f"ΔSOC中位={r['d_soc_med']:.0f}%  里程={r['mileage_med'] / 1000:.0f}k"
              + ("  (ds3)" if r["is_ds3"] else ""))

    print("[2/3] 误差分解…", flush=True)
    # 三个属性块必须在**同一完整样本**上比较，否则 R² 不可比（样本不同）
    cols_all = ATTR_COLS + CAR_COLS
    tt = t.dropna(subset=cols_all + ["rmse"]).reset_index(drop=True)
    r2_attr, r2_car, r2_both = block_r2(tt, ATTR_COLS), block_r2(tt, CAR_COLS), block_r2(tt, cols_all)
    decomp = {
        "r2_attr": r2_attr, "r2_car": r2_car, "r2_both": r2_both,
        "attr_marginal_pp": 100.0 * (r2_both - r2_car),
        "car_marginal_pp": 100.0 * (r2_both - r2_attr),
        "n_cars_used": int(len(tt)),
        "coef_attr": {}, "spearman": {},
    }
    Xa = tt[ATTR_COLS].to_numpy(float)
    ya = tt["rmse"].to_numpy(float)
    sd_a = Xa.std(0)
    sd_a[sd_a == 0] = 1.0
    Za = np.column_stack([np.ones(len(Xa)), (Xa - Xa.mean(0)) / sd_a])
    beta_a, *_ = np.linalg.lstsq(Za, ya, rcond=None)
    for c, b in zip(ATTR_COLS, beta_a[1:]):
        decomp["coef_attr"][c] = float(b)   # 标准化系数：RMSE 变化多少 Ah / 该特征 1 个标准差
    for c in cols_all:
        s_ = tt[[c, "rmse"]].dropna()
        if len(s_) > 20:
            decomp["spearman"][c] = float(s_[c].corr(s_["rmse"], method="spearman"))
    print(f"  片段属性 {decomp['r2_attr'] * 100:.1f}% / 车身份 {decomp['r2_car'] * 100:.1f}%"
          f" / 合并 {decomp['r2_both'] * 100:.1f}%（n={len(tt)} 车）", flush=True)
    print(f"  边际贡献：片段属性 {decomp['attr_marginal_pp']:+.1f}pp，"
          f"车身份 {decomp['car_marginal_pp']:+.1f}pp", flush=True)

    print("[3/3] 留一厂商 + CORAL…", flush=True)
    cache_coral = os.path.join(cfg["paths"]["metrics_dir"], "lovo_domain_transfer.json")
    if os.environ.get("REUSE_CORAL") == "1" and os.path.exists(cache_coral):
        with open(cache_coral, "r", encoding="utf-8") as f:
            coral = json.load(f)
        print("  复用已有 CORAL 结果", flush=True)
    else:
        coral = coral_transfer(df, feats, y, cfg)
        with open(cache_coral, "w", encoding="utf-8") as f:
            json.dump(coral, f, ensure_ascii=False, indent=2)

    out = {
        "n_sessions": int(len(df)), "n_cars": int(df["car"].nunique()),
        "lovo_overall": m,
        "lovo_car_rmse": {"mean": float(t["rmse"].mean()), "median": float(t["rmse"].median()),
                          "p90": float(t["rmse"].quantile(0.9)), "max": float(t["rmse"].max()),
                          "min": float(t["rmse"].min())},
        "worst_cars": worst.to_dict(orient="records"),
        "decomposition": decomp,
        "domain_transfer": coral,
        "note": ("LOVO 用 191 折逐车留出（每折单线程 XGB，进程池并行）；"
                 "误差分解以逐车 RMSE 为因变量做 OLS，R² 即该属性块解释的方差比例。"
                 "三个块必须在同一完整样本上拟合才可比，且需先标准化——"
                 "原始尺度下条件数 2.7e6，sklearn LinearRegression 会退化成次优解。"),
    }
    save_json(out, "lovo_decomposition")

    fig_lovo(t, decomp, os.path.join(cfg["paths"]["figures_dir"], "fig20_留一车误差分布与分解.png"))
    fig_domain(coral, os.path.join(cfg["paths"]["figures_dir"], "fig21_域对齐三种方式.png"))
    print("完成")


if __name__ == "__main__":
    main()
