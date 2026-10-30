# -*- coding: utf-8 -*-
"""11 D1 深化：IC（dQ/dV）峰特征提取与消融。

深化思路文档 A2-D1 要求"把手工统计特征升级为 dQ/dV 增量容量特征"。上一版只把 IC
按 50 mV 固定网格分箱（25 个绝对电量 + 25 个归一化形状），分箱值本身可被模型使用，
但**没有把 IC 的物理语义显式表达出来**：老化在 IC 上的表现是"峰高下降 + 峰位右移"，
这两件事只有做过峰检测才能直接读出。

本脚本从已有的 ic_raw_00..24 反推 IC 曲线（3.00–4.25 V，50 mV/格），做峰检测，
提取最多 3 个峰的峰位/峰高/峰宽/峰面积，再跑三组特征组合的 P2 五折对比：

    full         114 维（原有）
    peak_add     full + 峰特征
    peak_replace full 里 25 维 ic_raw 换成峰特征（检验"分箱是否可以用峰替代"）

输出
    results/cache/ic_peaks.parquet      峰特征（38744 × 21）
    results/metrics/ic_peaks.json       峰统计 + 三组消融
    results/figures/fig18_IC峰随SOH演化.png
    results/figures/fig19_IC峰特征消融.png
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import group_kfold_by_car, regression_metrics  # noqa: E402
from evsoh.features import active_features, feature_columns  # noqa: E402
from evsoh.models import build_model  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_PURPLE, C_RED, plt, savefig  # noqa: E402
from evsoh.session import V_GRID  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402

IC_BINS = [f"ic_raw_{k:02d}" for k in range(len(V_GRID) - 1)]
IC_CENTERS = (V_GRID[:-1] + V_GRID[1:]) / 2.0   # 3.025 … 4.225 V
MAX_PEAKS = 3
PEAK_FEATURES = (
    [f"ic_pk{i}_{s}" for i in range(MAX_PEAKS) for s in ("v", "h", "w", "a")]
    + [f"ic_npk{i}_h" for i in range(MAX_PEAKS)]
    + ["ic_pk_n", "ic_pk_vw", "ic_pk_ratio21", "ic_pk_cov_lo", "ic_pk_cov_hi"]
)


def _fill_contiguous(y: np.ndarray) -> np.ndarray:
    """把首个到末个有效点之间的内部缺失线性插补（覆盖区间本身是连续的）。"""
    idx = np.where(np.isfinite(y))[0]
    if idx.size < 2:
        return np.array([])
    lo, hi = idx[0], idx[-1]
    seg = y[lo:hi + 1].astype(float)
    miss = ~np.isfinite(seg)
    if miss.any() and (~miss).sum() >= 2:
        seg[miss] = np.interp(np.where(miss)[0], np.where(~miss)[0], seg[~miss])
    return seg


def peak_features(row: pd.Series) -> dict:
    """单条会话的 IC 峰特征。峰按高度降序，取前 MAX_PEAKS 个。"""
    y_all = row[IC_BINS].to_numpy(dtype=float)
    out = {k: np.nan for k in PEAK_FEATURES}
    seg = _fill_contiguous(y_all)
    if seg.size < 4:
        return out

    idx = np.where(np.isfinite(y_all))[0]
    out["ic_pk_cov_lo"] = float(IC_CENTERS[idx[0]])
    out["ic_pk_cov_hi"] = float(IC_CENTERS[idx[-1]])

    total = float(np.nansum(seg))
    if not np.isfinite(total) or total <= 0:
        return out

    # 峰检测：显著度取最大值的 12%，相邻峰至少隔 2 格（100 mV）
    peaks, props = find_peaks(seg, prominence=0.12 * np.nanmax(seg), distance=2)
    if peaks.size == 0:
        peaks = np.array([int(np.nanargmax(seg))])
        props = {"prominences": np.array([np.nanmax(seg)])}

    order = peaks[np.argsort(seg[peaks])[::-1]][:MAX_PEAKS]
    out["ic_pk_n"] = float(len(peaks))

    heights = []
    for i, p in enumerate(order):
        h = float(seg[p])
        half = h / 2.0
        # 峰宽：向两侧扩展到低于半高为止（以格数计，×0.05 换成 V）
        left = p
        while left > 0 and seg[left - 1] >= half:
            left -= 1
        right = p
        while right < seg.size - 1 and seg[right + 1] >= half:
            right += 1
        out[f"ic_pk{i}_v"] = float(IC_CENTERS[idx[0] + p])
        out[f"ic_pk{i}_h"] = h
        out[f"ic_pk{i}_w"] = float((right - left) * 0.05)
        out[f"ic_pk{i}_a"] = float(seg[left:right + 1].sum())   # Ah
        out[f"ic_npk{i}_h"] = h / total
        heights.append(h)

    if len(heights) >= 2 and heights[0] > 0:
        out["ic_pk_ratio21"] = heights[1] / heights[0]
    # 峰高加权平均峰位（"整体峰位"，比单个峰位更稳）
    v = np.array([out[f"ic_pk{i}_v"] for i in range(len(heights))], dtype=float)
    hh = np.array(heights, dtype=float)
    out["ic_pk_vw"] = float(np.sum(v * hh) / np.sum(hh))
    return out


def add_peak_columns(df: pd.DataFrame) -> pd.DataFrame:
    peaks = df.apply(peak_features, axis=1, result_type="expand")
    peaks.columns = list(PEAK_FEATURES)
    return pd.concat([df.reset_index(drop=True), peaks.reset_index(drop=True)], axis=1)


def run_p2(df: pd.DataFrame, cols: list, cfg: dict, y: np.ndarray, cars: np.ndarray,
           folds, tag: str) -> tuple[dict, np.ndarray]:
    X = df[cols].astype(float)
    pred = np.full(len(df), np.nan)
    t0 = time.time()
    for tr, te in folds:
        mdl = build_model("xgb", cfg["models"])
        mdl.fit(X.iloc[tr], y[tr])
        pred[te] = mdl.predict(X.iloc[te])
    m = regression_metrics(y, pred)
    m["n_features"] = len(cols)
    print(f"  [{tag:13s}] {len(cols):3d} 特征  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
          f"R2={m['r2']:.4f}  ({time.time() - t0:.0f}s)")
    return m, pred


def fig_ic_evolution(df: pd.DataFrame, path: str) -> None:
    """主图：IC 曲线按 SOH 分档叠加 + 主峰峰位随 SOH 的偏移。"""
    df = df.copy()
    # 固定 SOH 分档（qcut 会把 0.60–1.08 的分布压成 91–100%，看不出对比）
    edges = [0.0, 0.85, 0.92, 0.97, 1.02, 1.20]
    names = ["<85%", "85–92%", "92–97%", "97–102%", ">102%"]
    df["band"] = pd.cut(df["soh"] * 100, bins=[e * 100 for e in edges], labels=names)
    colors = [C_RED, C_ACC, C_GRAY, C_MAIN, C_PURPLE]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    for name, col in zip(names, colors):
        sub = df[df["band"] == name]
        if len(sub) < 100:
            continue
        curve = sub[IC_BINS].mean().to_numpy()
        ax.plot(IC_CENTERS, curve, color=col, lw=1.9,
                label=f"SOH {name}  (n={len(sub):,})")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Incremental capacity dQ/dV (Ah per 50 mV)")
    ax.set_title("IC curve flattens as the pack ages", pad=8)
    ax.set_xlim(3.55, 4.25)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    # 右图按 SOH 分档画柱状图（而不是把 SOH 当连续横轴再反转），
    # 这样左右两图的分档顺序一致，读者不用来回换算；峰位数值直接标在柱内。
    sub = df.dropna(subset=["ic_pk0_v"])
    order = [n for n in names if int((sub["band"] == n).sum()) > 50]
    cnt = np.array([int((sub["band"] == n).sum()) for n in order], dtype=float)
    h = np.array([sub.loc[sub["band"] == n, "ic_pk0_h"].mean() for n in order])
    e = np.array([1.96 * sub.loc[sub["band"] == n, "ic_pk0_h"].std() / np.sqrt(c)
                  for n, c in zip(order, cnt)])
    pos = [sub.loc[sub["band"] == n, "ic_pk0_v"].mean() for n in order]
    xs = np.arange(len(order))
    ax.bar(xs, h, yerr=e, capsize=3, width=0.62, color=C_MAIN,
           error_kw={"lw": 1.1, "ecolor": C_GRAY})
    for x, v, p in zip(xs, h, pos):
        ax.text(x, v + e[list(xs).index(x)] + 0.5, f"{v:.1f}", ha="center", fontsize=7.8)
        ax.text(x, 0.7, f"{p:.3f} V", ha="center", fontsize=7.2, color="white")
    ax.set_xticks(xs, order, fontsize=8)
    ax.set_xlabel("SOH band")
    ax.set_ylabel("Dominant peak height (Ah per 50 mV)")
    ax.set_title("Height tracks SOH, position drifts only 0.09 V\n"
                 "(96.8% of sessions yield exactly one peak)", pad=8, fontsize=9.3)
    ax.set_ylim(0, float(np.nanmax(h + e)) * 1.16)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    fig.suptitle("Incremental-capacity (dQ/dV) features: 25 bins at 50 mV, peak-detected",
                 fontsize=10.5, y=1.03)
    savefig(fig, path)


def fig_ablation(res: dict, path: str) -> None:
    """三组特征组合的对比。用基线参照点图而不是从 0 起的柱状图：
    三档 RMSE 都在 0.82–0.89，从 0 起柱子高度几乎一样，"加了峰特征也没用"看不出来。
    """
    order = [("full", "All features"), ("peak_add", "All + peak features"),
             ("peak_replace", "Bins replaced by peaks")]
    keys = [k for k, _b in order]
    vals = [res[k]["rmse"] for k in keys]
    base = vals[0]
    ys = np.arange(len(vals))
    cols = [C_MAIN, C_PURPLE, C_ACC]

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.hlines(ys, base, vals, color=cols, lw=2.2, alpha=0.75)
    for i, (y_, v, c) in enumerate(zip(ys, vals, cols)):
        ax.plot([v], [y_], marker="o", ms=8, color=c, zorder=3)
        txt = f"{v:.4f}  ({res[keys[i]]['n_features']} feats)"
        txt += "" if i == 0 else f"   ({(v / base - 1) * 100:+.1f}%)"
        if i == len(vals) - 1:
            ax.text(v + 0.006, y_ - 0.26, txt, ha="right", va="center", fontsize=8.2, color=c)
        else:
            ax.text(v + 0.006, y_, txt, ha="left", va="center", fontsize=8.2, color=c)
    ax.axvline(base, color=C_GRAY, ls="--", lw=1.1, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([b for _k, b in order], fontsize=8.8)
    ax.invert_yaxis()
    ax.set_ylim(len(vals) - 0.45, -0.55)
    ax.set_xlim(base - 0.022, max(vals) + 0.028)
    ax.set_xlabel("RMSE on unseen cars (Ah), P2 5-fold by car")
    ax.set_title("Explicit IC peaks add nothing over the raw 50 mV bins\n"
                 "(dashed line = all-114-feature baseline)", pad=8, fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    savefig(fig, path)


def main() -> None:
    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, lab = add_soh_labels(ses, cfg)
    df = df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)
    print(f"会话 {len(df)} / 车 {df['car'].nunique()}")

    t0 = time.time()
    df = add_peak_columns(df)
    print(f"峰特征提取完成（{time.time() - t0:.0f}s）")

    peaks = df[["car", "segment"] + PEAK_FEATURES].copy()
    peaks.to_parquet(os.path.join(cfg["paths"]["cache_dir"], "ic_peaks.parquet"), index=False)

    base = active_features(df, feature_columns())
    peak_ok = [c for c in PEAK_FEATURES if df[c].notna().any()]
    peak_ok = [c for c in peak_ok if c not in base]
    bin_cols = [c for c in base if c.startswith("ic_raw_")]
    no_bins = [c for c in base if c not in bin_cols]

    y = df["capacity"].values.astype(float)
    cars = df["car"].values
    folds = list(group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                    seed=cfg["splits"]["random_state"]))

    combos = {
        "full": base,
        "peak_add": base + peak_ok,
        "peak_replace": no_bins + peak_ok,
    }
    print("P2 五折消融：")
    res, oof = {}, {}
    for tag, cols in combos.items():
        res[tag], oof[tag] = run_p2(df, cols, cfg, y, cars, folds, tag)

    # 峰特征的可解释性：与 SOH 的秩相关
    corr = {}
    for c in peak_ok:
        s = df[[c, "soh"]].dropna()
        if len(s) > 100:
            corr[c] = float(s[c].corr(s["soh"], method="spearman"))
    top = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:8]

    summary = {
        "n_sessions": int(len(df)), "n_cars": int(df["car"].nunique()),
        "n_peak_features": len(peak_ok), "peak_features": peak_ok,
        "peak_count_hist": df["ic_pk_n"].value_counts().sort_index().to_dict(),
        "ablation": res,
        "spearman_with_soh": dict(top),
        "note": ("peak_replace 用 15 个峰特征替代 25 维 ic_raw 分箱；"
                 "两者都保留 ic_norm 形状分箱（峰特征不含形状全谱）。"),
    }
    save_json(summary, "ic_peaks")

    fig_ic_evolution(df, os.path.join(cfg["paths"]["figures_dir"], "fig18_IC峰随SOH演化.png"))
    fig_ablation(res, os.path.join(cfg["paths"]["figures_dir"], "fig19_IC峰特征消融.png"))

    print("\n峰特征与 SOH 的秩相关（|rho| 前 8）：")
    for k, v in top:
        print(f"  {k:14s} rho={v:+.3f}")
    print(f"\n峰值个数分布：{summary['peak_count_hist']}")


if __name__ == "__main__":
    if "--replot" in sys.argv:      # 只重画图 18–19（复用已落盘的峰特征与消融结果）
        import json

        _cfg = load_config()
        with open(os.path.join(_cfg["paths"]["metrics_dir"], "ic_peaks.json"), encoding="utf-8") as f:
            _res = json.load(f)
        _ses = pd.read_parquet(_cfg["paths"]["session_features"])
        _df, _ = add_soh_labels(_ses, _cfg)
        _df = _df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)
        _pk = pd.read_parquet(os.path.join(_cfg["paths"]["cache_dir"], "ic_peaks.parquet"))
        for _c in _pk.columns:
            if _c.startswith(("ic_pk", "ic_npk")):
                _df[_c] = _pk[_c].to_numpy()
        fig_ic_evolution(_df, os.path.join(_cfg["paths"]["figures_dir"], "fig18_IC峰随SOH演化.png"))
        fig_ablation(_res["ablation"],
                     os.path.join(_cfg["paths"]["figures_dir"], "fig19_IC峰特征消融.png"))
        print("重绘完成")
    else:
        main()
