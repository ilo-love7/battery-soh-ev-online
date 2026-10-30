# -*- coding: utf-8 -*-
"""15 D1 深化（二）：10 mV 高分辨增量容量，峰结构是否真的存在。

脚本 11 在已有的 50 mV 分箱上做峰检测，结果是"96.8% 的会话只检出一个峰"——
不是数据没有峰，而是 50 mV 一格太粗，把多峰结构抹平了。于是这一版回到原始 pkl，
把 IC 重算到 **10 mV/格（125 格）**，回答三个问题：

  1. 高分辨 IC 上到底有几个峰？峰位随 SOH 怎么移动？（这是 IC 分析的本体）
  2. 10 mV 分箱比 50 mV 分箱多带来多少精度？
  3. 显式峰特征叠在高分辨分箱上还有没有增量？

为了控制成本，只在**分层抽样的 3000 段会话**上重算（按 SOH 五分位等量抽），
读取约 3 万个窗口 pkl，几十秒即可完成。

输出
    results/cache/ic_fine.parquet       抽样会话的 10 mV IC 分箱与峰特征
    results/metrics/ic_fine.json
    results/figures/fig25_高分辨IC峰结构.png
    results/figures/fig26_IC分辨率消融.png
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
from evsoh.data import COL_SOC, COL_T, COL_V, load_window  # noqa: E402
from evsoh.evaluate import group_kfold_by_car, regression_metrics  # noqa: E402
from evsoh.models import build_model  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_PURPLE, C_RED, plt, savefig  # noqa: E402
from evsoh.session import _interp_at, concat_windows  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402

FINE_GRID = np.round(np.arange(3.00, 4.2501, 0.01), 3)   # 10 mV
FINE_CENTERS = (FINE_GRID[:-1] + FINE_GRID[1:]) / 2.0
N_SAMPLE = 3000
MAX_PEAKS = 4
PEAK_COLS = ([f"fpk{i}_{s}" for i in range(MAX_PEAKS) for s in ("v", "h", "w", "a")]
             + [f"fnpk{i}_h" for i in range(MAX_PEAKS)]
             + ["fpk_n", "fpk_vw", "fpk_ratio21", "fpk_cov_lo", "fpk_cov_hi"])
CTX_COLS = ["d_soc", "quality", "i_abs_mean", "t_mean", "n_windows", "dur_s",
            "soc_start", "soc_end", "v_start", "v_end", "cell_gap_mean",
            "q_ah", "q_per_soc"] + [f"socbin_ah_{a}_{b}" for a in range(0, 100, 10) for b in [a + 10]]
CTX_COLS = [c for c in CTX_COLS if not c.startswith("socbin_ah_")] + \
           [f"socbin_ah_{a}_{a + 10}" for a in range(0, 100, 10)]


def ic_curve(feat: np.ndarray, grid: np.ndarray) -> tuple:
    """从单会话轨迹算 IC 曲线（与 build_session 同一套口径，仅网格更细）。"""
    v = np.nan_to_num(feat[:, COL_V], nan=0.0)
    soc = np.nan_to_num(feat[:, COL_SOC], nan=0.0)
    cur = feat[:, 1]
    q_cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(np.abs(cur[:-1])) * 10.0 / 3600.0)])
    v_mono = np.maximum.accumulate(v)
    q_at = _interp_at(v_mono, q_cum, grid)
    return np.diff(q_at), float(soc[-1] - soc[0])


def fine_peaks(ic: np.ndarray) -> dict:
    """10 mV 网格上的峰特征。"""
    out = {k: np.nan for k in PEAK_COLS}
    valid = np.where(np.isfinite(ic))[0]
    if valid.size < 10:
        return out
    lo, hi = valid[0], valid[-1]
    seg = ic[lo:hi + 1].astype(float)
    miss = ~np.isfinite(seg)
    if miss.any() and (~miss).sum() >= 2:
        seg[miss] = np.interp(np.where(miss)[0], np.where(~miss)[0], seg[~miss])
    out["fpk_cov_lo"] = float(FINE_CENTERS[lo])
    out["fpk_cov_hi"] = float(FINE_CENTERS[hi])
    total = float(np.nansum(seg))
    if not np.isfinite(total) or total <= 0:
        return out

    peaks, _ = find_peaks(seg, prominence=0.05 * np.nanmax(seg), distance=8)  # 至少隔 80 mV
    if peaks.size == 0:
        peaks = np.array([int(np.nanargmax(seg))])
    out["fpk_n"] = float(len(peaks))
    order = peaks[np.argsort(seg[peaks])[::-1]][:MAX_PEAKS]
    hs = []
    for i, p in enumerate(order):
        h = float(seg[p])
        half = h / 2.0
        left = p
        while left > 0 and seg[left - 1] >= half:
            left -= 1
        right = p
        while right < seg.size - 1 and seg[right + 1] >= half:
            right += 1
        out[f"fpk{i}_v"] = float(FINE_CENTERS[lo + p])
        out[f"fpk{i}_h"] = h
        out[f"fpk{i}_w"] = float((right - left) * 0.01)
        out[f"fpk{i}_a"] = float(seg[left:right + 1].sum())
        out[f"fnpk{i}_h"] = h / total
        hs.append(h)
    if len(hs) >= 2 and hs[0] > 0:
        out["fpk_ratio21"] = hs[1] / hs[0]
    v = np.array([out[f"fpk{i}_v"] for i in range(len(hs))], float)
    hh = np.array(hs, float)
    out["fpk_vw"] = float(np.sum(v * hh) / np.sum(hh))
    return out


def fig_structure(fine: pd.DataFrame, fic_cols: list, path: str, corr: dict | None = None) -> None:
    """图 25：高分辨 IC 的峰结构（左）与分档峰高（右）。"""
    f2 = fine.copy()
    edges = [0, 85, 92, 97, 102, 120]
    names = ["<85%", "85-92%", "92-97%", "97-102%", ">102%"]
    f2["band"] = pd.cut(f2["soh"] * 100, bins=edges, labels=names)
    colors = [C_RED, C_ACC, C_GRAY, C_MAIN, C_PURPLE]
    fig, axes = plt.subplots(1, 2, figsize=(9.9, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    bin_idx = [int(c.split("_")[1]) for c in fic_cols]
    for name, col in zip(names, colors):
        sub = f2[f2["band"] == name]
        if len(sub) < 40:
            continue
        ax.plot(FINE_CENTERS[bin_idx], sub[fic_cols].mean().to_numpy(), color=col, lw=1.7,
                label=f"SOH {name}  (n={len(sub)})")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("dQ/dV (Ah per 10 mV)")
    ax.set_title("10 mV IC: multi-peak structure, flattening with age", pad=8)
    ax.set_xlim(3.55, 4.22)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    xs = np.arange(len(names))
    h0 = [f2.loc[f2["band"] == n, "fpk0_h"].mean() for n in names]
    h1 = [f2.loc[f2["band"] == n, "fpk1_h"].mean() for n in names]
    w = 0.38
    b1 = ax.bar(xs - w / 2, h0, width=w, color=C_MAIN, label="1st peak height")
    b2 = ax.bar(xs + w / 2, h1, width=w, color=C_ACC, label="2nd peak height")
    for bars, vals in ((b1, h0), (b2, h1)):
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                        ha="center", fontsize=7.5)
    ax.set_xticks(xs, names, fontsize=8)
    ax.set_xlabel("SOH band")
    ax.set_ylabel("Peak height (Ah per 10 mV)")
    if corr:
        ax.set_title("Peak heights grow with SOH\n"
                     f"(Spearman $\\rho$: 1st {corr.get('fpk0_h', float('nan')):+.2f}, "
                     f"2nd {corr.get('fpk1_h', float('nan')):+.2f})", pad=8, fontsize=9.5)
    else:
        ax.set_title("Peak heights grow with SOH", pad=8)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, np.nanmax(h0) * 1.20)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    fig.suptitle("High-resolution incremental capacity (10 mV, 125 bins, n=3000 sessions)",
                 fontsize=10.5, y=1.03)
    savefig(fig, path)


def fig_ablation(abl: dict, path: str) -> None:
    """分辨率消融图：以 50 mV 基线为参照，横轴只画有意义的区间。

    不用从 0 起的柱状图——四档 RMSE 都在 0.90–1.13 之间，从 0 起柱子高度几乎一样，
    "更细的分辨率没带来精度"这个结论反而看不出来。
    """
    order = [("coarse_50mV", "50 mV bins"), ("fine_10mV", "10 mV bins"),
             ("fine_plus_peaks", "10 mV bins + peaks"), ("peaks_only", "Peak features only")]
    keys = [k for k, _b in order]
    vals = [abl[k]["rmse"] for k in keys]
    base = vals[0]
    ys = np.arange(len(vals))
    cols = [C_GRAY, C_MAIN, C_PURPLE, C_ACC]

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.hlines(ys, base, vals, color=cols, lw=2.2, alpha=0.75)
    for i, (y_, v, c) in enumerate(zip(ys, vals, cols)):
        ax.plot([v], [y_], marker="o", ms=8, color=c, zorder=3)
        txt = f"{v:.4f}" + ("" if i == 0 else f"   ({(v / base - 1) * 100:+.1f}%)")
        if i == len(vals) - 1:      # 最后一档离右边界近，标注放左侧
            ax.text(v - 0.009, y_ - 0.26, txt, ha="right", va="center",
                    fontsize=8.2, color=c)
        else:
            ax.text(v + 0.009, y_, txt, ha="left", va="center", fontsize=8.2, color=c)
    ax.axvline(base, color=C_GRAY, ls="--", lw=1.1, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{b} ({abl[k]['n_features']} feats)" for k, b in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_ylim(len(vals) - 0.45, -0.55)
    ax.set_xlim(base - 0.028, max(vals) + 0.02)
    ax.set_xlabel("RMSE (Ah), 5-fold by car within the 3000-session sample")
    ax.set_title("Finer IC resolution does not buy accuracy\n"
                 "(dashed line = 50 mV baseline)", pad=8, fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    savefig(fig, path)


def main() -> None:
    cfg = load_config()
    root = cfg["paths"]["evbattery_root"]
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, _ = add_soh_labels(ses, cfg)
    df = df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)

    # 分层抽样：按 SOH 五分位等量
    df["_dec"] = pd.qcut(df["soh"], 5, labels=False)
    per = N_SAMPLE // 5
    parts = [g.sample(min(per, len(g)), random_state=42) for _k, g in df.groupby("_dec")]
    sample = pd.concat(parts).reset_index(drop=True)
    print(f"抽样 {len(sample)} 会话 / {sample['car'].nunique()} 车", flush=True)

    idx = pd.read_parquet(cfg["paths"]["window_index"],
                          columns=["dataset", "file", "car", "segment", "valid"])
    idx["fnum"] = idx["file"].str.replace(".pkl", "", regex=False).astype(int)
    idx = idx[idx["valid"]]
    key = list(zip(idx["dataset"], idx["car"].astype(int), idx["segment"].astype(str)))
    buckets: dict = {}
    for (ds, car, seg), f in zip(key, idx["file"].values):
        buckets.setdefault((ds, car, seg), []).append(f)

    rows, t0, miss = [], time.time(), 0
    for si, (_, r) in enumerate(sample.iterrows()):
        k = (r["dataset"], int(r["car"]), str(r["segment"]))
        files = buckets.get(k)
        if not files:
            miss += 1
            continue
        wins = []
        for f in sorted(files, key=lambda s: int(s.split(".")[0])):
            p = os.path.join(root, k[0], "data", f)
            if not os.path.exists(p):
                continue
            feat, meta = load_window(p)
            wins.append((f, feat, meta))
        if len(wins) < 2:
            miss += 1
            continue
        traj = concat_windows(wins)
        if traj.size == 0:
            miss += 1
            continue
        ic, d_soc = ic_curve(traj, FINE_GRID)
        rec = {"_row": si, "car": int(r["car"]), "segment": str(r["segment"]), "dataset": r["dataset"],
               "soh": float(r["soh"]), "capacity": float(r["capacity"]),
               "bol_ref": float(r["bol_ref"]), "d_soc_fine": d_soc}
        for j in range(len(FINE_GRID) - 1):
            rec[f"fic_{j:03d}"] = float(ic[j]) if np.isfinite(ic[j]) else np.nan
        rec.update(fine_peaks(ic))
        rows.append(rec)
        if len(rows) % 600 == 0:
            print(f"  {len(rows)} 段  ({time.time() - t0:.0f}s)", flush=True)
    fine = pd.DataFrame(rows).set_index("_row").sort_index()
    keep = fine.index.to_numpy()
    sample = sample.iloc[keep].reset_index(drop=True)
    fine = fine.reset_index(drop=True)
    print(f"完成 {len(fine)} 段（跳过 {miss}）  用时 {time.time() - t0:.0f}s", flush=True)
    fine.to_parquet(os.path.join(cfg["paths"]["cache_dir"], "ic_fine.parquet"), index=False)

    fic_cols = [c for c in fine.columns if c.startswith("fic_") and fine[c].notna().sum() > 50]
    pk_cols = [c for c in PEAK_COLS if fine[c].notna().sum() > 50]
    ctx = [c for c in CTX_COLS if c in sample.columns]
    print(f"有效细网格 {len(fic_cols)} 格 / 峰特征 {len(pk_cols)} 个 / 上下文 {len(ctx)} 个")

    # ---------------- 分辨率消融（同一批抽样、同一折划分） ----------------
    coarse_cols = [c for c in sample.columns if c.startswith("ic_raw_")]
    base = sample[coarse_cols + ctx].copy()
    y = sample["capacity"].values.astype(float)
    cars = sample["car"].values
    folds = list(group_kfold_by_car(cars, n_splits=5, seed=42))

    def run(cols: list, tag: str) -> dict:
        X = pd.concat([base[ctx].reset_index(drop=True),
                       fine[cols].reset_index(drop=True)], axis=1).astype(float)
        pred = np.full(len(fine), np.nan)
        for tr, te in folds:
            mdl = build_model("xgb", cfg["models"])
            mdl.fit(X.iloc[tr], y[tr])
            pred[te] = mdl.predict(X.iloc[te])
        m = regression_metrics(y, pred)
        m["n_features"] = X.shape[1]
        print(f"  [{tag:12s}] {X.shape[1]:3d} 特征  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}", flush=True)
        return m

    print("分辨率消融（抽样内 5 折，跨车）：", flush=True)
    abl = {}
    Xc = base[coarse_cols + ctx].astype(float)
    predc = np.full(len(fine), np.nan)
    for tr, te in folds:
        mdl = build_model("xgb", cfg["models"])
        mdl.fit(Xc.iloc[tr], y[tr])
        predc[te] = mdl.predict(Xc.iloc[te])
    abl["coarse_50mV"] = {**regression_metrics(y, predc), "n_features": len(coarse_cols) + len(ctx)}
    print(f"  [coarse_50mV ] {abl['coarse_50mV']['n_features']:3d} 特征  "
          f"RMSE={abl['coarse_50mV']['rmse']:.4f}  R2={abl['coarse_50mV']['r2']:.4f}", flush=True)
    abl["fine_10mV"] = run(fic_cols, "fine_10mV")
    abl["fine_plus_peaks"] = run(fic_cols + pk_cols, "fine+peaks")
    abl["peaks_only"] = run(pk_cols, "peaks_only")

    # ---------------- 峰统计 ----------------
    pk_hist = fine["fpk_n"].value_counts().sort_index().to_dict()
    corr = {}
    for c in ("fpk0_v", "fpk0_h", "fpk1_v", "fpk1_h", "fpk_ratio21", "fpk_n"):
        s = fine[[c, "soh"]].dropna()
        if len(s) > 100:
            corr[c] = float(s[c].corr(s["soh"], method="spearman"))

    res = {
        "n_sampled": int(len(fine)), "n_cars": int(fine["car"].nunique()),
        "grid_mV": 10, "n_fine_bins": len(fic_cols), "n_peak_features": len(pk_cols),
        "peak_count_hist": {str(k): int(v) for k, v in pk_hist.items()},
        "spearman_with_soh": corr,
        "ablation": abl,
        "note": ("抽样按 SOH 五分位等量取 3000 段；IC 与主流程同口径（q_cum 用固定 10 s 步长），"
                 "仅把电压网格从 50 mV 加密到 10 mV。消融在抽样内做，绝对值不可与全量 0.826 Ah 直接比。"),
    }
    save_json(res, "ic_fine")

    # ---------------- 图 25：峰结构 ----------------
    fig_structure(fine, fic_cols,
                  os.path.join(cfg["paths"]["figures_dir"], "fig25_高分辨IC峰结构.png"), corr)

    # ---------------- 图 26：分辨率消融 ----------------
    fig_ablation(abl, os.path.join(cfg["paths"]["figures_dir"], "fig26_IC分辨率消融.png"))

    print(f"\n峰个数分布：{res['peak_count_hist']}")
    print("与 SOH 的秩相关：", {k: round(v, 3) for k, v in corr.items()})


if __name__ == "__main__":
    if "--replot" in sys.argv:      # 只重画图 26（复用已落盘的消融结果）
        import json

        cfg_ = load_config()
        with open(os.path.join(cfg_["paths"]["metrics_dir"], "ic_fine.json"), encoding="utf-8") as f:
            _res = json.load(f)
        _fine = pd.read_parquet(os.path.join(cfg_["paths"]["cache_dir"], "ic_fine.parquet"))
        _fic = [c for c in _fine.columns if c.startswith("fic_") and _fine[c].notna().sum() > 50]
        fig_structure(_fine, _fic,
                      os.path.join(cfg_["paths"]["figures_dir"], "fig25_高分辨IC峰结构.png"),
                      _res.get("spearman_with_soh"))
        fig_ablation(_res["ablation"],
                     os.path.join(cfg_["paths"]["figures_dir"], "fig26_IC分辨率消融.png"))
        print("重绘完成")
    else:
        main()
