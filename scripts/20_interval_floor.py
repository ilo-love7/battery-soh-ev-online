# -*- coding: utf-8 -*-
"""20 区间还能不能更窄：同车多段聚合，以及标签噪声的地板。

第三轮（§16.3）把 90% 区间压到 2.452 Ah 就压不动了，并给出诊断：
`2 × P90(|残差|) = 2.436 Ah`，两者几乎相等，所以"宽度由标签噪声决定"。
但那个结论有一个隐含前提——**一个片段只报一个数**。真实部署不是这样：

    一辆车一年跑几百次充电，每次都能估一个容量。
    如果误差里有相当一部分是"这次片段特有的"，那么平均 k 次就能把它压掉。

于是本脚本回答三个问题：

  ① 同车连续 k 段取平均，误差和区间宽度各降多少？降的规律是不是 1/√k？
     如果不是，说明车身份偏差（§15.2 的 55.8%）在主导，平均救不回来。
  ② 标签噪声到底多大？用"同车近邻会话"直接估——相邻两次充电之间真实容量几乎不变，
     标签之差就近似是噪声之差，`σ_noise = std(Δe)/√2`。
  ③ 模型误差里有多少是"可解释"的？`σ_model = √(σ_total² − σ_noise²)`。
     如果 σ_model 已经很小，那再换模型/换表示都是白费，只能去改标签。

聚合口径（避免"长历史车被过度加权"）：每辆车在每个 k 上最多取 12 个窗，
于是每辆车等权。k=1 的等车权重值会与主实验的片段加权值（0.826 Ah）略有差异，
两个都报，便于对齐。

输出
  results/metrics/interval_floor.json
  results/figures/fig35_同车聚合的收益.png
  results/figures/fig36_标签噪声地板.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import group_kfold_by_car, regression_metrics  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_PURPLE, C_RED, plt, savefig  # noqa: E402

K_LIST = [1, 2, 3, 5, 8, 12]
MAX_WIN = 12              # 每车每 k 最多取多少窗（等车权重）
NOMINAL = 0.90
CAL_FRAC = 0.25           # 折内校准车占比，与脚本 13/18 一致
SPAN_BINS = [20, 30, 40, 50, 60, 70, 100]
SPAN_LABELS = ["20-30", "30-40", "40-50", "50-60", "60-70", "70-100"]
NOISE_GAP_KM = 3000.0     # "近邻会话"的里程差上限
PRED_COL = "pred_xgb"


def span_key(d_soc: np.ndarray) -> np.ndarray:
    k = np.full(d_soc.shape, "", dtype=object)
    for lo, hi, lab in zip(SPAN_BINS[:-1], SPAN_BINS[1:], SPAN_LABELS):
        k[(d_soc >= lo) & (d_soc < hi)] = lab
    return np.asarray(k, dtype=object)


def build_windows(df: pd.DataFrame, k: int, max_win: int = MAX_WIN,
                  seed: int = 7) -> pd.DataFrame:
    """把每辆车按里程排序，切成长度 k 的连续窗，返回窗级表。

    每车最多取 max_win 个窗（随机不重复抽样），保证车辆等权；
    k=1 时退化为"每车最多 12 个片段"的等权子集。
    """
    rng = np.random.default_rng(seed + k)
    rows = []
    for car, g in df.groupby("car", sort=False):
        g = g.sort_values("mileage")
        n = len(g)
        if n < k:
            continue
        m = n - k + 1
        if m > max_win:
            idx = np.sort(rng.choice(m, size=max_win, replace=False))
        else:
            idx = np.arange(m)
        p = g[PRED_COL].to_numpy(dtype=float)
        y = g["capacity"].to_numpy(dtype=float)
        d = g["d_soc"].to_numpy(dtype=float)
        for s in idx:
            rows.append((car, float(p[s:s + k].mean()), float(y[s:s + k].mean()),
                         float(d[s:s + k].mean()), k))
    return pd.DataFrame(rows, columns=["car", "pred", "capacity", "d_soc", "k"])


def conformal_by_car(w: pd.DataFrame, seed: int = 42, n_splits: int = 5) -> dict:
    """窗级 split-conformal：折内 25% 的车做校准，其余车拟合"中心=预测值"。"""
    cars = w["car"].to_numpy()
    y = w["capacity"].to_numpy(dtype=float)
    p = w["pred"].to_numpy(dtype=float)
    lo = np.full(len(w), np.nan)
    hi = np.full(len(w), np.nan)
    for fi, (tr, te) in enumerate(group_kfold_by_car(cars, n_splits=n_splits, seed=seed)):
        tr_cars = np.unique(cars[tr])
        rng = np.random.default_rng(1000 + fi)
        n_cal = max(1, int(round(len(tr_cars) * CAL_FRAC)))
        cal_cars = set(rng.choice(tr_cars, size=n_cal, replace=False).tolist())
        is_cal = np.array([c in cal_cars for c in cars[tr]])
        cal = tr[is_cal]
        q = float(np.quantile(np.abs(p[cal] - y[cal]), NOMINAL, method="higher"))
        lo[te], hi[te] = p[te] - q, p[te] + q
    cov = float(np.mean((y >= lo) & (y <= hi)))
    return {"width": float(np.mean(hi - lo)), "coverage": cov,
            "below": float(np.mean(y < lo)), "above": float(np.mean(y > hi)),
            "q": float(np.mean((hi - lo) / 2.0))}


def variance_components(r: np.ndarray, cars: np.ndarray) -> dict:
    """一元随机效应的方差分解：r_ij = b_i + e_ij。"""
    s = pd.DataFrame({"r": r, "car": cars})
    g = s.groupby("car")["r"]
    n_c = g.size().to_numpy(dtype=float)
    mean_c = g.mean().to_numpy(dtype=float)
    ss_within = float(np.sum((s["r"] - s.groupby("car")["r"].transform("mean")) ** 2))
    df_within = float(len(s) - len(n_c))
    sig_w2 = ss_within / max(df_within, 1.0)
    var_means = float(np.var(mean_c, ddof=1))
    n_h = float(len(n_c) / np.sum(1.0 / n_c))       # 调和均值，用于校正抽样误差
    sig_b2 = max(var_means - sig_w2 / n_h, 0.0)
    return {"sigma_within": float(np.sqrt(sig_w2)), "sigma_between": float(np.sqrt(sig_b2)),
            "icc": float(sig_b2 / (sig_b2 + sig_w2)) if (sig_b2 + sig_w2) > 0 else float("nan"),
            "n_cars": int(len(n_c)), "n_obs": int(len(s)),
            "harm_mean_n": n_h,
            "pred_rmse_k": {str(k): float(np.sqrt(sig_b2 + sig_w2 / k)) for k in K_LIST}}


def _pair_diffs(df: pd.DataFrame, gap_km: float = NOISE_GAP_KM):
    """同车相邻会话的 (残差之差, 容量之差, 里程差)，已按里程差截断。"""
    rows = []
    for _car, g in df.groupby("car", sort=False):
        g = g.sort_values("mileage")
        if len(g) < 8:
            continue
        x = g["mileage"].to_numpy(dtype=float) / 1.0e4
        y = g["capacity"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8:
            continue
        b, a = np.polyfit(x[ok], y[ok], 1)          # 车级线性衰减趋势
        e = y - (a + b * x)
        rows.append(pd.DataFrame({"de": np.diff(e), "dy": np.diff(y),
                                  "dm": np.diff(x) * 1.0e4}))
    p = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        {"de": [], "dy": [], "dm": []})
    p = p[np.abs(p["dm"]) <= gap_km]
    return (p["de"].to_numpy(dtype=float), p["dy"].to_numpy(dtype=float),
            p["dm"].to_numpy(dtype=float))


def label_noise(df: pd.DataFrame, gap_km: float = NOISE_GAP_KM) -> dict:
    """用同车近邻会话估标签噪声。

    先扣掉每车容量-里程的线性衰减，再看相邻会话的残差之差 Δe：
    若两次充电的真实容量几乎不变，Δe 就近似两次标签误差之差，σ = std(Δe)/√2。

    两个必须报告的事实（否则会误读成一个高斯噪声）：
      - **重复标签**：相当一部分相邻会话的容量值完全相同（连续量精确相等只可能是复制），
        说明标签是"一次测量挂到附近若干会话"上的，不是每会话独立测的；
      - **重尾**：std 与 MAD 相差数倍，说明噪声是"窄核 + 厚尾"的混合，
        任何基于高斯假设的宽度推断都会低估尾部。
    """
    de, dy, dm = _pair_diffs(df, gap_km)
    dup = np.abs(dy) < 1e-9                          # 标签完全相同 = 同一次测量
    de_u = de[~dup]

    def _sigma(v):
        return float(np.std(v, ddof=1) / np.sqrt(2.0)) if v.size > 10 else float("nan")

    # 连续重复的"标签游程"长度：一次测量平均挂到几个会话上
    runs = []
    for _car, g in df.groupby("car", sort=False):
        y = g.sort_values("mileage")["capacity"].to_numpy(dtype=float)
        n = 1
        for i in range(1, len(y)):
            if abs(y[i] - y[i - 1]) < 1e-9:
                n += 1
            else:
                runs.append(n)
                n = 1
        runs.append(n)
    runs = np.asarray(runs, dtype=float)

    mad = float(1.4826 * np.median(np.abs(de_u - np.median(de_u)))) if de_u.size else np.nan
    return {"sigma_noise": _sigma(de_u),          # 只用"不同标签"的对，避开复制造成的假 0
            "sigma_noise_all": _sigma(de),
            "sigma_noise_mad": float(mad / np.sqrt(2.0)) if np.isfinite(mad) else float("nan"),
            "dup_frac": float(dup.mean()) if len(de) else float("nan"),
            "mean_run": float(runs.mean()) if runs.size else float("nan"),
            "max_run": int(runs.max()) if runs.size else 0,
            "n_pairs": int(len(de)), "n_distinct_pairs": int(de_u.size),
            "median_gap_km": float(np.median(np.abs(dm))) if len(dm) else float("nan"),
            "gap_km_cap": gap_km}


def main() -> None:
    cfg = load_config()
    oof = pd.read_parquet(os.path.join(cfg["paths"]["metrics_dir"], "oof_predictions.parquet"))
    need = ["car", "capacity", PRED_COL, "mileage", "d_soc"]
    oof = oof.dropna(subset=need).reset_index(drop=True)
    print(f"样本 {len(oof)} 片段 / {oof['car'].nunique()} 车", flush=True)

    y = oof["capacity"].to_numpy(dtype=float)
    p = oof[PRED_COL].to_numpy(dtype=float)
    r = p - y
    ref_mean = float(oof["bol_ref"].mean()) if "bol_ref" in oof else float(y.mean())
    seg = regression_metrics(y, p)
    print(f"片段级（全量、按片段加权）：RMSE {seg['rmse']:.4f} / MAE {seg['mae']:.4f}",
          flush=True)

    # ---------------- ① 同车 k 段聚合 ----------------
    # 只保留"历史足够长"的车，保证 k=1 与 k=12 是同一批车，曲线不被样本组成带偏
    n_seg = oof.groupby("car").size()
    keep_cars = set(n_seg[n_seg >= max(K_LIST)].index)
    sub = oof[oof["car"].isin(keep_cars)].reset_index(drop=True)
    print(f"聚合实验子集：{len(sub)} 片段 / {len(keep_cars)} 车（≥{max(K_LIST)} 段）", flush=True)

    agg = {}
    for k in K_LIST:
        w = build_windows(sub, k)
        mm = regression_metrics(w["capacity"].to_numpy(dtype=float),
                                w["pred"].to_numpy(dtype=float))
        cv = conformal_by_car(w)
        agg[str(k)] = {"n_windows": int(len(w)),
                       "n_cars": int(w["car"].nunique()),
                       **mm, **cv,
                       "rmse_pp": mm["rmse"] / ref_mean * 100.0}
        print(f"  k={k:2d}: RMSE {mm['rmse']:.4f}  MAE {mm['mae']:.4f}  "
              f"宽度 {cv['width']:.3f}  覆盖 {cv['coverage']:.3f}  ({len(w)} 窗)", flush=True)

    # ---------------- ② 方差分解 ----------------
    vc = variance_components(r, oof["car"].to_numpy())
    print(f"方差分解：σ_within {vc['sigma_within']:.4f} / σ_between {vc['sigma_between']:.4f}"
          f"  ICC {vc['icc']:.3f}", flush=True)

    # ---------------- ③ 标签噪声与三段误差预算 ----------------
    ln = label_noise(oof)
    sig_n = ln["sigma_noise"]
    sig_b = vc["sigma_between"]
    sig_w = vc["sigma_within"]
    # 车内方差 = 车内模型误差 + 标签噪声（两者独立）
    sig_wm = float(np.sqrt(max(sig_w ** 2 - sig_n ** 2, 0.0)))
    sig_tot = float(np.sqrt(sig_b ** 2 + sig_w ** 2))
    mae_floor = float(np.sqrt(2.0 / np.pi) * sig_n)
    print(f"标签噪声：σ={sig_n:.4f} Ah（只用不同标签的对，{ln['n_distinct_pairs']} 对）；"
          f"MAD 法 σ={ln['sigma_noise_mad']:.4f}；重复标签占比 {ln['dup_frac']:.3f}"
          f"（平均游程 {ln['mean_run']:.2f}，最长 {ln['max_run']}）", flush=True)
    print(f"误差预算：车级偏差 {sig_b:.3f}（{sig_b ** 2 / sig_tot ** 2 * 100:.0f}%）+ "
          f"车内模型 {sig_wm:.3f}（{sig_wm ** 2 / sig_tot ** 2 * 100:.0f}%）+ "
          f"标签噪声 {sig_n:.3f}（{sig_n ** 2 / sig_tot ** 2 * 100:.0f}%）"
          f" = {sig_tot:.3f} Ah", flush=True)
    print(f"高斯假设下的 MAE 地板 {mae_floor:.4f} vs 实测 {seg['mae']:.4f}"
          f"（{seg['mae'] / mae_floor - 1:+.0%}）", flush=True)

    # ---------------- ④ 聚合收益是否只是"长片段的代理" ----------------
    # 逐箱 conformal 分位在 60–70% 这类小样本箱上抖动很大（校准窗不足 20 个），
    # 所以这里只比**逐箱 RMSE**（稳定、无校准噪声），宽度对比留在全局口径。
    cond = {}
    for k in (1, 4):
        w = build_windows(sub, k)
        sk = span_key(w["d_soc"].to_numpy(dtype=float))
        yy = w["capacity"].to_numpy(dtype=float)
        pp = w["pred"].to_numpy(dtype=float)
        rows = {}
        for lab in SPAN_LABELS:
            m = sk == lab
            if m.sum() < 80:
                continue
            rows[lab] = {"n": int(m.sum()),
                         "rmse": regression_metrics(yy[m], pp[m])["rmse"],
                         "mae": regression_metrics(yy[m], pp[m])["mae"]}
        cond[str(k)] = rows
    # 同一批车、同一批跨度箱下 k=1 → k=4 的相对收益
    common = [b for b in cond["1"] if b in cond["4"]]
    cond["gain_k1_to_k4"] = {
        b: {"rmse_1": cond["1"][b]["rmse"], "rmse_4": cond["4"][b]["rmse"],
            "gain_pct": (cond["4"][b]["rmse"] / cond["1"][b]["rmse"] - 1.0) * 100.0}
        for b in common}

    # 实用对照：想拿到某个宽度，至少要平均几次会话
    recipe = {}
    for tgt in (2.5, 2.0, 1.8, 1.5):
        hit = [k for k in K_LIST if agg[str(k)]["width"] <= tgt]
        recipe[str(tgt)] = {"min_k": int(min(hit)) if hit else None,
                            "width": agg[str(min(hit))]["width"] if hit else None,
                            "reachable": bool(hit)}

    # 等效独立样本数：把实测曲线反解回 σ_b² + σ_w²/k_eff
    sb2 = vc["sigma_between"] ** 2
    sw2 = vc["sigma_within"] ** 2
    k_eff = {}
    for k in K_LIST:
        r2 = agg[str(k)]["rmse"] ** 2
        k_eff[str(k)] = float(sw2 / (r2 - sb2)) if r2 > sb2 else float("nan")

    print("等效独立样本数 k_eff：" +
          "  ".join(f"k={k}→{k_eff[str(k)]:.1f}" for k in K_LIST), flush=True)
    print("跨度箱内 k=1→k=4 的 RMSE 收益：" +
          "  ".join(f"{b} {v['gain_pct']:+.0f}%" for b, v in cond["gain_k1_to_k4"].items()),
          flush=True)

    res = {
        "n_segments": int(len(oof)), "n_cars": int(oof["car"].nunique()),
        "subset": {"n_segments": int(len(sub)), "n_cars": int(len(keep_cars)),
                   "min_segments": max(K_LIST)},
        "segment_level": seg,
        "aggregate": agg, "variance": vc,
        "label_noise": {**ln,
                        "budget": {"car_bias": sig_b, "within_car_model": sig_wm,
                                   "label_noise": sig_n, "total": sig_tot,
                                   "share_car_bias": sig_b ** 2 / sig_tot ** 2,
                                   "share_model": sig_wm ** 2 / sig_tot ** 2,
                                   "share_noise": sig_n ** 2 / sig_tot ** 2},
                        "mae_floor": mae_floor, "mae_observed": seg["mae"],
                        "mae_excess_pct": float((seg["mae"] / mae_floor - 1.0) * 100.0)},
        "conditional": cond, "k_eff": k_eff, "recipe": recipe,
        "k_list": K_LIST, "max_win_per_car": MAX_WIN,
        "note": ("聚合口径：每车按里程排序切连续窗，每车每 k 最多取 12 个窗（车辆等权）；"
                 "只保留 ≥12 段的车辆，保证各 k 是同一批车。"
                 "标签噪声：车级容量~里程线性去趋势后，里程差 ≤3000 km 的相邻会话残差之差，"
                 "σ_noise = std(Δe)/√2；若标签误差在车内正相关，该估计偏保守（低估）。"),
    }
    save_json(res, "interval_floor")

    fig_agg(res, os.path.join(cfg["paths"]["figures_dir"], "fig35_同车聚合的收益.png"))
    fig_noise(oof, res, os.path.join(cfg["paths"]["figures_dir"], "fig36_标签噪声地板.png"))
    print("\n完成", flush=True)


# ---------------------------------------------------------------- 图 35
def fig_agg(res: dict, path: str) -> None:
    """图 35：RMSE 与区间宽度随聚合段数 k 的变化，叠加 1/√k 理论线。"""
    ks = [int(k) for k in res["aggregate"]]
    rmse = [res["aggregate"][str(k)]["rmse"] for k in ks]
    width = [res["aggregate"][str(k)]["width"] for k in ks]
    cov = [res["aggregate"][str(k)]["coverage"] for k in ks]
    theo = [res["variance"]["pred_rmse_k"][str(k)] for k in ks]
    flat = res["variance"]["sigma_between"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))

    # (a) RMSE
    axes[0].plot(ks, rmse, marker="o", ms=5.5, lw=1.9, color=C_MAIN, label="Measured")
    axes[0].plot(ks, theo, ls="--", lw=1.4, color=C_GRAY,
                 label=r"$\sqrt{\sigma_b^2+\sigma_w^2/k}$")
    axes[0].axhline(flat, color=C_RED, ls=":", lw=1.3,
                    label=rf"Floor from car bias $\sigma_b$={flat:.3f}")
    axes[0].set_xscale("log")
    axes[0].set_xticks(ks)
    axes[0].set_xticklabels([str(k) for k in ks])
    axes[0].set_xlabel("Charging sessions averaged per car (k)")
    axes[0].set_ylabel("RMSE (Ah)")
    axes[0].set_title("(a) Averaging over sessions", pad=8)
    axes[0].legend(fontsize=7.8, loc="upper right")
    for k, v in zip(ks, rmse):
        if k in (1, max(ks)):
            axes[0].annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=8)
    ke = res.get("k_eff", {}).get(str(max(ks)))
    if ke is not None:
        axes[0].text(0.035, 0.10,
                     f"{max(ks)} sessions ≈ {ke:.1f} independent\n"
                     r"(errors within a car are correlated)",
                     transform=axes[0].transAxes, fontsize=7.6, color="#444444")

    # (b) 宽度 + 覆盖率
    axes[1].plot(ks, width, marker="s", ms=5.5, lw=1.9, color=C_ACC,
                 label="90% interval width")
    axes[1].set_xscale("log")
    axes[1].set_xticks(ks)
    axes[1].set_xticklabels([str(k) for k in ks])
    axes[1].set_xlabel("Charging sessions averaged per car (k)")
    axes[1].set_ylabel("Interval width (Ah)")
    axes[1].set_title("(b) Interval width vs. coverage", pad=8)
    axes[1].set_ylim(min(width) * 0.82, max(width) * 1.14)
    ax2 = axes[1].twinx()
    ax2.plot(ks, cov, marker="^", ms=4.5, lw=1.2, ls=":", color=C_PURPLE,
             label="Coverage")
    ax2.axhline(NOMINAL, color=C_PURPLE, ls="--", lw=0.9, alpha=0.7)
    ax2.set_ylabel("Coverage", color=C_PURPLE)
    ax2.set_ylim(0.80, 1.0)
    ax2.tick_params(axis="y", colors=C_PURPLE)
    ax2.grid(visible=False)
    for k, v in zip(ks, width):
        if k in (1, max(ks)):
            axes[1].annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                             xytext=(0, 9), ha="center", fontsize=8, color=C_ACC)
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, fontsize=7.8, loc="upper right")
    fig.tight_layout()
    savefig(fig, path)


# ---------------------------------------------------------------- 图 36
def fig_noise(df: pd.DataFrame, res: dict, path: str) -> None:
    """图 36：标签抖动的分布形状（尖峰 + 重尾）+ 三段误差预算。"""
    ln = res["label_noise"]
    bud = ln["budget"]
    sig_n = ln["sigma_noise"]
    de, dy, _dm = _pair_diffs(df)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})

    # (a) 对数纵轴：既看得见 0 处的尖峰（重复标签），也看得见尾部
    lo, hi = float(np.percentile(de, 0.3)), float(np.percentile(de, 99.7))
    axes[0].hist(de, bins=120, range=(lo, hi), density=True, color=C_GRAY, alpha=0.55,
                 label=f"adjacent-session $\\Delta$label (n={len(de)})")
    xs = np.linspace(lo, hi, 400)
    axes[0].plot(xs, np.exp(-xs ** 2 / (2 * (np.sqrt(2) * sig_n) ** 2))
                 / (np.sqrt(2) * sig_n * np.sqrt(2 * np.pi)),
                 color=C_RED, lw=1.6,
                 label=rf"Gaussian fit, $\sigma_n$={sig_n:.2f} Ah")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-4, 3.0)
    axes[0].set_xlabel("$\\Delta$ residual capacity between adjacent sessions (Ah)")
    axes[0].set_ylabel("Density (log scale)")
    axes[0].set_title(f"(a) Label jitter: a spike plus heavy tails "
                      f"(std $\\sigma$ {sig_n:.2f} vs MAD $\\sigma$ "
                      f"{ln['sigma_noise_mad']:.2f} Ah)", fontsize=10, pad=8)
    axes[0].legend(fontsize=7.2, loc="upper right")
    axes[0].text(0.02, 0.97,
                 f"{ln['dup_frac'] * 100:.0f}% of adjacent pairs share an identical label",
                 transform=axes[0].transAxes, fontsize=7.4, color="#333333",
                 ha="left", va="top")

    # (b) 三段预算：车级偏差 / 车内模型误差 / 标签噪声（方差可加）
    keys = [("car_bias", "share_car_bias", "Car-level bias\n(cannot be averaged out)", C_PURPLE),
            ("within_car_model", "share_model", "Within-car model error", C_MAIN),
            ("label_noise", "share_noise", "Label noise", C_GRAY)]
    vals = [bud[k] for k, _s, _l, _c in keys]
    shares = [bud[s] for _k, s, _l, _c in keys]
    cols = [c for _k, _s, _l, c in keys]
    bars = axes[1].bar(range(3), vals, color=cols, width=0.56)
    for b, v, sh in zip(bars, vals, shares):
        axes[1].text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                     f"{v:.3f} Ah\n{sh * 100:.0f}%", ha="center", fontsize=8.4)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels([l for _k, _s, l, _c in keys], fontsize=7.8)
    axes[1].set_ylabel("Std. dev. (Ah)")
    axes[1].set_ylim(0, max(vals) * 1.36)
    axes[1].set_title("(b) Where the error of one snippet comes from", pad=8)
    axes[1].text(0.5, 0.95, rf"$\sigma_b^2+\sigma_m^2+\sigma_n^2$ = {bud['total']:.3f} Ah",
                 transform=axes[1].transAxes, ha="center", fontsize=8.6, color="#333333")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, path)


if __name__ == "__main__":
    if "--replot" in sys.argv:
        _cfg = load_config()
        with open(os.path.join(_cfg["paths"]["metrics_dir"], "interval_floor.json"),
                  encoding="utf-8") as f:
            _res = json.load(f)
        _oof = pd.read_parquet(os.path.join(_cfg["paths"]["metrics_dir"],
                                            "oof_predictions.parquet"))
        fig_agg(_res, os.path.join(_cfg["paths"]["figures_dir"], "fig35_同车聚合的收益.png"))
        fig_noise(_oof.dropna(subset=["capacity", PRED_COL]).reset_index(drop=True), _res,
                  os.path.join(_cfg["paths"]["figures_dir"], "fig36_标签噪声地板.png"))
        print("重绘完成")
    else:
        main()
