# -*- coding: utf-8 -*-
"""21 把两个互补的杠杆合起来：同车聚合 × 个性化。

第四轮（§17.3）把误差拆成三块，并指出每一块各有各的解法：

    车级偏差 0.449（28%）  → 个性化（§15.4）：用该车少量带标签会话校正
    车内模型 0.476（31%）  → 同车多段平均（§17.2）：免费，只要多跑几次
    标签噪声 0.547（41%）  → 只能改标签，算法无解

但这两个杠杆**从来没有一起测过**：§15.4 的个性化是在单片段上做的，§17.2 的聚合
是完全不做校正的。既然它们打的是误差里互不重叠的两块，合起来应该近似可加——
本脚本验证这个预测，并回答三个部署问题：

  ① 聚合与个性化是不是可加？把 (k, m) 铺成网格，实测 RMSE 与
     `√(σ_b²(m) + σ_w²(m)/k)` 的预测值比。
  ② 聚合器本身还能不能优化？中位数／截尾均值／按新近加权／按质量分加权／按跨度加权，
     都是**不需要标签**的口径，比一律取算术平均更"抗噪"的直觉能不能兑现。
  ③ 标签值多少钱？把"多跑 k 次"与"多标 m 段"放在同一张 RMSE 表上，
     才知道现场该去攒哪个。

评测口径（与 §15.4 一致，避免"用自己校正自己"）：
每辆车按里程排序，前 50% 段做校准池（只从里面取 m 段标签），后 50% 段做评估集，
评估集里再按里程切连续 k 段窗。bias 只从校准池**最近**的 m 段估计（部署时能拿到的
就是最近的标签，而且车级偏差随老化漂移），收缩系数 λ=5 沿用 §15.4 的设定。

输出
  results/metrics/aggregate_personalize.json
  results/figures/fig37_聚合与个性化网格.png
  results/figures/fig38_误差预算的可加性.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import conformal_by_car, regression_metrics  # noqa: E402
from evsoh.personalize import fit_bias  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_PURPLE, C_RED, plt, savefig  # noqa: E402

K_LIST = [1, 2, 3, 5, 8, 12]
M_LIST = [0, 1, 2, 3, 5, 8]
MIN_SESS = 24             # 前后各 12 段，保证 k=12 与 m=8 都有料
EVAL_FRAC = 0.5
MAX_WIN = 12              # 每车每 k 最多取多少窗（车辆等权，与脚本 20 一致）
NOMINAL = 0.90
BIAS_LAM = 5.0            # 与 §15.4 / personalize.fit_bias 的默认一致
AGGS = ["mean", "median", "trim", "recency", "quality", "span"]
PRED_COL = "pred_xgb"


# ---------------------------------------------------------------- 口径
def split_chrono(df: pd.DataFrame, min_sess: int = MIN_SESS,
                 eval_frac: float = EVAL_FRAC) -> pd.DataFrame:
    """按里程排序后前后切分：前段做校准池，后段做评估集。

    只用"历史足够长"的车，保证所有 (k, m) 格子都在同一批车上算出来。
    """
    d = df.sort_values(["car", "mileage"]).reset_index(drop=True)
    n = d.groupby("car")["car"].transform("size").to_numpy()
    rank = d.groupby("car").cumcount().to_numpy()
    n_eval = np.maximum(np.round(n * eval_frac).astype(int), 1)
    part = np.where(rank < n - n_eval, "cal", "eval")
    d = d.assign(part=part)
    keep = set(d.loc[n >= min_sess, "car"].unique())
    return d[d["car"].isin(keep)].reset_index(drop=True)


def _weights(how: str, q: np.ndarray, span: np.ndarray, k: int) -> np.ndarray | None:
    """聚合器对应的权重；返回 None 表示用等权。"""
    if how == "quality":
        w = np.clip(np.asarray(q, dtype=float), 0.02, None)
    elif how == "span":
        w = np.clip(np.asarray(span, dtype=float), 1.0, None)
    elif how == "recency":
        # 半衰期 = 窗长的一半：窗内越靠后（越新）权重越高
        age = np.arange(k)[::-1].astype(float)
        return 0.5 ** (age / max(k / 2.0, 1.0))
    else:
        return None
    return w if np.sum(w) > 0 else None


def _aggregate(p: np.ndarray, how: str, q: np.ndarray, span: np.ndarray) -> float:
    """窗内把 k 个片段预测合成一个数。目标值始终是窗内容量的算术平均。"""
    p = np.asarray(p, dtype=float)
    k = p.size
    if k == 1 or how == "mean":
        return float(p.mean())
    if how == "median":
        return float(np.median(p))
    if how == "trim":
        t = int(np.floor(k * 0.2))
        if t == 0:
            return float(p.mean())
        return float(np.sort(p)[t:k - t].mean())
    w = _weights(how, q, span, k)
    if w is None:
        return float(p.mean())
    return float(np.dot(w, p) / np.sum(w))


def build_windows(df: pd.DataFrame, k: int, how: str = "mean",
                  max_win: int = MAX_WIN, seed: int = 7) -> pd.DataFrame:
    """把每辆车按里程切成长度 k 的连续窗；窗集合与 ``how`` 无关，便于成对比较。"""
    rng = np.random.default_rng(seed + k)
    rows = []
    for car, g in df.groupby("car", sort=False):
        g = g.sort_values("mileage")
        if len(g) < k:
            continue
        m = len(g) - k + 1
        idx = (np.sort(rng.choice(m, size=max_win, replace=False))
               if m > max_win else np.arange(m))
        p = g[PRED_COL].to_numpy(dtype=float)
        y = g["capacity"].to_numpy(dtype=float)
        d = g["d_soc"].to_numpy(dtype=float)
        q = g["quality"].to_numpy(dtype=float) if "quality" in g else np.ones_like(p)
        for s in idx:
            rows.append((car,
                         _aggregate(p[s:s + k], how, q[s:s + k], d[s:s + k]),
                         float(y[s:s + k].mean()),
                         float(d[s:s + k].mean()), k))
    return pd.DataFrame(rows, columns=["car", "pred", "capacity", "d_soc", "k"])


def car_bias(cal: pd.DataFrame, m: int, lam: float = BIAS_LAM) -> dict:
    """从校准池**最近**的 m 段估每车偏差（收缩到 0，与 §15.4 同一套）。"""
    out = {}
    for car, g in cal.groupby("car", sort=False):
        g = g.sort_values("mileage").tail(m)
        out[car] = 0.0 if m == 0 else fit_bias(g[PRED_COL].to_numpy(dtype=float),
                                               g["capacity"].to_numpy(dtype=float),
                                               lam=lam)
    return out


def decompose(w: pd.DataFrame) -> dict:
    """窗级误差的方差分解：车级偏差 / 车内（模型+标签）。"""
    r = w["pred"].to_numpy(dtype=float) - w["capacity"].to_numpy(dtype=float)
    s = pd.DataFrame({"r": r, "car": w["car"].to_numpy()})
    g = s.groupby("car")["r"]
    n_c = g.size().to_numpy(dtype=float)
    mean_c = g.mean().to_numpy(dtype=float)
    ss_within = float(np.sum((s["r"] - s.groupby("car")["r"].transform("mean")) ** 2))
    sig_w2 = ss_within / max(float(len(s) - len(n_c)), 1.0)
    var_means = float(np.var(mean_c, ddof=1))
    sig_b2 = max(var_means - sig_w2 * float(np.mean(1.0 / n_c)), 0.0)
    return {"sigma_between": float(np.sqrt(sig_b2)),
            "sigma_within": float(np.sqrt(sig_w2)),
            "rmse": float(np.sqrt(np.mean(r ** 2)))}


# ---------------------------------------------------------------- 主流程
def main() -> None:
    cfg = load_config()
    mdir = cfg["paths"]["metrics_dir"]
    oof = pd.read_parquet(os.path.join(mdir, "oof_predictions.parquet"))
    oof = oof.dropna(subset=["car", "capacity", PRED_COL, "mileage", "d_soc"]).reset_index(drop=True)
    ref_mean = float(oof["bol_ref"].mean()) if "bol_ref" in oof else float(oof["capacity"].mean())

    d = split_chrono(oof)
    cal_all = d[d["part"] == "cal"].reset_index(drop=True)
    ev = d[d["part"] == "eval"].reset_index(drop=True)
    n_cars = ev["car"].nunique()
    print(f"样本 {len(oof)} 段 / {oof['car'].nunique()} 车；"
          f"进入实验 {len(d)} 段 / {n_cars} 车（≥{MIN_SESS} 段）", flush=True)
    print(f"校准池 {len(cal_all)} 段，评估集 {len(ev)} 段", flush=True)

    base = regression_metrics(ev["capacity"].to_numpy(dtype=float),
                              ev[PRED_COL].to_numpy(dtype=float))
    print(f"评估集上的零样本片段级 RMSE {base['rmse']:.4f} / MAE {base['mae']:.4f}", flush=True)

    # ---------------- ① 聚合器比较（m=0，纯免标签口径） ----------------
    aggs = {}
    for k in (3, 8):
        rows = {}
        for how in AGGS:
            w = build_windows(ev, k, how)
            mm = regression_metrics(w["capacity"].to_numpy(dtype=float),
                                    w["pred"].to_numpy(dtype=float))
            rows[how] = {**mm, "n_windows": int(len(w))}
        best = min(rows.values(), key=lambda r: r["rmse"])
        for how, r in rows.items():
            r["vs_mean_pct"] = (r["rmse"] / rows["mean"]["rmse"] - 1.0) * 100.0
        aggs[str(k)] = rows
        print(f"  k={k} 聚合器：" + "  ".join(
            f"{how} {r['rmse']:.4f}({r['vs_mean_pct']:+.1f}%)" for how, r in rows.items()),
            flush=True)
        assert best is not None

    # ---------------- ② (k, m) 网格 ----------------
    biases = {m: car_bias(cal_all, m) for m in M_LIST}
    grid, intervals = {}, {}
    for k in K_LIST:
        w = build_windows(ev, k, "mean")
        y = w["capacity"].to_numpy(dtype=float)
        p0 = w["pred"].to_numpy(dtype=float)
        cars = w["car"].to_numpy()
        grid[str(k)], intervals[str(k)] = {}, {}
        for m in M_LIST:
            b = np.array([biases[m].get(c, 0.0) for c in cars])
            p = p0 + b
            mm = regression_metrics(y, p)
            cv = conformal_by_car(cars, y, p, nominal=NOMINAL)
            grid[str(k)][str(m)] = {**mm, "rmse_pp": mm["rmse"] / ref_mean * 100.0}
            intervals[str(k)][str(m)] = cv
        print(f"  k={k:2d}：" + "  ".join(
            f"m={m} {grid[str(k)][str(m)]['rmse']:.3f}" for m in M_LIST), flush=True)

    # ---------------- ③ 可加性检验 ----------------
    # 只用 k=1（单片段窗）估出每个 m 的 (σ_b, σ_w)，再去预测**整张网格**：
    #     RMSE(k, m) ≈ √(σ_b(m)² + σ_w(m)²/k)
    # 这是真检验而不是拟合——(σ_b, σ_w) 没有见过 k>1 的任何数据。
    add, comp = {}, {}
    for m in M_LIST:
        w1 = build_windows(ev, 1, "mean")
        cars = w1["car"].to_numpy()
        p1 = w1["pred"].to_numpy(dtype=float) + np.array([biases[m].get(c, 0.0) for c in cars])
        dec = decompose(w1.assign(pred=p1))
        comp[str(m)] = dec
        for k in K_LIST:
            pred_rmse = float(np.sqrt(dec["sigma_between"] ** 2 + dec["sigma_within"] ** 2 / k))
            measured = grid[str(k)][str(m)]["rmse"]
            add.setdefault(str(k), {})[str(m)] = {
                "measured": measured, "predicted": pred_rmse,
                "sigma_between": dec["sigma_between"], "sigma_within": dec["sigma_within"],
                "delta_pct": (measured / pred_rmse - 1.0) * 100.0}

    # 车级偏差被个性化消掉了多少（相对 σ_b 的方差份额），以及收缩系数给的上限
    bias_var = {}
    for m in M_LIST:
        shrink = 0.0 if m == 0 else m / (m + BIAS_LAM)
        bias_var[str(m)] = {"sigma_between": comp[str(m)]["sigma_between"],
                            "removed_pct": None,  # 下面用 m=0 补齐
                            "shrink_ceiling_pct": (1.0 - (1.0 - shrink) ** 2) * 100.0}
    s0 = bias_var["0"]["sigma_between"]
    for m in M_LIST:
        v = bias_var[str(m)]["sigma_between"]
        bias_var[str(m)]["removed_pct"] = (1.0 - (v / s0) ** 2) * 100.0 if s0 > 0 else 0.0

    # 偏差是"常数偏移"还是"会漂移"？分别用校准池与评估集各估一次每车偏差，看两者的关系。
    # 若偏差稳定，两半应高度相关、斜率接近 1；若漂移，斜率 <1 且差值方差变大。
    w1 = build_windows(ev, 1, "mean")
    cars1 = w1["car"].to_numpy()
    b_ev = pd.Series(w1["pred"].to_numpy(dtype=float) - w1["capacity"].to_numpy(dtype=float)) \
        .groupby(cars1).mean()
    b_cal = pd.Series(cal_all[PRED_COL].to_numpy(dtype=float)
                      - cal_all["capacity"].to_numpy(dtype=float)) \
        .groupby(cal_all["car"].to_numpy()).mean()
    common = b_ev.index.intersection(b_cal.index)
    x = b_cal.loc[common].to_numpy(dtype=float)
    z = b_ev.loc[common].to_numpy(dtype=float)
    slope = float(np.polyfit(x, z, 1)[0]) if x.size > 2 else float("nan")
    drift = {"n_cars": int(x.size),
             "corr": float(np.corrcoef(x, z)[0, 1]) if x.size > 2 else float("nan"),
             "slope": slope,
             "var_cal": float(np.var(x, ddof=1)),
             "var_eval": float(np.var(z, ddof=1)),
             "var_diff": float(np.var(z - x, ddof=1)),
             "half_share_pct": float(np.var(z - x, ddof=1) / (2.0 * np.var(x, ddof=1)) * 100.0)}

    # ---------------- ④ 部署对照：多跑 k 次 vs 多标 m 段 ----------------
    k_best = 8
    deploy = {
        "k1_m0": {"rmse": grid["1"]["0"]["rmse"], "width": intervals["1"]["0"]["width"]},
        "k8_m0": {"rmse": grid["8"]["0"]["rmse"], "width": intervals["8"]["0"]["width"]},
        "k8_m2": {"rmse": grid["8"]["2"]["rmse"], "width": intervals["8"]["2"]["width"]},
        "k8_m8": {"rmse": grid["8"]["8"]["rmse"], "width": intervals["8"]["8"]["width"]},
        "k12_m8": {"rmse": grid["12"]["8"]["rmse"], "width": intervals["12"]["8"]["width"]},
    }
    # 反解等效独立样本数：聚合与个性化各自把误差降到哪一步
    k_eff = {}
    for m in M_LIST:
        sb2 = comp[str(m)]["sigma_between"] ** 2
        sw2 = comp[str(m)]["sigma_within"] ** 2
        r2 = grid["12"][str(m)]["rmse"] ** 2
        k_eff[str(m)] = float(sw2 / (r2 - sb2)) if r2 > sb2 else float("nan")

    out = {
        "n_sessions": int(len(oof)), "n_cars": int(n_cars),
        "n_eval_sessions": int(len(ev)), "n_cal_sessions": int(len(cal_all)),
        "min_sessions": MIN_SESS, "eval_frac": EVAL_FRAC, "max_win": MAX_WIN,
        "bias_lam": BIAS_LAM, "k_list": K_LIST, "m_list": M_LIST,
        "baseline": base, "aggregators": aggs, "grid": grid,
        "intervals": intervals, "additivity": add, "bias_variance": bias_var,
        "bias_drift": drift, "k_eff_m": k_eff, "deploy": deploy,
        "note": ("每车按里程前 50% 做校准池、后 50% 做评估集；bias 只从校准池最近 m 段估计，"
                 "收缩 λ=5；窗集合与聚合器无关，六种聚合器成对可比。"),
    }
    save_json(out, "aggregate_personalize")
    print("写入 results/metrics/aggregate_personalize.json", flush=True)

    print("可加性：实测 vs √(σ_b²+σ_w²/k)：" + "  ".join(
        f"k={k},m=0 {add[str(k)]['0']['measured']:.3f}/{add[str(k)]['0']['predicted']:.3f}"
        for k in K_LIST), flush=True)
    print("车级偏差被消掉：" + "  ".join(
        f"m={m} {bias_var[str(m)]['removed_pct']:.0f}%(上限{bias_var[str(m)]['shrink_ceiling_pct']:.0f}%)"
        for m in M_LIST), flush=True)
    print(f"偏差漂移：两半相关 {drift['corr']:.3f}、斜率 {drift['slope']:.3f}，"
          f"不共享的方差占 {drift['half_share_pct']:.0f}%", flush=True)
    print("部署对照：" + "  ".join(
        f"{k} {v['rmse']:.3f}Ah/{v['width']:.2f}Ah" for k, v in deploy.items()), flush=True)

    plots(out, cfg["paths"]["figures_dir"])


# ---------------------------------------------------------------- 图
def plots(d: dict, fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8),
                             gridspec_kw={"width_ratios": [1.08, 1.0]})

    # 左：(k, m) 网格。颜色编码"相对单片段零样本的改善"，格子里写绝对 RMSE——
    # 两者分开，否则 k=1..3 与 k>=5 两片色块会把行内的 m 效应吃掉。
    M = np.array([[d["grid"][str(k)][str(m)]["rmse"] for m in M_LIST] for k in K_LIST])
    base = M[0, 0]
    G = (1.0 - M / base) * 100.0
    im = axes[0].imshow(G, cmap="YlGnBu", aspect="auto", vmin=0.0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axes[0].text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7.6,
                         color="white" if G[i, j] > G.max() * 0.58 else "#222222")
    axes[0].set_xticks(range(len(M_LIST)))
    axes[0].set_xticklabels([f"m={m}" for m in M_LIST], fontsize=8)
    axes[0].set_yticks(range(len(K_LIST)))
    axes[0].set_yticklabels([f"k={k}" for k in K_LIST], fontsize=8)
    axes[0].set_xlabel("标注会话数 m（个性化，需真值）")
    axes[0].set_ylabel("聚合段数 k（同车平均，免费）")
    axes[0].set_title(f"RMSE (Ah) 与相对单片段零样本的改善（{base:.3f} Ah）",
                      fontsize=9.3, pad=8)
    axes[0].grid(visible=False)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.038, pad=0.02)
    cb.set_label("改善 (%)", fontsize=7.6)
    cb.ax.tick_params(labelsize=7)

    # 右：区间宽度随 k 的变化，一个 m 一条线（部署时看的就是这个数）
    colors = [C_GRAY, C_ACC, C_MAIN, C_PURPLE, C_RED, "#8c564b"]
    for j, m in enumerate(M_LIST):
        w = [d["intervals"][str(k)][str(m)]["width"] for k in K_LIST]
        axes[1].plot(K_LIST, w, marker="o", ms=3.6, lw=1.5, color=colors[j % len(colors)],
                     label=f"m={m}")
    w0 = d["intervals"]["1"]["0"]["width"]
    w1 = d["intervals"]["8"]["8"]["width"]
    axes[1].set_ylim(min(w0, w1) - 0.06, w0 + 0.10)
    axes[1].annotate(f"{w0:.2f}", (1, w0), textcoords="offset points", xytext=(7, -3),
                     fontsize=7.6, color=C_GRAY)
    axes[1].annotate(f"{w1:.2f}", (8, w1), textcoords="offset points", xytext=(-4, -12),
                     fontsize=7.6, color="#8c564b")
    axes[1].set_xscale("log")
    axes[1].set_xticks(K_LIST)
    axes[1].set_xticklabels([str(k) for k in K_LIST], fontsize=8)
    axes[1].set_xlabel("聚合段数 k")
    axes[1].set_ylabel("90% 区间宽度 (Ah)")
    axes[1].set_title("区间也跟着收，覆盖率不变", fontsize=9.5, pad=8)
    axes[1].legend(fontsize=7.2, ncol=2, loc="upper right", framealpha=0.92)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, "fig37_聚合与个性化网格.png"))

    # ---- 图 38：可加性与偏差削减 ----
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    xs, ys, cs = [], [], []
    for k in K_LIST:
        for m in M_LIST:
            a = d["additivity"][str(k)][str(m)]
            xs.append(a["predicted"])
            ys.append(a["measured"])
            cs.append(m)
    lim = [min(xs + ys) * 0.97, max(xs + ys) * 1.02]
    axes[0].plot(lim, lim, ls="--", lw=1.0, color=C_GRAY, label="y = x")
    sc = axes[0].scatter(xs, ys, c=cs, cmap="viridis", s=26, edgecolor="white", lw=0.5)
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("预测 √(σ_b²+σ_w²/k)  (Ah)")
    axes[0].set_ylabel("实测 RMSE  (Ah)")
    axes[0].set_title("误差预算能预测整张网格吗？", fontsize=9.5, pad=8)
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.038, pad=0.02)
    cb.set_label("m", fontsize=7.6)
    cb.ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7.4, loc="upper left")
    axes[0].text(0.97, 0.06, "全部落在 y=x 之上：1/√k 高估了收益",
                 transform=axes[0].transAxes, ha="right", fontsize=7.2, color=C_RED)

    ms = [m for m in M_LIST if m > 0]
    rem = [d["bias_variance"][str(m)]["removed_pct"] for m in ms]
    ceil = [d["bias_variance"][str(m)]["shrink_ceiling_pct"] for m in ms]
    xi = np.arange(len(ms))
    axes[1].bar(xi - 0.19, rem, width=0.36, color=C_MAIN, label="实测消掉的偏差方差")
    axes[1].bar(xi + 0.19, ceil, width=0.36, color=C_GRAY, alpha=0.75,
                label="收缩系数给的上限")
    for i, (a, b) in enumerate(zip(rem, ceil)):
        axes[1].text(i - 0.19, a + 1.5, f"{a:.0f}%", ha="center", fontsize=7.2)
        axes[1].text(i + 0.19, b + 1.5, f"{b:.0f}%", ha="center", fontsize=7.2, color="#555555")
    axes[1].set_xticks(xi)
    axes[1].set_xticklabels([f"m={m}" for m in ms], fontsize=8)
    axes[1].set_ylabel("车级偏差方差削减 (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("个性化只买得起偏差的一部分（m=0 为参照，不画）", fontsize=9.2, pad=8)
    axes[1].legend(fontsize=7.4, loc="upper left", framealpha=0.92)
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, "fig38_误差预算的可加性.png"))


if __name__ == "__main__":
    main()
