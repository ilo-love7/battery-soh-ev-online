# -*- coding: utf-8 -*-
"""24 折算的边界：一条线折不动不同的车。

第七轮给出一条全局温度斜率（+0.0449 Ah/°C）并证明折算到 25°C 不花精度。
但那条线是**所有车平均**出来的，分厂商看就已经差一倍（ds1 0.0327 / ds2 0.0682）。
于是"能不能给每辆车单独折"成了绕不过去的问题。这一轮的答案是"不能，而且不需要"：

  A. 单车斜率**估得出、但不可复现**。单看一辆车，斜率标准误中位只有 0.0061，
     矩估计给出"异质性" τ=0.043（占观测离散度的 84%）；但把同一辆车的历史
     切成前 60% / 后 40%，两半估出的斜率相关只有 **0.235**，差值 rms 0.087
     ——是抽样噪声的 **5.5 倍**。也就是说"车与车不一样"里能把持住的很少，
     大部分是随历史长度、季节覆盖、块噪声一起变的东西。
     学习曲线：要 180–250 段历史（≈ 一整段记录）才能把斜率误差压到 0.03 以下。
     λ 曲线（前 60% 估、后 40% 验收）：不折算 0.561 → 全局 0.530 → 最优 λ=0.2
     0.525 → 全用自己的斜率 λ=1 反而 0.546；自己的斜率只在 **43%** 的车上赢过全局，
     即使把全史斜率当 oracle 用，也只与全局持平（0.530）。逐车折算的全部空间 <1%。
  B. 按层（数据集/厂商）折算**只对 ds2 的一小部分车值钱**：留一车检验的合并残差
     0.744 → 0.684（全局）→ 0.676（层内，比全局好 1.2%）→ 0.642（oracle）。
     ds2 的逐车配对差 −0.015 Ah（胜率 59%），ds1 −0.003（54%），ds3 +0.005（43%）。
     拿不到层标签时，全局线已经拿到九成收益。
  C. 折算后仍剩一段**偏置漂移**，但它已经不是温度：车级前后半偏差差值与 ΔT 的
     R² 只有 0.040（系数 −0.010，符号还反了）。工况构成解释 17%，合计 20%，
     四分之三无法归因；而偏置方差的 **61% 落在复制块之间**——它是台阶，不是斜坡，
     更像车端容量估计口径按块切换。第七轮"不是季节"的判断由此收口。
  D. 车队分档：车级折算中位 0.71 pp（p95 1.28），与窗口温度偏移相关 −0.826。
     纯采样误差只有 0.49 pp，而实测车级误差 1.05 pp——档宽必须按后者设计：
     5 pp 档宽错档 17%、2.5 pp 档宽 34%、1 pp 档宽 65%；不折算时，
     5 pp 档宽下 **15% 的车会因为季节换档**。

口径说明：所有"哪种斜率更好"的比较一律用**逐车配对差**与**合并平方误差**两个口径，
不用"层内水平中位数之差"——中位数不可加，层内会与配对差反号（见 §21.6 踩坑）。

输出
  results/metrics/debias_transfer.json
  results/metrics/debias_transfer_cars.parquet     （A 的逐车斜率与两半估计）
  results/metrics/debias_transfer_drift.parquet    （C 的逐车偏置漂移）
  results/metrics/debias_transfer_fleet.parquet    （D 的车级 SOH 与折算量）
  results/figures/fig45_单车斜率的可识别性.png
  results/figures/fig46_分层折算的边际价值.png
  results/figures/fig47_残余漂移的归因.png
  results/figures/fig48_车队分档的分辨率.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import tier_error_rate  # noqa: E402
from evsoh.panel import (fe_ols, shrink_slopes, tau2_moments,  # noqa: E402
                         trend_resid_sd)
from evsoh.plotting import (C_ACC, C_GRAY, C_GREEN, C_MAIN, C_PURPLE,  # noqa: E402
                            C_RED, plt, savefig)

REF_T = 25.0
MIN_SESS = 24
K_WIN = 8                       # 车级上报窗口，沿用 §18/§19/§20 的饱和点
SPLIT = 0.6                     # 前 60% 估斜率、后 40% 验收
LAMS = np.round(np.arange(0.0, 1.01, 0.1), 2)
LEARN_N = [10, 20, 40, 60, 90, 130, 180, 250]
SIGMA_PP = [0.5, 1.05]          # 车级误差情景（1.05 = 第六轮实测车级 MAE）
WIDTHS = [1.0, 2.0, 2.5, 3.0, 5.0]
SMOKE = bool(os.environ.get("SMOKE"))


# ---------------------------------------------------------------- 数据
def load(cfg: dict) -> pd.DataFrame:
    """OOF 表 + 第七轮的折算标签/折算模型预测，按 (car, segment) 对齐。"""
    a = pd.read_parquet(os.path.join(cfg["paths"]["metrics_dir"], "oof_predictions.parquet"))
    t = pd.read_parquet(os.path.join(cfg["paths"]["metrics_dir"], "temperature_oof.parquet"))
    d = a.merge(t[["car", "segment", "y_ref", "pred_ref"]], on=["car", "segment"], how="inner")
    d = d.dropna(subset=["car", "capacity", "y_ref", "pred_ref", "pred_xgb",
                         "mileage", "t_mean", "bol_ref"]).copy()
    d = d.sort_values(["car", "mileage"]).reset_index(drop=True)
    n = d.groupby("car")["car"].transform("size").to_numpy()
    d = d[n >= MIN_SESS].reset_index(drop=True)
    # 复制块：容量与上一段完全相同（与 §19/§20 同口径）
    same = d.groupby("car")["capacity"].diff().eq(0).fillna(False).to_numpy()
    d["block_id"] = np.cumsum(~same)
    if SMOKE:
        keep = sorted(d["car"].unique())[:36]
        d = d[d["car"].isin(keep)].reset_index(drop=True)
    return d


def _pooled_sd(vars_: np.ndarray, ns: np.ndarray) -> float:
    """合并方差：按自由度加权（不是简单平均，样本多的车权重大）。"""
    vars_, ns = np.asarray(vars_, float), np.asarray(ns, float)
    dof = np.clip(ns - 1, 1.0, None)
    ok = np.isfinite(vars_)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.sum(vars_[ok] * dof[ok]) / np.sum(dof[ok])))


def _car_slope(s: pd.DataFrame, y: str = "capacity") -> dict:
    f = fe_ols(s, y, ["t_mean", "mileage"], cluster=False)
    return {"b": float(f["beta"][0]), "se": float(f["se"][0]),
            "resid": f["resid"], "n": int(len(s))}


# ---------------------------------------------------------------- A 可识别性
def part_a(d: pd.DataFrame) -> dict:
    rows = []
    for c, s in d.groupby("car"):
        s = s.sort_values("mileage")
        full = _car_slope(s)
        k = int(len(s) * SPLIT)
        if k < 8 or len(s) - k < 6:
            continue
        h1, h2 = _car_slope(s.iloc[:k]), _car_slope(s.iloc[k:])
        u = full["resid"]
        rows.append({"car": c, "b_full": full["b"], "se_full": full["se"],
                     "b1": h1["b"], "se1": h1["se"], "b2": h2["b"], "se2": h2["se"],
                     "n": len(s), "span": float(s["t_mean"].max() - s["t_mean"].min()),
                     "span1": float(s.iloc[:k]["t_mean"].max() - s.iloc[:k]["t_mean"].min()),
                     "span2": float(s.iloc[k:]["t_mean"].max() - s.iloc[k:]["t_mean"].min()),
                     "ac": float(np.corrcoef(u[:-1], u[1:])[0, 1]) if len(u) > 5 else np.nan})
    r = pd.DataFrame(rows).set_index("car")

    # 残差自相关造成的方差低估：se 近似乘 sqrt((1+ρ)/(1−ρ))
    rho = float(np.nanmedian(r["ac"].to_numpy()))
    infl = (1 + rho) / (1 - rho)
    het = tau2_moments(r["b_full"], r["se_full"], inflate=infl)

    # 稳定性：前后半差值里有多少是"抽样噪声解释不了"的
    obs = float(np.mean((r["b1"] - r["b2"]) ** 2) / 2)
    noise = float(np.mean((r["se1"] ** 2 + r["se2"] ** 2)) / 2)
    stab = {"corr_half": float(r["b1"].corr(r["b2"])),
            "corr_half_full": float(r["b1"].corr(r["b_full"])),
            "rms_diff": float(np.sqrt(np.mean((r["b1"] - r["b2"]) ** 2))),
            "rms_diff_vs_full": float(np.sqrt(np.mean((r["b1"] - r["b_full"]) ** 2))),
            "var_excess": obs - noise, "var_excess_ratio": obs / noise if noise > 0 else np.nan,
            "se_full_median": float(r["se_full"].median()),
            "se_half_median": float(np.median(np.r_[r["se1"], r["se2"]])),
            "b_sd": float(r["b_full"].std())}

    # 学习曲线：前 n 段估出的斜率离"全史斜率"还有多远
    learn = []
    for n in LEARN_N:
        dif, ses = [], []
        for c, s in d.groupby("car"):
            s = s.sort_values("mileage")
            if len(s) < n:
                continue
            try:
                fit = _car_slope(s.iloc[:n])
            except (ValueError, np.linalg.LinAlgError):
                continue
            dif.append(fit["b"] - r.loc[c, "b_full"] if c in r.index else np.nan)
            ses.append(fit["se"])
        if len(dif) >= 5:
            learn.append({"n": n, "n_cars": len(dif),
                          "rms": float(np.sqrt(np.nanmean(np.square(dif)))),
                          "bias": float(np.nanmean(dif)),
                          "se_median": float(np.median(ses)),
                          "share_se_lt_02": float(np.mean(np.array(ses) < 0.02))})

    # λ 曲线：前半段估自己的斜率、后半段验收；λ=0 全用别人，λ=1 全用自己
    # 口径说明：合并方差会被极少数残差 2 Ah 以上的车主导（逐车 sd 从 0.4 到 2.9），
    # 所以主口径改成"每辆车自己相对不折算的改善，再取中位数"，合并值只作对照。
    b_global = float(fe_ols(d, "capacity", ["t_mean", "mileage"])["beta"][0])
    tails = {}
    for c, s in d.groupby("car"):
        if c not in r.index:
            continue
        s = s.sort_values("mileage")
        tail = s.iloc[int(len(s) * SPLIT):]
        tails[c] = (tail, tail["t_mean"].to_numpy(dtype=float) - REF_T)
    n_tail = np.array([len(t) for t, _ in tails.values()])

    def sd_by_car(weights: dict) -> pd.Series:
        return pd.Series({c: trend_resid_sd(tail, "capacity", "mileage",
                                            shift=weights[c] * tt)
                          for c, (tail, tt) in tails.items()})

    base = sd_by_car({c: 0.0 for c in tails})
    lam_tbl = []
    for lam in LAMS:
        sd = sd_by_car({c: lam * r.loc[c, "b1"] + (1 - lam) * b_global for c in tails})
        rel = (1 - sd / base) * 100
        lam_tbl.append({"lam": float(lam),
                        "median_sd": float(sd.median()),
                        "median_rel_gain_pct": float(rel.median()),
                        "pooled_sd": _pooled_sd(sd.to_numpy() ** 2, n_tail),
                        "share_better_than_none": float((sd < base).mean())})
    best = min(lam_tbl, key=lambda x: x["median_sd"])

    # 经验贝叶斯权重：λ_c = τ²/(τ²+se_c²)，估得糊的车自动退回全局值
    keys = list(tails)
    wt = shrink_slopes(r.loc[keys, "b1"].to_numpy(), r.loc[keys, "se1"].to_numpy(),
                       b_global, tau2=het["tau2"])["weight"]
    eb = sd_by_car({c: b_global + float(w) * (r.loc[c, "b1"] - b_global)
                    for c, w in zip(keys, wt)})
    orc = sd_by_car({c: r.loc[c, "b_full"] for c in tails})
    sd0 = sd_by_car({c: b_global for c in tails})
    sd1 = sd_by_car({c: r.loc[c, "b1"] for c in tails})

    out = {
        "hetero": het,
        "stability": stab,
        "learning": learn,
        "lam_curve": lam_tbl,
        "lam_best": best,
        "none_sd_median": float(base.median()),
        "global_sd_median": float(sd0.median()),
        "oracle_sd_median": float(orc.median()),
        "eb_sd_median": float(eb.median()),
        "own_sd_median": float(sd1.median()),
        "gain_global_pct": float((1 - sd0.median() / base.median()) * 100),
        "gain_oracle_pct": float((1 - orc.median() / base.median()) * 100),
        "gain_own_pct": float((1 - sd1.median() / base.median()) * 100),
        "gain_eb_pct": float((1 - eb.median() / base.median()) * 100),
        "eb_weight_mean": float(np.mean(wt)),
        "eb_weight_median": float(np.median(wt)),
        "share_global_better": float((sd0 < base).mean()),
        "share_own_better_than_global": float((sd1 < sd0).mean()),
        "share_oracle_better_than_global": float((orc < sd0).mean()),
        "pooled": {"none": float(_pooled_sd(base.to_numpy() ** 2, n_tail)),
                   "global": float(_pooled_sd(sd0.to_numpy() ** 2, n_tail)),
                   "oracle": float(_pooled_sd(orc.to_numpy() ** 2, n_tail))},
        "tail_span_median": float(np.median(r["span2"])),
        "n_cars": int(len(r)),
        "b_global": b_global,
    }
    out["_r"] = r
    out["_sd"] = pd.DataFrame({"none": base, "global": sd0, "own": sd1,
                               "oracle": orc, "eb": eb})
    return out


# ---------------------------------------------------------------- B 分层
def part_b(d: pd.DataFrame, a: dict) -> dict:
    b_global = a["b_global"]
    tiers = sorted(d["dataset"].unique())
    # 层内全样本斜率（按车聚类，得到诚实的区间）
    tier_fit = {}
    for t in tiers:
        s = d[d["dataset"] == t]
        f = fe_ols(s, "capacity", ["t_mean", "mileage"])
        tier_fit[t] = {"b": float(f["beta"][0]), "se": float(f["se"][0]),
                       "n": int(len(s)), "n_cars": int(s["car"].nunique())}

    # 留一车：全局斜率与其他车的层内斜率都不许看这辆车
    rows = []
    for c, s in d.groupby("car"):
        t = s["dataset"].iloc[0]
        other = d[d["car"] != c]
        g = fe_ols(other, "capacity", ["t_mean", "mileage"], cluster=False)["beta"][0]
        tt = other[other["dataset"] == t]
        if tt["car"].nunique() >= 3:
            bt = fe_ols(tt, "capacity", ["t_mean", "mileage"], cluster=False)["beta"][0]
        else:
            bt = g
        tv = s["t_mean"].to_numpy(dtype=float) - REF_T
        own = a["_r"].loc[c, "b_full"] if c in a["_r"].index else g
        rows.append({"car": c, "dataset": t, "n": int(len(s)),
                     "b_global": g, "b_tier": bt, "b_own": own,
                     "sd_none": trend_resid_sd(s, "capacity", "mileage"),
                     "sd_global": trend_resid_sd(s, "capacity", "mileage", shift=g * tv),
                     "sd_tier": trend_resid_sd(s, "capacity", "mileage", shift=bt * tv),
                     "sd_own": trend_resid_sd(s, "capacity", "mileage", shift=own * tv)})
    r = pd.DataFrame(rows)

    def agg(sub):
        ns = sub["n"].to_numpy(dtype=float)
        med = {k: float(sub["sd_" + k].median()) for k in ("none", "global", "tier", "own")}
        pool = {k: _pooled_sd(sub["sd_" + k] ** 2, ns) for k in ("none", "global", "tier", "own")}
        out = {"n_cars": int(len(sub))}
        out.update({f"pool_{k}": v for k, v in pool.items()})
        out.update({f"med_{k}": v for k, v in med.items()})
        for k in ("global", "tier", "own"):
            out[f"pool_gain_{k}_pct"] = (1 - pool[k] / pool["none"]) * 100
            out[f"med_gain_{k}_pct"] = (1 - med[k] / med["none"]) * 100
        # 逐车配对差：中位数不可加，"水平中位之差"与"配对差的中位"会反号，
        # 所以比较两种斜率时一律用配对差，层内水平中位只作描述。
        out["paired_tier_vs_global"] = float((sub["sd_tier"] - sub["sd_global"]).median())
        out["paired_own_vs_global"] = float((sub["sd_own"] - sub["sd_global"]).median())
        out["pool_tier_vs_global_pct"] = (1 - pool["tier"] / pool["global"]) * 100
        out["share_tier_better_than_global"] = float((sub["sd_tier"] < sub["sd_global"]).mean())
        out["share_own_better_than_global"] = float((sub["sd_own"] < sub["sd_global"]).mean())
        return out

    overall = agg(r)
    by_tier = {}
    for t in tiers:
        sub = r[r["dataset"] == t]
        if len(sub) < 3:
            continue
        by_tier[t] = agg(sub)
    return {"tier_fit": tier_fit, "overall": overall, "by_tier": by_tier,
            "_r": r}


# ---------------------------------------------------------------- C 残余漂移
def part_c(d: pd.DataFrame) -> dict:
    rows = []
    for c, s in d.groupby("car"):
        s = s.sort_values("mileage")
        k = len(s) // 2
        a1, a2 = s.iloc[:k], s.iloc[k:]
        if len(a1) < 6 or len(a2) < 6:
            continue
        res1 = (a1["y_ref"] - a1["pred_ref"]).to_numpy()
        res2 = (a2["y_ref"] - a2["pred_ref"]).to_numpy()
        rows.append({
            "car": c, "dataset": s["dataset"].iloc[0],
            "db": float(res2.mean() - res1.mean()),
            "db_raw": float((a2["capacity"] - a2["pred_xgb"]).mean()
                            - (a1["capacity"] - a1["pred_xgb"]).mean()),
            "dt": float(a2["t_mean"].mean() - a1["t_mean"].mean()),
            "d_soc": float(a2["d_soc"].mean() - a1["d_soc"].mean()),
            "d_i": float(a2["i_abs_mean"].mean() - a1["i_abs_mean"].mean()),
            "d_q": float(a2["quality"].mean() - a1["quality"].mean()),
            "d_km": float(a2["mileage"].mean() - a1["mileage"].mean()),
            "d_copy": float((s["block_id"].diff().fillna(1) == 0).iloc[k:].mean()
                            - (s["block_id"].diff().fillna(1) == 0).iloc[:k].mean()),
            "sd_within": float(np.mean([res1.std(ddof=1), res2.std(ddof=1)])),
        })
    r = pd.DataFrame(rows)

    def r2(y, cols, sub=None):
        s = r if sub is None else sub
        X = np.c_[np.ones(len(s)), s[cols].to_numpy(dtype=float)]
        yy = s[y].to_numpy(dtype=float)
        b = np.linalg.lstsq(X, yy, rcond=None)[0]
        e = yy - X @ b
        return 1 - float(np.var(e) / np.var(yy)), b

    conds = ["d_soc", "d_i", "d_q"]
    m_t, b_t = r2("db", ["dt"])
    m_c, b_c = r2("db", conds)
    m_all, b_all = r2("db", ["dt"] + conds)
    m_full, b_full = r2("db", ["dt"] + conds + ["d_km", "d_copy"])
    # 纯工况（先扣掉 Δt 能解释的部分）的增量
    resid_dt = r["db"].to_numpy() - (b_t[0] + b_t[1] * r["dt"].to_numpy())
    inc = 1 - float(np.var(resid_dt - (np.c_[np.ones(len(r)), r[conds].to_numpy(float)]
                                       @ np.linalg.lstsq(np.c_[np.ones(len(r)),
                                                               r[conds].to_numpy(float)],
                                                         resid_dt, rcond=None)[0]))
                     / np.var(resid_dt))

    # 块结构：偏置是"连续漂移"还是"块与块之间的台阶"
    between, within, tot = [], [], []
    for c, s in d.groupby("car"):
        s = s.assign(res=s["y_ref"] - s["pred_ref"])
        gm = s.groupby("block_id")["res"].agg(["mean", "var", "size"]).dropna()
        if len(gm) < 3:
            continue
        w = gm["size"] / gm["size"].sum()
        mu = float((w * gm["mean"]).sum())
        between.append(float((w * (gm["mean"] - mu) ** 2).sum()))
        within.append(float((w * gm["var"].fillna(0)).sum()))
    bv, wv = float(np.mean(between)), float(np.mean(within))
    return {"n_cars": int(len(r)),
            "db_sd": float(r["db"].std()), "db_mean": float(r["db"].mean()),
            "db_sd_raw": float(r["db_raw"].std()),
            "r2_dt": float(m_t), "r2_cond": float(m_c), "r2_all": float(m_all),
            "r2_plus_km_copy": float(m_full),
            "coef_dt": float(b_t[1]), "coef_cond": dict(zip(conds, np.round(b_c[1:], 5).tolist())),
            "coef_all": dict(zip(["dt"] + conds, np.round(b_all[1:], 5).tolist())),
            "coef_plus_km_copy": dict(zip(["d_km", "d_copy"],
                                          np.round(b_full[-2:], 8).tolist())),
            "cond_increment_after_dt": float(inc),
            "share_dt": float(m_t), "share_cond_only": float(m_c),
            "share_unexplained": float(1 - m_all),
            "block_var_between": bv, "block_var_within": wv,
            "block_share_between": bv / (bv + wv) if (bv + wv) > 0 else np.nan,
            "corr_db_dt": float(r["db"].corr(r["dt"])),
            "corr_dbraw_dt": float(r["db_raw"].corr(r["dt"])),
            "_r": r}


# ---------------------------------------------------------------- D 分档
def _tier(x, edges):
    return np.digitize(np.asarray(x, dtype=float), edges)


def part_d(d: pd.DataFrame) -> dict:
    # 车级：最近 K_WIN 段，折算后标签 / 未折算标签
    rows = []
    for c, s in d.groupby("car"):
        s = s.sort_values("mileage").iloc[-K_WIN:]
        bol = float(s["bol_ref"].iloc[0])
        rows.append({"car": c, "dataset": s["dataset"].iloc[0], "bol": bol,
                     "soh_raw": float(s["capacity"].mean() / bol * 100),
                     "soh_ref": float(s["y_ref"].mean() / bol * 100),
                     "t_win": float(s["t_mean"].mean()),
                     "t_dev": float(s["t_mean"].mean() - d[d["car"] == c]["t_mean"].mean()),
                     "resid_sd": float((s["y_ref"] - s["pred_ref"]).std(ddof=1)),
                     "n_win": int(len(s))})
    cl = pd.DataFrame(rows)
    cl["shift_pp"] = cl["soh_ref"] - cl["soh_raw"]
    cl["se_sampling_pp"] = cl["resid_sd"] / np.sqrt(cl["n_win"]) / cl["bol"] * 100

    out = {"car_shift_pp": {"median_abs": float(cl["shift_pp"].abs().median()),
                            "mean": float(cl["shift_pp"].mean()),
                            "p95_abs": float(cl["shift_pp"].abs().quantile(0.95)),
                            "corr_tdev": float(cl["shift_pp"].corr(cl["t_dev"]))},
           "se_sampling_pp_median": float(cl["se_sampling_pp"].median()),
           "soh_ref_range": [float(cl["soh_ref"].min()), float(cl["soh_ref"].max())]}
    grid = []
    for w in WIDTHS:
        edges = np.arange(80.0, 100.0 + w, w)
        edges = edges[(edges >= 80) & (edges <= 105)]
        if len(edges) < 3:
            continue
        t_ref = _tier(cl["soh_ref"], edges)
        t_raw = _tier(cl["soh_raw"], edges)
        flip = float(np.mean(t_raw != t_ref))
        rec = {"width_pp": float(w), "n_edges": int(len(edges)),
               "tier_flip_by_season_pct": flip * 100}
        for sg in SIGMA_PP:
            te = tier_error_rate(cl["soh_ref"].to_numpy(), edges, sg,
                                 n_draw=3000, seed=int(42 + 10 * w))
            rec[f"mis_pct_sigma{sg}"] = te["mis_rate"] * 100
            rec[f"boundary_share_sigma{sg}"] = te["boundary_share"] * 100
        grid.append(rec)
    out["grid"] = grid
    # 折算 vs 不折算的分档差异（在 5 pp 档宽下）
    edges5 = np.arange(80.0, 106.0, 5.0)
    t_ref, t_raw = _tier(cl["soh_ref"], edges5), _tier(cl["soh_raw"], edges5)
    out["tier5_shift_cars"] = int((t_ref != t_raw).sum())
    out["tier5_shift_pct"] = float(np.mean(t_ref != t_raw) * 100)
    out["tier5_counts_ref"] = {str(int(k)): int(v) for k, v in
                               zip(*np.unique(t_ref, return_counts=True))}
    out["_cl"] = cl
    return out


# ---------------------------------------------------------------- 图
def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.55)


def plots(A: dict, B: dict, C: dict, D: dict, fig_dir: str) -> None:
    r = A["_r"]

    # ---- 图 45：单车斜率的可识别性
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7))
    axes[0].scatter(r["se_full"] * 1e3, r["b_full"], s=11, color=C_MAIN, alpha=0.72,
                    edgecolor="white", lw=0.3)
    axes[0].axhline(A["b_global"], color=C_RED, lw=1.2, ls="--",
                    label=f"全局 {A['b_global']:.4f}")
    axes[0].set_xlabel("单车斜率标准误 (×0.001 Ah/°C)")
    axes[0].set_ylabel("单车斜率 (Ah/°C)")
    axes[0].set_title(f"① 单车斜率“看起来”估得准：se 中位 "
                      f"{A['stability']['se_full_median']:.4f}", fontsize=9.6, pad=8)
    axes[0].legend(fontsize=7.6, loc="upper right")
    _style(axes[0])

    axes[1].scatter(r["b1"], r["b2"], s=11, color=C_ACC, alpha=0.72,
                    edgecolor="white", lw=0.3)
    lim = [min(r["b1"].min(), r["b2"].min()) - 0.02, max(r["b1"].max(), r["b2"].max()) + 0.02]
    axes[1].plot(lim, lim, color=C_GRAY, lw=1.0, ls=":")
    axes[1].axhline(0, color=C_GRAY, lw=0.7, ls=":")
    axes[1].axvline(0, color=C_GRAY, lw=0.7, ls=":")
    axes[1].set_xlim(lim); axes[1].set_ylim(lim)
    axes[1].set_xlabel("前半段估的斜率")
    axes[1].set_ylabel("后半段估的斜率")
    axes[1].set_title(f"② 前后半不相关：ρ={A['stability']['corr_half']:.2f}，"
                      f"差值 rms {A['stability']['rms_diff']:.3f}"
                      f"（{A['stability']['var_excess_ratio']:.1f}× 抽样噪声）",
                      fontsize=9.0, pad=8)
    _style(axes[1])

    lam = np.array([x["lam"] for x in A["lam_curve"]])
    sd = np.array([x["median_sd"] for x in A["lam_curve"]])
    axes[2].plot(lam, sd, "-o", ms=3.6, lw=1.4, color=C_MAIN, label="λ·自己 + (1−λ)·全局")
    axes[2].axhline(A["none_sd_median"], color=C_GRAY, lw=1.0, ls=":",
                    label=f"不折算 {A['none_sd_median']:.3f}")
    axes[2].axhline(A["oracle_sd_median"], color=C_GREEN, lw=1.3, ls="--",
                    label=f"知道真斜率 {A['oracle_sd_median']:.3f}")
    axes[2].axvline(A["lam_best"]["lam"], color=C_RED, lw=1.0, ls=":")
    axes[2].annotate(f"最优 λ={A['lam_best']['lam']:.1f}", (0.2, 1.0),
                     xycoords=("data", "axes fraction"), xytext=(3, -2),
                     textcoords="offset points", va="top", ha="left",
                     fontsize=7.6, color=C_RED)
    axes[2].set_ylim(A["lam_best"]["median_sd"] - 0.004,
                     max(A["own_sd_median"], A["none_sd_median"]) + 0.012)
    axes[2].set_xlabel("自己斜率的权重 λ")
    axes[2].set_ylabel("后半段残差 SD 中位数 (Ah)")
    axes[2].set_title(f"③ λ 在 0–0.4 之间是平的，自己的斜率只在 "
                      f"{A['share_own_better_than_global']*100:.0f}% 的车上赢过全局",
                      fontsize=9.2, pad=8)
    axes[2].legend(fontsize=7.0, loc="upper right", labelspacing=0.3)
    _style(axes[2])
    savefig(fig, os.path.join(fig_dir, "fig45_单车斜率的可识别性.png"))

    # ---- 图 46：分层折算的边际价值
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    tiers = [t for t in B["tier_fit"]]
    xi = np.arange(len(tiers))
    vals = [B["tier_fit"][t]["b"] for t in tiers]
    errs = [1.96 * B["tier_fit"][t]["se"] for t in tiers]
    axes[0].bar(xi, vals, yerr=errs, capsize=3.2, color=[C_MAIN, C_ACC, C_PURPLE][:len(tiers)],
                width=0.5, error_kw={"lw": 1.0, "ecolor": C_GRAY})
    axes[0].axhline(A["b_global"], color=C_RED, lw=1.2, ls="--",
                    label=f"全局 {A['b_global']:.4f}")
    for i, t in enumerate(tiers):
        axes[0].text(i, vals[i] * 0.5, f"{vals[i]:.4f}\n({B['tier_fit'][t]['n_cars']} 辆)",
                     ha="center", va="center", fontsize=7.6, color="white")
    axes[0].set_xticks(xi)
    axes[0].set_xticklabels([t.replace("battery_dataset", "ds") for t in tiers])
    axes[0].set_ylabel("层内温度斜率 (Ah/°C)")
    axes[0].set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.32)
    axes[0].set_title("① 层间差一倍，只有 ds2 的区间不含全局值", fontsize=9.6, pad=8)
    axes[0].legend(fontsize=7.4, loc="upper left")
    _style(axes[0])

    names = [("pool_none", "不折算", C_GRAY), ("pool_global", "全局斜率", C_MAIN),
             ("pool_tier", "层内斜率", C_ACC), ("pool_own", "自己的斜率\n(oracle)", C_GREEN)]
    xi = np.arange(len(names))
    ov = B["overall"]
    axes[1].bar(xi, [ov[k] for k, _, _ in names], color=[c for _, _, c in names], width=0.56)
    for i, (k, _, _) in enumerate(names):
        axes[1].text(i, ov[k] + 0.012, f"{ov[k]:.3f}", ha="center", fontsize=7.8)
    axes[1].set_xticks(xi)
    axes[1].set_xticklabels([n for _, n, _ in names], fontsize=7.6)
    axes[1].set_ylim(0, max(ov[k] for k, _, _ in names) * 1.18)
    axes[1].set_ylabel("留一车残差 SD (Ah，合并)")
    axes[1].set_title(f"② 层内比全局好 {ov['pool_tier_vs_global_pct']:.1f}%，"
                      f"oracle 好 {ov['pool_gain_own_pct']:.1f}%", fontsize=9.2, pad=8)
    _style(axes[1])
    savefig(fig, os.path.join(fig_dir, "fig46_分层折算的边际价值.png"))

    # ---- 图 47：残余漂移的归因
    rr = C["_r"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    axes[0].scatter(rr["dt"], rr["db"], s=13, color=C_PURPLE, alpha=0.75,
                    edgecolor="white", lw=0.3)
    xs = np.linspace(rr["dt"].min(), rr["dt"].max(), 30)
    axes[0].plot(xs, C["coef_dt"] * xs, color=C_RED, lw=1.4,
                 label=f"斜率 {C['coef_dt']:.4f} Ah/°C（ρ={C['corr_db_dt']:+.2f}）")
    axes[0].axhline(0, color=C_GRAY, lw=0.8, ls=":")
    axes[0].axvline(0, color=C_GRAY, lw=0.8, ls=":")
    axes[0].set_xlabel("后半段比前半段热多少 (°C)")
    axes[0].set_ylabel("偏置漂移 (Ah)")
    axes[0].set_title(f"① 折算后偏置与温度基本无关（只解释 {C['r2_dt']*100:.0f}%）",
                      fontsize=9.4, pad=8)
    axes[0].legend(fontsize=7.4, loc="upper left")
    _style(axes[0])

    parts = [("温度偏移", C["share_dt"], C_ACC),
             ("工况构成\n(增量)", C["cond_increment_after_dt"], C_MAIN),
             ("无法归因", C["share_unexplained"], C_GRAY)]
    xi = np.arange(len(parts))
    axes[1].bar(xi, [p[1] * 100 for p in parts], color=[p[2] for p in parts], width=0.52)
    for i, p in enumerate(parts):
        axes[1].text(i, p[1] * 100 + 1.2, f"{p[1]*100:.0f}%", ha="center", fontsize=8.2)
    axes[1].set_xticks(xi)
    axes[1].set_xticklabels([p[0] for p in parts], fontsize=8.0)
    axes[1].set_ylabel("偏置漂移方差解释比例 (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title(f"② 温度 + 工况只能解释 {(1-C['share_unexplained'])*100:.0f}%",
                      fontsize=9.4, pad=8)
    axes[1].text(0.5, 62, f"偏置方差的 {C['block_share_between']*100:.0f}% 落在块与块之间\n"
                          f"（台阶，不是斜坡）", ha="center", fontsize=8.0, color=C_RED,
                 bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=C_RED, lw=0.8, alpha=0.9))
    _style(axes[1])
    savefig(fig, os.path.join(fig_dir, "fig47_残余漂移的归因.png"))

    # ---- 图 48：分档分辨率
    cl = D["_cl"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    w = [g["width_pp"] for g in D["grid"]]
    for sg, col in zip(SIGMA_PP, [C_MAIN, C_RED]):
        axes[0].plot(w, [g[f"mis_pct_sigma{sg}"] for g in D["grid"]], "-o", ms=4,
                     lw=1.4, color=col, label=f"车级误差 σ={sg} pp")
    axes[0].axhline(5, color=C_GRAY, lw=1.0, ls=":")
    axes[0].annotate("5% 错档", (w[0], 5), xytext=(2, 5), textcoords="offset points",
                     fontsize=7.2, color=C_GRAY, ha="left")
    axes[0].set_xlabel("档宽 (pp)")
    axes[0].set_ylabel("错档率 (%)")
    axes[0].set_ylim(0, 72)
    axes[0].set_yticks([0, 5, 10, 20, 30, 40, 50, 65])
    axes[0].set_title("① 档宽必须 ≥ 3–4σ 才谈得上分辨", fontsize=9.6, pad=8)
    axes[0].legend(fontsize=7.4)
    _style(axes[0])

    axes[1].scatter(cl["t_dev"], cl["shift_pp"], s=13, color=C_MAIN, alpha=0.75,
                    edgecolor="white", lw=0.3)
    axes[1].axhline(0, color=C_GRAY, lw=0.8, ls=":")
    axes[1].axvline(0, color=C_GRAY, lw=0.8, ls=":")
    axes[1].set_xlabel("上报窗口比全历史热多少 (°C)")
    axes[1].set_ylabel("折算带来的 SOH 变化 (pp)")
    axes[1].set_title(f"② 不折算就按季节翻档：5 pp 档宽下 "
                      f"{D['tier5_shift_pct']:.0f}% 的车换档", fontsize=9.4, pad=8)
    _style(axes[1])
    savefig(fig, os.path.join(fig_dir, "fig48_车队分档的分辨率.png"))


# ---------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    d = load(cfg)
    print(f"样本 {len(d)} 段 / {d['car'].nunique()} 车", flush=True)

    if os.environ.get("REPLOT"):
        import json
        md = cfg["paths"]["metrics_dir"]
        with open(os.path.join(md, "debias_transfer.json"), encoding="utf-8") as f:
            res = json.load(f)
        A, B, C, D = res["A"], res["B"], res["C"], res["D"]
        A["_r"] = pd.read_parquet(os.path.join(md, "debias_transfer_cars.parquet")
                                  ).set_index("car")
        C["_r"] = pd.read_parquet(os.path.join(md, "debias_transfer_drift.parquet"))
        D["_cl"] = pd.read_parquet(os.path.join(md, "debias_transfer_fleet.parquet"))
        plots(A, B, C, D, cfg["paths"]["figures_dir"])
        print("REPLOT 完成", flush=True)
        return

    print("A 单车斜率的可识别性", flush=True)
    A = part_a(d)
    print(f"  异质性 τ={A['hetero']['tau']:.4f}（占 var(b) "
          f"{A['hetero']['share_signal']*100:.0f}%）；前后半 ρ="
          f"{A['stability']['corr_half']:.3f}；差值/噪声="
          f"{A['stability']['var_excess_ratio']:.1f}×；中位残差 "
          f"{A['none_sd_median']:.4f}→{A['global_sd_median']:.4f}(全局)"
          f"→{A['oracle_sd_median']:.4f}(oracle)；最优 λ={A['lam_best']['lam']:.1f}", flush=True)
    print("B 分层折算", flush=True)
    B = part_b(d, A)
    o = B["overall"]
    print(f"  合并残差 SD：none {o['pool_none']:.3f} → global {o['pool_global']:.3f} "
          f"→ tier {o['pool_tier']:.3f} → own {o['pool_own']:.3f}；tier 比 global "
          f"{o['pool_tier_vs_global_pct']:+.2f}%（胜率 {o['share_tier_better_than_global']*100:.0f}%）",
          flush=True)
    for t, v in B["by_tier"].items():
        print(f"    {t}: 配对差 tier−global {v['paired_tier_vs_global']:+.3f} Ah，"
              f"胜率 {v['share_tier_better_than_global']*100:.0f}%，"
              f"层内斜率 {B['tier_fit'][t]['b']:.4f}", flush=True)
    print("C 残余漂移", flush=True)
    C = part_c(d)
    print(f"  db sd {C['db_sd']:.3f}；R²(Δt) {C['r2_dt']:.3f}，+工况 {C['r2_all']:.3f}；"
          f"块间占 {C['block_share_between']*100:.0f}%", flush=True)
    print("D 分档", flush=True)
    D = part_d(d)
    print(f"  5pp 档宽季节换档 {D['tier5_shift_pct']:.1f}%；σ=1.05 时错档 "
          f"{[round(g['mis_pct_sigma1.05'],1) for g in D['grid']]}", flush=True)

    res = {"n_sessions": int(len(d)), "n_cars": int(d["car"].nunique()),
           "ref_t": REF_T, "k_win": K_WIN, "split": SPLIT,
           "A": {k: v for k, v in A.items() if not k.startswith("_")},
           "B": {k: v for k, v in B.items() if not k.startswith("_")},
           "C": {k: v for k, v in C.items() if not k.startswith("_")},
           "D": {k: v for k, v in D.items() if not k.startswith("_")},
           "note": ("A 的 λ 曲线用前 60% 估斜率、后 40% 验收；B 的留一车检验中"
                    "全局/层内斜率都不含被评估车；C 的偏置用折算后的标签与折算后"
                    "训练的模型之差；D 的错档率由 σ 的正态抽样 3000 次得到。")}
    save_json(res, "debias_transfer")
    print("写入 results/metrics/debias_transfer.json", flush=True)

    md = cfg["paths"]["metrics_dir"]
    A["_r"].reset_index().to_parquet(os.path.join(md, "debias_transfer_cars.parquet"),
                                     index=False)
    C["_r"].to_parquet(os.path.join(md, "debias_transfer_drift.parquet"), index=False)
    D["_cl"].to_parquet(os.path.join(md, "debias_transfer_fleet.parquet"), index=False)
    plots(A, B, C, D, cfg["paths"]["figures_dir"])


if __name__ == "__main__":
    main()
