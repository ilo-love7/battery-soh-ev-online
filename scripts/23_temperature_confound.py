# -*- coding: utf-8 -*-
"""23 温度的混杂：标签带温度响应、模型照抄、SOH 跨季节不可比。

前六轮一直在追问「模型离标签还有多远」，却始终把一个前提当成理所当然：
**标签本身是同一把尺子量出来的**。第六轮（§19）已经证明标签有块状噪声（34.4% 的会话
落在复制块里），但那还是"同一个数被抄了几遍"。这一轮问的是更根本的问题：
**标签的数值本身会不会随测量条件漂移？**

实测答案是很硬的"会"：同一辆车内部，控制里程后容量标签随电池温度 **+0.045 Ah/°C**
（≈0.11 %/°C），而模型预测的温度斜率几乎相同。后果不是"模型不准"（恰恰相反，
它准得很——因为它复制了尺子的偏差），而是**不同季节报出来的 SOH 不可比**：
一个季节循环的摆幅相当于约 3.5 万公里的真实老化，而车级上报窗口（连续 8 段）
几乎消不掉它，因为同一季节里连续 8 段的温度几乎一样。

  A. 标签的温度响应：车内固定效应 + 一阶差分 + 去复制块三种口径互相校验，
     按车整群 bootstrap 给区间，按厂商拆开看异质性。
  B. 模型是否继承：同样的回归跑在预测上，量"复制比例"；再看残差里的温度斜率。
  C. 参考温度重训：把标签折算到 25°C 再训练，检验①精度不掉、②预测不再随温度漂、
     ③车级 SOH 与阈值判定改变多少。
  D. 报告口径：季节摆幅换算成等效里程，量化"连续 k 段聚合能消掉多少温度偏差"。

输出
  results/metrics/temperature_confound.json
  results/metrics/temperature_oof.parquet（逐会话的原始/折算标签与两版预测）
  results/figures/fig42_温度响应的三层证据.png
  results/figures/fig43_参考温度重训.png
  results/figures/fig44_季节摆幅与报告口径.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import group_kfold_by_car, regression_metrics  # noqa: E402
from evsoh.features import active_features, feature_columns  # noqa: E402
from evsoh.models import build_model  # noqa: E402
from evsoh.panel import (debias, fe_ols, first_difference,  # noqa: E402
                         piecewise_fit, temperature_bins, within_demean)
from evsoh.plotting import (C_ACC, C_GRAY, C_GREEN, C_MAIN, C_PURPLE,  # noqa: E402
                            C_RED, plt, savefig)
from evsoh.target import add_soh_labels  # noqa: E402

REF_T = 25.0               # 参考温度：容量折算的基准
KNOT = 33.0                # 温度响应的拐点（实测 35°C 以上趋于饱和）
K_WIN = 8                  # 车级上报窗口，沿用 §18/§19 的饱和点
MIN_SESS = 24
PRED = "pred_xgb"
N_BOOT = 200
SMOKE = bool(os.environ.get("SMOKE"))

COND = ["d_soc", "i_abs_mean", "quality"]     # 工况控制变量（会话内可观测）
SPECS = [
    ("fe_t", ["t_mean"]),
    ("fe_t_km", ["t_mean", "mileage"]),
    ("fe_full", ["t_mean", "mileage"] + COND),
]


# ---------------------------------------------------------------- 数据
def load(cfg: dict) -> pd.DataFrame:
    """用 OOF 表（已含 t_mean / d_soc / i_abs_mean / quality），不另做 merge。"""
    d = pd.read_parquet(os.path.join(cfg["paths"]["metrics_dir"], "oof_predictions.parquet"))
    d = d.dropna(subset=["car", "capacity", PRED, "mileage", "t_mean"]).copy()
    d = d.sort_values(["car", "mileage"]).reset_index(drop=True)
    n = d.groupby("car")["car"].transform("size").to_numpy()
    d = d[n >= MIN_SESS].reset_index(drop=True)
    # 复制块：容量与上一段完全相同（§19 的口径），块首才算独立测量
    same = d.groupby("car")["capacity"].diff().eq(0).fillna(False).to_numpy()
    d["is_copy"] = same
    d["run_id"] = np.cumsum(~same)
    if SMOKE:
        keep = sorted(d["car"].unique())[:40]
        d = d[d["car"].isin(keep)].reset_index(drop=True)
    return d


# ---------------------------------------------------------------- A 标签侧
def _fit_all(d: pd.DataFrame, y: str) -> dict:
    """三种口径的斜率：固定效应、一阶差分、只留块首。"""
    out = {}
    for name, xs in SPECS:
        f = fe_ols(d, y, xs)
        out[name] = {"beta": float(f["beta"][0]), "se": float(f["se"][0]),
                     "r2": float(f["r2"]), "n": f["n"], "n_cars": f["n_groups"],
                     "coefs": {k: float(v) for k, v in zip(f["names"], f["beta"])}}
    # 一阶差分：消掉车效应与线性老化趋势
    diff = first_difference(d[["car", "mileage", y, "t_mean"]], [y, "t_mean"])
    b = np.polyfit(diff["t_mean"], diff[y], 1)[0]
    resid = diff[y] - b * diff["t_mean"]
    out["fd"] = {"beta": float(b),
                 "r2": float(1 - np.var(resid) / np.var(diff[y])),
                 "n": int(len(diff)), "n_cars": int(diff["car"].nunique())}
    # 只取相邻 3000 km 内的会话对：季节相近，老化几乎为零
    srt = d[["car", "mileage", y, "t_mean"]].sort_values(["car", "mileage"])
    gap = pd.DataFrame({
        "dY": srt.groupby("car")[y].diff(),
        "dT": srt.groupby("car")["t_mean"].diff(),
        "dM": srt.groupby("car")["mileage"].diff(),
    }).dropna()
    near = gap[gap["dM"].abs() < 3000]
    out["fd_near"] = {"beta": float(np.polyfit(near["dT"], near["dY"], 1)[0]),
                      "n": int(len(near))}
    # 去复制块：每块只用第一段
    head = d[~d["is_copy"]]
    f = fe_ols(head, y, SPECS[1][1])
    out["fe_head"] = {"beta": float(f["beta"][0]), "se": float(f["se"][0]),
                      "r2": float(f["r2"]), "n": f["n"], "n_cars": f["n_groups"]}
    return out


def part_a(d: pd.DataFrame) -> dict:
    out = {}
    out["label"] = _fit_all(d, "capacity")
    out["pred"] = _fit_all(d, PRED)

    # 按车整群 bootstrap：标签与预测用同一批重抽样车，得到配对差的区间
    boots = {"capacity": [], PRED: []}
    rng = np.random.default_rng(42)
    cars = d["car"].unique()
    for _ in range(N_BOOT if not SMOKE else 40):
        pick = rng.choice(cars, size=cars.size, replace=True)
        parts = [d[d["car"] == c] for c in pick]
        boot = pd.concat(parts, ignore_index=True)
        boot["car"] = np.repeat(np.arange(len(pick)), [len(p) for p in parts])
        for y in ("capacity", PRED):
            try:
                boots[y].append(float(fe_ols(boot, y, SPECS[1][1])["beta"][0]))
            except (ValueError, np.linalg.LinAlgError):
                boots[y].append(np.nan)
    bl = np.asarray(boots["capacity"], float)
    bp = np.asarray(boots[PRED], float)
    ratio = bp / bl
    out["boot"] = {
        "n_boot": int(np.isfinite(bl).sum()),
        "label_ci": [float(np.nanpercentile(bl, 5)), float(np.nanpercentile(bl, 95))],
        "pred_ci": [float(np.nanpercentile(bp, 5)), float(np.nanpercentile(bp, 95))],
        "ratio_median": float(np.nanmedian(ratio)),
        "ratio_ci": [float(np.nanpercentile(ratio, 5)), float(np.nanpercentile(ratio, 95))],
    }

    # 分厂商
    per = {}
    for ds, g in d.groupby("dataset"):
        if len(g) < 300:
            per[str(ds)] = {"n": int(len(g)), "skipped": True}
            continue
        f = fe_ols(g, "capacity", SPECS[1][1])
        per[str(ds)] = {"n": int(len(g)), "n_cars": f["n_groups"],
                        "beta": float(f["beta"][0]), "se": float(f["se"][0])}
    out["per_dataset"] = per

    # 分段线性（拐点 33°C）：给 C 用
    pw = piecewise_fit(d, "capacity", "t_mean", KNOT, ref=REF_T,
                       controls=["mileage"])
    out["piecewise"] = {k: v for k, v in pw.items() if not k.startswith("_")}
    out["_pw"] = pw

    # 温度分箱：车内去均值后的容量与预测（图上用）
    dm = within_demean(d, ["capacity", PRED], group="car")
    d2 = d.assign(cap_w=dm["capacity"].to_numpy(), pred_w=dm[PRED].to_numpy())
    lab, edges = temperature_bins(d2["t_mean"].to_numpy(), n=8)
    d2["_bin"] = lab
    g = d2.groupby("_bin").agg(t=("t_mean", "mean"), cap_w=("cap_w", "mean"),
                               pred_w=("pred_w", "mean"), n=("cap_w", "size"))
    out["bins"] = {"edges": edges,
                   "rows": [{"t": float(r.t), "cap_w": float(r.cap_w),
                             "pred_w": float(r.pred_w), "n": int(r.n)}
                            for r in g.itertuples()]}
    out["_d2"] = d2

    # 每车温度斜率（固定效应，只有一辆车时即为该车斜率）
    slopes = []
    for c, g2 in d.groupby("car"):
        if len(g2) < 12:
            continue
        T = g2["t_mean"].to_numpy(float)
        if T.std() < 1.0:
            continue
        y = g2["capacity"].to_numpy(float)
        slopes.append(float(np.polyfit(T - T.mean(), y - y.mean(), 1)[0]))
    out["per_car_slopes"] = {"n_cars": len(slopes), "median": float(np.median(slopes)),
                             "share_positive": float(np.mean(np.asarray(slopes) > 0)),
                             "p10": float(np.percentile(slopes, 10)),
                             "p90": float(np.percentile(slopes, 90)),
                             "values": [float(v) for v in slopes]}

    # 标签自身的不稳定度是否也随温度变化：与同车近邻中位数的差（§19 的去噪口径）
    lab_noise = np.full(len(d), np.nan)
    for _c, g in d.groupby("car", sort=False):
        y = g["capacity"].to_numpy(float)
        nb = np.full(y.size, np.nan)
        for i in range(y.size):
            win = np.r_[y[max(0, i - 3):i], y[i + 1:i + 4]]
            if win.size:
                nb[i] = np.median(win)
        lab_noise[g.index.to_numpy()] = y - nb
    d_noise = d.assign(nl=lab_noise).dropna(subset=["nl"])
    lab3, _e3 = temperature_bins(d_noise["t_mean"].to_numpy(), n=4)
    d_noise = d_noise.assign(_tb=lab3)
    rows_n = []
    for k, g in d_noise.groupby("_tb"):
        rows_n.append({"t": float(g["t_mean"].mean()), "n": int(len(g)),
                       "sd": float(g["nl"].std()),
                       "mad": float(np.median(np.abs(g["nl"]))),
                       "rmse_pred": float(np.sqrt(np.mean(
                           (g[PRED] - g["capacity"]).to_numpy() ** 2)))})
    out["noise_by_temp"] = rows_n
    print("  A 标签不稳定度：" + "  ".join(f"{r['t']:.0f}°C:{r['sd']:.3f}" for r in rows_n)
          + "（越冷越不稳）", flush=True)
    print(f"  A 标签温度斜率 {out['label']['fe_t_km']['beta']:+.4f} Ah/°C "
          f"[{out['boot']['label_ci'][0]:+.4f}, {out['boot']['label_ci'][1]:+.4f}]；"
          f"预测 {out['pred']['fe_t_km']['beta']:+.4f}，复制比 "
          f"{out['boot']['ratio_median']:.3f}", flush=True)
    return out


# ---------------------------------------------------------------- C 参考温度重训
def part_c(cfg: dict, d: pd.DataFrame) -> dict:
    if os.environ.get("SKIP_TRAIN"):
        print("  C SKIP_TRAIN=1，跳过重训", flush=True)
        return {"skipped": True}

    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, _ = add_soh_labels(ses, cfg)
    df = df.dropna(subset=["capacity", "bol_ref", "t_mean", "mileage"])
    df = df.sort_values(["car", "mileage"]).reset_index(drop=True)
    n = df.groupby("car")["car"].transform("size").to_numpy()
    df = df[n >= MIN_SESS].reset_index(drop=True)
    if SMOKE:
        keep = sorted(df["car"].unique())[:40]
        df = df[df["car"].isin(keep)].reset_index(drop=True)

    cols = active_features(df, feature_columns())
    X = df[cols].astype(float)
    X = X.fillna(X.median()).fillna(0.0)
    y = df["capacity"].to_numpy(float)
    T = df["t_mean"].to_numpy(float)
    cars = df["car"].to_numpy()
    folds = list(group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                    seed=cfg["splits"]["random_state"]))
    pred = {"base": np.full(len(df), np.nan), "ref": np.full(len(df), np.nan)}
    y_ref = np.full(len(df), np.nan)
    shifts = []
    for k, (tr, te) in enumerate(folds):
        # 折算参数只用训练车估，避免把测试车的温度响应漏进标签
        pw = piecewise_fit(df.iloc[tr], "capacity", "t_mean", KNOT, ref=REF_T,
                           controls=["mileage"])
        yr = debias(y, T, pw)
        y_ref[te] = yr[te]
        shifts.append({"fold": k, "slope_below": pw["slope_below_knot"],
                       "slope_above": pw["slope_above"]})
        m0 = build_model("xgb", cfg["models"])
        m0.fit(X.iloc[tr], y[tr])
        pred["base"][te] = m0.predict(X.iloc[te])
        m1 = build_model("xgb", cfg["models"])
        m1.fit(X.iloc[tr], yr[tr])
        pred["ref"][te] = m1.predict(X.iloc[te])
        print(f"  C fold {k+1}/{len(folds)} 折内温度斜率 {pw['slope_below_knot']:+.4f}", flush=True)

    ev = df.copy()
    ev["y_ref"] = y_ref
    for k, v in pred.items():
        ev[f"pred_{k}"] = v
    ev = ev.dropna(subset=["y_ref", "pred_base", "pred_ref"])

    out = {"target_spec": {"knot": KNOT, "ref_t": REF_T, "n": int(len(ev)),
                           "fold_slopes": shifts}}
    # 2×2：目标（原始 / 折算）× 模型（原始 / 折算）
    for tgt in ("capacity", "y_ref"):
        for mdl in ("base", "ref"):
            m = regression_metrics(ev[tgt].to_numpy(), ev[f"pred_{mdl}"].to_numpy())
            out[f"{tgt}|{mdl}"] = {"rmse": m["rmse"], "mae": m["mae"], "n": m["n"]}
    # 去掉温度后，原始模型的误差会被放大多少（说明它的"准"有多少来自复制尺子）
    d_base = out["y_ref|base"]["rmse"] / out["capacity|base"]["rmse"] - 1
    d_ref = out["y_ref|ref"]["rmse"] / out["capacity|base"]["rmse"] - 1
    out["cost_of_debiasing_pct"] = float(d_base * 100)
    out["ref_model_vs_base_pct"] = float(d_ref * 100)

    # 温度三分位内的误差：原始模型在两端应该更差
    lab, edges = temperature_bins(ev["t_mean"].to_numpy(), n=3)
    ev["_tb"] = lab
    names = ["冷", "中", "热"]
    by_t = {}
    for k, g in ev.groupby("_tb"):
        row = {"t_mean": float(g["t_mean"].mean()), "n": int(len(g))}
        for mdl in ("base", "ref"):
            row[mdl] = float(regression_metrics(g["y_ref"].to_numpy(),
                                                g[f"pred_{mdl}"].to_numpy())["rmse"])
        by_t[names[int(k)] if int(k) < 3 else str(k)] = row
    out["by_temp"] = {"edges": edges, "rows": by_t}

    # 重训后预测的温度斜率（应当接近 0）
    for mdl in ("base", "ref"):
        f = fe_ols(ev, f"pred_{mdl}", ["t_mean", "mileage"])
        out[f"slope_{mdl}"] = {"beta": float(f["beta"][0]), "se": float(f["se"][0])}
    fl = fe_ols(ev, "capacity", ["t_mean", "mileage"])
    out["slope_label"] = {"beta": float(fl["beta"][0]), "se": float(fl["se"][0])}

    # 车级：最近 K 段的均值 / BOL
    rows = []
    for c, g in ev.groupby("car"):
        g = g.sort_values("mileage")
        if len(g) < K_WIN:
            continue
        w = g.iloc[-K_WIN:]
        rows.append({"car": c, "bol": float(g["bol_ref"].iloc[0]),
                     "t_mean": float(w["t_mean"].mean()),
                     "soh_label": float(w["capacity"].mean() / g["bol_ref"].iloc[0]),
                     "soh_label_ref": float(w["y_ref"].mean() / g["bol_ref"].iloc[0]),
                     "soh_base": float(w["pred_base"].mean() / g["bol_ref"].iloc[0]),
                     "soh_ref": float(w["pred_ref"].mean() / g["bol_ref"].iloc[0]),
                     "t_dev": float(w["t_mean"].mean() - g["t_mean"].mean())})
    cl = pd.DataFrame(rows)
    delta = (cl["soh_label_ref"] - cl["soh_label"]) * 100
    # 阈值敏感性：样本里 187 辆车全部高于 80%（BOL 是每车自己最高 10% 的中位数，
    # 所以 SOH 天然贴近 95%），80% 这条线在这批车上一辆都不跨越——照抄它没有意义。
    # 改用实际分布上的几条线，看"同一车队、换个阈值、名单会换多少车"。
    thresholds = {}
    for th in (0.85, 0.90, 0.92, 0.95):
        a = cl["soh_label"] < th
        b = cl["soh_label_ref"] < th
        thresholds[f"{th:.2f}"] = {"raw": int(a.sum()), "ref": int(b.sum()),
                                    "crossed": int((a != b).sum())}
    k = max(1, int(round(0.10 * len(cl))))
    old_set = set(cl.nsmallest(k, "soh_label")["car"])
    new_set = set(cl.nsmallest(k, "soh_label_ref")["car"])
    out["car_level"] = {
        "n_cars": int(len(cl)),
        "soh_min": float(cl["soh_label"].min() * 100),
        "soh_median": float(cl["soh_label"].median() * 100),
        "delta_pp_mean": float(delta.mean()),
        "delta_pp_abs_median": float(delta.abs().median()),
        "delta_pp_p5": float(delta.quantile(0.05)),
        "delta_pp_p95": float(delta.quantile(0.95)),
        "corr_delta_tdev": float(delta.corr(cl["t_dev"])),
        "spearman": float(cl["soh_label"].corr(cl["soh_label_ref"], method="spearman")),
        "thresholds": thresholds,
        "worst_decile_overlap": int(len(old_set & new_set)),
        "worst_decile_k": k,
        "rmse_label": float(np.sqrt(np.mean(((cl["soh_base"] - cl["soh_label"]) * 100) ** 2))),
        "rmse_ref": float(np.sqrt(np.mean(((cl["soh_ref"] - cl["soh_label_ref"]) * 100) ** 2))),
        "rmse_cross": float(np.sqrt(np.mean(((cl["soh_ref"] - cl["soh_label"]) * 100) ** 2))),
    }
    print(f"  C 重训：原始模型对原始标签 {out['capacity|base']['rmse']:.4f}；"
          f"折算标签上原始模型 {out['y_ref|base']['rmse']:.4f}、折算模型 "
          f"{out['y_ref|ref']['rmse']:.4f}；预测温度斜率 {out['slope_base']['beta']:+.4f} → "
          f"{out['slope_ref']['beta']:+.4f}", flush=True)
    print(f"  C 车级：折算改变 {out['car_level']['delta_pp_abs_median']:.2f} pp（中位绝对值），"
          f"90% 线跨越 {out['car_level']['thresholds']['0.90']['crossed']} 辆、"
          f"最差 10% 名单重合 {out['car_level']['worst_decile_overlap']}/"
          f"{out['car_level']['worst_decile_k']}", flush=True)

    pd.DataFrame({"car": ev["car"], "segment": ev["segment"],
                  "t_mean": ev["t_mean"], "capacity": ev["capacity"],
                  "y_ref": ev["y_ref"], "pred_base": ev["pred_base"],
                  "pred_ref": ev["pred_ref"], "bol_ref": ev["bol_ref"]}
                 ).to_parquet(os.path.join(cfg["paths"]["metrics_dir"],
                                           "temperature_oof.parquet"), index=False)
    out["_ev"] = ev
    out["_cl"] = cl
    return out


# ---------------------------------------------------------------- D 报告口径
def part_d(d: pd.DataFrame, pw: dict, C: dict) -> dict:
    out = {}
    # 老化速率：车内 FE，容量对里程
    f = fe_ols(d, "capacity", ["mileage", "t_mean"])
    ah_per_km = float(f["beta"][0])
    mean_cap = float(d["capacity"].mean())
    out["aging"] = {"ah_per_10k_km": ah_per_km * 1e4,
                    "pp_per_10k_km": ah_per_km * 1e4 / mean_cap * 100,
                    "n": f["n"], "mean_capacity": mean_cap}

    # 季节摆幅：p5–p95 的温度跨度
    t_lo, t_hi = float(d["t_mean"].quantile(0.05)), float(d["t_mean"].quantile(0.95))
    f_ = pw["_f"]
    amp_ah = float(f_(t_hi) - f_(t_lo))
    out["season"] = {"t_p5": t_lo, "t_p95": t_hi, "amp_ah": amp_ah,
                     "amp_pp": amp_ah / mean_cap * 100,
                     "equiv_km": amp_ah / abs(ah_per_km) if ah_per_km != 0 else float("nan")}

    # 车级上报窗口（连续 K 段）残留的温度偏差：同一季节内温度高度相关，平均不掉
    dev = []
    for _c, g in d.groupby("car"):
        T = g["t_mean"].to_numpy(float)
        if T.size < K_WIN + 4:
            continue
        dev.append(float(T[-K_WIN:].mean() - T.mean()))
    dev = np.asarray(dev)
    beta = float(pw["slope_below_knot"])
    out["reporting"] = {
        "k_win": K_WIN, "n_cars": int(dev.size),
        "t_dev_sd": float(np.std(dev)),
        "t_dev_p5": float(np.percentile(dev, 5)), "t_dev_p95": float(np.percentile(dev, 95)),
        "bias_ah_sd": float(np.std(dev) * beta),
        "bias_pp_sd": float(np.std(dev) * beta / mean_cap * 100),
        "bias_pp_p95_abs": float(np.percentile(np.abs(dev), 95) * beta / mean_cap * 100),
        "within_car_t_sd": float(d.groupby("car")["t_mean"].std().median()),
        "lag1_t_autocorr": float(np.nanmean([
            np.corrcoef(g["t_mean"].to_numpy(float)[:-1], g["t_mean"].to_numpy(float)[1:])[0, 1]
            for _c, g in d.groupby("car") if len(g) >= 8])),
    }
    # 与第六轮车级精度对比：季节偏差占多大比重
    # 为什么"连续 8 段"平均不掉温度：窗口内的温度几乎一样（自相关 0.88），
    # 平均掉的是标签噪声，不是季节。量化一下：窗口内温度跨度 vs 全历史跨度。
    spans = []
    for _c, g in d.groupby("car"):
        g = g.sort_values("mileage")
        if len(g) < K_WIN + 8:
            continue
        T = g["t_mean"].to_numpy(float)
        spans.append({"win_span": float(T[-K_WIN:].max() - T[-K_WIN:].min()),
                      "all_span": float(T.max() - T.min())})
    sp = pd.DataFrame(spans)
    if len(sp) > 10:
        out["reporting"]["win_t_span_median"] = float(sp["win_span"].median())
        out["reporting"]["all_t_span_median"] = float(sp["all_span"].median())
        out["reporting"]["win_over_all_span"] = float(
            sp["win_span"].median() / max(sp["all_span"].median(), 1e-9))
    ref = C.get("car_level", {}) if not C.get("skipped") else {}
    if ref:
        out["reporting"]["car_level_mae_pp_round6"] = 1.05
        out["reporting"]["bias_over_mae"] = float(
            out["reporting"]["bias_pp_sd"] / 1.05)
    print(f"  D 季节摆幅 {out['season']['amp_ah']:.2f} Ah = {out['season']['amp_pp']:.2f} pp "
          f"= {out['season']['equiv_km']/1000:.1f} 千公里老化；连续 {K_WIN} 段报告的"
          f"温度偏差 {out['reporting']['bias_pp_sd']:.2f} pp", flush=True)
    return out


# ---------------------------------------------------------------- 图
def _style(ax):
    ax.grid(axis="x", visible=False)


def plots(res: dict, d: pd.DataFrame, fig_dir: str) -> None:
    A, C, D = res["A"], res["C"], res["D"]
    beta_lab = A["label"]["fe_t_km"]["beta"]
    beta_prd = A["pred"]["fe_t_km"]["beta"]

    # ---- 图 42：温度响应的四层证据
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2))
    axes = axes.ravel()
    rows = A["bins"]["rows"]
    ts = np.array([r["t"] for r in rows])
    axes[0].plot(ts, [r["cap_w"] for r in rows], "o-", color=C_MAIN, ms=4.6, lw=1.6,
                 label=f"容量标签（斜率 {beta_lab:+.4f}）")
    axes[0].plot(ts, [r["pred_w"] for r in rows], "s--", color=C_ACC, ms=4.2, lw=1.5,
                 label=f"模型预测（斜率 {beta_prd:+.4f}）")
    axes[0].axhline(0, color=C_GRAY, lw=0.9, ls=":")
    axes[0].axvline(REF_T, color=C_GRAY, lw=0.9, ls=":")
    axes[0].set_xlabel("会话平均电池温度 (°C)")
    axes[0].set_ylabel("车内去均值的容量 (Ah)")
    axes[0].set_title("① 标签随温度升高，模型几乎等量跟随", fontsize=9.8, pad=8)
    axes[0].legend(fontsize=7.2, loc="lower right", framealpha=0.92)

    sl = np.asarray(A["per_car_slopes"]["values"], float)
    axes[1].hist(sl, bins=28, color=C_PURPLE, alpha=0.78, edgecolor="white", lw=0.6)
    axes[1].axvline(0, color=C_GRAY, lw=1.0, ls=":")
    axes[1].axvline(np.median(sl), color=C_RED, lw=1.4,
                    label=f"中位 {np.median(sl):+.4f}（{(sl>0).mean()*100:.0f}% 为正）")
    axes[1].set_xlabel("每辆车自己的温度斜率 (Ah/°C)")
    axes[1].set_ylabel("车辆数")
    axes[1].set_title(f"② 不是少数车驱动（{len(sl)} 辆，跨度 "
                      f"{sl.min():+.2f} ~ {sl.max():+.2f}）", fontsize=9.8, pad=8)
    axes[1].legend(fontsize=7.2, loc="upper left", framealpha=0.92)

    diff = first_difference(d[["car", "mileage", "capacity", "t_mean"]], ["capacity", "t_mean"])
    # 散点只作背景：ΔC 的离群值会把 y 轴撑到 ±15，把斜率压成一条水平线。
    # 真正读斜率要看分箱均值（每箱 ~3800 段，SEM 小到看不见）。
    b = A["label"]["fd"]["beta"]
    axes[2].scatter(diff["t_mean"], diff["capacity"], s=3, color=C_GRAY, alpha=0.10,
                    edgecolor="none")
    edges = np.quantile(diff["t_mean"], np.linspace(0, 1, 15))
    mids, means, sems = [], [], []
    for i in range(len(edges) - 1):
        m = (diff["t_mean"] >= edges[i]) & (diff["t_mean"] <= edges[i + 1])
        if m.sum() < 50:
            continue
        mids.append(float(diff["t_mean"][m].mean()))
        means.append(float(diff["capacity"][m].mean()))
        sems.append(float(diff["capacity"][m].std() / np.sqrt(m.sum())))
    axes[2].errorbar(mids, means, yerr=np.asarray(sems) * 1.96, fmt="o", ms=4.4,
                     color=C_MAIN, ecolor=C_MAIN, elinewidth=1.0, capsize=2.4,
                     label="分箱均值（±95%）")
    xs = np.linspace(min(mids), max(mids), 20)
    axes[2].plot(xs, b * xs, color=C_RED, lw=1.8,
                 label=f"差分斜率 {b:+.4f}（已消掉老化趋势）")
    axes[2].axhline(0, color=C_GRAY, lw=0.9, ls=":")
    axes[2].axvline(0, color=C_GRAY, lw=0.9, ls=":")
    axes[2].set_ylim(min(means) - 0.6, max(means) + 0.6)
    axes[2].set_xlabel("相邻会话的温差 ΔT (°C)")
    axes[2].set_ylabel("相邻会话的容量差 ΔC (Ah)")
    axes[2].set_title("③ 一阶差分：去掉车效应与线性老化后仍在", fontsize=9.8, pad=8)
    axes[2].legend(fontsize=7.2, loc="upper left", framealpha=0.92)

    nb = A["noise_by_temp"]
    xs_n = [r["t"] for r in nb]
    axes[3].plot(xs_n, [r["sd"] for r in nb], "o-", color=C_RED, ms=5, lw=1.7,
                 label="标签自身的不稳定度 (SD)")
    axes[3].plot(xs_n, [r["rmse_pred"] for r in nb], "s--", color=C_MAIN, ms=4.4, lw=1.4,
                 label="预测对标签的 RMSE")
    for x, r in zip(xs_n, nb):
        axes[3].annotate(f"{r['sd']:.2f}", (x, r["sd"]), textcoords="offset points",
                         xytext=(0, -12), fontsize=7.0, color=C_RED, ha="center")
    axes[3].set_xlabel("会话平均电池温度 (°C)")
    axes[3].set_ylabel("Ah")
    axes[3].set_title("④ 冷天连标签自己都不稳：误差只是跟着标签走", fontsize=9.8, pad=8)
    axes[3].legend(fontsize=7.2, loc="upper right", framealpha=0.92)
    axes[3].set_ylim(0, max(r["rmse_pred"] for r in nb) * 1.22)
    savefig(fig, os.path.join(fig_dir, "fig42_温度响应的三层证据.png"))

    if C.get("skipped"):
        return

    # ---- 图 43：参考温度重训
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    names = [k for k in ("冷", "中", "热") if k in C["by_temp"]["rows"]]
    xi = np.arange(len(names))
    base_v = [C["by_temp"]["rows"][k]["base"] for k in names]
    ref_v = [C["by_temp"]["rows"][k]["ref"] for k in names]
    axes[0].bar(xi - 0.19, base_v, width=0.36, color=C_GRAY, label="原始模型")
    axes[0].bar(xi + 0.19, ref_v, width=0.36, color=C_GREEN, label="折算重训")
    for i in range(len(names)):
        axes[0].text(i - 0.19, base_v[i] + max(base_v) * 0.015, f"{base_v[i]:.3f}",
                     ha="center", fontsize=7.4)
        axes[0].text(i + 0.19, ref_v[i] + max(base_v) * 0.015, f"{ref_v[i]:.3f}",
                     ha="center", fontsize=7.4)
    axes[0].set_xticks(xi)
    axes[0].set_xticklabels([f"{k}\n({C['by_temp']['rows'][k]['t_mean']:.1f}°C)"
                             for k in names], fontsize=8)
    axes[0].set_ylim(0, max(base_v) * 1.22)
    axes[0].set_ylabel("对折算标签的 RMSE (Ah)")
    axes[0].set_title("① 原始模型在两端更差（因为它复制的尺子会漂）", fontsize=9.8, pad=8)
    axes[0].legend(fontsize=7.4, loc="upper left", framealpha=0.92)
    _style(axes[0])

    cl = C["_cl"]
    axes[1].scatter((cl["soh_label"] * 100), (cl["soh_label_ref"] * 100), s=13,
                    color=C_MAIN, alpha=0.66, edgecolor="none")
    lo, hi = 80, 100
    axes[1].plot([lo, hi], [lo, hi], color=C_GRAY, lw=1.0, ls="--", label="y = x")
    th = C["car_level"]["thresholds"]["0.90"]
    axes[1].axhline(90, color=C_RED, lw=0.9, ls=":")
    axes[1].axvline(90, color=C_RED, lw=0.9, ls=":")
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[1].set_xlabel("原始标签下的车级 SOH (%)")
    axes[1].set_ylabel(f"折算到 {REF_T:.0f}°C 的 SOH (%)")
    axes[1].set_title(f"② 车级 SOH 整体上移（中位 |Δ| "
                      f"{C['car_level']['delta_pp_abs_median']:.2f} pp）", fontsize=9.8, pad=8)
    axes[1].annotate(f"90% 线：{th['raw']} 辆 → {th['ref']} 辆\n（{th['crossed']} 辆换边）",
                     (99.5, 81.0), fontsize=7.2, color=C_RED, va="bottom", ha="right")
    axes[1].legend(fontsize=7.4, loc="upper left", framealpha=0.92)

    delta = (cl["soh_label_ref"] - cl["soh_label"]) * 100
    axes[2].hist(delta, bins=26, color=C_PURPLE, alpha=0.78, edgecolor="white", lw=0.6)
    axes[2].axvline(0, color=C_GRAY, lw=1.0, ls=":")
    axes[2].axvline(delta.mean(), color=C_RED, lw=1.4, label=f"均值 {delta.mean():+.2f} pp")
    axes[2].set_xlabel("折算带来的 SOH 变化 (pp)")
    axes[2].set_ylabel("车辆数")
    axes[2].set_title(f"③ 方向由温度偏移决定（ρ={C['car_level']['corr_delta_tdev']:+.2f}）",
                      fontsize=9.8, pad=8)
    axes[2].legend(fontsize=7.4, loc="upper left", framealpha=0.92)
    _style(axes[2])
    savefig(fig, os.path.join(fig_dir, "fig43_参考温度重训.png"))

    # ---- 图 44：季节摆幅与报告口径
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    # 单人单车读不出来（三个 twinx 会糊成一团）；改成"温度是慢波"这一机制图：
    # 窗口内温度几乎不变，所以 8 段平均掉的是噪声，不是季节。
    car_show = max(((len(g), c) for c, g in d.groupby("car")))[1]
    g = d[d["car"] == car_show].sort_values("mileage")
    idx = np.arange(len(g))
    T = g["t_mean"].to_numpy(float)
    axes[0].plot(idx, T, "-", lw=1.2, color=C_ACC, alpha=0.9)
    axes[0].plot(idx, T, ".", ms=3.0, color=C_ACC, alpha=0.8)
    axes[0].axvspan(len(g) - K_WIN, len(g), color=C_MAIN, alpha=0.16)
    axes[0].annotate(f"最近 {K_WIN} 段"
                     f"跨度 {D['reporting']['win_t_span_median']:.1f}°C",
                     (len(g) - K_WIN / 2, T.max()), ha="center", va="top",
                     fontsize=7.2, color=C_MAIN, xytext=(0, -6), textcoords="offset points")
    axes[0].set_xlabel(f"会话序号（car {car_show}，共 {len(g)} 段）")
    axes[0].set_ylabel("电池温度 (°C)")
    axes[0].set_title(f"① 温度是慢波：全历史跨度 {D['reporting']['all_t_span_median']:.0f}°C，"
                      f"而 {K_WIN} 段窗口只有 {D['reporting']['win_t_span_median']:.1f}°C",
                      fontsize=9.4, pad=8)

    bars = [("季节摆幅\n(p5–p95 温度)", D["season"]["amp_pp"], C_RED),
            (f"连续 {K_WIN} 段\n报告偏差 (SD)", D["reporting"]["bias_pp_sd"], C_ACC),
            ("第六轮车级\nMAE", 1.05, C_GREEN)]
    xi = np.arange(len(bars))
    axes[1].bar(xi, [b[1] for b in bars], color=[b[2] for b in bars], width=0.56)
    for i, b in enumerate(bars):
        axes[1].text(i, b[1] + 0.06, f"{b[1]:.2f} pp", ha="center", fontsize=8)
    axes[1].set_xticks(xi)
    axes[1].set_xticklabels([b[0] for b in bars], fontsize=7.8)
    axes[1].set_ylabel("SOH 变化幅度 (pp)")
    axes[1].set_ylim(0, max(b[1] for b in bars) * 1.24)
    axes[1].set_title("② 季节偏差与「已达成的精度」同量级", fontsize=9.8, pad=8)
    _style(axes[1])
    savefig(fig, os.path.join(fig_dir, "fig44_季节摆幅与报告口径.png"))


# ---------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    d = load(cfg)
    print(f"样本 {len(d)} 段 / {d['car'].nunique()} 车；温度 "
          f"{d['t_mean'].min():.1f}–{d['t_mean'].max():.1f}°C", flush=True)

    if os.environ.get("REPLOT"):
        import json
        with open(os.path.join(cfg["paths"]["metrics_dir"],
                               "temperature_confound.json"), encoding="utf-8") as f:
            res = json.load(f)
        A = res["A"]
        A["_pw"] = piecewise_fit(d, "capacity", "t_mean", KNOT, ref=REF_T,
                                 controls=["mileage"])
        C = {"skipped": "capacity|base" not in res["C"]}
        if not C["skipped"]:
            C.update({k: res["C"][k] for k in ("by_temp", "car_level", "slope_base",
                                               "slope_ref", "capacity|base", "y_ref|base",
                                               "y_ref|ref")})
            C["_cl"] = pd.read_parquet(os.path.join(cfg["paths"]["metrics_dir"],
                                                    "temperature_cars.parquet"))
        plots({"A": A, "C": C, "D": res["D"]}, d, cfg["paths"]["figures_dir"])
        print("REPLOT 完成", flush=True)
        return

    print("A 标签与预测的温度响应", flush=True)
    A = part_a(d)
    print("C 参考温度重训", flush=True)
    if os.environ.get("REUSE_C"):
        # 重训一次约 5 分钟；改 D 或改图时复用上一轮的 C，不必每次重训
        import json
        with open(os.path.join(cfg["paths"]["metrics_dir"],
                               "temperature_confound.json"), encoding="utf-8") as f:
            prev = json.load(f)
        C = dict(prev["C"])
        C["_cl"] = pd.read_parquet(os.path.join(cfg["paths"]["metrics_dir"],
                                                "temperature_cars.parquet"))
        print("C REUSE_C=1，复用已有重训结果", flush=True)
    else:
        C = part_c(cfg, d)
    print("D 报告口径", flush=True)
    D = part_d(d, A["_pw"], C)

    res = {"n_sessions": int(len(d)), "n_cars": int(d["car"].nunique()),
           "ref_t": REF_T, "knot": KNOT, "k_win": K_WIN,
           "A": {k: v for k, v in A.items() if not k.startswith("_")},
           "C": {k: v for k, v in C.items() if not k.startswith("_")},
           "D": D,
           "note": ("A 的斜率是车内固定效应估计：t_mean 与里程同时入模，"
                    "单位 Ah/°C；C 的折算参数按折内训练车估计，测试车不参与。")}
    save_json(res, "temperature_confound")
    print("写入 results/metrics/temperature_confound.json", flush=True)

    if not C.get("skipped"):
        C["_cl"].to_parquet(os.path.join(cfg["paths"]["metrics_dir"],
                                         "temperature_cars.parquet"), index=False)
    plots({"A": A, "C": C, "D": D}, d, cfg["paths"]["figures_dir"])


if __name__ == "__main__":
    main()
