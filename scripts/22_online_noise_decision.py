# -*- coding: utf-8 -*-
"""22 在线序贯标定、标签噪声感知与决策层。

第五轮（§18）把「多跑 k 次」与「多标 m 段」放在同一张网格上，得到 0.888 → 0.469 Ah。
但那个 m 段标定是**静态**的：一次性取校准池最靠后的 m 段，算一个偏差，此后不再更新。
真实部署是**流式**的——车一直在跑，带标签的会话陆续到达，估计器应该边走边改。
§18.6 还留下两个没被利用的事实：车级偏差会漂移（ρ=0.688，只有约六成可迁移），
19.5% 的标签是成段复制的。本脚本把这两条从「解释」变成「方法」：

  A. 在线序贯标定：把标定当成状态估计问题，逐个会话向前推，只用**过去**的标签。
     比较 累积均值 / 静态窗 / 指数遗忘 / 卡尔曼，并扫描标签到达节奏（每 1/3/10 段一个标签）。
     产出真正可部署的「误差随获得标签数下降」曲线（定安文档点名的 A2 主图）。
  B. 标签噪声感知：复制块可检测 → ①换一个去噪评估目标，量「报告误差里有多少是标签假象」；
     ②训练时按块降权，检验块状复制的标签是否真的在拖累模型。
  C. 决策层：残值评估要的不是 Ah，而是「这台车比别人差多少」。用车级估计做三件事——
     排序分辨率（差 1 pp 能不能分出来）、最差 10% 召回（批量筛查能不能捞出来）、
     区间下界的保守性（敢不敢用它压价）。
  D. 尾部与分层：确认平均改善没有牺牲个别车。

输出
  results/metrics/online_noise_decision.json
  results/figures/fig39_在线序贯标定.png
  results/figures/fig40_标签噪声感知.png
  results/figures/fig41_决策层分辨率与召回.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import (conformal_by_car, group_kfold_by_car,  # noqa: E402
                            regression_metrics)
from evsoh.features import active_features, feature_columns  # noqa: E402
from evsoh.models import build_model  # noqa: E402
from evsoh.plotting import (C_ACC, C_GRAY, C_GREEN, C_MAIN, C_PURPLE,  # noqa: E402
                            C_RED, plt, savefig)
from evsoh.target import add_soh_labels  # noqa: E402

BURN = 10                  # 前 10 段算冷启动期：还没有任何标签可用
MIN_SESS = 24
CADENCE = [1, 3, 10]       # 标签到达节奏（每几段会话产生一个带容量标签的会话）
HALF_GRID = [2, 3, 5, 8, 12, 20, 40]
K_WIN = 8                  # 车级上报窗口（沿用 §18 的饱和点）
BIAS_LAM = 5.0             # 偏差收缩强度（与 §15.4 / §18 的 fit_bias 一致）
NOMINAL = 0.90
LOO_W = 3                  # 去噪目标：局部中位数的半窗
CURVE_EDGES = [10, 15, 20, 30, 50, 100, 10 ** 9]   # 收敛曲线分箱（按累计标签数）
PRED_COL = "pred_xgb"
SMOKE = bool(os.environ.get("SMOKE"))
_STREAM_CACHE: dict = {}


# ---------------------------------------------------------------- 通用
def load_oof(cfg: dict) -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(cfg["paths"]["metrics_dir"], "oof_predictions.parquet"))
    d = d.dropna(subset=["car", "capacity", PRED_COL, "mileage", "d_soc"])
    d = d.sort_values(["car", "mileage"]).reset_index(drop=True)
    n = d.groupby("car")["car"].transform("size").to_numpy()
    d = d[n >= MIN_SESS].reset_index(drop=True)
    if SMOKE:
        keep = sorted(d["car"].unique())[:40]
        d = d[d["car"].isin(keep)].reset_index(drop=True)
    return d


def label_runs(df: pd.DataFrame) -> pd.DataFrame:
    """标签块：同一车内里程相邻、容量**精确相同**的连续会话同属一块。

    连续量不会偶然重复，重复只能来自"从稀疏测量复制到相邻会话"。块的第二个元素起
    是 copy，样本权重应当按块而不是按会话计。
    """
    d = df.sort_values(["car", "mileage"]).reset_index(drop=True).copy()
    same = d.groupby("car")["capacity"].diff().eq(0)
    same = same & d["car"].eq(d["car"].shift())
    d["is_copy"] = same.fillna(False).to_numpy(bool)
    d["run_id"] = ((~d["is_copy"].to_numpy()).cumsum()).astype(int)
    d["run_len"] = d.groupby("run_id")["run_id"].transform("size").to_numpy()
    return d


def loo_median(y: np.ndarray, w: int = LOO_W) -> np.ndarray:
    """留一局部中位数：不含自身的邻域中位数，作为"去噪后的容量"代理。

    不含自身是关键——否则目标里包含被预测的那一点，误差会被自己稀释掉。
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    out = np.full(n, np.nan)
    for i in range(n):
        nb = np.concatenate([y[max(0, i - w):i], y[i + 1:min(n, i + w + 1)]])
        if nb.size:
            out[i] = np.median(nb)
    return out


def _shrink(b: float, n_eff: float, lam: float = BIAS_LAM) -> float:
    """向 0 收缩：n_eff 少时不敢全信这个偏差（与 §15.4 的 fit_bias 同一个 λ）。"""
    return float(b) * float(n_eff) / (float(n_eff) + lam)


def _bias_for(mode: str, e: np.ndarray, avail: np.ndarray, t: int,
              half: float, m: int) -> float:
    """给定"可用标签的下标"，算预测第 t 段时应扣掉的偏差。

    ``e = pred - y``，所以最终修正是 ``e - b``（把系统性高估扣掉）。
    偏差估计按有效样本数收缩，与 §15.4/§18 的 ``fit_bias`` 共用 λ=5——
    这样"最近 8 段"这一格与第五轮的口径可以直接对比。
    """
    if avail.size == 0:
        return 0.0
    vals = e[avail]
    if mode == "none":
        return 0.0
    if mode == "cum":
        return _shrink(vals.mean(), vals.size)
    if mode == "win":
        v = vals[-m:]
        return _shrink(v.mean(), v.size)
    if mode in ("lab_ewma", "time_ewma"):
        # 标签节拍 vs 日历节拍：差的只是"龄"怎么算
        age = avail.size - 1 - np.arange(avail.size) if mode == "lab_ewma" else t - 1 - avail
        w = 0.5 ** (age / half)
        n_eff = float(w.sum() ** 2 / np.dot(w, w))   # Kish 有效样本数
        return _shrink(float(np.dot(w, vals) / w.sum()), n_eff)
    if mode == "lin":
        # 漂移是直线吗？取最近 m 个标签拟一条线再前推
        idx = avail[-m:]
        if idx.size < 3:
            return _shrink(vals[-m:].mean(), idx.size)
        x = idx.astype(float) - idx[-1]
        a0, a1 = np.polyfit(x, e[idx], 1)
        return float(a0 + a1 * (t - 1 - idx[-1]))
    raise ValueError(f"未知估计器: {mode}")


def stream(df: pd.DataFrame, mode: str, cadence: int = 1, half: float = 5.0,
           m: int = K_WIN, burn: int = BURN, tag: str = "") -> pd.DataFrame:
    """逐车逐会话向前推：预测第 t 段时只允许用 t 之前到达的标签。

    返回长表 (car, t, n_labels, err)，err 是"扣掉偏差之后"的残差。
    同一组 (mode, cadence, half, m) 会被主流程重复请求（网格、收敛曲线都要），
    在这里做记忆化，省掉重复的 O(n²) 递推。
    """
    key = (mode, cadence, float(half), int(m), int(burn), tag, int(len(df)))
    if key in _STREAM_CACHE:
        return _STREAM_CACHE[key]
    rows = []
    for car, g in df.groupby("car", sort=False):
        e = g[PRED_COL].to_numpy(dtype=float) - g["capacity"].to_numpy(dtype=float)
        n = e.size
        labels = np.arange(0, n, cadence)
        for t in range(burn, n):
            avail = labels[labels <= t - 1]
            b = _bias_for(mode, e, avail, t, half, m)
            rows.append((car, t, int(avail.size), float(e[t] - b)))
    out = pd.DataFrame(rows, columns=["car", "t", "n_labels", "err"])
    _STREAM_CACHE[key] = out
    return out


def _summ(s: pd.DataFrame) -> dict:
    r = s["err"].to_numpy(dtype=float)
    per_car = s.groupby("car")["err"].apply(lambda x: float(np.sqrt(np.mean(x ** 2))))
    return {"n_pred": int(r.size),
            "rmse": float(np.sqrt(np.mean(r ** 2))),
            "mae": float(np.mean(np.abs(r))),
            "car_rmse_median": float(per_car.median()),
            "car_rmse_p90": float(np.percentile(per_car, 90))}


# ---------------------------------------------------------------- A 在线序贯标定
def part_a(d: pd.DataFrame) -> dict:
    est_main = ["none", "cum", "win", "lab_ewma", "time_ewma", "lin"]
    out = {"burn": BURN, "cadence": CADENCE, "half_grid": HALF_GRID,
           "estimators": {}, "by_cadence": {}, "curve": {}, "half_sweep": {}}

    # 主口径：每段都有标签（节奏 1），静态窗取 K_WIN
    base = None
    for mode in est_main:
        s = stream(d, mode, cadence=1, half=5.0, m=K_WIN)
        r = _summ(s)
        if base is None:
            base = r["rmse"]
        r["vs_zero_pct"] = (r["rmse"] / base - 1.0) * 100.0
        out["estimators"][mode] = r
        print(f"  A 节奏1 {mode:11s} RMSE {r['rmse']:.4f} ({r['vs_zero_pct']:+.1f}%)", flush=True)

    # 半衰期扫描：遗忘多快才合适
    for mode, half in [("lab_ewma", h) for h in HALF_GRID]:
        r = _summ(stream(d, mode, cadence=1, half=half))
        out["half_sweep"][str(half)] = r["rmse"]
    best_half = min(out["half_sweep"], key=lambda k: out["half_sweep"][k])
    out["best_half"] = {"half": float(best_half), "rmse": out["half_sweep"][best_half]}
    print("  A 半衰期扫描 " + "  ".join(f"H={k}:{v:.3f}"
                                       for k, v in out["half_sweep"].items()), flush=True)

    # 标签节奏：稀疏标签下"按日历遗忘"会不会把历史一次性打折
    for cad in CADENCE:
        sub = {}
        for mode, half, m in [("none", 5.0, K_WIN), ("win", 5.0, K_WIN),
                              ("win", 5.0, 20), ("lab_ewma", 5.0, K_WIN),
                              ("time_ewma", 5.0, K_WIN)]:
            key = f"{mode}|m{m}" if mode == "win" else mode
            r = _summ(stream(d, mode, cadence=cad, half=half, m=m))
            sub[key] = r
        ref = sub["none"]["rmse"]
        for v in sub.values():
            v["vs_zero_pct"] = (v["rmse"] / ref - 1.0) * 100.0
        out["by_cadence"][str(cad)] = sub
        print(f"  A 节奏{cad:2d} " + "  ".join(f"{k}:{v['rmse']:.3f}" for k, v in sub.items()),
              flush=True)

    # 收敛曲线：拿到第 n 个标签之后，误差还剩多少（节奏 1）。
    # 两处坑：① 按"恰好 n 个标签"单点聚合，用 40 辆车时一个离群车就能把某个点顶到 1.19 Ah，
    # 曲线看着像"越标越差"；② 不做长历史筛选，n 大的分箱里只剩长历史车，换了一批分母。
    # 这里改成"区间分箱 + 只用历史足够长的同一批车"，曲线才可读。
    long_cars = d.groupby("car")["car"].size()
    long_cars = set(long_cars[long_cars >= CURVE_EDGES[-2] + BURN].index)
    d_long = d[d["car"].isin(long_cars)]
    print(f"  A 收敛曲线只用 {len(long_cars)} 辆长历史车（≥{CURVE_EDGES[-2]+BURN} 段）", flush=True)
    for mode, half in [("none", 5.0), ("win", 5.0), ("lab_ewma", float(best_half))]:
        s = stream(d_long, mode, cadence=1, half=half, m=K_WIN, tag="long")
        nb = s["n_labels"].to_numpy()
        pts = []
        for lo, hi in zip(CURVE_EDGES[:-1], CURVE_EDGES[1:]):
            sub = s[(nb >= lo) & (nb < hi)]
            if len(sub) < 100:
                continue
            pts.append({"lo": int(lo), "hi": int(hi), "n_pred": int(len(sub)),
                        "rmse": float(np.sqrt(np.mean(sub["err"] ** 2)))})
        key = f"{mode}_H{int(half)}" if mode == "lab_ewma" else mode
        out["curve"][key] = pts
        print(f"  A 收敛曲线 {key}: " + "  ".join(f"{p['lo']}+→{p['rmse']:.3f}" for p in pts),
              flush=True)
    out["curve_cars"] = int(len(long_cars))

    # 绝对 RMSE 曲线有个绕不开的混淆：同一批车的残差本身随车龄变化
    # （实测 t=50–99 的零样本 RMSE 0.689、t≥100 又回到 0.844），所以曲线的下降
    # 不能全记在"标签攒得多"头上。这里再给一条**同档内相对零样本的削减**曲线，
    # 把车龄效应当分母除掉，才是标定本身的贡献。
    base_curve = {p["lo"]: p["rmse"] for p in out["curve"]["none"]}
    out["curve_gain"] = {}
    for key, pts in out["curve"].items():
        if key == "none":
            continue
        out["curve_gain"][key] = [
            {"lo": p["lo"], "zero_rmse": base_curve[p["lo"]],
             "cut_pct": (1.0 - p["rmse"] / base_curve[p["lo"]]) * 100.0}
            for p in pts if p["lo"] in base_curve]
    for key, pts in out["curve_gain"].items():
        print(f"  A 同档内削减 {key}: " + "  ".join(f"{p['lo']}+→{p['cut_pct']:.1f}%"
                                                 for p in pts), flush=True)
    return out


# ---------------------------------------------------------------- B 标签噪声感知
def part_b(d: pd.DataFrame, cfg: dict) -> dict:
    out = {}
    # ① 块结构
    runs = d.groupby("run_id")["run_id"].size()
    out["blocks"] = {
        "copy_share": float(d["is_copy"].mean()),
        "sessions_in_multi_blocks": float((d["run_len"] > 1).mean()),
        "run_len_mean": float(runs.mean()),
        "run_len_max": int(runs.max()),
        "run_len_hist": {str(k): int(v) for k, v in
                         runs.value_counts().sort_index().head(8).items()},
        "n_runs": int(len(runs)),
    }
    print(f"  B 复制块：{out['blocks']['copy_share']*100:.1f}% 标签是复制来的，"
          f"平均块长 {out['blocks']['run_len_mean']:.2f}、最长 {out['blocks']['run_len_max']}",
          flush=True)

    # ② 去噪评估目标：误差里有多少是标签假象
    y_loo = np.concatenate([loo_median(g["capacity"].to_numpy(dtype=float))
                            for _c, g in d.groupby("car", sort=False)])
    pred = d[PRED_COL].to_numpy(dtype=float)
    lab = d["capacity"].to_numpy(dtype=float)
    m = np.isfinite(y_loo)
    r_lab = regression_metrics(lab[m], pred[m])
    r_loo = regression_metrics(y_loo[m], pred[m])
    out["denoise"] = {
        "window": LOO_W, "n": int(m.sum()),
        "rmse_vs_label": r_lab["rmse"], "mae_vs_label": r_lab["mae"],
        "rmse_vs_loo": r_loo["rmse"], "mae_vs_loo": r_loo["mae"],
        "delta_pct": (r_loo["rmse"] / r_lab["rmse"] - 1.0) * 100.0,
        "loo_noise_std": float(np.std((y_loo - lab)[m])),
        "loo_gap_mean_abs": float(np.mean(np.abs((y_loo - lab)[m]))),
        "note": ("去噪目标 = 留一局部中位数（半窗 %d 段）。它自己也会抹平真实容量随里程的变化，"
                 "所以 rmse_vs_loo 是『标签噪声剔除后』的乐观口径，不是真值。" % LOO_W),
    }
    print(f"  B 去噪目标：对标签 {r_lab['rmse']:.4f} → 对留一中位数 {r_loo['rmse']:.4f} "
          f"({out['denoise']['delta_pct']:+.1f}%)", flush=True)

    # ③ 训练侧：块级权重能不能换成精度
    out["weighted_training"] = _weighted_training(cfg)
    return out


def _weighted_training(cfg: dict) -> dict:
    """同一个按车分组 5 折下，只改样本权重：每段一票 / 每块一票 / 直接丢掉复制段。"""
    if os.environ.get("SKIP_TRAIN"):
        print("  B 训练侧：SKIP_TRAIN=1，跳过", flush=True)
        return {}
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, _lab = add_soh_labels(ses, cfg)
    df = df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)
    df = label_runs(df)
    if SMOKE:
        keep = sorted(df["car"].unique())[:60]
        df = df[df["car"].isin(keep)].reset_index(drop=True)
    cols = active_features(df, feature_columns())
    X = df[cols].astype(float)
    X = X.fillna(X.median()).fillna(0.0)
    y = df["capacity"].to_numpy(dtype=float)
    cars = df["car"].to_numpy()
    w_run = (1.0 / df["run_len"].to_numpy(dtype=float))
    res = {}
    folds = list(group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                    seed=cfg["splits"]["random_state"]))
    for name, w in (("base", np.ones_like(y)), ("per_run", w_run),
                    ("drop_copied", np.where(df["is_copy"].to_numpy(), 0.0, 1.0))):
        pred = np.full(len(df), np.nan)
        for tr, te in folds:
            keep = w[tr] > 0
            mdl = build_model("xgb", cfg["models"])
            mdl.fit(X.iloc[tr[keep]], y[tr[keep]], m__sample_weight=w[tr[keep]])
            pred[te] = mdl.predict(X.iloc[te])
        mm = regression_metrics(y, pred)
        res[name] = {"rmse": mm["rmse"], "mae": mm["mae"], "n": mm["n"],
                     "n_eff": float(w.sum())}
        res[name]["delta_vs_base_pct"] = (mm["rmse"] / res["base"]["rmse"] - 1.0) * 100.0 \
            if "base" in res and res["base"]["rmse"] > 0 else 0.0
        print(f"  B 训练权重 {name:12s} RMSE {mm['rmse']:.4f} "
              f"({res[name]['delta_vs_base_pct']:+.2f}%)", flush=True)
    return res


# ---------------------------------------------------------------- C 决策层
def car_level(d: pd.DataFrame) -> pd.DataFrame:
    """车级上报：最近 K_WIN 段的预测均值 + 只用窗口之前标签估的偏差。

    真值同口径取这 K_WIN 段标签的均值——问的是"这台车现在的健康度"。
    偏差一律按 ``y - pred`` 的方向估（正的偏差意味着模型**低估**），并加收缩；
    方向搞反会把校正变成加倍（这是第五轮 fit_bias 的约定，必须对齐）。
    """
    rows = []
    for car, g in d.groupby("car", sort=False):
        g = g.sort_values("mileage")
        p = g[PRED_COL].to_numpy(dtype=float)
        y = g["capacity"].to_numpy(dtype=float)
        e = p - y
        if y.size < K_WIN + 8:
            continue
        win = slice(y.size - K_WIN, y.size)
        hist = np.arange(0, y.size - K_WIN)
        row = {"car": car, "pred_zero": float(p[win].mean()), "y": float(y[win].mean()),
               "d_soc": float(g["d_soc"].to_numpy(dtype=float)[win].mean()),
               "t_mean": float(g["t_mean"].to_numpy(dtype=float)[win].mean()) if "t_mean" in g else np.nan,
               "bol_ref": float(g["bol_ref"].to_numpy(dtype=float)[0]),
               "n_sess": int(y.size)}
        # 静态：窗口之前最近 K_WIN 段
        row["b_static"] = -_shrink(e[hist][-K_WIN:].mean(), K_WIN) if hist.size >= K_WIN else 0.0
        # 在线：窗口之前全部标签的指数遗忘（按标签节拍，半衰期 5 个标签）
        if hist.size:
            age = hist.size - 1 - np.arange(hist.size)
            w = 0.5 ** (age / 5.0)
            n_eff = float(w.sum() ** 2 / np.dot(w, w))
            row["b_online"] = -_shrink(float(np.dot(w, e[hist]) / w.sum()), n_eff)
        else:
            row["b_online"] = 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:      # 一辆车都不满足时不构造派生列（否则 KeyError: 'pred_zero'）
        return out
    for k in ("static", "online"):
        out[f"pred_{k}"] = out["pred_zero"] + out[f"b_{k}"]
    return out


def _screening(cl: pd.DataFrame, picked=(0.05, 0.10, 0.15, 0.20, 0.30)) -> dict:
    """按估计的 SOH 从差到好排序，取前 x% 名额送去复检，能覆盖多少真正的差车。

    名额等于真差车数（10%）时，召回与命中率在数学上必然相等，所以这张图看的是
    **名额放宽后召回涨多快**，而不是两个指标。
    """
    y = cl["y"].to_numpy(dtype=float)
    bad = y <= float(np.quantile(y, 0.10))
    n = len(y)
    out = {"picked_pct": [p * 100 for p in picked], "n_bad": int(bad.sum()),
           "zero": [], "online": []}
    for scheme in ("zero", "online"):
        s = cl[f"pred_{scheme}"].to_numpy(dtype=float)
        order = np.argsort(s)
        for p in picked:
            k = max(1, int(round(p * n)))
            out[scheme].append(float(bad[order[:k]].sum() / max(bad.sum(), 1)))
    return out


def part_c(cl: pd.DataFrame) -> dict:
    out = {"n_cars": int(len(cl)), "k_win": K_WIN}
    ref = float(cl["bol_ref"].mean())
    out["ref_capacity"] = ref
    for scheme, col in (("zero", "pred_zero"), ("static8", "pred_static"), ("online", "pred_online")):
        mm = regression_metrics(cl["y"], cl[col])
        # 车级区间：以车为单位的 split-conformal（车就是独立单位，不再分组）
        cv = _conformal_cars(cl["y"], cl[col], nominal=NOMINAL)
        out[scheme] = {**mm, "rmse_pp": mm["rmse"] / ref * 100.0,
                       "mae_pp": mm["mae"] / ref * 100.0, **cv}
        print(f"  C 车级 {scheme:8s} RMSE {mm['rmse']:.4f} Ah / {mm['rmse']/ref*100:.2f} pp，"
              f"90% 宽度 {cv['width']:.3f} Ah，覆盖 {cv['coverage']*100:.1f}%", flush=True)

    # ① 排序分辨率：两台车差 X pp 时，能不能分出先后
    y = cl["y"].to_numpy(dtype=float)
    est = cl["pred_online"].to_numpy(dtype=float)
    n = len(cl)
    ii, jj = np.triu_indices(n, k=1)
    dy = np.abs(y[ii] - y[jj]) / ref * 100.0
    ok = (est[ii] - est[jj]) * (y[ii] - y[jj]) > 0
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 99.0)]
    out["resolution"] = {"ref_capacity": ref, "n_pairs": int(len(dy)),
                         "bins": [{"lo": a, "hi": b, "n_pairs": int(((dy >= a) & (dy < b)).sum()),
                                   "order_acc": float(ok[(dy >= a) & (dy < b)].mean())
                                   if ((dy >= a) & (dy < b)).sum() else float("nan")}
                                  for a, b in bins],
                         "overall_order_acc": float(ok.mean())}
    print("  C 排序一致率 " + "  ".join(
        f"{b['lo']}-{b['hi']}pp:{b['order_acc']*100:.0f}%(n={b['n_pairs']})"
        for b in out["resolution"]["bins"]), flush=True)

    # ② 最差 10% 召回：批量筛查能不能把该复检的车捞出来
    q = 0.10
    thr = float(np.quantile(y, q))
    bad = y <= thr
    out["worst_decile"] = {"q": q, "threshold_ah": thr, "n_bad": int(bad.sum()),
                           "threshold_pp": (1 - thr / ref) * 100.0}
    for scheme in ("zero", "online"):
        s = cl[f"pred_{scheme}"].to_numpy(dtype=float)
        pick = np.zeros(n, dtype=bool)
        pick[np.argsort(s)[:int(round(q * n))]] = True
        rec = float((pick & bad).sum() / max(bad.sum(), 1))
        prec = float((pick & bad).sum() / max(pick.sum(), 1))
        out["worst_decile"][scheme] = {"recall": rec, "precision": prec,
                                       "n_picked": int(pick.sum())}
        print(f"  C 最差 {int(q*100)}% 召回（{scheme}）: 召回 {rec*100:.0f}% / 精确 {prec*100:.0f}%",
              flush=True)
    # 名额放宽时的召回曲线：现场不可能只复检 10%，看放宽到 20%/30% 值不值
    out["screening"] = _screening(cl)
    print("  C 筛查召回曲线：" + "  ".join(
        f"{p}%名额→{out['screening']['online'][i]*100:.0f}%(零样本 "
        f"{out['screening']['zero'][i]*100:.0f}%)"
        for i, p in enumerate(out["screening"]["picked_pct"])), flush=True)

    # ③ 区间下界：敢不敢用它压价
    lo, hi = _conformal_bounds(cl["y"], cl["pred_online"], nominal=NOMINAL)
    below = (cl["y"].to_numpy(dtype=float) < lo.to_numpy(dtype=float))
    out["lower_bound"] = {"coverage": float((~below).mean()),
                          "below_rate": float(below.mean()),
                          "mean_conservatism_ah": float((cl["y"] - lo).mean()),
                          "mean_conservatism_pp": float((cl["y"] - lo).mean() / ref * 100),
                          "width_ah": float((hi - lo).mean())}
    print(f"  C 区间下界：{out['lower_bound']['below_rate']*100:.1f}% 的车真值低于下界，"
          f"平均保守 {out['lower_bound']['mean_conservatism_pp']:.2f} pp", flush=True)
    return out


def _conformal_bounds(y, pred, nominal: float = NOMINAL, n_splits: int = 5,
                      cal_frac: float = 0.25, seed: int = 42):
    """车级 split-conformal，五折交叉（每折从其余车抽 25% 做校准）。

    与 ``evaluate.conformal_by_car`` 同一套配方，只是这里要把上下界留下来算"下界保守量"。
    单折 30% 的写法在 187 辆车上只有 56 个校准点，分位数估计本身就有 ±2 pp 的抖动。
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    idx = np.arange(y.size)
    lo = np.full(y.shape, np.nan)
    hi = np.full(y.shape, np.nan)
    for fi, (tr, te) in enumerate(group_kfold_by_car(idx, n_splits=n_splits, seed=seed)):
        rng = np.random.default_rng(1000 + fi)
        n_cal = max(1, int(round(len(tr) * cal_frac)))
        cal = rng.choice(tr, size=n_cal, replace=False)
        q = float(np.quantile(np.abs(pred[cal] - y[cal]), nominal, method="higher"))
        lo[te], hi[te] = pred[te] - q, pred[te] + q
    return pd.Series(lo), pd.Series(hi)


def _conformal_cars(y, pred, nominal: float = NOMINAL) -> dict:
    lo, hi = _conformal_bounds(y, pred, nominal)
    m = lo.notna().to_numpy()
    yy = np.asarray(y, dtype=float)
    inside = (yy[m] >= lo.to_numpy()[m]) & (yy[m] <= hi.to_numpy()[m])
    return {"width": float((hi.to_numpy()[m] - lo.to_numpy()[m]).mean()),
            "coverage": float(inside.mean())}


# ---------------------------------------------------------------- D 尾部与分层
def part_d(d: pd.DataFrame, cl: pd.DataFrame) -> dict:
    out = {}
    # 车级每辆车只有一条观测，所以"每车 RMSE"退化成 |误差|，这里就按误差分布报
    e = (cl["pred_online"] - cl["y"]).to_numpy(dtype=float)
    ref = float(cl["bol_ref"].mean())
    out["car_error"] = {"median_abs": float(np.median(np.abs(e))),
                        "p90_abs": float(np.percentile(np.abs(e), 90)),
                        "worst_abs": float(np.max(np.abs(e))),
                        "worst_car": int(cl["car"].to_numpy()[int(np.argmax(np.abs(e)))]),
                        "median_abs_pp": float(np.median(np.abs(e)) / ref * 100),
                        "p90_abs_pp": float(np.percentile(np.abs(e), 90) / ref * 100),
                        "n_cars": int(len(cl))}
    q = out["car_error"]
    print(f"  D 车级误差中位 {q['median_abs']:.3f} / p90 {q['p90_abs']:.3f} / "
          f"最差 {q['worst_abs']:.3f} Ah（car {q['worst_car']}）", flush=True)

    per = cl.assign(err=e)
    subs = {}
    for col, label in (("d_soc", "SOC 跨度"), ("t_mean", "平均温度")):
        if col not in per:
            continue
        per["_bin"] = pd.qcut(per[col], 3, labels=False, duplicates="drop")
        g = per.groupby("_bin")["err"].apply(lambda x: float(np.sqrt(np.mean(x ** 2))))
        names = {0: "低", 1: "中", 2: "高"}
        subs[label] = {names.get(int(k), str(int(k))): float(v) for k, v in g.items()}
    if "dataset" in d:
        ds = cl.merge(d.groupby("car")["dataset"].first().reset_index(), on="car", how="left")
        rmse = ds.assign(err=ds["pred_online"] - ds["y"]).groupby("dataset")["err"].apply(
            lambda x: float(np.sqrt(np.mean(x ** 2))))
        cnt = ds.groupby("dataset")["car"].size()
        subs["数据集"] = {str(k): float(rmse[k]) for k in rmse.index if cnt[k] >= 10}
    out["subgroups"] = subs
    for k, v in subs.items():
        print(f"  D 分层 {k}: " + "  ".join(f"{a}:{b:.3f}" for a, b in v.items()), flush=True)
    return out


# ---------------------------------------------------------------- 图
def plots(res: dict, cl: pd.DataFrame, fig_dir: str) -> None:
    A, B, C = res["online"], res["noise"], res["decision"]

    # ---- 图 39：在线收敛 + 标签节奏 ----
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    hb = int(A["best_half"]["half"])
    styles = {"win": (C_ACC, "-.", "静态最近 8 段"),
              f"lab_ewma_H{hb}": (C_MAIN, "-", f"指数遗忘（按标签节拍，半衰期 {hb}）")}
    for key, (c, ls, lab) in styles.items():
        pts = A["curve_gain"].get(key)
        if not pts:
            continue
        xs = [p["lo"] for p in pts]
        ys = [p["cut_pct"] for p in pts]
        axes[0].plot(xs, ys, marker="o", ms=3.8, lw=1.7, color=c, ls=ls, label=lab)
        axes[0].annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), textcoords="offset points",
                         xytext=(-4, 8), fontsize=7.2, color=c, ha="right")
    axes[0].axhline(0, color=C_GRAY, ls=":", lw=1.0)
    axes[0].text(A["curve"]["none"][0]["lo"], 0.5, "标定无效（=零样本）", fontsize=7.0, color=C_GRAY)
    axes[0].set_xscale("log")
    axes[0].set_xticks([p["lo"] for p in A["curve"]["none"]])
    axes[0].set_xticklabels([f"≥{p['lo']}" for p in A["curve"]["none"]], fontsize=7.8)
    axes[0].set_xlabel("累计已获得的带标签会话数")
    axes[0].set_ylabel("同档内相对零样本的误差削减 (%)")
    axes[0].set_ylim(0, max(p["cut_pct"] for p in A["curve_gain"][f"lab_ewma_H{hb}"]) * 1.32)
    axes[0].set_title("真正可部署的收敛曲线：只用过去的标签", fontsize=9.6, pad=8)
    axes[0].legend(fontsize=7.2, loc="lower right", framealpha=0.92)

    cads = [1, 3, 10]
    width = 0.26
    keys = [("win|m8", "静态最近 8 段", C_ACC), ("lab_ewma", "按标签节拍遗忘", C_MAIN),
            ("time_ewma", "按日历节拍遗忘", C_RED)]
    for i, (key, lab, c) in enumerate(keys):
        vals = [A["by_cadence"][str(cd)][key]["rmse"] for cd in cads]
        xi = np.arange(len(cads)) + (i - 1) * width
        axes[1].bar(xi, vals, width=width, color=c, label=lab)
        for x, v in zip(xi, vals):
            axes[1].text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.9, rotation=90)
    axes[1].axhline(A["by_cadence"]["1"]["none"]["rmse"], color=C_GRAY, ls="--", lw=1.0)
    axes[1].text(len(cads) - 0.5, A["by_cadence"]["1"]["none"]["rmse"] + 0.004,
                 "零样本", fontsize=7.2, color=C_GRAY, ha="right")
    axes[1].set_xticks(np.arange(len(cads)))
    axes[1].set_xticklabels([f"每 {c} 段一个标签" for c in cads], fontsize=8)
    axes[1].set_ylabel("会话级 RMSE (Ah，y 轴截断)")
    axes[1].set_ylim(min(A["by_cadence"]["1"]["win|m8"]["rmse"], 0.6) - 0.09,
                     A["by_cadence"]["1"]["none"]["rmse"] + 0.06)
    axes[1].set_title("标签越稀疏，越不能按日历遗忘", fontsize=9.6, pad=8)
    axes[1].legend(fontsize=7.2, loc="upper left", framealpha=0.92)
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, "fig39_在线序贯标定.png"))

    # ---- 图 40：标签块结构 + 去噪口径 + 训练权重 ----
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6),
                             gridspec_kw={"width_ratios": [1.0, 0.95, 1.05]})
    hist = B["blocks"]["run_len_hist"]
    ks = [int(k) for k in hist]
    axes[0].bar(ks, [hist[str(k)] for k in ks], color=C_PURPLE, width=0.62)
    for k in ks:
        axes[0].text(k, hist[str(k)], str(hist[str(k)]), ha="center", va="bottom", fontsize=7)
    axes[0].set_xlabel("同一标签连续覆盖的会话数（块长）")
    axes[0].set_ylabel("块的数量")
    share1 = hist[str(ks[0])] / sum(hist.values()) * 100
    axes[0].text(0.97, 0.72, f"块长 1（独有测量）\n占块数 {share1:.0f}%、占会话 "
                 f"{(1 - B['blocks']['copy_share']) * 100:.0f}%",
                 transform=axes[0].transAxes, ha="right", fontsize=7.0, color=C_GRAY)
    axes[0].set_title(f"19.5% 的标签是复制品（最长 {B['blocks']['run_len_max']} 段）",
                      fontsize=9.3, pad=8)

    dn = B["denoise"]
    bars = axes[1].bar([0, 1], [dn["rmse_vs_label"], dn["rmse_vs_loo"]],
                       color=[C_GRAY, C_MAIN], width=0.55)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["对原始标签", "对去噪目标"], fontsize=8.4)
    for b, v in zip(bars, [dn["rmse_vs_label"], dn["rmse_vs_loo"]]):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                     ha="center", fontsize=8)
    axes[1].set_ylim(min(dn["rmse_vs_label"], dn["rmse_vs_loo"]) - 0.12,
                     max(dn["rmse_vs_label"], dn["rmse_vs_loo"]) + 0.05)
    axes[1].set_ylabel("RMSE (Ah，y 轴截断)")
    axes[1].annotate(f"{dn['delta_pct']:+.1f}%", (0.5, (dn["rmse_vs_label"] + dn["rmse_vs_loo"]) / 2),
                     ha="center", fontsize=10, color=C_RED)
    axes[1].set_title("报告误差里约 9% 是标签假象", fontsize=9.3, pad=8)
    axes[1].grid(axis="x", visible=False)

    wt = B.get("weighted_training") or {}
    if wt:
        names = {"base": "每段一票", "per_run": "每块一票", "drop_copied": "丢弃复制段"}
        ks = [k for k in ("base", "per_run", "drop_copied") if k in wt]
        vals = [wt[k]["rmse"] for k in ks]
        bars = axes[2].bar(range(len(ks)), vals, color=[C_GRAY, C_MAIN, C_ACC][:len(ks)], width=0.55)
        for i, (b, k) in enumerate(zip(bars, ks)):
            axes[2].text(i, vals[i] + 0.002, f"{vals[i]:.4f}\n({wt[k]['delta_vs_base_pct']:+.2f}%)",
                         ha="center", fontsize=7.2)
        axes[2].set_xticks(range(len(ks)))
        axes[2].set_xticklabels([names[k] for k in ks], fontsize=8.4)
        axes[2].set_ylim(0, max(vals) * 1.18)   # 从 0 起：三条柱子看不出差别，正是结论
        axes[2].set_ylabel("RMSE (Ah)")
        axes[2].set_title("按块降权重训：噪声在目标侧，不在特征侧", fontsize=9.3, pad=8)
        axes[2].grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, "fig40_标签噪声感知.png"))

    # ---- 图 41：决策层 ----
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))
    bins = [b for b in C["resolution"]["bins"] if np.isfinite(b["order_acc"])]
    xs = np.arange(len(bins))
    axes[0].plot(xs, [b["order_acc"] * 100 for b in bins], marker="o", color=C_MAIN, lw=1.8)
    axes[0].axhline(50, color=C_GRAY, ls=":", lw=1.0)
    for i, b in enumerate(bins):
        axes[0].annotate(f"{b['order_acc']*100:.0f}%", (i, b["order_acc"] * 100),
                         textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.4)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([f"{b['lo']:g}–{b['hi']:g}" for b in bins], fontsize=8)
    axes[0].set_xlabel("两台车真实 SOH 差异 (pp)")
    axes[0].set_ylabel("排序判对的概率 (%)")
    axes[0].set_ylim(45, 102)
    axes[0].set_title("分辨率：差多少 pp 才敢说谁更差", fontsize=9.3, pad=8)

    wd = C["worst_decile"]
    scr = C.get("screening") or _screening(cl)
    xs = scr["picked_pct"]
    axes[1].plot(xs, [v * 100 for v in scr["zero"]], marker="s", ms=3.4, lw=1.5,
                 color=C_GRAY, label="零样本")
    axes[1].plot(xs, [v * 100 for v in scr["online"]], marker="o", ms=3.6, lw=1.8,
                 color=C_MAIN, label="在线标定")
    # 随机抽检：名额 p% 只能覆盖 p/10 的差车（差车占车队 10%），到 10% 名额即 100%，之后转平
    xs_r = [0, 10, max(xs)]
    axes[1].plot(xs_r, [0, 100, 100], ls=":", lw=1.0, color=C_RED, label="随机抽检")
    axes[1].axvline(10, color=C_GRAY, ls="--", lw=0.9)
    axes[1].annotate(f"10% 名额：{wd['zero']['recall']*100:.0f}% → "
                     f"{wd['online']['recall']*100:.0f}%", (10, wd["online"]["recall"] * 100),
                     textcoords="offset points", xytext=(6, -14), fontsize=7.0, color=C_MAIN)
    axes[1].set_xlabel("复检名额（占车队比例 %）")
    axes[1].set_ylabel(f"最差 {int(wd['q']*100)}% 车队的召回率 (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("批量筛查：名额放宽，召回涨多快", fontsize=9.3, pad=8)
    axes[1].legend(fontsize=7.2, loc="lower right", framealpha=0.92)

    lb = C["lower_bound"]
    y = cl["y"].to_numpy(dtype=float)
    lo, hi = _conformal_bounds(cl["y"], cl["pred_online"], nominal=NOMINAL)
    m = lo.notna().to_numpy()
    pv = cl["pred_online"].to_numpy(dtype=float)[m]
    axes[2].scatter(pv, y[m], s=11, color=C_MAIN, alpha=0.6, edgecolor="none",
                    label="车级点估计")
    # 抽 40 辆画竖线，否则 187 条会糊成一片；区间是等宽的（split-conformal）
    rng = np.random.default_rng(0)
    pick = rng.choice(np.where(m)[0], size=min(40, m.sum()), replace=False)
    axes[2].vlines(cl["pred_online"].to_numpy(dtype=float)[pick],
                   lo.to_numpy(dtype=float)[pick], hi.to_numpy(dtype=float)[pick],
                   color=C_RED, lw=0.9, alpha=0.55, label="90% 区间（抽 40 辆）")
    axes[2].plot(pv, lo.to_numpy(dtype=float)[m], ".", ms=1.1, color=C_RED, alpha=0.5)
    lim = [y.min() - 0.8, y.max() + 0.8]
    axes[2].plot(lim, lim, ls="--", lw=0.9, color=C_GRAY)
    axes[2].set_xlim(lim)
    axes[2].set_ylim(lim)
    axes[2].set_xlabel("估计容量 (Ah)")
    axes[2].set_ylabel("真实容量 (Ah)")
    axes[2].set_title(f"下界只比真值低 {lb['mean_conservatism_pp']:.2f} pp，敢用来定价",
                      fontsize=9.3, pad=8)
    axes[2].legend(fontsize=7.0, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, "fig41_决策层分辨率与召回.png"))


# ---------------------------------------------------------------- 主流程
def main() -> None:
    cfg = load_config()
    d = label_runs(load_oof(cfg))
    cl = car_level(d)
    print(f"样本 {len(d)} 段 / {d['car'].nunique()} 车；车级 {len(cl)} 车", flush=True)

    if os.environ.get("REPLOT"):
        # 只重画图（调图时不必重跑重训那 5 分钟）
        import json
        path = os.path.join(cfg["paths"]["metrics_dir"], "online_noise_decision.json")
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        plots(res, cl, cfg["paths"]["figures_dir"])
        print("REPLOT 完成", flush=True)
        return

    print("A 在线序贯标定", flush=True)
    A = part_a(d)
    print("B 标签噪声感知", flush=True)
    B = part_b(d, cfg)
    print("C 决策层", flush=True)
    C = part_c(cl)
    print("D 尾部与分层", flush=True)
    D = part_d(d, cl)

    res = {"n_sessions": int(len(d)), "n_cars": int(d["car"].nunique()),
           "burn": BURN, "min_sessions": MIN_SESS, "k_win": K_WIN,
           "online": A, "noise": B, "decision": C, "stratify": D,
           "note": ("A 是会话级流式：预测第 t 段时只用 t 之前到达的标签，前 10 段为冷启动；"
                    "C 是车级上报：最近 8 段的预测均值 + 窗口之前标签估的偏差；"
                    "两条口径的绝对值不可与 §18（只用后 50% 评估集）直接比。")}
    save_json(res, "online_noise_decision")
    print("写入 results/metrics/online_noise_decision.json", flush=True)
    plots(res, cl, cfg["paths"]["figures_dir"])


if __name__ == "__main__":
    main()
