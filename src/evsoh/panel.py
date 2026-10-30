# -*- coding: utf-8 -*-
"""面板（同车重复观测）回归工具：车内去均值固定效应、聚类标准误、一阶差分。

为什么单独一个模块：本项目所有"同一辆车多个充电会话"的分析，都必须把车级差异
（BOL 容量、装配一致性、使用习惯）先扣掉，否则跨车的横截面差异会污染任何斜率估计。
第七轮的温度混杂分析尤其依赖这一点——温度与车强相关（南方车常年偏热），
不扣车效应的话，量到的是"南方车容量偏大"，不是"同一辆车在热天容量偏大"。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def within_demean(df: pd.DataFrame, cols: Sequence[str], group: str = "car") -> pd.DataFrame:
    """对 `cols` 做车内去均值（等价于加车固定效应）。

    返回新表，列名不变，只保留 ``group`` + ``cols``。缺失值不参与均值。
    """
    cols = list(cols)
    out = df[[group] + cols].copy()
    out[cols] = out[cols].astype(float)
    out[cols] = out[cols] - out.groupby(group)[cols].transform("mean")
    return out


def fe_ols(df: pd.DataFrame, y: str, xs: Sequence[str], group: str = "car",
           cluster: bool = True) -> Dict[str, object]:
    """车内固定效应 OLS：先去均值，再跑无截距回归。

    返回 ``beta``（与 ``xs`` 同序）、``se``（按 ``group`` 聚类的稳健标准误）、
    ``r2``（去均值后的组内 R²）、``n``、``n_groups``、``resid``。
    """
    xs = list(xs)
    sub = df[[group, y] + xs].dropna()
    dm = within_demean(sub, [y] + xs, group=group)
    A = dm[xs].to_numpy(dtype=float)
    b = dm[y].to_numpy(dtype=float)
    if A.size == 0:
        raise ValueError("fe_ols: 去均值后没有样本")
    xtx = A.T @ A
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (A.T @ b)
    resid = b - A @ beta
    ss_tot = float(np.sum(b ** 2))          # 已去均值，故以 0 为中心
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")

    if cluster:
        g = sub[group].to_numpy()
        meat = np.zeros_like(xtx)
        for c in np.unique(g):
            m = g == c
            u = resid[m]
            xg = A[m]
            s = xg.T @ u
            meat += np.outer(s, s)
        cov = xtx_inv @ meat @ xtx_inv
    else:
        sigma2 = float(np.sum(resid ** 2)) / max(len(b) - len(xs), 1)
        cov = xtx_inv * sigma2
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {"beta": beta, "se": se, "r2": r2, "n": int(len(b)),
            "n_groups": int(sub[group].nunique()), "resid": resid,
            "names": xs, "cov": cov}


def cluster_bootstrap(df: pd.DataFrame, y: str, xs: Sequence[str], group: str = "car",
                      n_boot: int = 200, seed: int = 42) -> np.ndarray:
    """按车整群重抽样，返回 (n_boot, k) 的斜率矩阵。

    车是独立单位（同一辆车内的会话高度相关），所以只能整群抽车，不能抽会话。
    """
    xs = list(xs)
    sub = df[[group, y] + xs].dropna()
    groups = sub[group].unique()
    rng = np.random.default_rng(seed)
    out = np.full((n_boot, len(xs)), np.nan)
    for i in range(n_boot):
        pick = rng.choice(groups, size=groups.size, replace=True)
        parts = [sub[sub[group] == c] for c in pick]
        boot = pd.concat(parts, ignore_index=True)
        boot[group] = np.repeat(np.arange(len(pick)), [len(p) for p in parts])
        try:
            out[i] = fe_ols(boot, y, xs, group=group, cluster=False)["beta"]
        except (ValueError, np.linalg.LinAlgError):
            continue
    return out


def first_difference(df: pd.DataFrame, cols: Sequence[str], group: str = "car",
                     order: str = "mileage") -> pd.DataFrame:
    """按车内顺序做一阶差分：同时消掉车固定效应与任何线性时间趋势。

    温度混杂最容易担心的替代解释是"车越用越旧、恰好也越用越热"，
    差分后线性老化项被完全消掉，剩下的斜率就是温度带来的净效应。
    """
    cols = list(cols)
    sub = df[[group, order] + cols].sort_values([group, order]).copy()
    sub[cols] = sub[cols].astype(float)
    diff = sub.groupby(group)[cols].diff()
    diff[group] = sub[group].to_numpy()
    return diff.dropna()


def hinge(x: np.ndarray, knot: float) -> np.ndarray:
    """线性样条基：``max(x - knot, 0)``，用来让斜率在 ``knot`` 之后拐弯。"""
    x = np.asarray(x, dtype=float)
    return np.clip(x - knot, 0.0, None)


def piecewise_fit(df: pd.DataFrame, y: str, x: str, knot: float, group: str = "car",
                  ref: float = 25.0, controls: Sequence[str] = ()) -> Dict[str, float]:
    """车内固定效应的分段线性拟合，返回 ``delta(T) - delta(ref)`` 的修正函数参数。

    模型：``y = a·T + b·max(T-knot,0) + Σc·controls + 车固定效应``。
    修正量定义为 ``f(T) - f(ref)``，所以在参考温度处恒为 0（去条件化不改变基准点）。

    ``controls`` 必须带上里程：温度与里程在一辆车的历史里高度相关（车越用越旧，
    也经历完整的季节循环），不控制里程会把老化折进温度斜率。实测不控制时
    斜率由 +0.045 抬到 +0.069（+53%），直接拿去折算会**过度修正**。
    """
    controls = list(controls)
    sub = df[[group, y, x] + controls].dropna().copy()
    sub["_h"] = hinge(sub[x].to_numpy(dtype=float), knot)
    xs = [x, "_h"] + controls
    fit = fe_ols(sub, y, xs, group=group)
    a, b = float(fit["beta"][0]), float(fit["beta"][1])

    def f(t: float) -> float:
        return a * float(t) + b * float(max(float(t) - knot, 0.0))

    out = {"slope_below_knot": a, "slope_above": a + b, "knot": float(knot),
           "ref": float(ref), "shift_at_ref": f(ref) - f(ref),
           "_f": f, "r2": float(fit["r2"]), "n": int(fit["n"])}
    for name, v in zip(xs[2:], fit["beta"][2:]):
        out[f"coef_{name}"] = float(v)
    return out


def debias(y: np.ndarray, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """把"在温度 T 下测到的容量"折算到参考温度：``y - (f(T) - f(ref))``。"""
    f = params["_f"]
    ref = float(params["ref"])
    t = np.asarray(t, dtype=float)
    shift = np.array([f(v) - f(ref) for v in t], dtype=float)
    return np.asarray(y, dtype=float) - shift


def temperature_bins(t: np.ndarray, n: int = 3) -> Tuple[np.ndarray, List[float]]:
    """按温度分位切箱，返回 (标签, 边界)。样本不足时自动减少箱数。"""
    t = np.asarray(t, dtype=float)
    edges = np.quantile(t, np.linspace(0, 1, n + 1)[1:-1])
    lab = np.digitize(t, edges)
    return lab, [float(e) for e in edges]


# ---------------------------------------------------------------- 收缩（partial pooling）
def tau2_moments(b_hat: np.ndarray, se: np.ndarray, inflate: float = 1.0) -> Dict[str, float]:
    """斜率异质性的矩估计：``τ² = var(b_hat) − 平均抽样方差``。

    ``b_hat`` 是每辆车单独估的斜率，``se`` 是它的标准误。观测到的离散度里
    有一部分只是抽样噪声（``se²``），扣掉之后的 ``τ²`` 才是"车与车真的不一样"。
    结果截断在 0：若 ``var(b_hat) < 噪声``，说明根本没有可识别的异质性，
    收缩权重应当全部给全局值。

    ``inflate`` 用来吸收残差自相关造成的方差低估（用 ``(1+ρ)/(1−ρ)`` 近似）。
    """
    b = np.asarray(b_hat, dtype=float)
    s = np.asarray(se, dtype=float) * np.sqrt(float(inflate))
    ok = np.isfinite(b) & np.isfinite(s)
    b, s = b[ok], s[ok]
    if b.size < 3:
        raise ValueError("tau2_moments: 有效车数不足")
    var_b = float(np.var(b, ddof=1))
    noise = float(np.mean(s ** 2))
    tau2 = max(0.0, var_b - noise)
    return {"tau2": tau2, "tau": float(np.sqrt(tau2)), "var_b": var_b,
            "noise_var": noise, "share_signal": float(tau2 / var_b) if var_b > 0 else 0.0,
            "share_noise": float(noise / var_b) if var_b > 0 else 1.0,
            "n": int(b.size), "inflate": float(inflate)}


def shrink_slopes(b_hat: np.ndarray, se: np.ndarray, global_b: float,
                  tau2: float | None = None, lam: float | None = None) -> Dict[str, np.ndarray]:
    """把每辆车的斜率往全局值收缩：``b_c = λ_c·b_hat_c + (1−λ_c)·b_global``。

    给了 ``tau2`` 就走经验贝叶斯权重 ``λ_c = τ² / (τ² + se_c²)``——
    估得越准的车（se 小）越保留自己的斜率，估得糊的车自动退回全局值。
    给了 ``lam`` 就用固定权重（用于画 λ 曲线做敏感性分析）。
    """
    b = np.asarray(b_hat, dtype=float)
    s = np.asarray(se, dtype=float)
    if lam is not None:
        w = np.full_like(b, float(lam))
    else:
        if tau2 is None:
            raise ValueError("shrink_slopes: 需要 tau2 或 lam")
        w = tau2 / (tau2 + s ** 2)
    return {"slope": global_b + w * (b - global_b), "weight": w}


def trend_resid_sd(df: pd.DataFrame, y: str, order: str = "mileage",
                   shift: np.ndarray | None = None, group: str = "car") -> float:
    """先减掉折算量 ``shift``，再车内去均值并对 ``order`` 去趋势，返回残差标准差。

    这是本轮统一的"折算好不好"的度量：折算若正确，剩下的就只是老化趋势 + 噪声，
    残差越小说明温度项被吃得越干净。用共同斜率 + 车内截距（等价于
    车固定效应回归 ``y ~ order``），与 ``fe_ols`` 的口径一致。
    """
    sub = df[[group, y, order]].copy()
    z = sub[y].to_numpy(dtype=float)
    if shift is not None:
        z = z - np.asarray(shift, dtype=float)
    x = sub[order].to_numpy(dtype=float)
    g = sub[group].to_numpy()
    zc = z - pd.Series(z).groupby(g).transform("mean").to_numpy()
    xc = x - pd.Series(x).groupby(g).transform("mean").to_numpy()
    denom = float(xc @ xc)
    beta = float(xc @ zc) / denom if denom > 0 else 0.0
    resid = zc - beta * xc
    return float(np.std(resid, ddof=1))
