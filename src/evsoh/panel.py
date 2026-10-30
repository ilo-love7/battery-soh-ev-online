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
