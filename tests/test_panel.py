# -*- coding: utf-8 -*-
"""第七轮（§20）单元测试：面板回归、温度折算与参考温度重训的可辨识性。

这些测试守的是"结论会不会被实现细节偷走"：固定效应是否真的扣掉了车级差异、
一阶差分是否真的消掉了线性老化、折算在参考温度处是否恒等、聚类标准误是否
比朴素标准误更保守。温度斜率是本轮全部结论的支点，估错了整节都站不住。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.panel import (debias, fe_ols, first_difference, hinge, piecewise_fit,
                         temperature_bins, within_demean)


def make_panel(n_cars: int = 40, n_sess: int = 30, beta: float = 0.05,
               trend: float = -3e-5, noise: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """合成面板：车级截距 + 里程线性老化 + 温度效应 + 噪声。"""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cars):
        a = rng.normal(41.0, 2.0)          # 车级 BOL 差异
        for k in range(n_sess):
            mile = k * 1200.0 + rng.normal(0, 200)
            t = 25.0 + 10.0 * np.sin(k / 6.0 + rng.uniform(0, 6.28)) + rng.normal(0, 2)
            y = a + trend * mile + beta * (t - 25.0) + rng.normal(0, noise)
            rows.append({"car": c, "mileage": mile, "t_mean": t, "capacity": y})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 固定效应
def test_fe_recovers_slope_with_car_effects():
    d = make_panel(beta=0.05)
    fit = fe_ols(d, "capacity", ["t_mean"])
    assert fit["beta"][0] == pytest.approx(0.05, abs=0.004)


def test_fe_with_km_separates_trend_from_temperature():
    # 让老化与温度强相关：不给里程项就会被老化污染
    d = make_panel(beta=0.05, trend=-3e-5)
    d["t_mean"] = d["t_mean"] + d["mileage"] * 3e-5 * 1e4   # 里程越大越热
    naive = fe_ols(d, "capacity", ["t_mean"])["beta"][0]
    both = fe_ols(d, "capacity", ["t_mean", "mileage"])["beta"][0]
    assert abs(both - 0.05) < abs(naive - 0.05)


def test_fe_demean_kills_between_car_variance():
    d = make_panel(n_cars=20, n_sess=10, beta=0.0, trend=0.0, noise=0.0)
    dm = within_demean(d, ["capacity"], group="car")
    assert np.allclose(dm.groupby("car")["capacity"].mean().to_numpy(), 0.0, atol=1e-9)


def test_cluster_se_wider_than_iid_when_within_car_correlated():
    d = make_panel(n_cars=20, n_sess=40, noise=0.8)
    cl = fe_ols(d, "capacity", ["t_mean"], cluster=True)
    iid = fe_ols(d, "capacity", ["t_mean"], cluster=False)
    assert cl["se"][0] > iid["se"][0]


# ---------------------------------------------------------------- 一阶差分
def test_fd_is_invariant_to_linear_trend():
    """加一条任意陡的线性老化，差分斜率必须纹丝不动。

    里程必须是确定步长（等间距会话）才能"纹丝不动"：若里程带随机抖动，
    ``Δ(mileage)`` 本身有噪声，加进去的老化项会漏一点进斜率——那不是 bug，是
    里程测量误差的必然结果，所以这里刻意用严格等间距。
    """
    d = make_panel(beta=0.05, trend=0.0, noise=0.05)
    d["mileage"] = d.groupby("car").cumcount().to_numpy() * 1200.0
    diff_a = first_difference(d, ["capacity", "t_mean"])
    slope_a = np.polyfit(diff_a["t_mean"], diff_a["capacity"], 1)[0]

    d2 = d.copy()
    d2["capacity"] = d2["capacity"] + 1e-3 * d2["mileage"]   # 陡得离谱的老化
    diff_b = first_difference(d2, ["capacity", "t_mean"])
    slope_b = np.polyfit(diff_b["t_mean"], diff_b["capacity"], 1)[0]
    assert slope_a == pytest.approx(slope_b, abs=1e-9)


def test_fd_drops_first_session_per_car():
    d = make_panel(n_cars=5, n_sess=6)
    diff = first_difference(d, ["capacity", "t_mean"])
    assert len(diff) == 5 * 5
    assert diff["car"].nunique() == 5


def test_fd_recovers_slope():
    d = make_panel(beta=0.06, trend=-2e-5, noise=0.08, n_cars=60, n_sess=30)
    diff = first_difference(d, ["capacity", "t_mean"])
    assert np.polyfit(diff["t_mean"], diff["capacity"], 1)[0] == pytest.approx(0.06, abs=0.006)


# ---------------------------------------------------------------- 温度折算
def test_hinge_is_continuous_and_flat_below_knot():
    x = np.array([0.0, 10.0, 33.0, 40.0])
    h = hinge(x, 33.0)
    assert np.allclose(h, [0.0, 0.0, 0.0, 7.0])


def test_piecewise_slopes_below_and_above():
    d = make_panel(beta=0.05, noise=0.05)
    # 35°C 以上把斜率折半
    d["capacity"] = d["capacity"] - 0.025 * hinge(d["t_mean"].to_numpy(), 30.0)
    pw = piecewise_fit(d, "capacity", "t_mean", 30.0, ref=25.0)
    assert pw["slope_below_knot"] == pytest.approx(0.05, abs=0.004)
    assert pw["slope_above"] == pytest.approx(0.025, abs=0.006)


def test_piecewise_needs_mileage_control_to_avoid_over_correction():
    """温度与里程正相关时，不控制里程会把老化折进温度斜率（实测 +53%）。"""
    d = make_panel(beta=0.05, trend=-4e-5, noise=0.05)
    # 让温度随里程一起爬：车越旧越"热"（季节与使用年限混在一起）
    d["t_mean"] = d["t_mean"] + d["mileage"] * 4e-6
    no_ctl = piecewise_fit(d, "capacity", "t_mean", 33.0, ref=25.0)
    ctl = piecewise_fit(d, "capacity", "t_mean", 33.0, ref=25.0, controls=["mileage"])
    assert abs(ctl["slope_below_knot"] - 0.05) < abs(no_ctl["slope_below_knot"] - 0.05)


def test_debias_is_identity_at_reference():
    d = make_panel(beta=0.05, noise=0.0)
    pw = piecewise_fit(d, "capacity", "t_mean", 33.0, ref=25.0)
    y = d["capacity"].to_numpy(float)
    t = np.full_like(y, 25.0)
    assert np.allclose(debias(y, t, pw), y)


def test_debias_removes_known_temperature_term():
    """注入 +0.05 Ah/°C 后折算到 25°C，残差不应再有温度趋势。"""
    d = make_panel(beta=0.05, trend=0.0, noise=0.02, n_cars=50, n_sess=40)
    pw = piecewise_fit(d, "capacity", "t_mean", 33.0, ref=25.0)
    d["y_ref"] = debias(d["capacity"].to_numpy(float), d["t_mean"].to_numpy(float), pw)
    slope = fe_ols(d, "y_ref", ["t_mean"])["beta"][0]
    assert abs(slope) < 0.005


def test_temperature_bins_are_monotone():
    t = np.linspace(0, 40, 100)
    lab, edges = temperature_bins(t, n=4)
    assert len(edges) == 3
    assert np.all(np.diff(edges) > 0)
    assert lab.min() == 0 and lab.max() == 3
    assert np.all(np.diff([t[lab == k].mean() for k in range(4)]) > 0)


# ---------------------------------------------------------------- 报告口径
def test_k_window_bias_shrinks_when_temperature_random():
    """温度完全随机时，连续 k 段均温的偏差应随 k 缩小（对照：同季节则不会）。"""
    rng = np.random.default_rng(0)
    def dev_sd(k: int) -> float:
        vals = []
        for _ in range(400):
            T = rng.normal(25, 7, 60)
            vals.append(T[-k:].mean() - T.mean())
        return float(np.std(vals))
    assert dev_sd(8) < dev_sd(2)


def test_k_window_bias_persists_when_temperature_is_seasonal():
    """温度按季节成块时，k 段平均几乎压不掉偏差——这是 §20.5 的核心机制。"""
    def dev_sd(k: int) -> float:
        vals = []
        rng = np.random.default_rng(1)
        for _ in range(300):
            season = np.repeat(rng.normal(25, 8, 5), 12)   # 每「季」12 段
            vals.append(season[-k:].mean() - season.mean())
        return float(np.std(vals))
    assert dev_sd(8) > 0.4 * dev_sd(2)     # 远不如独立情形下降得快
