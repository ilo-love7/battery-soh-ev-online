# -*- coding: utf-8 -*-
"""第八轮（§21）单元测试：收缩、折算残差与车队分档。

这一轮的结论全部是"某个数字该多大"型的（τ 有多大、λ 该取多少、档宽要多宽），
实现上任何一处口径错位都会让结论变成"看起来合理但其实没意义"。测试守四件事：
1. 方差分解真的把抽样噪声扣掉了（不给噪声会高估异质性）；
2. 收缩权重在边界上退化正确（τ²=0 时全部退回全局，se=0 时全部保留自己）；
3. 折算残差的度量对"车内加常数"不变（否则会把车级水平当成温度效应）；
4. 错档率对误差与档宽单调（这是"档宽 ≥ 3–4σ"这条建议的全部依据）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.evaluate import tier_error_rate
from evsoh.panel import shrink_slopes, tau2_moments, trend_resid_sd


def make_temps(n_cars: int = 60, n_sess: int = 40, beta: float = 0.05,
               trend: float = -2e-5, noise: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """合成面板：车级截距 + 里程线性老化 + 温度效应 + 噪声。"""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cars):
        a = rng.normal(41.0, 1.5)
        for k in range(n_sess):
            mile = k * 1500.0 + rng.normal(0, 150)
            t = 25.0 + 9.0 * np.sin(k / 5.0 + rng.uniform(0, 6.28)) + rng.normal(0, 1.5)
            y = a + trend * mile + beta * (t - 25.0) + rng.normal(0, noise)
            rows.append({"car": c, "mileage": mile, "t_mean": t, "capacity": y})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- τ² 矩估计
def test_tau2_recovers_known_heterogeneity():
    rng = np.random.default_rng(1)
    true_b = rng.normal(0.05, 0.03, 200)          # 真异质性 sd 0.03
    se = np.full(200, 0.01)                       # 抽样噪声 sd 0.01
    b_hat = true_b + rng.normal(0, 0.01, 200)
    est = tau2_moments(b_hat, se)
    assert est["tau"] == pytest.approx(0.03, abs=0.006)


def test_tau2_is_zero_when_all_spread_is_noise():
    rng = np.random.default_rng(2)
    se = np.full(400, 0.02)
    b_hat = 0.05 + rng.normal(0, 0.02, 400)       # 完全由抽样噪声产生
    est = tau2_moments(b_hat, se)
    # 400 个样本的方差本身有 ~7% 抖动，所以只能要求"几乎全部是噪声"，
    # 不能要求严格为 0；真正的截断由下面那个确定性用例守。
    assert est["tau2"] < 0.2 * est["noise_var"]
    assert est["share_noise"] > 0.8


def test_tau2_clipped_at_zero():
    b = np.array([0.05, 0.05, 0.05])               # 毫无离散
    est = tau2_moments(b, np.full(3, 0.02))
    assert est["tau2"] == 0.0
    assert est["tau"] == 0.0


def test_tau2_ignores_missing_se():
    b = np.array([0.04, 0.05, np.nan, 0.06, 0.03])
    se = np.array([0.01, 0.01, 0.01, 0.01, np.nan])
    est = tau2_moments(b, se)
    assert est["n"] == 3


# ---------------------------------------------------------------- 收缩
def test_shrink_weight_endpoints():
    """lam=0 全部退回全局，lam=1 全部保留自己 —— 这是 λ 曲线两端必须成立的锚点。"""
    b = np.array([0.01, 0.09, 0.05])
    se = np.array([0.01, 0.01, 0.01])
    np.testing.assert_allclose(shrink_slopes(b, se, 0.045, lam=0.0)["slope"], [0.045] * 3)
    np.testing.assert_allclose(shrink_slopes(b, se, 0.045, lam=1.0)["slope"], b)


def test_shrink_weights_follow_precision():
    """se 越大权重越小：估得糊的车自动退回全局值。"""
    b = np.array([0.09, 0.09])
    se = np.array([0.005, 0.05])
    out = shrink_slopes(b, se, 0.045, tau2=0.002)
    assert out["weight"][0] > out["weight"][1]
    assert out["slope"][0] > out["slope"][1]


def test_shrink_with_zero_tau2_is_pure_pooling():
    b = np.array([0.01, 0.09, 0.05])
    se = np.array([0.001, 0.001, 0.001])
    out = shrink_slopes(b, se, 0.045, tau2=0.0)
    np.testing.assert_allclose(out["slope"], [0.045] * 3)
    np.testing.assert_allclose(out["weight"], [0.0] * 3)


def test_shrink_reduces_error_when_estimates_are_noisy():
    """当单车估计噪声大于真异质性时，收缩后的 MSE 必须低于直接用估计值。"""
    rng = np.random.default_rng(3)
    true_b = rng.normal(0.05, 0.01, 400)
    se = np.full(400, 0.03)                        # 噪声 3 倍于异质性
    b_hat = true_b + rng.normal(0, 0.03, 400)
    tau2 = tau2_moments(b_hat, se)["tau2"]
    shrunk = shrink_slopes(b_hat, se, 0.05, tau2=tau2)["slope"]
    mse_raw = float(np.mean((b_hat - true_b) ** 2))
    mse_shrunk = float(np.mean((shrunk - true_b) ** 2))
    assert mse_shrunk < mse_raw


# ---------------------------------------------------------------- 折算残差
def test_resid_sd_invariant_to_within_car_level():
    """车内整体平移（换一把尺子）不该改变残差 —— 否则量到的是车级差异。"""
    d = make_temps(n_cars=8, n_sess=20)
    d["capacity"] += d.groupby("car")["capacity"].transform("mean") * 0.0
    shifted = d.copy()
    shifted["capacity"] += 5.0
    assert trend_resid_sd(d, "capacity") == pytest.approx(
        trend_resid_sd(shifted, "capacity"), abs=1e-9)


def test_resid_sd_invariant_to_reference_temperature():
    """折算量是 b·(T−ref)；换参考温度只加一个车内常数，残差必须不变。"""
    d = make_temps(n_cars=6, n_sess=20)
    b = 0.05
    a = trend_resid_sd(d, "capacity", shift=b * (d["t_mean"] - 25.0))
    c = trend_resid_sd(d, "capacity", shift=b * (d["t_mean"] - 0.0))
    assert a == pytest.approx(c, abs=1e-9)


def test_resid_sd_is_minimised_near_true_slope():
    d = make_temps(n_cars=30, n_sess=30, beta=0.05, noise=0.1)
    sds = {b: trend_resid_sd(d, "capacity", shift=b * (d["t_mean"] - 25.0))
           for b in (0.0, 0.05, 0.10)}
    assert sds[0.05] < sds[0.0]
    assert sds[0.05] < sds[0.10]


def test_wrong_sign_correction_makes_it_worse():
    d = make_temps(n_cars=30, n_sess=30, beta=0.05, noise=0.1)
    base = trend_resid_sd(d, "capacity")
    flipped = trend_resid_sd(d, "capacity", shift=-0.05 * (d["t_mean"] - 25.0))
    assert flipped > base


# ---------------------------------------------------------------- 分档
def test_tier_error_zero_without_uncertainty():
    v = np.array([81.0, 86.0, 91.0, 96.0])
    out = tier_error_rate(v, np.arange(80, 106, 5.0), sigma=0.0)
    assert out["mis_rate"] == 0.0
    assert out["boundary_share"] == 0.0


def test_tier_error_grows_with_sigma():
    rng = np.random.default_rng(4)
    v = rng.uniform(82, 98, 300)
    edges = np.arange(80, 106, 5.0)
    mis = [tier_error_rate(v, edges, s, n_draw=800)["mis_rate"] for s in (0.3, 1.0, 2.0)]
    assert mis[0] < mis[1] < mis[2]


def test_tier_error_grows_as_tiers_narrow():
    rng = np.random.default_rng(5)
    v = rng.uniform(82, 98, 300)
    wide = tier_error_rate(v, np.arange(80, 106, 5.0), 1.0, n_draw=800)["mis_rate"]
    narrow = tier_error_rate(v, np.arange(80, 106, 1.0), 1.0, n_draw=800)["mis_rate"]
    assert narrow > wide


def test_tier_boundary_share_is_monotone_in_sigma():
    v = np.linspace(81, 99, 40)
    edges = np.arange(80, 106, 5.0)
    a = tier_error_rate(v, edges, 0.5, n_draw=50)["boundary_share"]
    b = tier_error_rate(v, edges, 1.5, n_draw=50)["boundary_share"]
    assert b > a
