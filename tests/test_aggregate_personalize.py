# -*- coding: utf-8 -*-
"""第五轮深化的单元测试：时序切分、窗内聚合器、偏差估计、方差分解与公共 conformal。

这一轮把"聚合"和"个性化"两个杠杆放进同一张网格，最容易写错的地方恰好都在口径上：

  - 时序切分若把同一辆车的前后两半混在一起，bias 就变成"用未来校正过去"；
  - 窗内聚合器若把目标值也一起"稳健化"，就变成在猜标签而不是在估容量；
  - 偏差估计若不收缩，m=1 时一个噪声就能把整车的预测拉走；
  - 方差分解若用总体方差，ICC 会系统性偏高（与第四轮同一处坑）。
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from evsoh.evaluate import conformal_by_car  # noqa: E402
from evsoh.personalize import fit_bias  # noqa: E402


def _load_script(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m21():
    return _load_script("_deep_21", "scripts/21_aggregate_personalize.py")


def _toy(n_car: int = 4, n_sess: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_car):
        for i in range(n_sess):
            rows.append((f"car{c}", float(i * 100.0), 40.0 + c + rng.normal(0, 0.3),
                         30.0 + rng.normal(0, 5), 0.8, rng.normal(0, 0.5)))
    return pd.DataFrame(rows, columns=["car", "mileage", "capacity", "d_soc",
                                       "quality", "pred_xgb"])


# ---------------------------------------------------------------- 时序切分
def test_split_chrono_keeps_each_car_on_one_side_per_part(m21):
    """同一辆车只能整体落在 cal 或 eval 一侧，且 cal 的里程严格早于 eval。"""
    d = m21.split_chrono(_toy(), min_sess=10)
    assert set(d["part"].unique()) == {"cal", "eval"}
    for car, g in d.groupby("car"):
        cal = g.loc[g["part"] == "cal", "mileage"]
        ev = g.loc[g["part"] == "eval", "mileage"]
        assert len(cal) and len(ev)
        assert cal.max() < ev.min(), car


def test_split_chrono_drops_short_cars(m21):
    """历史太短的车进不了实验，否则 k=12 与 m=8 的格子会由不同车辆拼出来。"""
    d = _toy(n_car=3, n_sess=30)
    short = _toy(n_car=1, n_sess=5).assign(car="short")
    out = m21.split_chrono(pd.concat([d, short], ignore_index=True), min_sess=24)
    assert "short" not in set(out["car"])
    assert out["car"].nunique() == 3


# ---------------------------------------------------------------- 窗内聚合
def test_aggregate_mean_is_plain_average(m21):
    p = np.array([40.0, 41.0, 39.0, 40.5])
    for how in ("mean", "median", "trim", "recency", "quality", "span"):
        if how == "mean":
            assert m21._aggregate(p, how, np.ones(4), np.ones(4) * 30) == pytest.approx(p.mean())


def test_aggregate_median_rejects_single_spike(m21):
    """一个离群预测不该把窗内估计拖走——这是"稳健聚合"的直觉，后面被实测否掉。"""
    p = np.array([40.0, 40.2, 40.1, 39.9, 60.0])
    q, s = np.ones(5), np.ones(5) * 30
    assert m21._aggregate(p, "median", q, s) < 41.0
    assert m21._aggregate(p, "mean", q, s) > 44.0


def test_aggregate_trim_falls_back_to_mean_when_too_short(m21):
    """k<5 时 20% 截尾一个都截不掉，必须退回均值而不是返回空。"""
    p = np.array([40.0, 41.0, 42.0])
    out = m21._aggregate(p, "trim", np.ones(3), np.ones(3) * 30)
    assert out == pytest.approx(p.mean())


def test_aggregate_recency_weights_the_newest_session(m21):
    """新近加权的半衰期是窗长的一半，最后一个片段权重必须最大。"""
    p = np.array([40.0, 40.0, 50.0])
    out = m21._aggregate(p, "recency", np.ones(3), np.ones(3) * 30)
    assert out > p.mean()


def test_aggregate_weighted_falls_back_when_all_weights_zero(m21):
    p = np.array([40.0, 42.0])
    assert m21._aggregate(p, "quality", np.zeros(2), np.ones(2) * 30) == pytest.approx(p.mean())


def test_build_windows_target_stays_arithmetic_mean(m21):
    """不管用哪种聚合器，目标值都必须是窗内容量的算术平均（估计量只动预测侧）。"""
    df = _toy(n_car=2, n_sess=20)
    ref = m21.build_windows(df, 4, "mean")[["car", "capacity"]].reset_index(drop=True)
    for how in ("median", "trim", "recency", "quality", "span"):
        alt = m21.build_windows(df, 4, how)[["car", "capacity"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(ref, alt)


def test_build_windows_identical_across_aggregators(m21):
    """六种聚合器必须落在同一批窗上，否则比较不是成对的。"""
    df = _toy(n_car=3, n_sess=40)
    a = m21.build_windows(df, 5, "mean")[["car", "capacity", "d_soc"]]
    b = m21.build_windows(df, 5, "median")[["car", "capacity", "d_soc"]]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


# ---------------------------------------------------------------- 偏差估计
def test_car_bias_zero_when_no_labels(m21):
    """m=0 必须严格不校正，这是"零样本"那一列的参照点。"""
    cal = _toy(n_car=2, n_sess=10)
    assert all(v == 0.0 for v in m21.car_bias(cal, 0).values())


def test_car_bias_is_shrunk_toward_zero(m21):
    """收缩系数 λ=5：m 段标签只能买到 m/(m+λ) 的偏差，避免小样本过拟合。"""
    cal = pd.DataFrame({
        "car": ["A"] * 6,
        "mileage": np.arange(6, dtype=float),
        "capacity": np.full(6, 40.0),
        "pred_xgb": np.full(6, 42.0),          # 恒定 +2 Ah 的系统偏差
    })
    b1 = m21.car_bias(cal, 1)["A"]
    b6 = m21.car_bias(cal, 6)["A"]
    assert b1 == pytest.approx(-2.0 * 1 / (1 + m21.BIAS_LAM))
    assert b6 == pytest.approx(-2.0 * 6 / (6 + m21.BIAS_LAM))
    assert b1 > b6  # 校正量随 m 单调变强（负偏差，绝对值变大）


def test_car_bias_uses_most_recent_sessions(m21):
    """只取校准池里最近 m 段——部署时拿到的就是最近的标签。

    ``fit_bias`` 返回的是 mean(y − pred) 的收缩值：后两段 pred=50、y=40，所以校正量
    应当是负的。若错取了最早两段（pred=30），符号会反过来，这条断言就能抓住。
    """
    cal = pd.DataFrame({
        "car": ["A"] * 4,
        "mileage": np.arange(4, dtype=float),
        "capacity": np.array([40.0, 40.0, 40.0, 40.0]),
        "pred_xgb": np.array([30.0, 30.0, 50.0, 50.0]),
    })
    b2 = m21.car_bias(cal, 2)["A"]
    assert b2 == pytest.approx(-10.0 * 2 / (2 + m21.BIAS_LAM))
    assert b2 < 0


# ---------------------------------------------------------------- 方差分解
def test_decompose_recovers_known_sigmas(m21):
    """r_ij = b_i + e_ij 的分解必须把两半还原回来（与第四轮同一套去偏逻辑）。"""
    rng = np.random.default_rng(3)
    n_car, n_sess, sb, sw = 60, 40, 0.5, 0.7
    b = rng.normal(0, sb, n_car)
    e = rng.normal(0, sw, (n_car, n_sess))
    w = pd.DataFrame({
        "car": np.repeat(np.arange(n_car), n_sess),
        "pred": (b[:, None] + e).ravel(),
        "capacity": np.zeros(n_car * n_sess),
    })
    out = m21.decompose(w)
    assert out["sigma_between"] == pytest.approx(sb, rel=0.12)
    assert out["sigma_within"] == pytest.approx(sw, rel=0.06)


# ---------------------------------------------------------------- 公共 conformal
def test_conformal_by_car_achieves_nominal_coverage_shared():
    """evaluate 里的公共版本必须与脚本 20 的本地版本同配方：覆盖率落在名义值附近。"""
    rng = np.random.default_rng(11)
    cars = np.repeat(np.arange(80), 25)
    y = rng.normal(40.0, 2.0, cars.size)
    pred = y + rng.normal(0.0, 0.5, cars.size)
    out = conformal_by_car(cars, y, pred, nominal=0.90)
    assert 0.86 <= out["coverage"] <= 0.94
    assert out["width"] > 0


def test_fit_bias_shrinkage_matches_manual_formula():
    """fit_bias 的收缩形式在 m 段上就是 m/(m+λ)——网格里所有 m 列都依赖它。"""
    pred = np.full(8, 42.0)
    y = np.full(8, 40.0)
    for m in (1, 3, 8):
        b = fit_bias(pred[:m], y[:m], lam=5.0)
        assert b == pytest.approx(-2.0 * m / (m + 5.0))
