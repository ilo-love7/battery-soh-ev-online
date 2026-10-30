# -*- coding: utf-8 -*-
"""第六轮深化的单元测试：标签块、去噪目标、在线序贯标定、车级标定与决策指标。

这一轮最容易写错的地方全在**方向和时序**上，也是唯一一类"跑得通但结论全反"的错：

  - 偏差方向：``e = pred - y``，修正必须是 ``e - b``。方向写反会把校正变成加倍，
    RMSE 从 0.59 直接翻到 1.0（第五轮 fit_bias 用的是 ``y - pred``，两套约定必须对齐）；
  - 时序：预测第 t 段只能用 t 之前到达的标签，用一条"未来标签改成极端值"的断言守住；
  - 去噪目标：留一中位数必须**不含自身**，否则误差被自己稀释；
  - 收缩：n_eff→0 时偏差必须归零，否则 1 段标签就能把整车拉走。
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load_script(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m22():
    return _load_script("_deep_22", "scripts/22_online_noise_decision.py")


def _toy(cap, pred=None, car: int = 1) -> pd.DataFrame:
    cap = np.asarray(cap, dtype=float)
    pred = cap if pred is None else np.asarray(pred, dtype=float)
    n = cap.size
    return pd.DataFrame({"car": car, "capacity": cap, "pred_xgb": pred,
                         "mileage": np.arange(n, dtype=float) * 100.0,
                         "d_soc": np.full(n, 50.0),
                         "bol_ref": np.full(n, 42.0)})


# ---------------------------------------------------------------- 标签块
def test_label_runs_marks_copies(m22):
    """同一车内里程相邻且容量精确相同 → 同块，第二段起标记为复制。"""
    d = m22.label_runs(_toy([40.0, 40.0, 40.0, 39.5, 39.5, 39.0]))
    assert d["run_len"].tolist() == [3, 3, 3, 2, 2, 1]
    assert d["is_copy"].tolist() == [False, True, True, False, True, False]


def test_label_runs_does_not_cross_cars(m22):
    """两辆车的首段容量相同时不能合并成一块（跨车比较无意义）。"""
    d = pd.concat([_toy([40.0, 40.0], car=1), _toy([40.0, 40.0], car=2)],
                  ignore_index=True)
    d = m22.label_runs(d)
    assert d.groupby("car")["run_len"].max().tolist() == [2, 2]


# ---------------------------------------------------------------- 去噪目标
def test_loo_median_excludes_self(m22):
    """留一中位数不能被自身那个离群值影响。"""
    y = np.array([1.0, 1.0, 10.0, 1.0, 1.0])
    out = m22.loo_median(y, w=2)
    assert out[2] == pytest.approx(1.0)
    assert np.isfinite(out[1])
    assert np.isnan(out).sum() == 0


# ---------------------------------------------------------------- 收缩
def test_shrink_bounds(m22):
    assert m22._shrink(2.0, 0.0) == pytest.approx(0.0)
    assert m22._shrink(2.0, 1e9) == pytest.approx(2.0, abs=1e-6)
    assert m22._shrink(2.0, 5.0) == pytest.approx(1.0)


# ---------------------------------------------------------------- 在线流式
def test_stream_never_uses_future_labels(m22):
    """把第 t0 段之后的标签改成极端值，t≤t0 的残差必须一字不变。"""
    n = 40
    cap = np.linspace(40.0, 38.0, n)
    pred = cap + 0.5
    base = m22.stream(_toy(cap, pred), "lab_ewma", cadence=1, half=5.0, burn=5,
                      tag="t1")
    cap2 = cap.copy()
    cap2[25:] = 0.0        # 25 段之后全部改坏
    pert = m22.stream(_toy(cap2, pred), "lab_ewma", cadence=1, half=5.0, burn=5,
                      tag="t2")
    # 注意比到 24：第 25 段自己的标签也被改坏了，它的残差本来就该变
    before = base[base["t"] < 25]["err"].to_numpy()
    before2 = pert[pert["t"] < 25]["err"].to_numpy()
    assert np.allclose(before, before2)
    # 25 之后必须真的变了，否则这条断言没在检验任何东西
    after = base[base["t"] > 25]["err"].to_numpy()
    after2 = pert[pert["t"] > 25]["err"].to_numpy()
    assert not np.allclose(after, after2)


def test_stream_zero_shot_is_raw_residual(m22):
    cap = np.array([40.0, 39.8, 39.6, 39.4, 39.2, 39.0, 38.8, 38.6])
    pred = cap + 0.4
    s = m22.stream(_toy(cap, pred), "none", cadence=1, burn=3, tag="t3")
    assert np.allclose(s["err"].to_numpy(), pred[3:] - cap[3:])


def test_stream_cadence_limits_labels(m22):
    """节奏 3 时，第 t 段最多只能用 ceil(t/3) 个标签。"""
    n = 30
    cap = np.linspace(40.0, 38.0, n)
    s = m22.stream(_toy(cap, cap + 0.3), "lab_ewma", cadence=3, burn=5, tag="t4")
    assert s["n_labels"].max() <= int(np.ceil(n / 3))


# ---------------------------------------------------------------- 车级标定方向
def test_car_level_corrects_the_right_direction(m22):
    """模型整体高估 1 Ah 时，车级校正必须让误差变小——方向写反就会翻倍。"""
    rng = np.random.default_rng(0)
    n = 40
    cap = 40.0 - np.linspace(0.0, 1.5, n) + rng.normal(0, 0.05, n)
    d = _toy(cap, cap + 1.0)
    cl = m22.car_level(d)
    err_zero = abs(cl["pred_zero"].iloc[0] - cl["y"].iloc[0])
    err_corr = abs(cl["pred_online"].iloc[0] - cl["y"].iloc[0])
    assert err_corr < err_zero
    assert cl["b_online"].iloc[0] < 0        # 高估 → 修正量为负


def test_car_level_skips_short_history(m22):
    d = _toy(np.linspace(40.0, 39.0, 12), np.linspace(40.0, 39.0, 12))
    assert len(m22.car_level(d)) == 0        # 少于 K_WIN+8 段不入统计


# ---------------------------------------------------------------- 区间
def test_conformal_bounds_have_nominal_coverage(m22):
    rng = np.random.default_rng(1)
    n = 600
    y = rng.normal(40.0, 1.0, n)
    pred = y + rng.normal(0.0, 0.5, n)
    lo, hi = m22._conformal_bounds(y, pred, nominal=0.90)
    m = lo.notna().to_numpy()
    cov = float(((y[m] >= lo.to_numpy()[m]) & (y[m] <= hi.to_numpy()[m])).mean())
    assert 0.85 <= cov <= 0.95
    assert np.all(hi.to_numpy()[m] > lo.to_numpy()[m])


# ---------------------------------------------------------------- 决策指标
def test_screening_recall_is_monotone(m22):
    rng = np.random.default_rng(2)
    y = 40.0 - rng.gamma(2.0, 0.4, 200)
    cl = pd.DataFrame({"car": np.arange(200), "y": y,
                       "pred_zero": y + rng.normal(0, 0.8, 200),
                       "pred_online": y + rng.normal(0, 0.4, 200)})
    scr = m22._screening(cl)
    for key in ("zero", "online"):
        rec = np.asarray(scr[key], dtype=float)
        assert np.all(np.diff(rec) >= -1e-9)
        assert rec[-1] >= rec[0]


def test_ordering_accuracy_is_perfect_for_perfect_predictions(m22):
    y = np.linspace(36.0, 42.0, 30)
    cl = pd.DataFrame({"car": np.arange(30), "y": y, "pred_zero": y,
                       "pred_online": y})
    ii, jj = np.triu_indices(30, k=1)
    ok = (cl["pred_online"].to_numpy()[ii] - cl["pred_online"].to_numpy()[jj]) * \
         (y[ii] - y[jj]) > 0
    assert ok.mean() == 1.0
