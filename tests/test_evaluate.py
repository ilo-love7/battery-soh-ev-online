# -*- coding: utf-8 -*-
"""评测协议、SOH 标签、个性化校正的单元测试。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.evaluate import (group_kfold_by_car, interval_metrics,  # noqa: E402
                           leave_one_dataset_out, regression_metrics,
                           random_split_within_car)
from evsoh.personalize import apply_affine, coldstart_curve, fit_affine, fit_bias  # noqa: E402
from evsoh.target import add_soh_labels, bol_reference  # noqa: E402


def test_regression_metrics_exact():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)

    m2 = regression_metrics(y, y + 1.0)
    assert m2["bias"] == pytest.approx(1.0)
    assert m2["rmse"] == pytest.approx(1.0)


def test_group_kfold_no_car_leakage():
    cars = np.repeat(np.arange(20), 5)
    X = np.zeros((cars.size, 1))
    seen = 0
    for tr, te in group_kfold_by_car(cars, n_splits=4, seed=0):
        assert set(cars[tr]).isdisjoint(set(cars[te]))
        assert len(tr) + len(te) == cars.size
        seen += len(te)
    assert seen == cars.size


def test_random_split_within_car():
    cars = np.repeat(np.arange(10), 10)
    tr, te = random_split_within_car(cars, test_frac=0.2, seed=1)
    assert len(set(tr) & set(te)) == 0
    assert len(tr) + len(te) == cars.size


def test_leave_one_dataset_out():
    ds = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
    outs = list(leave_one_dataset_out(ds))
    assert len(outs) == 3
    for name, tr, te in outs:
        assert set(ds[te]) == {name}
        assert name not in set(ds[tr])


def test_bol_reference_robust():
    cap = np.array([30, 35, 39, 40, 40.5, 41, 41, 41.2, 41.5, 42])
    ref = bol_reference(cap, top_frac=0.2)
    assert 41.0 <= ref <= 42.0
    assert bol_reference(np.array([0.0, 0.0])) != bol_reference(np.array([0.0, 0.0]))  # nan


def test_add_soh_labels():
    cfg = {"label": {"min_labeled_sessions": 3, "soh_clip": [0.6, 1.15]}}
    rows = []
    for car, caps in [(1, [40, 41, 42, 38, 30]), (2, [0, 0, 41, 40, 39])]:
        for c in caps:
            rows.append({"car": car, "capacity": float(c), "mileage": 1000.0})
    df = pd.DataFrame(rows)
    out, stats = add_soh_labels(df, cfg)
    assert set(out["car"].unique()) == {1, 2}
    assert out["soh"].max() <= 1.0 + 1e-9
    assert stats["cars_kept"] == 2


def test_fit_affine_recovers_mapping():
    rng = np.random.default_rng(0)
    p = rng.normal(40, 3, 200)
    y = 1.1 * p - 2.0
    a, b = fit_affine(p, y, lam=1e-6)
    assert a == pytest.approx(1.1, abs=1e-3)
    assert b == pytest.approx(-2.0, abs=1e-2)


def test_affine_shrinks_to_identity():
    p = np.array([38.0, 40.0, 42.0, 44.0])
    y = 1.2 * p - 5.0  # 真实映射偏离恒等（a=1.2, b=-5）
    a_hard, b_hard = fit_affine(p, y, lam=1e-9)
    a_soft, b_soft = fit_affine(p, y, lam=1e4)
    assert abs(a_hard - 1.2) < 1e-3
    assert abs(a_soft - 1.0) < abs(a_hard - 1.0)
    assert abs(b_soft) < abs(b_hard)


def test_fit_bias_and_coldstart_curve():
    rng = np.random.default_rng(1)
    cars = np.repeat(np.arange(6), 20)
    y = rng.normal(40, 2, cars.size)
    pred = y + 2.0 + rng.normal(0, 0.5, cars.size)  # 系统性 +2 Ah 偏差
    b = fit_bias(pred[:20], y[:20], lam=0.0)
    assert b == pytest.approx(-2.0, abs=0.3)

    curve = coldstart_curve(pred, y, cars, k_list=[0, 1, 5], n_repeats=3)
    assert [c["k"] for c in curve] == [0, 1, 5]
    assert curve[-1]["car_mae_mean"] < curve[0]["car_mae_mean"]


def test_interval_metrics():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    lo = np.array([0.0, 1.0, 2.0, 0.0])
    hi = np.array([2.0, 3.0, 4.0, 1.0])
    m = interval_metrics(y, lo, hi)
    assert m["coverage"] == pytest.approx(0.75)
    assert m["above_rate"] == pytest.approx(0.25)
