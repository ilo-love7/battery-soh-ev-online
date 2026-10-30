# -*- coding: utf-8 -*-
"""第四轮深化的单元测试：同车聚合窗口、方差分解、标签噪声、conformal 与模型工厂。

这一轮的分析结论全部建立在"口径"上，所以测试的重点不是数值精度，而是**口径不能被写错**：

  - 窗口聚合若把不同车的片段混进同一个窗，k 段的收益会被高估（等于偷偷用了同车信息）；
  - 方差分解若用总体方差而不是"减去抽样误差的车间方差"，ICC 会系统性偏高；
  - 标签噪声若把"重复标签"的对算进去，σ 会被 20% 的假零稀释，噪声被低估；
  - conformal 若在校准集上取分位后又在同一批数据上报覆盖率，覆盖率必然虚高。
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
def m20():
    return _load_script("_deep_20", "scripts/20_interval_floor.py")


@pytest.fixture(scope="module")
def m19():
    return _load_script("_deep_19", "scripts/19_model_frontier.py")


# ---------------------------------------------------------------- 窗口聚合
def test_build_windows_never_mixes_cars(m20):
    """窗内必须同车；k 段均值要等于该车连续 k 个片段的均值。"""
    df = pd.DataFrame({
        "car": ["A"] * 6 + ["B"] * 3,
        "mileage": [0, 1, 2, 3, 4, 5, 0, 1, 2],
        "pred_xgb": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 100.0, 102.0, 104.0],
        "capacity": [9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 99.0, 101.0, 103.0],
        "d_soc": [50.0] * 9,
    })
    w = m20.build_windows(df, k=3, max_win=100)
    # 车 A 有 6 段 → 4 个窗；车 B 有 3 段 → 1 个窗
    assert len(w) == 5
    assert w["car"].value_counts().to_dict() == {"A": 4, "B": 1}
    # 车 A 第一个窗 = 前三段均值，绝不能掺进车 B 的 100
    assert w.iloc[0]["pred"] == pytest.approx(12.0)
    assert w.iloc[0]["capacity"] == pytest.approx(11.0)


def test_build_windows_caps_per_car_and_drops_short_cars(m20):
    """每车最多 max_win 个窗；片段数不足 k 的车直接不进子集。"""
    rng = np.random.default_rng(0)
    n_a, n_b = 40, 2
    df = pd.DataFrame({
        "car": ["A"] * n_a + ["B"] * n_b,
        "mileage": list(range(n_a)) + list(range(n_b)),
        "pred_xgb": rng.normal(size=n_a + n_b),
        "capacity": rng.normal(size=n_a + n_b),
        "d_soc": [50.0] * (n_a + n_b),
    })
    w = m20.build_windows(df, k=5, max_win=7)
    assert w["car"].unique().tolist() == ["A"], "B 只有 2 段，k=5 时应被剔除"
    assert len(w) == 7


# ---------------------------------------------------------------- 方差分解
def test_variance_components_recovers_known_sigmas(m20):
    """已知 σ_b/σ_w 的合成数据上，分解结果应接近真值且 ICC ∈ (0,1)。

    注意比对的是**这次抽样实际实现的** std(b)，不是总体参数 —— 80 辆车的抽样
    std 本身就有 ~8% 的抖动，直接比总体参数会得到一个假失败。
    """
    rng = np.random.default_rng(42)
    n_cars, n_obs = 80, 120
    sig_w = 0.70
    b = rng.normal(scale=0.45, size=n_cars)
    r = np.repeat(b, n_obs) + rng.normal(scale=sig_w, size=n_cars * n_obs)
    cars = np.repeat([f"c{i}" for i in range(n_cars)], n_obs)

    vc = m20.variance_components(r, cars)
    assert vc["sigma_between"] == pytest.approx(float(b.std(ddof=1)), rel=0.10)
    assert vc["sigma_within"] == pytest.approx(sig_w, rel=0.05)
    assert 0.0 < vc["icc"] < 1.0
    # ICC 的定义必须与两个 σ 一致
    assert vc["icc"] == pytest.approx(
        vc["sigma_between"] ** 2 / (vc["sigma_between"] ** 2 + vc["sigma_within"] ** 2))


def test_variance_components_debiases_small_groups(m20):
    """每车只有几个片段时，车间方差的朴素估计会被抽样误差抬高，必须扣掉 σ_w²/n。

    这里 n=3、σ_w=1.0：朴素 var(车均值) ≈ σ_b² + 1/3，会把 σ_b 高估一倍多。
    """
    rng = np.random.default_rng(9)
    n_cars, n_obs = 300, 3
    b = rng.normal(scale=0.30, size=n_cars)
    r = np.repeat(b, n_obs) + rng.normal(scale=1.0, size=n_cars * n_obs)
    cars = np.repeat([f"c{i}" for i in range(n_cars)], n_obs)

    vc = m20.variance_components(r, cars)
    naive = float(np.std(pd.Series(r).groupby(cars).mean(), ddof=1))
    assert naive > 0.55, "朴素估计应被抽样误差显著抬高（否则这个测试没有意义）"
    assert vc["sigma_between"] == pytest.approx(0.30, abs=0.10)


def test_variance_components_prediction_shrinks_with_k(m20):
    """理论 RMSE_k 随 k 单调下降，并收敛到 σ_b（车级偏差平均不掉）。"""
    rng = np.random.default_rng(7)
    b = rng.normal(scale=0.4, size=50)
    r = np.repeat(b, 60) + rng.normal(scale=0.6, size=50 * 60)
    cars = np.repeat([f"c{i}" for i in range(50)], 60)
    vc = m20.variance_components(r, cars)
    vals = [vc["pred_rmse_k"][str(k)] for k in (1, 2, 5, 12)]
    assert all(x > y for x, y in zip(vals, vals[1:])), "k 增大应单调下降"
    assert vals[-1] > vc["sigma_between"] * 0.95, "极限应趋近 σ_b，而不是 0"


# ---------------------------------------------------------------- 标签噪声
def test_label_noise_ignores_duplicated_labels(m20):
    """重复标签的对必须单独统计：否则 20% 的假零会把 σ 稀释成偏小值。"""
    rng = np.random.default_rng(3)
    rows = []
    for car in range(6):
        mileage = np.arange(200) * 100.0
        base = 40.0 - 3.0e-6 * mileage + rng.normal(scale=0.3, size=200)
        # 每隔一个片段复制前一个标签 → 50% 的相邻对标签相同
        y = base.copy()
        y[1::2] = y[0::2]
        rows.append(pd.DataFrame({"car": f"c{car}", "mileage": mileage,
                                  "capacity": y, "pred_xgb": base}))
    df = pd.concat(rows, ignore_index=True)
    ln = m20.label_noise(df)
    assert ln["dup_frac"] == pytest.approx(0.5, abs=0.05)
    assert ln["sigma_noise"] > ln["sigma_noise_all"], \
        "只用不同标签的对，σ 必须大于把重复对也算进去的版本"
    assert ln["sigma_noise"] == pytest.approx(0.3, rel=0.35)
    assert ln["max_run"] >= 2


# ---------------------------------------------------------------- 跨度分箱
def test_span_key_covers_edges_and_drops_out_of_range(m20):
    """边界值落进上侧箱；低于 20 / 高于 100 的片段必须落空而不是被硬塞进首末箱。"""
    d = np.array([19.9, 20.0, 29.9, 30.0, 99.9, 100.0, 120.0])
    k = m20.span_key(d)
    assert k[0] == "" and k[1] == "20-30" and k[3] == "30-40"
    assert k[5] == "" and k[6] == "", "超出 [20,100) 的跨度不应被分箱"


# ---------------------------------------------------------------- conformal
def test_conformal_by_car_achieves_nominal_coverage(m20):
    """覆盖率要接近 90%，且宽度为正 —— 覆盖虚高通常意味着校准/评估用了同一批数据。"""
    rng = np.random.default_rng(11)
    cars, pred, y = [], [], []
    for c in range(60):
        for _ in range(20):
            p = 40.0 + rng.normal(scale=0.5)
            cars.append(f"c{c}")
            pred.append(p)
            y.append(p + rng.normal(scale=0.8))
    w = pd.DataFrame({"car": cars, "pred": pred, "capacity": y})
    out = m20.conformal_by_car(w)
    assert 0.84 <= out["coverage"] <= 0.94
    assert out["width"] > 0
    # 覆盖不足时下尾/上尾都应接近 5%
    assert out["below"] < 0.10 and out["above"] < 0.10


# ---------------------------------------------------------------- 模型工厂
def test_fit_config_returns_finite_predictions(m19):
    """19 脚本的配置工厂：线性模型走通、输出有限、形状对齐。"""
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(300, 6)), columns=[f"f{i}" for i in range(6)])
    X.iloc[:10, 0] = np.nan
    y = 40.0 + rng.normal(size=300)
    p, sec = m19.fit_config("elasticnet", X.iloc[:240], y[:240], X.iloc[240:],
                            {"models": {}})
    assert p.shape == (60,)
    assert np.isfinite(p).all()
    assert sec >= 0.0
