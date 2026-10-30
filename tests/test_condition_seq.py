# -*- coding: utf-8 -*-
"""第三轮深化的单元测试：神经网络预处理与训练循环、序列重采样、工况分箱。

这三个函数都有"看起来对但会静默出错"的失败模式：
  - prep_tabular 若用测试集的统计量，就是泄漏（结果会虚高而看不出来）；
  - 不做 winsorize，MLP 会在极端离群值上梯度爆炸（曾出现 RMSE ~1e11）；
  - 序列按 SOC 重采样时若 SOC 不单调，np.interp 会静默给出错误的曲线。
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from evsoh.nn import fit_torch, make_mlp, prep_tabular  # noqa: E402


def _load_script(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m17():
    return _load_script("_deep_17", "scripts/17_sequence_model.py")


@pytest.fixture(scope="module")
def m16():
    return _load_script("_deep_16", "scripts/16_condition_norm.py")


# ---------------------------------------------------------------- prep_tabular
def test_prep_tabular_imputes_and_winsorizes():
    """NaN 被中位数填补；极端离群值被 0.5/99.5 分位裁剪，不进入标准化。"""
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(500, 4))
    Xtr[10, 1] = np.nan
    Xtr[20, 2] = 1e6            # 极端离群
    A, _ = prep_tabular(Xtr, Xtr)

    assert np.isfinite(A).all(), "插补后不应残留 NaN/inf"
    assert A[:, 2].max() < 20.0, "离群值应被 winsorize 掉，而不是标准化后仍巨大"


def test_prep_tabular_uses_train_statistics_only():
    """测试集必须用训练集的均值/标准差变换 —— 否则是泄漏。

    注意测试集要落在训练集的分位范围内，否则会被 winsorize 截平，
    就看不出"统计量来自哪里"了（那是另一个正确行为，见上一个测试）。
    """
    rng = np.random.default_rng(1)
    Xtr = rng.normal(loc=0.0, scale=1.0, size=(400, 3))
    # 测试集必须整体落在训练集的 0.5–99.5 分位内（约 ±2.58），否则会被 winsorize 截平
    Xte_a = rng.normal(loc=1.0, scale=0.30, size=(50, 3))
    Xte_b = Xte_a + 0.30

    _, B_a = prep_tabular(Xtr, Xte_a)
    _, B_b = prep_tabular(Xtr, Xte_b)
    # 同一变换下，平移 0.3 应让输出恰好差 0.3/sd
    diff = (B_b - B_a).mean()
    assert diff == pytest.approx(0.30 / Xtr.std(axis=0).mean(), rel=0.05)
    # 测试集均值是 1.0，若误用测试集统计量会被归零
    assert abs(B_a.mean()) > 0.5, "测试集被中心化了，说明用了测试集统计量"


# ---------------------------------------------------------------- fit_torch
def test_fit_torch_is_reproducible_and_learns():
    """同种子两次结果完全一致；且能学到 y = 3*x0 - 2*x1 这样的线性关系。"""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 5))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.1 * rng.normal(size=600)

    A, B = prep_tabular(X[:400], X[400:])
    p1 = fit_torch(lambda: make_mlp(5, width=32), A, y[:400], B, seed=7, epochs=60)
    p2 = fit_torch(lambda: make_mlp(5, width=32), A, y[:400], B, seed=7, epochs=60)
    np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-6)

    r2 = 1.0 - np.sum((y[400:] - p1) ** 2) / np.sum((y[400:] - y[400:].mean()) ** 2)
    assert r2 > 0.85, f"简单线性关系应能学到，实测 R²={r2:.3f}"


def test_fit_torch_returns_original_scale():
    """预测必须回到原始尺度（容量 ~30 Ah），不是标准化尺度。"""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 4))
    y = 30.0 + 2.0 * X[:, 0]
    A, B = prep_tabular(X[:300], X[300:])
    p = fit_torch(lambda: make_mlp(4, width=32), A, y[:300], B, seed=1, epochs=60)
    assert 20.0 < np.median(p) < 40.0, "预测尺度不对，可能忘了反标准化"


# ---------------------------------------------------------------- 序列重采样
def test_trajectory_to_grid_shape_and_mask(m17):
    """输出形状固定 (5, 256)；掩码只在 SOC 覆盖区间内为 1。"""
    traj = np.zeros((128, 8))
    traj[:, 0] = np.linspace(3.6, 4.1, 128)          # V
    traj[:, 1] = 50.0                                 # I
    traj[:, 2] = np.linspace(40.0, 70.0, 128)         # SOC：只覆盖 40–70%
    traj[:, 7] = np.arange(128) * 10.0                # T

    g = m17.trajectory_to_grid(traj)
    assert g is not None
    assert g.shape == (5, m17.N_GRID)

    mask = g[4]
    grid = m17.SOC_GRID
    inside = (grid >= 40.0) & (grid <= 70.0)
    assert mask[inside].min() == 1.0, "覆盖区间内掩码应为 1"
    assert mask[~inside].max() == 0.0, "覆盖区间外掩码应为 0"


def test_trajectory_to_grid_rejects_short_coverage(m17):
    """SOC 只跨 1% 的轨迹应被拒绝（返回 None），而不是硬塞进网格。"""
    traj = np.zeros((128, 8))
    traj[:, 0] = 3.9
    traj[:, 1] = 50.0
    traj[:, 2] = np.linspace(50.0, 51.0, 128)
    assert m17.trajectory_to_grid(traj) is None


def test_resample_handles_non_monotonic_soc(m17):
    """SOC 轻微回落（真实数据里约 0.1%）不能让插值崩掉或给出乱序曲线。"""
    base = np.linspace(20.0, 40.0, 60)
    # 每 6 个点制造一次 1.5% 的回落，模拟 SOC 报数抖动
    soc = np.concatenate([base[i:i + 6] if i % 12 else base[i:i + 6][::-1]
                          for i in range(0, 60, 6)])
    x = np.linspace(3.5, 4.0, soc.size)
    out = m17._resample_on_soc(x, soc, np.array([22.0, 30.0, 38.0]))
    assert np.isfinite(out).all()
    assert np.all(np.diff(out) > -1e-9), "按 SOC 重采样后电压应单调不减"


# ---------------------------------------------------------------- 工况分箱
def test_temp_bin_covers_edges(m16):
    """温度分箱的边界值应落进相邻箱，而不是变成 NaN。"""
    t = np.array([-5.0, 9.99, 10.0, 17.99, 18.0, 24.0, 30.0, 36.0, 50.0])
    b = m16.temp_bin(t)
    assert "" not in set(b), "不应有样本落在所有箱之外"
    assert b[0] == "<10" and b[-1] == ">36"
