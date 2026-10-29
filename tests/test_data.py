# -*- coding: utf-8 -*-
"""数据读取与窗口统计的单元测试（合成数据，不依赖真实数据集）。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.data import window_stats  # noqa: E402


def make_window(n=128, soc0=20.0, soc_rate=0.1, current=-10.0, v0=3.7, v_rate=0.001):
    feat = np.zeros((n, 8), dtype=np.float64)
    t = np.arange(n) * 10.0
    feat[:, 0] = v0 + v_rate * t          # 平均电压
    feat[:, 1] = current                  # 电流（负=充电）
    feat[:, 2] = soc0 + soc_rate * np.arange(n)  # SOC
    feat[:, 3] = feat[:, 0] + 0.02        # 最高单体
    feat[:, 4] = feat[:, 0] - 0.02        # 最低单体
    feat[:, 5] = 25.0 + 0.001 * t         # 最高温度
    feat[:, 6] = 24.0 + 0.001 * t         # 最低温度
    feat[:, 7] = t                        # 时间戳
    return feat


def meta(**kw):
    base = {"car": 7, "charge_segment": "3", "label": "00", "mileage": 12345.0, "capacity": 41.0}
    base.update(kw)
    return base


QC = {"n_steps": 128, "min_current_a": 0.5, "max_soc_jump": 30.0}


def test_window_stats_basic():
    feat = make_window()
    s = window_stats(feat, meta(), "battery_dataset1", "1.pkl", QC)
    assert s["car"] == 7 and s["segment"] == "3"
    assert s["valid"] is True
    assert s["soc_start"] == pytest.approx(20.0)
    assert s["soc_end"] == pytest.approx(20.0 + 0.1 * 127)
    assert s["i_abs_mean"] == pytest.approx(10.0)
    assert s["v_start"] == pytest.approx(3.7)
    assert s["q_ah"] > 0
    # 10 A × 1270 s / 3600 ≈ 3.53 Ah
    assert s["q_ah"] == pytest.approx(10.0 * 1270.0 / 3600.0, rel=1e-3)
    assert s["cell_gap_mean"] == pytest.approx(0.04)
    assert 0.0 <= s["soc_mono"] <= 1.0


def test_window_stats_rejects_idle_and_jumpy():
    idle = make_window(current=0.0)
    assert window_stats(idle, meta(), "d", "1.pkl", QC)["valid"] is False

    jumpy = make_window()
    jumpy[64:, 2] = 90.0  # SOC 突跳 70%
    assert window_stats(jumpy, meta(), "d", "2.pkl", QC)["valid"] is False


def test_window_stats_handles_nan():
    feat = make_window()
    feat[10:20, 0] = np.nan
    s = window_stats(feat, meta(), "d", "3.pkl", QC)
    assert np.isfinite(s["v_mean"])
    assert s["n_steps"] == 128
