# -*- coding: utf-8 -*-
"""会话级特征工程的单元测试：拼接、IC 特征、质量分、特征名一致性。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.session import (SOC_BINS, V_GRID, V_WINDOWS, build_session,  # noqa: E402
                           concat_windows, session_feature_names)


def make_window(start_soc, end_soc, v0, v1, current=-10.0, n=128, t_offset=0.0):
    feat = np.zeros((n, 8), dtype=np.float64)
    t = np.arange(n) * 10.0
    feat[:, 0] = np.linspace(v0, v1, n)
    feat[:, 1] = current
    feat[:, 2] = np.linspace(start_soc, end_soc, n)
    feat[:, 3] = feat[:, 0] + 0.01
    feat[:, 4] = feat[:, 0] - 0.01
    feat[:, 5] = np.linspace(25, 27, n)
    feat[:, 6] = np.linspace(24, 26, n)
    feat[:, 7] = t + t_offset
    return feat


def make_session(n_win=5, soc0=20.0, soc_step=10.0, v0=3.45, v1=4.05, current=-10.0):
    wins = []
    for k in range(n_win):
        s0 = soc0 + k * soc_step
        a = v0 + (v1 - v0) * k / n_win
        b = v0 + (v1 - v0) * (k + 1) / n_win
        wins.append((f"{k}.pkl", make_window(s0, s0 + soc_step, a, b, current),
                     {"car": 11, "charge_segment": "5", "label": "00",
                      "mileage": 50000.0, "capacity": 40.0, "dataset": "battery_dataset1"}))
    return wins


def test_concat_and_features():
    wins = make_session()
    traj = concat_windows(wins)
    assert traj.shape == (128 * 5, 8)

    f = build_session(wins)
    assert f["__invalid__"] is False
    assert f["car"] == 11 and f["segment"] == "5"
    assert f["d_soc"] == pytest.approx(50.0, abs=1.0)
    assert f["n_windows"] == 5
    # 全局时间轴：5 个窗口 × 1270 s
    assert f["dur_s"] == pytest.approx(1270.0 * 5, rel=1e-6)
    # 10 A × 6350 s / 3600 ≈ 17.6 Ah
    assert f["q_ah"] == pytest.approx(10 * 127 * 5 * 10 / 3600, rel=0.02)
    assert f["q_per_soc"] == pytest.approx(f["q_ah"] / 0.5, rel=0.02)
    assert 0.0 <= f["quality"] <= 1.0
    # SOC 分箱：20-30 与 30-40 应当有值
    assert np.isfinite(f["socbin_ah_20_30"])
    assert np.isfinite(f["socbin_s_20_30"])
    # 电压窗口 3.5-3.7 应当被穿越
    assert np.isfinite(f["vwin_s_3.50_3.70"])


def test_ic_features_sum_to_total():
    f = build_session(make_session(n_win=8, soc0=10, soc_step=10, v0=3.10, v1=4.15))
    ics = [f[f"ic_raw_{k:02d}"] for k in range(len(V_GRID) - 1)]
    total = np.nansum(ics)
    assert total > 0
    assert total == pytest.approx(f["ic_total_ah"], rel=1e-6)
    # 归一化形状求和为 1（在覆盖区间内）
    norms = [f[f"ic_norm_{k:02d}"] for k in range(len(V_GRID) - 1)]
    assert np.nansum(norms) == pytest.approx(1.0, rel=1e-6)


def test_duplicate_windows_dropped():
    wins = make_session(n_win=3)
    dup = wins + [wins[0]]  # 重复第一个窗口
    traj = concat_windows(dup)
    assert traj.shape[0] == 128 * 3


def test_quality_lower_for_short_span():
    short = build_session(make_session(n_win=1, soc_step=3.0))
    long = build_session(make_session(n_win=6, soc_step=10.0))
    assert long["quality"] > short["quality"]
    assert long["q_span"] > short["q_span"]


def test_feature_names_all_present():
    f = build_session(make_session())
    names = session_feature_names()
    missing = [n for n in names if n not in f]
    assert not missing, f"缺少特征: {missing[:5]}"
    assert len(names) == len(set(names)), "特征名重复"
    assert len(names) > 90
