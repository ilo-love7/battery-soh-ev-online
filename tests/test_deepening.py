# -*- coding: utf-8 -*-
"""第二轮深化实验的单元测试：CORAL 变换与 10 mV IC 峰检测。

这两个函数是 `scripts/12` 与 `scripts/15` 的核心数学步骤，且都容易被"看起来对"的
实现糊弄过去（CORAL 方向搞反、峰检测把噪声当峰），所以单独钉住。
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load_script(name: str, rel_path: str):
    """按文件路径加载 scripts/ 下的脚本（它们不是包的一部分）。"""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m12():
    return _load_script("_deep_12", "scripts/12_lovo_decomposition.py")


@pytest.fixture(scope="module")
def m15():
    return _load_script("_deep_15", "scripts/15_ic_fine.py")


def test_coral_matches_target_moments(m12):
    """CORAL 把源域搬到目标域后，均值与协方差必须等于目标域。"""
    rng = np.random.default_rng(0)
    Xs = rng.normal(0.0, 1.0, size=(400, 5))
    Xt = rng.normal(3.0, 2.0, size=(500, 5)) @ np.diag([1.0, 0.5, 2.0, 1.0, 1.5])

    Z = m12._coral_src_to_tgt(Xs, Xt, eps=1e-9)

    assert np.allclose(Z.mean(axis=0), Xt.mean(axis=0), atol=1e-6)
    assert np.allclose(np.cov(Z, rowvar=False), np.cov(Xt, rowvar=False), atol=1e-6)


def test_coral_is_identity_when_domains_agree(m12):
    """源域与目标域同分布时，CORAL 不应改变任何样本。"""
    rng = np.random.default_rng(1)
    X = rng.normal(1.0, 2.0, size=(300, 4))

    Z = m12._coral_src_to_tgt(X, X.copy(), eps=1e-9)

    assert np.allclose(Z, X, atol=1e-6)


def test_fine_peaks_recovers_bimodal_structure(m15):
    """10 mV 网格上的两个高斯峰应被检出，且按峰高降序排列。"""
    centers = m15.FINE_CENTERS
    ic = np.full(centers.size, 0.02)
    for pos, amp in ((3.45, 1.0), (3.95, 0.6)):
        ic += amp * np.exp(-0.5 * ((centers - pos) / 0.05) ** 2)

    out = m15.fine_peaks(ic)

    assert out["fpk_n"] == 2.0
    assert out["fpk0_v"] == pytest.approx(3.45, abs=0.03)   # 更高的峰排第一
    assert out["fpk1_v"] == pytest.approx(3.95, abs=0.03)
    assert out["fpk0_h"] > out["fpk1_h"]
    assert 0.0 < out["fpk_ratio21"] < 1.0                   # 第二峰更矮，比值 < 1
    assert out["fpk_cov_lo"] < out["fpk0_v"] < out["fpk_cov_hi"]


def test_fine_peaks_handles_all_nan(m15):
    """整段缺失时返回全 NaN，而不是抛异常。"""
    out = m15.fine_peaks(np.full(m15.FINE_CENTERS.size, np.nan))

    assert np.isnan(out["fpk_n"])
    assert np.isnan(out["fpk0_v"])
