# -*- coding: utf-8 -*-
"""特征分组：用于消融实验（库仑计 / 电压域形状 / SOC 区间 / 工况上下文）。"""
from __future__ import annotations

from typing import Dict, List

from .session import session_feature_names

_ALL = session_feature_names()

# 库仑计类：直接对电流积分得到的电量量纲特征
_COULOMB_PREFIX = ("q_ah", "q_per_soc", "q_per_hour", "socbin_ah_", "ic_raw_", "vwin_ah_", "ic_total_ah")
# 电压域形状：IC 归一化形状 + 电压窗口穿越时长 + 电压统计
_SHAPE_PREFIX = ("ic_norm_", "vwin_s_", "v_", "ic_covered_bins")
# SOC 区间时长
_SOCBIN_PREFIX = ("socbin_s_",)
# 工况上下文：电流/温度/一致性/质量
_CONTEXT_PREFIX = ("i_", "t_", "temp_gap", "cell_gap", "n_windows", "n_steps", "dur_s",
                   "soc_start", "soc_end", "d_soc", "soc_min", "soc_max",
                   "q_span", "q_mono", "q_current", "q_temp", "q_complete", "quality")


def _pick(prefixes) -> List[str]:
    return [c for c in _ALL if c.startswith(tuple(prefixes))]


FEATURE_GROUPS: Dict[str, List[str]] = {
    "coulomb": _pick(_COULOMB_PREFIX),
    "shape": _pick(_SHAPE_PREFIX),
    "socbin_time": _pick(_SOCBIN_PREFIX),
    "context": _pick(_CONTEXT_PREFIX),
}

ABLATIONS: Dict[str, List[str]] = {
    "full": list(_ALL),
    "curve_only": FEATURE_GROUPS["shape"] + FEATURE_GROUPS["socbin_time"] + FEATURE_GROUPS["context"],
    "coulomb_only": FEATURE_GROUPS["coulomb"] + FEATURE_GROUPS["context"],
    "no_coulomb": [c for c in _ALL if c not in FEATURE_GROUPS["coulomb"]],
}


def feature_columns() -> List[str]:
    return list(_ALL)


def active_features(df, cols: List[str] | None = None) -> List[str]:
    """剔除在整份数据里全为缺失的列。

    某些电压网格/分箱在所有样本上都不被覆盖（例如 LFP 体系永远到不了 4.1–4.2 V），
    这类列对模型零贡献，保留会让插补器报"跳过无观测值特征"，也让特征重要性对不齐。
    """
    cols = list(cols if cols is not None else _ALL)
    return [c for c in cols if c in df.columns and df[c].notna().any()]
