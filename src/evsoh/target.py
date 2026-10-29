# -*- coding: utf-8 -*-
"""SOH 标签构造。

数据集只给绝对容量 capacity（Ah，0 = 缺失），跨车不可比（不同车型包容量 24–46 Ah）。
因此 SOH 以「每车 BOL 参考容量」为分母：

    SOH_car = capacity_session / ref_capacity_car

ref_capacity_car 取该车全部有标签会话中容量最高 10% 的中位数（稳健上分位），
含义是"该车观测到的最健康状态"。这是本数据集下的必要假设（数据集未提供额定容量），
其合理性用"容量随里程单调下降"来验证（见 reports §2.2：80.6% 的车负相关）。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def bol_reference(cap: np.ndarray, top_frac: float = 0.10) -> float:
    """稳健 BOL 参考容量：最高 top_frac 分位的中位数。"""
    cap = np.asarray(cap, dtype=float)
    cap = cap[np.isfinite(cap) & (cap > 0)]
    if cap.size == 0:
        return np.nan
    n_top = max(1, int(round(cap.size * top_frac)))
    return float(np.median(np.sort(cap)[-n_top:]))


def add_soh_labels(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, Dict]:
    """加 soh / bol_ref 列，并剔除标签不可用的车。返回 (df, 统计信息)。"""
    lab = cfg["label"]
    min_n = int(lab["min_labeled_sessions"])
    clip_lo, clip_hi = lab["soh_clip"]

    df = df.copy()
    labeled = df["capacity"] > 0
    stats = {"sessions_in": int(len(df)), "cars_in": int(df["car"].nunique()),
             "labeled_sessions": int(labeled.sum())}

    refs = {}
    for c, g in df[labeled].groupby("car"):
        refs[c] = bol_reference(g["capacity"].values, top_frac=0.10)
    counts = df[labeled].groupby("car").size().to_dict()

    keep_cars = [c for c, r in refs.items() if np.isfinite(r) and counts.get(c, 0) >= min_n]
    df = df[df["car"].isin(keep_cars)].copy()
    df["bol_ref"] = df["car"].map(refs)
    df["soh"] = df["capacity"] / df["bol_ref"]

    before = len(df)
    ok = (df["soh"] >= clip_lo) & (df["soh"] <= clip_hi)
    df["soh_clipped"] = ~ok
    df = df[ok].copy()

    stats.update({
        "cars_kept": int(df["car"].nunique()),
        "sessions_kept": int(len(df)),
        "dropped_by_clip": int(before - len(df)),
        "soh_min": float(df["soh"].min()),
        "soh_median": float(df["soh"].median()),
        "soh_max": float(df["soh"].max()),
        "soh_p05": float(df["soh"].quantile(0.05)),
    })
    return df, stats


def mileage_normalized_soh(df: pd.DataFrame) -> pd.Series:
    """按车去趋势后的 SOH 残差（用于诊断标签噪声，不参与建模）。"""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _c, g in df.groupby("car"):
        if len(g) < 3 or g["mileage"].nunique() < 2:
            continue
        x = g["mileage"].values.astype(float)
        y = g["soh"].values.astype(float)
        k, b = np.polyfit(x, y, 1)
        out.loc[g.index] = y - (k * x + b)
    return out
