# -*- coding: utf-8 -*-
"""EVBattery 数据读取与窗口级统计。

窗口 pkl = 固定 128 步 × 8 列（见数据集说明文档），必须 torch.load(weights_only=False)。
本模块提供：
  - load_window()      读单个 pkl
  - window_stats()     单窗口统计（第一遍全量扫描用，只依赖单窗口数据）
  - list_windows()     枚举某分块全部 pkl
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

# 8 列顺序（与数据集说明文档一致）
COL_V, COL_I, COL_SOC, COL_VMAX, COL_VMIN, COL_TMAX, COL_TMIN, COL_T = range(8)

_EPS = 1e-9


def load_window(path: str) -> Tuple[np.ndarray, Dict]:
    """读一个窗口 pkl，返回 (features(128,8) float64, meta dict)。"""
    import torch

    feat, meta = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(feat, dtype=np.float64), dict(meta)


def list_windows(root: str, dataset: str) -> List[str]:
    """列出某分块 data/ 下全部 pkl 文件名（升序按数值编号）。"""
    d = os.path.join(root, dataset, "data")
    names = [f for f in os.listdir(d) if f.endswith(".pkl")]
    names.sort(key=lambda s: int(s.split(".")[0]))
    return names


def window_stats(feat: np.ndarray, meta: Dict, dataset: str, fname: str,
                 qc: Dict | None = None) -> Dict:
    """单窗口统计量 + 元数据。用于第一遍全量扫描（不跨窗口聚合）。"""
    qc = qc or {}
    n_steps = feat.shape[0]
    v = feat[:, COL_V]
    i = feat[:, COL_I]
    soc = feat[:, COL_SOC]
    vmax = feat[:, COL_VMAX]
    vmin = feat[:, COL_VMIN]
    tmax = feat[:, COL_TMAX]
    tmin = feat[:, COL_TMIN]
    t = feat[:, COL_T]

    def _f(x, default=np.nan):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        return float(x) if x.size else default

    # ---- 时间 ----
    t_valid = t[np.isfinite(t)]
    if t_valid.size >= 2:
        dt = np.diff(t_valid)
        dt = dt[dt > 0]
        dt_mean = float(np.median(dt)) if dt.size else np.nan
        t_start, t_end = float(t_valid[0]), float(t_valid[-1])
    else:
        dt_mean = t_start = t_end = np.nan

    # ---- SOC ----
    soc_f = soc[np.isfinite(soc)]
    if soc_f.size >= 2:
        soc_start, soc_end = float(soc_f[0]), float(soc_f[-1])
        soc_min, soc_max = float(soc_f.min()), float(soc_f.max())
        d = np.diff(soc_f)
        soc_mono = float((d >= -1e-6).mean())
        soc_max_jump = float(np.abs(d).max()) if d.size else 0.0
    else:
        soc_start = soc_end = soc_min = soc_max = np.nan
        soc_mono = soc_max_jump = np.nan

    # ---- 电压 ----
    v_f = v[np.isfinite(v)]
    if v_f.size >= 2:
        v_start, v_end = float(v_f[0]), float(v_f[-1])
        v_mean, v_std = float(v_f.mean()), float(v_f.std())
        # 对时间做线性回归取斜率（V/s）
        tt = t[np.isfinite(v)]
        if tt.size == v_f.size and tt.size >= 3 and np.ptp(tt) > 0:
            v_slope = float(np.polyfit(tt - tt[0], v_f, 1)[0])
        else:
            v_slope = np.nan
    else:
        v_start = v_end = v_mean = v_std = v_slope = np.nan

    # ---- 单体一致性（最高-最低单体压差）----
    gap = vmax - vmin
    gap = gap[np.isfinite(gap)]
    cell_gap_mean = float(gap.mean()) if gap.size else np.nan
    cell_gap_max = float(gap.max()) if gap.size else np.nan

    # ---- 电流 ----
    i_f = i[np.isfinite(i)]
    if i_f.size:
        i_mean = float(i_f.mean())
        i_abs_mean = float(np.abs(i_f).mean())
        i_std = float(i_f.std())
        i_abs_p95 = float(np.percentile(np.abs(i_f), 95))
    else:
        i_mean = i_abs_mean = i_std = i_abs_p95 = np.nan

    # ---- 温度 ----
    tm = tmax[np.isfinite(tmax)]
    tn = tmin[np.isfinite(tmin)]
    t_mean = float(tm.mean()) if tm.size else np.nan
    t_max = float(tm.max()) if tm.size else np.nan
    t_min = float(tn.min()) if tn.size else np.nan
    t_rise = (t_max - float(tn[0])) if (tm.size and tn.size) else np.nan
    tgap = tmax - tmin
    tgap = tgap[np.isfinite(tgap)]
    temp_gap_mean = float(tgap.mean()) if tgap.size else np.nan

    # ---- 窗口内充入电量（Ah，对 |I|dt 积分）----
    if np.isfinite(dt_mean) and t_valid.size >= 2:
        tt = t_valid
        ii = np.abs(i[np.isfinite(t)][: tt.size])
        if ii.size == tt.size and ii.size >= 2:
            q_ah = float(np.trapezoid(ii, tt) / 3600.0)
        else:
            q_ah = np.nan
    else:
        q_ah = np.nan

    # ---- 可用性 ----
    ok = (
        n_steps == qc.get("n_steps", 128)
        and np.isfinite(soc_start) and np.isfinite(soc_end)
        and np.isfinite(v_mean) and np.isfinite(i_abs_mean)
        and i_abs_mean >= qc.get("min_current_a", 0.0)
        and (not np.isfinite(soc_max_jump) or soc_max_jump <= qc.get("max_soc_jump", 1e9))
    )

    return {
        "dataset": dataset,
        "file": fname,
        "car": int(meta["car"]),
        "segment": str(meta["charge_segment"]),
        "label": str(meta["label"]),
        "mileage": float(meta["mileage"]),
        "capacity": float(meta["capacity"]),
        "n_steps": int(n_steps),
        "t_start": t_start,
        "t_end": t_end,
        "dt_mean": dt_mean,
        "soc_start": soc_start,
        "soc_end": soc_end,
        "soc_min": soc_min,
        "soc_max": soc_max,
        "soc_span": soc_end - soc_start if np.isfinite(soc_end) else np.nan,
        "soc_mono": soc_mono,
        "soc_max_jump": soc_max_jump,
        "v_start": v_start,
        "v_end": v_end,
        "v_mean": v_mean,
        "v_std": v_std,
        "v_slope": v_slope,
        "cell_gap_mean": cell_gap_mean,
        "cell_gap_max": cell_gap_max,
        "i_mean": i_mean,
        "i_abs_mean": i_abs_mean,
        "i_std": i_std,
        "i_abs_p95": i_abs_p95,
        "t_mean": t_mean,
        "t_max": t_max,
        "t_min": t_min,
        "t_rise": t_rise,
        "temp_gap_mean": temp_gap_mean,
        "q_ah": q_ah,
        "valid": bool(ok),
    }
