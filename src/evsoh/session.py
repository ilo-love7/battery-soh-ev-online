# -*- coding: utf-8 -*-
"""会话级（charge_segment）特征工程：把一个充电会话的多个 128 步窗口拼成一条充电轨迹，
再提取「容量估计」所需的三类特征：

1. 库仑计物理量：累计充入电量 ΔQ、SOC 跨度 ΔSOC、隐含容量 ΔQ/ΔSOC；
2. 固定 SOC 区间特征：每 10% SOC 的充电时长与充入电量（对 SOC 报数跳变不敏感的部分靠平滑）；
3. 增量容量（IC, dQ/dV）电压网格特征：在 3.00–4.25 V 固定网格上按 50 mV 分箱，得到与 SOC 字段
   无关的纯电压域容量形状；另有 5 个粗电压窗口的穿越时长/电量。

同时输出片段质量分（quality），用于 D5 的鲁棒性分析与低质片段降权。

注意（数据实况，已核验）：原始 Timestamp 列以 10 s/步 记录，但积分得到的 ΔQ/ΔSOC 与
数据集给出的 capacity 标签存在约 3.5 倍的固定标度差（见 reports 报告 §2.3）。本模块保留
原始物理量不做修正，标度校准放在物理基线脚本里用训练集拟合，避免泄漏。
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data import COL_I, COL_SOC, COL_T, COL_TMAX, COL_TMIN, COL_V, COL_VMAX, COL_VMIN, load_window

# 电压网格：3.00 → 4.25 V，步长 50 mV（覆盖 LFP 平台与三元高压区）
V_GRID = np.round(np.arange(3.00, 4.2501, 0.05), 3)
# 粗电压窗口（V）
V_WINDOWS = [(3.30, 3.50), (3.50, 3.70), (3.70, 3.90), (3.90, 4.10), (4.10, 4.20)]
# SOC 分箱（每 10%）
SOC_BINS = [(i, i + 10) for i in range(0, 100, 10)]

_EPS = 1e-9


def _sanitize(feat: np.ndarray) -> np.ndarray:
    """把非有限值替换为 np.nan 后原样返回（不插值，避免制造数据）。"""
    out = np.array(feat, dtype=np.float64, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def _fnum(name: str) -> int:
    """窗口文件名（全局窗口编号）转整数，用于恢复会话内的时间顺序。

    已核验：同一会话的窗口文件编号连续（步长 1 占 100%），按编号排序后 SOC 单调
    上升的比例为 99.9%。原始 Timestamp 列是**窗口内相对时间**（每个窗口都从 0 开始），
    不能用来排序或跨窗口积分。
    """
    base = os.path.basename(name).split(".")[0]
    try:
        return int(base)
    except ValueError:
        return 0


def concat_windows(windows: Sequence[Tuple[str, np.ndarray, Dict]]) -> np.ndarray:
    """按窗口编号拼接轨迹，并把各窗口的相对时间还原成会话内全局时间轴。

    windows: [(file, feat(128,8), meta), ...]；返回拼接后的 (N,8) 数组。
    """
    ordered = sorted(windows, key=lambda w: _fnum(w[0]))
    blocks: List[np.ndarray] = []
    t_acc = 0.0
    seen = set()
    for fname, feat, _m in ordered:
        if fname in seen:  # 同一文件重复给出时跳过
            continue
        seen.add(fname)
        f = np.array(feat, dtype=np.float64, copy=True)
        tloc = f[:, COL_T]
        if not np.isfinite(tloc).any():
            continue
        t0 = float(tloc[0])
        t1 = float(tloc[-1])
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 < t0:
            continue
        f[:, COL_T] = t_acc + (tloc - t0)
        t_acc += (t1 - t0)
        blocks.append(f)
    if not blocks:
        return np.empty((0, 8))
    return _sanitize(np.vstack(blocks))


def _interp_at(x: np.ndarray, y: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """在单调不减的 x 上插值 y；超出范围返回 nan。"""
    if x.size < 2:
        return np.full(targets.shape, np.nan)
    out = np.interp(targets, x, y, left=np.nan, right=np.nan)
    out[(targets < x[0]) | (targets > x[-1])] = np.nan
    return out


def build_session(windows: Sequence[Tuple[str, np.ndarray, Dict]]) -> Dict:
    """从若干窗口构造一个会话的特征字典。"""
    traj = concat_windows(windows)
    meta0 = windows[0][2]
    f: Dict[str, float] = {}
    f["n_windows"] = float(len(windows))

    if traj.shape[0] < 10 or traj.shape[0] < 100 * len(windows):
        return {"__invalid__": True, "car": int(meta0["car"]),
                "segment": str(meta0["charge_segment"]), "n_windows": float(len(windows))}

    t = traj[:, COL_T]
    soc = traj[:, COL_SOC]
    v = traj[:, COL_V]
    cur = traj[:, COL_I]
    vmax = traj[:, COL_VMAX]
    vmin = traj[:, COL_VMIN]
    tmax = traj[:, COL_TMAX]
    tmin = traj[:, COL_TMIN]

    # 会话总时长（concat_windows 已把窗口内相对时间还原为全局时间轴）
    n_win = len(windows)
    dur_s = float(np.nanmax(t) - np.nanmin(t)) if np.isfinite(t).any() else 0.0

    # ---------- 基础量 ----------
    soc_ok = np.isfinite(soc)
    soc0 = float(soc[soc_ok][0]) if soc_ok.any() else np.nan
    soc1 = float(soc[soc_ok][-1]) if soc_ok.any() else np.nan
    d_soc = soc1 - soc0
    f["soc_start"], f["soc_end"], f["d_soc"] = soc0, soc1, d_soc
    f["soc_min"], f["soc_max"] = float(np.nanmin(soc)), float(np.nanmax(soc))
    f["dur_s"] = dur_s
    f["n_steps"] = float(traj.shape[0])

    # 电流（充电为负）
    f["i_mean"] = float(np.nanmean(cur))
    f["i_abs_mean"] = float(np.nanmean(np.abs(cur)))
    f["i_std"] = float(np.nanstd(cur))
    f["i_cv"] = f["i_std"] / (f["i_abs_mean"] + _EPS)
    f["i_abs_p95"] = float(np.nanpercentile(np.abs(cur), 95))
    f["i_abs_max"] = float(np.nanmax(np.abs(cur)))

    # 电压
    f["v_start"], f["v_end"] = float(v[0]), float(v[-1])
    f["v_min"], f["v_max"] = float(np.nanmin(v)), float(np.nanmax(v))
    f["v_mean"] = float(np.nanmean(v))
    f["v_range"] = f["v_max"] - f["v_min"]

    # 温度
    f["t_mean"] = float(np.nanmean(tmax))
    f["t_max"] = float(np.nanmax(tmax))
    f["t_min"] = float(np.nanmin(tmin))
    f["t_rise"] = f["t_max"] - float(tmin[np.isfinite(tmin)][0]) if np.isfinite(tmin).any() else np.nan
    gap = tmax - tmin
    f["temp_gap_mean"] = float(np.nanmean(gap))
    cgap = vmax - vmin
    f["cell_gap_mean"] = float(np.nanmean(cgap))
    f["cell_gap_max"] = float(np.nanmax(cgap))

    # ---------- 库仑计（原始标度） ----------
    q_cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(np.abs(cur[:-1])) * 10.0 / 3600.0)])
    f["q_ah"] = float(q_cum[-1])
    f["q_per_soc"] = f["q_ah"] / (abs(d_soc) / 100.0) if abs(d_soc) > 1.0 else np.nan
    f["q_per_hour"] = f["q_ah"] / (dur_s / 3600.0) if dur_s > 0 else np.nan

    # ---------- 固定 SOC 区间 ----------
    # 用累计电量做 x 轴：q_at_soc 由 SOC→Q 的单调化插值得到
    soc_mono = np.maximum.accumulate(np.nan_to_num(soc, nan=0.0))
    for lo, hi in SOC_BINS:
        q_lo = _interp_at(soc_mono, q_cum, np.array([lo], dtype=float))[0]
        q_hi = _interp_at(soc_mono, q_cum, np.array([hi], dtype=float))[0]
        tag = f"{lo}_{hi}"
        f[f"socbin_ah_{tag}"] = q_hi - q_lo if (np.isfinite(q_lo) and np.isfinite(q_hi)) else np.nan
        # 该 SOC 区间的平均充电时长（s）
        t_at = _interp_at(soc_mono, t, np.array([lo, hi], dtype=float))
        f[f"socbin_s_{tag}"] = (t_at[1] - t_at[0]) if np.all(np.isfinite(t_at)) else np.nan

    # ---------- 增量容量 IC（dQ/dV）----------
    v_mono = np.maximum.accumulate(np.nan_to_num(v, nan=0.0))
    q_at_grid = _interp_at(v_mono, q_cum, V_GRID)
    ic_raw = np.diff(q_at_grid)  # 每 50 mV 的 Ah
    covered = np.isfinite(ic_raw)
    f["ic_covered_bins"] = float(covered.sum())
    ic_sum = float(np.nansum(ic_raw))
    f["ic_total_ah"] = ic_sum
    for k in range(len(V_GRID) - 1):
        f[f"ic_raw_{k:02d}"] = float(ic_raw[k]) if np.isfinite(ic_raw[k]) else np.nan
        f[f"ic_norm_{k:02d}"] = float(ic_raw[k] / ic_sum) if (np.isfinite(ic_raw[k]) and ic_sum > 0) else np.nan

    # ---------- 粗电压窗口穿越 ----------
    t_at_v = _interp_at(v_mono, t, np.array([w[0] for w in V_WINDOWS] + [w[1] for w in V_WINDOWS], dtype=float))
    q_at_v = _interp_at(v_mono, q_cum, np.array([w[0] for w in V_WINDOWS] + [w[1] for w in V_WINDOWS], dtype=float))
    n_w = len(V_WINDOWS)
    for j, (lo, hi) in enumerate(V_WINDOWS):
        t_lo, t_hi = t_at_v[j], t_at_v[j + n_w]
        q_lo, q_hi = q_at_v[j], q_at_v[j + n_w]
        f[f"vwin_s_{lo:.2f}_{hi:.2f}"] = float(t_hi - t_lo) if (np.isfinite(t_lo) and np.isfinite(t_hi)) else np.nan
        f[f"vwin_ah_{lo:.2f}_{hi:.2f}"] = float(q_hi - q_lo) if (np.isfinite(q_lo) and np.isfinite(q_hi)) else np.nan

    # ---------- 质量分 ----------
    f.update(quality_score(f, soc, cur, traj))

    f["car"] = int(meta0["car"])
    f["segment"] = str(meta0["charge_segment"])
    f["dataset"] = str(windows[0][2].get("dataset", ""))
    f["label"] = str(meta0["label"])
    f["mileage"] = float(meta0["mileage"])
    f["capacity"] = float(meta0["capacity"])
    f["__invalid__"] = False
    return f


def quality_score(f: Dict[str, float], soc: np.ndarray, cur: np.ndarray, traj: np.ndarray) -> Dict[str, float]:
    """片段质量分：SOC 覆盖度、单调性、电流稳定性、温区覆盖、数据完整性。

    全部为 0–1 之间的可解释分量，加权得到 quality（越大越可信）。
    """
    d_soc = abs(f.get("d_soc", np.nan))
    s_span = float(np.clip(d_soc / 60.0, 0.0, 1.0)) if np.isfinite(d_soc) else 0.0

    soc_f = soc[np.isfinite(soc)]
    if soc_f.size >= 2:
        d = np.diff(soc_f)
        s_mono = float(np.clip((d >= -1e-6).mean(), 0.0, 1.0))
    else:
        s_mono = 0.0

    s_cur = float(np.clip(1.0 - f.get("i_cv", 1.0), 0.0, 1.0)) if np.isfinite(f.get("i_cv", np.nan)) else 0.0

    t = f.get("t_mean", np.nan)
    s_temp = float(np.clip(1.0 - abs(t - 25.0) / 40.0, 0.0, 1.0)) if np.isfinite(t) else 0.0

    n_nan = float(np.isnan(traj).any(axis=1).mean())
    s_complete = float(np.clip(1.0 - n_nan * 5.0, 0.0, 1.0))

    q = 0.40 * s_span + 0.20 * s_mono + 0.20 * s_cur + 0.10 * s_temp + 0.10 * s_complete
    return {
        "q_span": s_span, "q_mono": s_mono, "q_current": s_cur,
        "q_temp": s_temp, "q_complete": s_complete, "quality": float(q),
    }


def session_feature_names() -> List[str]:
    """返回特征列名（供训练脚本区分特征与元数据列）。"""
    names = []
    for k in range(len(V_GRID) - 1):
        names += [f"ic_raw_{k:02d}", f"ic_norm_{k:02d}"]
    for lo, hi in SOC_BINS:
        names += [f"socbin_ah_{lo}_{hi}", f"socbin_s_{lo}_{hi}"]
    for lo, hi in V_WINDOWS:
        names += [f"vwin_s_{lo:.2f}_{hi:.2f}", f"vwin_ah_{lo:.2f}_{hi:.2f}"]
    base = ["n_windows", "n_steps", "dur_s", "soc_start", "soc_end", "d_soc", "soc_min", "soc_max",
            "i_mean", "i_abs_mean", "i_std", "i_cv", "i_abs_p95", "i_abs_max",
            "v_start", "v_end", "v_min", "v_max", "v_mean", "v_range",
            "t_mean", "t_max", "t_min", "t_rise", "temp_gap_mean", "cell_gap_mean", "cell_gap_max",
            "q_ah", "q_per_soc", "q_per_hour", "ic_covered_bins", "ic_total_ah",
            "q_span", "q_mono", "q_current", "q_temp", "q_complete", "quality"]
    return base + names


# ---------------- 并行读取入口（供脚本 02 调用） ----------------
_JOB_CTX: Dict[str, str] = {}


def init_worker(root: str) -> None:
    _JOB_CTX["root"] = root


def build_from_files(task: Tuple[str, str, List[str]]) -> Dict:
    """task = (dataset, segment_key, [文件名...])；在工作进程里读文件并构造会话特征。"""
    dataset, _seg_key, files = task
    root = _JOB_CTX["root"]
    windows = []
    for fn in files:
        path = os.path.join(root, dataset, "data", fn)
        try:
            feat, meta = load_window(path)
        except Exception:
            continue
        meta["dataset"] = dataset
        windows.append((fn, feat, meta))
    if not windows:
        return {"__invalid__": True, "dataset": dataset, "segment": _seg_key}
    try:
        return build_session(windows)
    except Exception as exc:  # 单会话失败不阻断批处理
        return {"__invalid__": True, "dataset": dataset, "segment": _seg_key, "error": repr(exc)}
