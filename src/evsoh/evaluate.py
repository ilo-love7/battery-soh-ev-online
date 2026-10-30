# -*- coding: utf-8 -*-
"""评测协议与指标：按车分组（跨车外推）/ 同车随机（乐观上界）/ 跨厂商留一域。"""
from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import numpy as np


# ---------------- 指标 ----------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """容量/SOH 通用回归指标。y 以 Ah 或 SOH 小数传入，报告时再换算。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size == 0:
        return {}
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(y_true.size),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mape_pct": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-9)) * 100.0),
        "p90_abs_err": float(np.percentile(np.abs(err), 90)),
    }


def group_metrics(df, y_col: str, pred_col: str, group_col: str = "car") -> Dict[str, float]:
    """先按车算误差再取平均（车级等权），避免高里程车主导结论。"""
    per = []
    for _g, sub in df.groupby(group_col):
        per.append(regression_metrics(sub[y_col], sub[pred_col]))
    if not per:
        return {}
    return {
        "n_cars": len(per),
        "car_rmse_mean": float(np.mean([p["rmse"] for p in per if p])),
        "car_rmse_median": float(np.median([p["rmse"] for p in per if p])),
        "car_mae_mean": float(np.mean([p["mae"] for p in per if p])),
        "car_bias_mean": float(np.mean([p["bias"] for p in per if p])),
        "car_r2_median": float(np.median([p["r2"] for p in per if p])),
    }


# ---------------- 划分 ----------------
def group_kfold_by_car(cars: np.ndarray, n_splits: int = 5, seed: int = 42) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """按车分组的 K 折索引：同一辆车的全部会话只出现在一侧（跨车外推协议）。"""
    uniq = np.unique(cars)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    folds = np.array_split(perm, n_splits)
    for k in range(n_splits):
        test_cars = set(folds[k].tolist())
        mask = np.array([c in test_cars for c in cars])
        yield np.where(~mask)[0], np.where(mask)[0]


def leave_one_dataset_out(datasets: np.ndarray) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """留一厂商域：用其余两个分块训练，在留出分块上测。"""
    for ds in np.unique(datasets):
        mask = datasets == ds
        yield str(ds), np.where(~mask)[0], np.where(mask)[0]


def random_split_within_car(cars: np.ndarray, test_frac: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """每车内部随机切分（同车回代 = 乐观上界，用于对照跨车外推的差距）。"""
    rng = np.random.default_rng(seed)
    test_idx: List[int] = []
    for c in np.unique(cars):
        idx = np.where(cars == c)[0]
        if idx.size <= 1:
            test_idx.extend(idx.tolist())
            continue
        n_test = max(1, int(round(idx.size * test_frac)))
        test_idx.extend(rng.choice(idx, size=n_test, replace=False).tolist())
    test_idx = np.array(sorted(set(test_idx)))
    train_idx = np.setdiff1d(np.arange(cars.size), test_idx)
    return train_idx, test_idx


# ---------------- 区间校准 ----------------
def interval_metrics(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Dict[str, float]:
    """区间覆盖率 / 平均宽度 / 越界方向。"""
    y_true, lo, hi = map(lambda a: np.asarray(a, dtype=float), (y_true, lo, hi))
    m = np.isfinite(y_true) & np.isfinite(lo) & np.isfinite(hi)
    y_true, lo, hi = y_true[m], lo[m], hi[m]
    if y_true.size == 0:
        return {}
    inside = (y_true >= lo) & (y_true <= hi)
    return {
        "n": int(y_true.size),
        "coverage": float(inside.mean()),
        "width_mean": float(np.mean(hi - lo)),
        "width_median": float(np.median(hi - lo)),
        "below_rate": float((y_true < lo).mean()),
        "above_rate": float((y_true > hi).mean()),
    }


def pit_histogram(y_true: np.ndarray, lo: np.ndarray, med: np.ndarray, hi: np.ndarray, bins: int = 10):
    """PIT（概率积分变换）直方图：把样本映射到区间内的分位位置，理想为均匀分布。

    以分位数回归给出的 (P5, P50, P95) 三点做分段线性 CDF 近似。
    """
    y_true, lo, med, hi = map(lambda a: np.asarray(a, dtype=float), (y_true, lo, med, hi))
    m = np.isfinite(y_true) & np.isfinite(lo) & np.isfinite(med) & np.isfinite(hi)
    y_true, lo, med, hi = y_true[m], lo[m], med[m], hi[m]
    u = np.full(y_true.shape, np.nan)
    lower = y_true <= med
    span_lo = np.maximum(med - lo, 1e-9)
    span_hi = np.maximum(hi - med, 1e-9)
    u[lower] = 0.5 * np.clip((y_true[lower] - lo[lower]) / span_lo[lower], 0, 1)
    u[~lower] = 0.5 + 0.5 * np.clip((y_true[~lower] - med[~lower]) / span_hi[~lower], 0, 1)
    hist, edges = np.histogram(u, bins=bins, range=(0, 1))
    hist = hist / max(hist.sum(), 1)
    return hist, edges


def tier_error_rate(values: np.ndarray, edges: np.ndarray, sigma: float,
                    n_draw: int = 2000, seed: int = 42) -> Dict[str, float]:
    """车队分档的错档率：把"报告值 ± 误差"落到档位上，统计被分错的比例。

    ``values`` 是每辆车的上报值（SOH，pp），``edges`` 是升序档位边界，
    ``sigma`` 是车级误差（pp）。真值按 ``N(value, sigma²)`` 抽 ``n_draw`` 次，
    逐车算"真档 ≠ 报档"的频率再对车取平均。

    另外返回 ``boundary_share``：上报值落在某个边界 ±σ 之内的车辆占比。
    这个量比错档率更早给出警报——它只依赖我们自己的估计，不需要知道真值。
    """
    v = np.asarray(values, dtype=float)
    e = np.asarray(edges, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or e.size < 2:
        raise ValueError("tier_error_rate: 样本或边界不足")
    if sigma < 0:
        raise ValueError("tier_error_rate: sigma 不能为负")
    assigned = np.digitize(v, e)
    if sigma == 0:
        return {"mis_rate": 0.0, "boundary_share": 0.0, "n": int(v.size),
                "n_tiers": int(len(np.unique(assigned)))}
    rng = np.random.default_rng(seed)
    draw = v[:, None] + rng.normal(0.0, float(sigma), (v.size, int(n_draw)))
    truth = np.digitize(draw, e)
    mis = (truth != assigned[:, None]).mean(axis=1)
    near = np.min(np.abs(v[:, None] - e[None, :]), axis=1) < sigma
    return {"mis_rate": float(mis.mean()), "boundary_share": float(near.mean()),
            "n": int(v.size), "n_tiers": int(len(np.unique(assigned)))}


def conformal_by_car(cars: np.ndarray, y: np.ndarray, pred: np.ndarray,
                     nominal: float = 0.90, cal_frac: float = 0.25,
                     n_splits: int = 5, seed: int = 42) -> Dict[str, float]:
    """窗级 split-conformal：折内抽 ``cal_frac`` 的车做校准，其余车直接以预测为中心。

    与脚本 13 / 18 / 20 里用的是同一套配方（``method="higher"`` 的分位），
    差别只在抽出来的窗口是"同车 k 段聚合"还是"单个片段"。抽出成公共函数是为了
    让聚合 × 个性化的网格不必再复制一份。
    """
    cars = np.asarray(cars)
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    lo = np.full(y.shape, np.nan)
    hi = np.full(y.shape, np.nan)
    for fi, (tr, te) in enumerate(group_kfold_by_car(cars, n_splits=n_splits, seed=seed)):
        tr_cars = np.unique(cars[tr])
        rng = np.random.default_rng(1000 + fi)
        n_cal = max(1, int(round(len(tr_cars) * cal_frac)))
        cal_cars = set(rng.choice(tr_cars, size=n_cal, replace=False).tolist())
        is_cal = np.array([c in cal_cars for c in cars[tr]])
        cal = tr[is_cal]
        q = float(np.quantile(np.abs(pred[cal] - y[cal]), nominal, method="higher"))
        lo[te], hi[te] = pred[te] - q, pred[te] + q
    m = interval_metrics(y, lo, hi)
    return {"width": float(m["width_mean"]), "coverage": m["coverage"],
            "below": m["below_rate"], "above": m["above_rate"],
            "q": float(np.nanmean((hi - lo) / 2.0))}
