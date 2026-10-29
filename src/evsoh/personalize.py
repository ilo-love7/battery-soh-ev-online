# -*- coding: utf-8 -*-
"""冷启动 + 个性化两阶段框架。

阶段 1（零样本）：全局模型对全新车辆直接预测，不接触该车任何标签。
阶段 2（个性化）：该车每多一条带容量标签的充电片段，就用少量参数校正全局预测。

校正形式取仿射 a·p + b（两参数），并加向恒等映射收缩的岭惩罚：
    min_{a,b} Σ (a·p_i + b - y_i)² + λ[(a-1)² + b²]
λ 越大越接近"不校正"，用于避免 k 很小时过拟合。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def fit_affine(pred: np.ndarray, y: np.ndarray, lam: float = 5.0) -> Tuple[float, float]:
    """拟合 a, b（收缩到 a=1, b=0）。"""
    pred = np.asarray(pred, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(pred) & np.isfinite(y)
    pred, y = pred[m], y[m]
    if pred.size == 0:
        return 1.0, 0.0
    X = np.column_stack([pred, np.ones_like(pred)])
    A = X.T @ X + lam * np.eye(2)
    b = X.T @ y + lam * np.array([1.0, 0.0])
    theta = np.linalg.solve(A, b)
    return float(theta[0]), float(theta[1])


def apply_affine(pred: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.asarray(pred, dtype=float) + b


def fit_bias(pred: np.ndarray, y: np.ndarray, lam: float = 5.0) -> float:
    """仅校正偏差（1 参数，k 极小时更稳）。"""
    pred = np.asarray(pred, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() == 0:
        return 0.0
    return float(np.mean(y[m] - pred[m]) * pred[m].size / (pred[m].size + lam))


def online_curve(pred_zero: np.ndarray, y: np.ndarray, cars: np.ndarray,
                 order: np.ndarray | None = None, k_list: List[int] = None,
                 correction: str = "bias", lam: float = 5.0,
                 eval_frac: float = 0.5, min_sessions: int = 20,
                 split: str = "chrono") -> Tuple[List[Dict], Dict]:
    """在线个性化收敛曲线（严格的"用过去校准、预测未来"设计）。

    每辆车按 order（建议里程序）排序后切成两段：
        cal_pool = 前 (1-eval_frac) 段  —— 可用来积累标签
        eval_set = 后 eval_frac 段      —— 固定不变的评估集（模拟"之后的新片段"）
    对每个 k：只用 cal_pool 的前 k 段做校正，在**固定的** eval_set 上评估。
    这样 k=0（零样本）与 k>0 的误差在同一评估集上可比，也不存在"用自己校正自己"的乐观偏差。

    返回 (曲线, 每车明细{k=0 与 k=k_mid 的 MAE/偏差})。
    """
    k_list = k_list or [0, 1, 2, 3, 5, 8, 12, 20]
    cars = np.asarray(cars)
    y = np.asarray(y, dtype=float)
    pred_zero = np.asarray(pred_zero, dtype=float)
    order = np.arange(cars.size) if order is None else np.asarray(order)

    plan: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for c in np.unique(cars):
        idx = np.where(cars == c)[0]
        idx = idx[np.argsort(order[idx])]
        if split == "random":  # 同分布切分：衡量"个性化潜力"上限
            idx = idx[np.random.default_rng(0).permutation(idx.size)]
        if idx.size < min_sessions:
            continue
        n_eval = max(1, int(round(idx.size * eval_frac)))
        plan[int(c)] = (idx[:-n_eval], idx[-n_eval:])

    curve: List[Dict] = []
    detail: Dict[int, Dict[str, float]] = {}
    for k in k_list:
        maes, meds, biases = [], [], []
        for c, (pool, ev) in plan.items():
            if k > len(pool):
                continue
            if k == 0:
                p = pred_zero[ev]
            elif correction == "bias":
                b = float(np.sum(y[pool[:k]] - pred_zero[pool[:k]]) / (k + lam))
                p = pred_zero[ev] + b
            else:
                a, b = fit_affine(pred_zero[pool[:k]], y[pool[:k]], lam)
                p = apply_affine(pred_zero[ev], a, b)
            mae = float(np.mean(np.abs(p - y[ev])))
            maes.append(mae)
            meds.append(mae)
            biases.append(float(np.mean(pred_zero[ev] - y[ev])))
            if k in (0, 5):
                detail.setdefault(c, {})[f"mae_k{k}"] = mae
                detail[c]["zero_shot_bias"] = float(np.mean(pred_zero[ev] - y[ev]))
        if maes:
            curve.append({
                "k": int(k), "n_cars": len(maes),
                "car_mae_mean": float(np.mean(maes)),
                "car_mae_median": float(np.median(meds)),
                "car_mae_p90": float(np.percentile(maes, 90)),
            })
    return curve, detail


def coldstart_curve(pred_zero: np.ndarray, y: np.ndarray, cars: np.ndarray,
                    order: np.ndarray | None = None, k_list: List[int] = None,
                    correction: str = "affine", lam: float = 5.0,
                    n_repeats: int = 20, seed: int = 42) -> List[Dict]:
    """误差随个性化片段数 k 的收敛曲线。

    pred_zero: 全局模型的零样本预测（每条会话一个）
    y:         真值（SOH 或容量）
    cars:      所属车辆
    order:     同一辆车内部的片段顺序（默认按数组顺序；建议传时间序）
    n_repeats: 每辆车随机抽取"用哪 k 条做校正"的重复次数（k 较小时降低偶然性）
    """
    k_list = k_list or [0, 1, 2, 3, 5, 8, 12, 20]
    rng = np.random.default_rng(seed)
    cars = np.asarray(cars)
    y = np.asarray(y, dtype=float)
    pred_zero = np.asarray(pred_zero, dtype=float)
    order = np.arange(cars.size) if order is None else np.asarray(order)

    # 每辆车内部的片段索引（按给定顺序）
    per_car: Dict[int, np.ndarray] = {}
    for c in np.unique(cars):
        idx = np.where(cars == c)[0]
        per_car[int(c)] = idx[np.argsort(order[idx])]

    out: List[Dict] = []
    for k in k_list:
        errs: List[float] = []
        per_car_mae: List[float] = []
        for c, idx in per_car.items():
            if idx.size <= k:
                continue
            if k == 0:
                cal_idx = np.array([], dtype=int)
                eval_idx = idx
            else:
                eval_idx = idx
                cal_idx = idx[:k]
            if k == 0:
                pred = pred_zero[eval_idx]
            else:
                # 用前 k 条校正，再在同一辆车的全部片段上评估（含校正样本，模拟在线更新后的整体状态）
                reps = 1 if idx.size <= k + 2 else n_repeats
                acc = []
                for _ in range(reps):
                    if reps > 1:
                        pick = rng.choice(idx, size=k, replace=False)
                    else:
                        pick = cal_idx
                    if correction == "affine":
                        a, b = fit_affine(pred_zero[pick], y[pick], lam)
                        p = apply_affine(pred_zero[eval_idx], a, b)
                    else:
                        b = fit_bias(pred_zero[pick], y[pick], lam)
                        p = pred_zero[eval_idx] + b
                    acc.append(float(np.mean(np.abs(p - y[eval_idx]))))
                per_car_mae.append(float(np.mean(acc)))
                pred = None
            if pred is not None:
                per_car_mae.append(float(np.mean(np.abs(pred - y[eval_idx]))))
        if per_car_mae:
            out.append({
                "k": int(k),
                "n_cars": len(per_car_mae),
                "car_mae_mean": float(np.mean(per_car_mae)),
                "car_mae_median": float(np.median(per_car_mae)),
                "car_mae_p90": float(np.percentile(per_car_mae, 90)),
            })
    return out
