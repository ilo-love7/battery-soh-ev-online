# -*- coding: utf-8 -*-
"""模型工厂：线性基线 / 树模型 / 分位数回归。统一返回 sklearn Pipeline（含中位数插补）。"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(name: str, cfg: Dict[str, Any] | None = None, **overrides):
    """构造模型。

    name:
        elasticnet      线性基线（标准化 + L1/L2 混合）
        rf              随机森林
        xgb             XGBoost 回归（主模型）
        hgb             直方图梯度提升（分位数损失时使用）
        quantile        HistGradientBoosting + quantile 损失（需 alpha）
    """
    cfg = cfg or {}
    kw = dict(overrides)

    if name == "elasticnet":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", ElasticNet(alpha=kw.get("alpha", 0.01), l1_ratio=kw.get("l1_ratio", 0.3),
                             max_iter=5000, random_state=42)),
        ])

    if name == "rf":
        p = {**cfg.get("rf", {}), **kw}
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", RandomForestRegressor(n_jobs=-1, random_state=42, **p)),
        ])

    if name == "xgb":
        from xgboost import XGBRegressor

        p = {**cfg.get("xgb", {}), **kw}
        n_jobs = p.pop("n_jobs", -1)      # 允许调用方限制线程数（LOVO 并行时需要）
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", XGBRegressor(
                objective="reg:squarederror", tree_method="hist", n_jobs=n_jobs,
                random_state=42, verbosity=0, **p)),
        ])

    if name == "hgb":
        p = {**kw}
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", HistGradientBoostingRegressor(random_state=42, **p)),
        ])

    if name == "quantile":
        alpha = kw.pop("alpha", 0.5)
        p = {**kw}
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", HistGradientBoostingRegressor(
                loss="quantile", quantile=alpha, random_state=42,
                learning_rate=p.pop("learning_rate", 0.06),
                max_iter=p.pop("max_iter", 400),
                max_leaf_nodes=p.pop("max_leaf_nodes", 31),
                min_samples_leaf=p.pop("min_samples_leaf", 20),
                l2_regularization=p.pop("l2_regularization", 1.0),
                **p)),
        ])

    raise ValueError(f"未知模型: {name}")


def fit_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def feature_importance(model, feature_names) -> np.ndarray:
    """取出 Pipeline 末端的特征重要性（树模型），线性模型返回 |coef|。

    注意：SimpleImputer 会跳过"整列全缺失"的特征，导致末端模型的维度小于输入列数。
    这里用 imputer.statistics_ 里的 NaN 还原被跳过的列，保证返回数组与 feature_names 对齐。
    """
    est = model[-1] if isinstance(model, Pipeline) else model
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        imp = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
    else:
        return np.zeros(len(feature_names))

    if imp.size == len(feature_names):
        return imp

    if isinstance(model, Pipeline) and "imp" in model.named_steps:
        imputer = model.named_steps["imp"]
        stats = getattr(imputer, "statistics_", None)
        if stats is not None:
            keep = ~np.isnan(np.asarray(stats, dtype=float))
            if keep.sum() == imp.size:
                full = np.zeros(len(feature_names))
                full[keep] = imp
                return full
    return np.zeros(len(feature_names))
