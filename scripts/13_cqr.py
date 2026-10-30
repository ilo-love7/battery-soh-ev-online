# -*- coding: utf-8 -*-
"""13 D3 深化：CQR（共形分位回归）与条件覆盖率。

上一版给了三条区间路线：分位数回归 Q（覆盖 81.2%）、CV+（90.0%）、归一化一致性 NC
（90.0%）。覆盖率达标只是及格线，真正决定"SOH = 87% ± 2%"能不能写进残值报告的是
**条件覆盖**：短片段（SOC 跨度小）上区间还准不准？如果整体 90% 是由长片段撑起来的，
那对最需要区间的短片段就是失真的。

本脚本补两件事：
  1. **CQR（conformalized quantile regression）**：在每折训练集内部再切出校准车集，
     用 conformity score E = max(lo - y, y - hi) 的 90% 分位把区间整体平移，
     得到有限样本覆盖保证（且保留分位数模型的不对称性，不像 NC 那样按比例缩放）。
  2. **条件覆盖审计**：按 SOC 跨度、质量分、厂商分箱，逐箱报告覆盖率，
     给出"最差箱偏离名义值多少"，四法同表对比。

输出
    results/metrics/cqr.json
    results/cache/cqr_oof.parquet
    results/figures/fig22_条件覆盖率审计.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import group_kfold_by_car, interval_metrics, regression_metrics  # noqa: E402
from evsoh.features import active_features, feature_columns  # noqa: E402
from evsoh.models import build_model  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_PURPLE, C_RED, plt, savefig  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402

SPAN_BINS = [20, 30, 40, 50, 60, 70, 100]


def cond_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  groups: pd.Series, min_n: int = 80) -> dict:
    """逐分箱覆盖率；返回各箱覆盖率与"最差箱相对名义值的偏离"。"""
    inside = ((y >= lo) & (y <= hi)).astype(float)
    d = pd.DataFrame({"c": inside, "g": groups})
    per = {}
    for g, sub in d.groupby("g", observed=True):
        if len(sub) >= min_n:
            per[str(g)] = {"n": int(len(sub)), "coverage": float(sub["c"].mean())}
    if not per:
        return {"bins": {}, "worst_dev": float("nan"), "worst_bin": ""}
    devs = {k: abs(v["coverage"] - 0.9) for k, v in per.items()}
    wk = max(devs, key=devs.get)
    return {"bins": per, "worst_dev": float(devs[wk]), "worst_bin": wk,
            "min_cov": float(min(v["coverage"] for v in per.values())),
            "max_cov": float(max(v["coverage"] for v in per.values()))}


def main() -> None:
    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, _ = add_soh_labels(ses, cfg)
    df = df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)
    feats = active_features(df, feature_columns())
    X = df[feats].astype(float)
    y = df["capacity"].values.astype(float)
    cars = df["car"].values
    ref = float(df["bol_ref"].mean())
    n = len(df)
    print(f"样本 {n} 会话 / {df['car'].nunique()} 车 / {len(feats)} 特征", flush=True)

    folds = list(group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                    seed=cfg["splits"]["random_state"]))

    # ---------------- 1. 分位数模型 + 折内再切校准车集 ----------------
    lo = np.full(n, np.nan)
    med = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    q_cqr = np.full(n, np.nan)      # 每折的 CQR 修正量
    n_calib = []
    for fi, (tr, te) in enumerate(folds):
        tr_cars = np.unique(cars[tr])
        rng = np.random.default_rng(1000 + fi)
        n_cal = max(1, int(round(len(tr_cars) * 0.25)))
        cal_cars = set(rng.choice(tr_cars, size=n_cal, replace=False).tolist())
        is_cal = np.array([c in cal_cars for c in cars[tr]])
        fit_idx, cal_idx = tr[~is_cal], tr[is_cal]
        n_calib.append(int(len(cal_idx)))

        cal_pred = {}
        for a, store in ((0.05, lo), (0.5, med), (0.95, hi)):
            mdl = build_model("quantile", alpha=a)
            mdl.fit(X.iloc[fit_idx], y[fit_idx])
            store[te] = mdl.predict(X.iloc[te])
            cal_pred[a] = mdl.predict(X.iloc[cal_idx])

        # CQR 修正量：校准车集上的 conformity score 的 90% 分位
        e = np.maximum(cal_pred[0.05] - y[cal_idx], y[cal_idx] - cal_pred[0.95])
        q_val = float(np.quantile(e, 0.9, method="higher"))
        q_cqr[te] = q_val
        print(f"  fold {fi + 1}/{len(folds)}  校准车 {n_cal} 辆 / {len(cal_idx)} 段  "
              f"Q={q_val:.3f} Ah", flush=True)

    # ---------------- 2. 四种区间 ----------------
    q_int = interval_metrics(y, lo, hi)
    cqr_int = interval_metrics(y, lo - q_cqr, hi + q_cqr)

    # 复用 07 的 CV+ / NC 结果（同折同模型，保证可比）
    prev = {}
    pth = os.path.join(cfg["paths"]["metrics_dir"], "uncertainty.json")
    if os.path.exists(pth):
        import json
        with open(pth, encoding="utf-8") as f:
            prev = json.load(f)

    # NC 需要在同一批 OOF 上重算才能做条件覆盖，这里按 07 的口径复算
    resid = y - med
    half_cv = np.full(n, np.nan)
    scale = np.maximum((hi - lo) / 2.0, 1e-6)
    score = np.abs(resid) / scale
    q_nc = np.full(n, np.nan)
    for _fi, (_tr, te) in enumerate(folds):
        mask = np.ones(n, bool)
        mask[te] = False
        half_cv[te] = np.percentile(np.abs(resid[mask]), 90)
        q_nc[te] = np.percentile(score[mask], 90)
    half_nc = q_nc * scale

    methods = {
        "Q_quantile": (lo, hi),
        "CVplus": (med - half_cv, med + half_cv),
        "NC": (med - half_nc, med + half_nc),
        "CQR": (lo - q_cqr, hi + q_cqr),
    }

    # ---------------- 3. 条件覆盖审计 ----------------
    span_g = pd.cut(df["d_soc"].abs(), bins=SPAN_BINS)
    qual_g = pd.cut(df["quality"], bins=[0.6, 0.8, 0.9, 1.01])
    data_g = df["dataset"]
    audit = {}
    for name, (l, h) in methods.items():
        m = interval_metrics(y, l, h)
        audit[name] = {
            "overall": m,
            "soh_pp_width": m["width_mean"] / ref * 100.0,
            "by_span": cond_coverage(y, l, h, span_g),
            "by_quality": cond_coverage(y, l, h, qual_g),
            "by_dataset": cond_coverage(y, l, h, data_g, min_n=50),
        }
        print(f"{name:11s} 覆盖={m['coverage']:.3f}  宽度={m['width_mean']:.3f} Ah "
              f"({m['width_mean'] / ref * 100:.2f} pp)  "
              f"跨度最差箱偏离={audit[name]['by_span']['worst_dev'] * 100:.1f}pp"
              f" ({audit[name]['by_span']['worst_bin']})", flush=True)

    point = regression_metrics(y, med)
    res = {
        "n_sessions": int(n), "n_cars": int(df["car"].nunique()),
        "point_rmse_Ah": point["rmse"], "point_soh_rmse_pp": point["rmse"] / ref * 100,
        "nominal_coverage": 0.9,
        "n_calib_sessions_per_fold": n_calib,
        "cqr_shift_Ah": {"mean": float(np.nanmean(q_cqr)), "min": float(np.nanmin(q_cqr)),
                         "max": float(np.nanmax(q_cqr))},
        "methods": audit,
        "reference_prev_run": {k: prev.get(k) for k in
                               ("quantile_interval", "cvplus_interval", "normalized_conformal")},
        "note": ("CQR 的校准集是每折训练集内部按车再切 25% 的车辆，"
                 "与测试折车辆互斥；Q 为 conformity score 的 90% 分位，加在区间两端（保不对称性）。"),
    }
    save_json(res, "cqr")

    pd.DataFrame({"car": df["car"], "segment": df["segment"], "capacity": y,
                  "d_soc": df["d_soc"].abs(), "quality": df["quality"],
                  "dataset": df["dataset"],
                  "lo_q": lo, "med": med, "hi_q": hi,
                  "lo_cqr": lo - q_cqr, "hi_cqr": hi + q_cqr,
                  "lo_cv": med - half_cv, "hi_cv": med + half_cv,
                  "lo_nc": med - half_nc, "hi_nc": med + half_nc}).to_parquet(
        os.path.join(cfg["paths"]["cache_dir"], "cqr_oof.parquet"), index=False)

    # ---------------- 4. 图 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    styles = [("Q_quantile", "Quantile Q", C_RED, "o"), ("CVplus", "CV+", C_GRAY, "s"),
              ("NC", "Normalized conformal", C_ACC, "^"), ("CQR", "CQR", C_MAIN, "D")]
    labels = [f"{b.left:.0f}-{b.right:.0f}" for b in span_g.cat.categories]
    for key, label, col, mk in styles:
        l, h = methods[key]
        inside = ((y >= l) & (y <= h)).astype(float)
        cov = pd.DataFrame({"c": inside, "g": span_g}).groupby("g", observed=True)["c"].mean()
        wid = pd.DataFrame({"w": h - l, "g": span_g}).groupby("g", observed=True)["w"].median()
        axes[0].plot(range(len(cov)), cov.values, marker=mk, ms=4.5, lw=1.5,
                     color=col, label=label)
        axes[1].plot(range(len(wid)), wid.values, marker=mk, ms=4.5, lw=1.5,
                     color=col, label=label)

    axes[0].axhline(0.9, color=C_GRAY, ls="--", lw=1.1)
    axes[0].set_ylim(0.6, 1.0)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("(a) Coverage by SOC span", pad=8)
    axes[1].set_ylabel("Median interval width (Ah)")
    axes[1].set_title("(b) Width by SOC span", pad=8)
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("SOC span (%)")
    axes[0].legend(loc="lower right", fontsize=8)

    fig.suptitle("Conditional coverage audit: the 90% interval on short snippets",
                 fontsize=10.5, y=1.03)
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig22_条件覆盖率审计.png"))
    print("完成")


if __name__ == "__main__":
    main()
