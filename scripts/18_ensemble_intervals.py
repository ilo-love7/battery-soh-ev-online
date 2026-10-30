# -*- coding: utf-8 -*-
"""18 D3 深化（二）：深度集成 vs 分位数回归，以及区间宽度的下限在哪。

脚本 13 已经证明 CQR 优于 CV+：同样宽度下条件覆盖最差箱只偏 1.8 pp（CV+ 偏 15.1 pp）。
深化文档 D3 要求"分位数回归 / MC Dropout / 深度集成 三选二对比"。本脚本补上第三档：
**深度集成**（7 个不同初始化的 MLP），并回答一个更根本的问题 ——

    90% 区间宽 2.45 Ah，到底是方法不够好，还是标签噪声决定了它不可能更窄？

四种区间（同一套按车分组 5 折、同一折内校准集）：
  quantile   分位数回归 P5/P95（基线，未校准）
  cqr        conformalized quantile regression（脚本 13 的最优方法）
  ens        7 个 MLP 的 5%/95% 分位（未校准）
  ens_cal    集成区间 + conformal 校准（用同一批校准车）

诊断（决定宽度的上限）：
  - 用特征预测 |残差|，看 R²。如果接近 0，说明残差里几乎没有可学的结构，
    宽度就由标签噪声决定，换方法只是重新分配同样的不确定性。
  - 两模型不一致度（集成标准差）与真实误差的相关性：如果很弱，
    说明模型不确定性解释不了误差，区间只能靠校准量撑开。

输出
  results/metrics/ensemble_intervals.json
  results/figures/fig31_区间宽度覆盖率权衡.png
  results/figures/fig32_集成区间的条件覆盖.png
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import group_kfold_by_car, interval_metrics  # noqa: E402
from evsoh.features import active_features, feature_columns  # noqa: E402
from evsoh.models import build_model  # noqa: E402
from evsoh.nn import fit_torch, make_mlp, prep_tabular  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_PURPLE, C_RED, plt, savefig  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402

ALPHAS = (0.05, 0.5, 0.95)
NOMINAL = 0.90
SPAN_BINS = [20, 30, 40, 50, 60, 70, 100]
SPAN_LABELS = ["20-30", "30-40", "40-50", "50-60", "60-70", "70-100"]
N_ENS = 7                     # 集成规模：7 个种子的经验折中（再多收益递减）
CAL_FRAC = 0.25               # 折内校准车占比，与脚本 13 保持一致


def cond_cov(y, lo, hi, keys, min_n: int = 80) -> dict:
    """逐箱条件覆盖：返回各箱覆盖率与"最差箱偏离名义值"的绝对值（百分点）。"""
    bins, devs = {}, []
    for k in sorted(set(keys)):
        m = keys == k
        if m.sum() < min_n:
            continue
        cov = float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))
        bins[str(k)] = {"n": int(m.sum()), "coverage": cov,
                        "width": float(np.mean(hi[m] - lo[m]))}
        devs.append(abs(cov - NOMINAL) * 100.0)
    return {"bins": bins, "worst_dev_pp": float(max(devs)) if devs else float("nan"),
            "min_cov": float(min(v["coverage"] for v in bins.values())) if bins else float("nan"),
            "max_cov": float(max(v["coverage"] for v in bins.values())) if bins else float("nan")}


def span_key(d_soc: np.ndarray) -> np.ndarray:
    k = np.full(d_soc.shape, "", dtype=object)
    for lo, hi, lab in zip(SPAN_BINS[:-1], SPAN_BINS[1:], SPAN_LABELS):
        k[(d_soc >= lo) & (d_soc < hi)] = lab
    return k.astype(str)


def main() -> None:
    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, lab = add_soh_labels(ses, cfg)
    feats = active_features(df, feature_columns())
    df = df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)
    print(f"样本 {len(df)} 会话 / {df['car'].nunique()} 车；特征 {len(feats)}", flush=True)

    X = df[feats].astype(float)
    y = df["capacity"].to_numpy(dtype=float)
    cars = df["car"].to_numpy()
    d_soc = df["d_soc"].to_numpy(dtype=float)
    ref_mean = float(df["bol_ref"].mean())
    skey = span_key(d_soc)

    folds = list(group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                    seed=cfg["splits"]["random_state"]))
    n = len(df)
    lo_q = np.full(n, np.nan)
    hi_q = np.full(n, np.nan)
    med_q = np.full(n, np.nan)
    lo_c = np.full(n, np.nan)
    hi_c = np.full(n, np.nan)
    lo_e = np.full(n, np.nan)
    hi_e = np.full(n, np.nan)
    med_e = np.full(n, np.nan)
    lo_ec = np.full(n, np.nan)
    hi_ec = np.full(n, np.nan)
    shift_log = []
    t0 = time.time()

    for fi, (tr, te) in enumerate(folds):
        tr_cars = np.unique(cars[tr])
        rng = np.random.default_rng(1000 + fi)
        n_cal = max(1, int(round(len(tr_cars) * CAL_FRAC)))
        cal_cars = set(rng.choice(tr_cars, size=n_cal, replace=False).tolist())
        is_cal = np.array([c in cal_cars for c in cars[tr]])
        fit_idx, cal_idx = tr[~is_cal], tr[is_cal]

        # ---- 分位数回归 + CQR ----
        cal_pred = {}
        for a in ALPHAS:
            mdl = build_model("quantile", alpha=a)
            mdl.fit(X.iloc[fit_idx], y[fit_idx])
            p_te = mdl.predict(X.iloc[te])
            cal_pred[a] = mdl.predict(X.iloc[cal_idx])
            if a == 0.05:
                lo_q[te] = p_te
            elif a == 0.95:
                hi_q[te] = p_te
            else:
                med_q[te] = p_te
        e = np.maximum(cal_pred[0.05] - y[cal_idx], y[cal_idx] - cal_pred[0.95])
        q_val = float(np.quantile(e, NOMINAL, method="higher"))
        lo_c[te] = lo_q[te] - q_val
        hi_c[te] = hi_q[te] + q_val
        shift_log.append({"fold": fi, "q_ah": q_val, "n_cal_cars": len(cal_cars)})

        # ---- 深度集成（7 个种子，同一训练折）----
        A_fit, _ = prep_tabular(X.iloc[fit_idx].to_numpy(dtype=float),
                                X.iloc[fit_idx].to_numpy(dtype=float))
        _, B_te = prep_tabular(X.iloc[fit_idx].to_numpy(dtype=float),
                               X.iloc[te].to_numpy(dtype=float))
        _, B_cal = prep_tabular(X.iloc[fit_idx].to_numpy(dtype=float),
                                X.iloc[cal_idx].to_numpy(dtype=float))
        p_te_ens = np.stack([fit_torch(lambda: make_mlp(A_fit.shape[1]), A_fit, y[fit_idx],
                                       B_te, seed=2000 + fi * 100 + s)
                             for s in range(N_ENS)])
        p_cal_ens = np.stack([fit_torch(lambda: make_mlp(A_fit.shape[1]), A_fit, y[fit_idx],
                                        B_cal, seed=2000 + fi * 100 + s)
                              for s in range(N_ENS)])
        lo_e[te] = np.quantile(p_te_ens, 0.05, axis=0)
        hi_e[te] = np.quantile(p_te_ens, 0.95, axis=0)
        med_e[te] = np.median(p_te_ens, axis=0)

        # ---- 集成 + conformal 校准 ----
        lo_cal = np.quantile(p_cal_ens, 0.05, axis=0)
        hi_cal = np.quantile(p_cal_ens, 0.95, axis=0)
        e2 = np.maximum(lo_cal - y[cal_idx], y[cal_idx] - hi_cal)
        q2 = float(np.quantile(e2, NOMINAL, method="higher"))
        lo_ec[te] = lo_e[te] - q2
        hi_ec[te] = hi_e[te] + q2
        shift_log[-1]["q_ens_ah"] = q2

        print(f"[折 {fi + 1}/5] CQR 修正 {q_val:.3f} Ah；集成修正 {q2:.3f} Ah  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---------------- 汇总 ----------------
    res = {"n_sessions": int(n), "n_cars": int(df["car"].nunique()),
           "nominal_coverage": NOMINAL, "n_ensemble": N_ENS, "label": lab,
           "shifts": shift_log, "methods": {},
           "note": ("四种区间在同一套折划分与同一批校准车上评估；"
                    "ens 的 7 个 MLP 与 ens_cal 共用同一批模型，只有校准不同。")}

    stack = {"quantile": (lo_q, med_q, hi_q), "cqr": (lo_c, med_q, hi_c),
             "ens": (lo_e, med_e, hi_e), "ens_cal": (lo_ec, med_e, hi_ec)}
    for name, (lo, med, hi) in stack.items():
        im = interval_metrics(y, lo, hi)
        res["methods"][name] = {
            **im,
            "soh_pp_width": im["width_mean"] / ref_mean * 100.0,
            "by_span": cond_cov(y, lo, hi, skey),
        }
        print(f"  {name:9s} 覆盖={im['coverage']:.3f}  宽度={im['width_mean']:.3f} Ah "
              f"({im['width_mean'] / ref_mean * 100:.2f} pp)  "
              f"低于={im['below_rate']:.3f} 高于={im['above_rate']:.3f}  "
              f"最差箱偏离={res['methods'][name]['by_span']['worst_dev_pp']:.1f} pp", flush=True)

    # ---------------- 诊断：宽度下限 ----------------
    resid = med_q - y
    abs_r = np.abs(resid)
    A_all, _ = prep_tabular(X.to_numpy(dtype=float), X.to_numpy(dtype=float))
    # 用特征预测 |残差|：R² 高说明误差有结构，可压缩；接近 0 说明是噪声
    pred_absr = fit_torch(lambda: make_mlp(A_all.shape[1], width=64), A_all, abs_r, A_all,
                          seed=777, epochs=40)
    ss_res = float(np.sum((abs_r - pred_absr) ** 2))
    ss_tot = float(np.sum((abs_r - abs_r.mean()) ** 2))
    r2_absr = 1.0 - ss_res / ss_tot

    # 集成离散度 vs 真实误差的相关性
    corr_sd = float("nan")
    if np.isfinite(lo_e).all():
        spread = hi_e - lo_e
        corr_sd = float(np.corrcoef(spread, abs_r)[0, 1])

    res["diagnostics"] = {
        "r2_predict_abs_residual": float(r2_absr),
        "corr_interval_width_vs_abs_error": corr_sd,
        "abs_residual_median": float(np.median(abs_r)),
        "abs_residual_p90": float(np.percentile(abs_r, 90)),
        "note": ("r2_predict_abs_residual 用同一批特征预测 |残差| 的 R²。"
                 "接近 0 意味着误差里几乎没有可学结构，宽度由标签噪声决定。"),
    }
    print(f"\n诊断：|残差| 的可预测 R² = {r2_absr:.4f}；"
          f"区间宽度与 |误差| 的相关 = {corr_sd:.3f}", flush=True)

    save_json(res, "ensemble_intervals")
    pd.DataFrame({"car": cars, "d_soc": d_soc, "capacity": y,
                  "lo_quantile": lo_q, "hi_quantile": hi_q,
                  "lo_cqr": lo_c, "hi_cqr": hi_c,
                  "lo_ens": lo_e, "hi_ens": hi_e,
                  "lo_ens_cal": lo_ec, "hi_ens_cal": hi_ec}).to_parquet(
        os.path.join(cfg["paths"]["cache_dir"], "interval_oof.parquet"), index=False)

    fig_tradeoff(res, os.path.join(cfg["paths"]["figures_dir"],
                                   "fig31_区间宽度覆盖率权衡.png"))
    fig_cond(res, os.path.join(cfg["paths"]["figures_dir"],
                               "fig32_集成区间的条件覆盖.png"))
    print(f"\n完成，用时 {time.time() - t0:.0f}s", flush=True)


def fig_tradeoff(res: dict, path: str) -> None:
    """图 31：宽度-覆盖率平面。理想点是"宽度小 + 覆盖率高"，名义 90% 线之上为过覆盖。"""
    meta = [("quantile", "Quantile regression", C_GRAY, "o", (8, 5), "left"),
            ("cqr", "CQR", C_MAIN, "D", (0, -15), "center"),
            ("ens", "Deep ensemble (7)", C_ACC, "s", (8, 5), "left"),
            ("ens_cal", "Ensemble + conformal", C_PURPLE, "^", (-2, 9), "center")]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for key, label, col, mk, off, ha in meta:
        m = res["methods"][key]
        x, yv = m["width_mean"], m["coverage"]
        ax.scatter([x], [yv], s=70, color=col, marker=mk, zorder=3, label=label)
        ax.annotate(f"{yv:.3f}", (x, yv), textcoords="offset points", xytext=off,
                    fontsize=8.2, color=col, ha=ha)
    ax.axhline(res["nominal_coverage"], color=C_RED, ls="--", lw=1.2)
    ax.text(0.02, res["nominal_coverage"] + 0.006, "nominal 90%",
            transform=ax.get_yaxis_transform(), fontsize=8, color=C_RED, va="bottom")
    ax.set_xlabel("Mean interval width (Ah)")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.0, 2.85)
    ax.set_ylim(0.15, 0.98)
    ax.set_title("Every method lands on the same frontier:\n"
                 "wider is the only way to cover more", pad=8, fontsize=9.5)
    ax.legend(loc="center right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    savefig(fig, path)


def fig_cond(res: dict, path: str) -> None:
    """图 32：条件覆盖（按 SOC 跨度）。CQR 与集成校准版的对比是核心。"""
    meta = [("quantile", "Quantile regression", C_GRAY, "o"),
            ("cqr", "CQR", C_MAIN, "D"),
            ("ens_cal", "Ensemble + conformal", C_PURPLE, "^")]
    bins = [b for b in SPAN_LABELS
            if all(b in res["methods"][k]["by_span"]["bins"] for k, _l, _c, _m in meta)]
    xs = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    for key, label, col, mk in meta:
        vals = [res["methods"][key]["by_span"]["bins"][b]["coverage"] for b in bins]
        ax.plot(xs, vals, marker=mk, ms=5, lw=1.7, color=col, label=label)
    ax.axhline(res["nominal_coverage"], color=C_RED, ls="--", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b}%" for b in bins], fontsize=8.5)
    ax.set_xlabel("SOC span of the charging snippet")
    ax.set_ylabel("Coverage of the 90% interval")
    ax.set_title("Only CQR holds coverage flat across snippet length:\n"
                 "a constant conformal margin still fails on short snippets",
                 pad=8, fontsize=9.5)
    ax.set_ylim(0.76, 0.94)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3,
              fontsize=8, frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    savefig(fig, path)


def diag_only() -> None:
    """只重算"残差是否可预测"这条诊断，并写回 JSON。

    原实现是在全数据上拟合再评估（样本内），R² 会明显偏高。这里改成按车分组 5 折
    交叉拟合：每折用 4/5 的车训练 |残差| 预测器，在留出的 1/5 车上评估。
    这个数字才是"误差里有多少可学结构"的无偏估计。
    """
    import json

    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, _ = add_soh_labels(ses, cfg)
    feats = active_features(df, feature_columns())
    df = df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)
    oof = pd.read_parquet(os.path.join(cfg["paths"]["metrics_dir"], "oof_predictions.parquet"))

    X = df[feats].astype(float)
    y = df["capacity"].to_numpy(dtype=float)
    cars = df["car"].to_numpy()
    resid = oof["pred_xgb"].to_numpy(dtype=float) - y
    abs_r = np.abs(resid)

    folds = list(group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                    seed=cfg["splits"]["random_state"]))
    p_cv = np.full(len(y), np.nan)
    for fi, (tr, te) in enumerate(folds):
        A, B = prep_tabular(X.iloc[tr].to_numpy(dtype=float),
                            X.iloc[te].to_numpy(dtype=float))
        p_cv[te] = fit_torch(lambda: make_mlp(A.shape[1], width=64), A, abs_r[tr], B,
                             seed=900 + fi, epochs=40)
        print(f"  折 {fi + 1}/5 完成", flush=True)

    ss_res = float(np.sum((abs_r - p_cv) ** 2))
    ss_tot = float(np.sum((abs_r - abs_r.mean()) ** 2))
    r2_cv = 1.0 - ss_res / ss_tot

    path = os.path.join(cfg["paths"]["metrics_dir"], "ensemble_intervals.json")
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    res["diagnostics"]["r2_predict_abs_residual_insample"] = \
        res["diagnostics"].get("r2_predict_abs_residual")
    res["diagnostics"]["r2_predict_abs_residual"] = float(r2_cv)
    res["diagnostics"]["note"] = (
        "r2_predict_abs_residual 为按车分组 5 折交叉拟合（无偏）；"
        "insample 版本仅作对照，偏高。接近 0 意味着误差里几乎没有可学结构，"
        "宽度由标签噪声决定。")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"交叉拟合 R² = {r2_cv:.4f}（样本内 {res['diagnostics']['r2_predict_abs_residual_insample']:.4f}）")


if __name__ == "__main__":
    if "--replot" in sys.argv:          # 只重画图，复用已落盘的结果
        import json

        _cfg = load_config()
        with open(os.path.join(_cfg["paths"]["metrics_dir"], "ensemble_intervals.json"),
                  encoding="utf-8") as f:
            _res = json.load(f)
        fig_tradeoff(_res, os.path.join(_cfg["paths"]["figures_dir"],
                                        "fig31_区间宽度覆盖率权衡.png"))
        fig_cond(_res, os.path.join(_cfg["paths"]["figures_dir"],
                                    "fig32_集成区间的条件覆盖.png"))
        print("重绘完成")
    elif "--diag-only" in sys.argv:
        diag_only()
    else:
        main()
