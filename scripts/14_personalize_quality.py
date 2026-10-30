# -*- coding: utf-8 -*-
"""14 D4/D5 深化：个性化策略升级 + 主动选样 + 质量门控 + 鲁棒特征。

D4（冷启动个性化）上一版只做了"偏差/仿射两参数校正"，并发现同分布 11.8%、时序 2.9%。
两个没回答的问题：
  ① 换成"每个车一个小岭回归头"（多几个参数）能否把时序收益拉起来？
  ② k 段是"最早 k 段"，还是"挑质量×跨度最高的 k 段"？—— 后者是部署时真正能做的选择。

D5（片段质量）上一版只做了质量分与分层报告。补两个实验：
  ③ **质量门控**：模型不动，只决定"低于某质量分的片段拒收"，看精度-覆盖率怎么换。
     这是部署侧唯一能落地的用法（训练时不该丢数据）。
  ④ **鲁棒特征**：剔除与倍率/温度强相关的特征后重训，看整体精度损失多少、
     换来的是不是低温/高温段误差更平。

输出
    results/metrics/personalize_v2.json
    results/metrics/quality_gate.json
    results/metrics/robust_features.json
    results/figures/fig23_个性化策略与质量门控.png
    results/figures/fig24_鲁棒特征代价.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import group_kfold_by_car, regression_metrics  # noqa: E402
from evsoh.features import active_features, feature_columns  # noqa: E402
from evsoh.models import build_model  # noqa: E402
from evsoh.personalize import fit_affine  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_GREEN, C_MAIN, C_PURPLE, C_RED, plt, savefig  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402

K_LIST = [0, 1, 2, 3, 5, 8, 12, 20]
EVAL_FRAC = 0.5
MIN_SESSIONS = 20
LAM = 5.0


# ---------------------------------------------------------------- 个性化
def build_plan(cars: np.ndarray, order: np.ndarray, split: str = "chrono"):
    """每车切"校准池 / 固定评估集"；返回 {car: (pool_idx, eval_idx)}。"""
    plan = {}
    for c in np.unique(cars):
        idx = np.where(cars == c)[0]
        idx = idx[np.argsort(order[idx])]
        if split == "random":
            idx = idx[np.random.default_rng(0).permutation(idx.size)]
        if idx.size < MIN_SESSIONS:
            continue
        n_eval = max(1, int(round(idx.size * EVAL_FRAC)))
        plan[int(c)] = (idx[:-n_eval], idx[-n_eval:])
    return plan


def _pick_k(pool: np.ndarray, k: int, policy: str, quality: np.ndarray, span: np.ndarray) -> np.ndarray:
    """从校准池里挑 k 段：时间序（部署默认）或按 质量×跨度 主动挑（部署可做的选择）。"""
    if policy == "active":
        score = quality[pool] * (span[pool] / 100.0)
        return pool[np.argsort(-score)[:k]]
    return pool[:k]


def curve(pred_zero: np.ndarray, y: np.ndarray, plan: dict, policy: str,
          quality: np.ndarray, span: np.ndarray) -> list:
    """一条收敛曲线。policy: bias / affine / ridge / active。"""
    out = []
    for k in K_LIST:
        maes = []
        for c, (pool, ev) in plan.items():
            if k > len(pool):
                continue
            if k == 0:
                p = pred_zero[ev]
            else:
                sel = _pick_k(pool, k, policy, quality, span)
                if policy in ("bias", "active"):
                    b = float(np.sum(y[sel] - pred_zero[sel]) / (len(sel) + LAM))
                    p = pred_zero[ev] + b
                elif policy == "affine":
                    a, b = fit_affine(pred_zero[sel], y[sel], LAM)
                    p = a * pred_zero[ev] + b
                else:  # ridge：残差小头，特征 = 跨度/质量/预测偏离
                    Z = np.column_stack([
                        (pred_zero[sel] - pred_zero.mean()) / 5.0,
                        (span[sel] - 50.0) / 25.0,
                        quality[sel] - 0.9,
                    ])
                    r = Ridge(alpha=LAM, fit_intercept=True).fit(Z, y[sel] - pred_zero[sel])
                    Ze = np.column_stack([
                        (pred_zero[ev] - pred_zero.mean()) / 5.0,
                        (span[ev] - 50.0) / 25.0,
                        quality[ev] - 0.9,
                    ])
                    p = pred_zero[ev] + r.predict(Ze)
            maes.append(float(np.mean(np.abs(p - y[ev]))))
        if maes:
            out.append({"k": int(k), "n_cars": len(maes),
                        "car_mae_mean": float(np.mean(maes)),
                        "car_mae_p90": float(np.percentile(maes, 90))})
    return out


def personalization(df: pd.DataFrame, pred: np.ndarray, y: np.ndarray) -> dict:
    cars = df["car"].values
    order = df["mileage"].values.astype(float)
    quality = df["quality"].values.astype(float)
    span = np.abs(df["d_soc"].values.astype(float))
    res = {"k_list": K_LIST, "lam": LAM, "eval_frac": EVAL_FRAC, "curves": {}}
    for split in ("chrono", "random"):
        plan = build_plan(cars, order, split)
        for policy in ("bias", "affine", "ridge", "active"):
            res["curves"][f"{split}_{policy}"] = curve(pred, y, plan, policy, quality, span)
    for key, c in res["curves"].items():
        base, last = c[0]["car_mae_mean"], c[-1]["car_mae_mean"]
        print(f"  {key:16s} k=0 {base:.4f} → k={c[-1]['k']} {last:.4f} "
              f"({(base - last) / base * 100:+.1f}%)", flush=True)
    return res


# ---------------------------------------------------------------- 质量门控
def quality_gate(df: pd.DataFrame, pred: np.ndarray, y: np.ndarray) -> dict:
    """模型不动，只按质量分拒收片段：精度 vs 保留比例。"""
    rows = []
    n0 = len(df)
    for thr in (0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        m = df["quality"].values >= thr
        if m.sum() < 500:
            continue
        err = np.abs(pred[m] - y[m])
        rows.append({
            "threshold": float(thr),
            "retained_frac": float(m.sum() / n0),
            "n": int(m.sum()),
            "mae": float(err.mean()),
            "p90_abs_err": float(np.percentile(err, 90)),
            "mae_pp": float(err.mean() / df["bol_ref"].values[m].mean() * 100),
        })
        print(f"  quality ≥ {thr:.2f}  保留 {m.sum() / n0 * 100:5.1f}%  "
              f"MAE {rows[-1]['mae']:.4f} Ah ({rows[-1]['mae_pp']:.2f} pp)  "
              f"P90 {rows[-1]['p90_abs_err']:.3f}", flush=True)
    return {"rows": rows}


# ---------------------------------------------------------------- 鲁棒特征
def robust_subset(df: pd.DataFrame, feats: list, thr: float = 0.35) -> tuple:
    """剔除与倍率/温度强相关的特征（|Spearman| ≥ thr）。"""
    i_abs = df["i_abs_mean"].values.astype(float)
    t_mean = df["t_mean"].values.astype(float)
    rho = {}
    for c in feats:
        v = df[c].values.astype(float)
        m = np.isfinite(v)
        if m.sum() < 500:
            continue
        s = pd.Series(v[m])
        rho[c] = (float(s.corr(pd.Series(i_abs[m]), method="spearman")),
                  float(s.corr(pd.Series(t_mean[m]), method="spearman")))
    keep = [c for c in feats if c in rho and max(abs(rho[c][0]), abs(rho[c][1])) < thr]
    drop = [c for c in feats if c in rho and c not in keep]
    return keep, drop, rho


def run_p2(df: pd.DataFrame, cols: list, cfg: dict, y: np.ndarray, cars: np.ndarray, folds) -> np.ndarray:
    X = df[cols].astype(float)
    pred = np.full(len(df), np.nan)
    for tr, te in folds:
        mdl = build_model("xgb", cfg["models"])
        mdl.fit(X.iloc[tr], y[tr])
        pred[te] = mdl.predict(X.iloc[te])
    return pred


def temp_strata(df: pd.DataFrame, pred: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    t = df["t_mean"].values.astype(float)
    for name, m in (("cold <15C", t < 15), ("mid 15-30C", (t >= 15) & (t < 30)),
                    ("hot >=30C", t >= 30)):
        if m.sum() >= 200:
            out[name] = {"n": int(m.sum()),
                         "mae": float(np.mean(np.abs(pred[m] - y[m]))),
                         "mae_pp": float(np.mean(np.abs(pred[m] - y[m])) /
                                         df["bol_ref"].values[m].mean() * 100)}
    return out


def main() -> None:
    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, _ = add_soh_labels(ses, cfg)
    df = df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)
    feats = active_features(df, feature_columns())
    y = df["capacity"].values.astype(float)
    cars = df["car"].values
    oof = pd.read_parquet(os.path.join(cfg["paths"]["metrics_dir"], "oof_predictions.parquet"))
    assert len(oof) == len(df), "OOF 与特征表行数不一致"
    pred = oof["pred_xgb"].values.astype(float)
    print(f"样本 {len(df)} 会话 / {df['car'].nunique()} 车", flush=True)

    reuse = os.environ.get("REUSE_PERS") == "1"
    if reuse and all(os.path.exists(os.path.join(cfg["paths"]["metrics_dir"], f"{n}.json"))
                     for n in ("personalize_v2", "quality_gate", "robust_features")):
        import json
        with open(os.path.join(cfg["paths"]["metrics_dir"], "personalize_v2.json"), encoding="utf-8") as f:
            pers = json.load(f)
        with open(os.path.join(cfg["paths"]["metrics_dir"], "quality_gate.json"), encoding="utf-8") as f:
            gate = json.load(f)
        with open(os.path.join(cfg["paths"]["metrics_dir"], "robust_features.json"), encoding="utf-8") as f:
            rob = json.load(f)
        print("复用已有实验结果，只重绘图", flush=True)
    else:
        print("[1/3] 个性化策略对比（bias / affine / ridge / active）…", flush=True)
        pers = personalization(df, pred, y)
        save_json(pers, "personalize_v2")

        print("[2/3] 质量门控（模型不动，只拒收低质片段）…", flush=True)
        gate = quality_gate(df, pred, y)
        save_json(gate, "quality_gate")

        print("[3/3] 鲁棒特征（剔除与倍率/温度强相关）…", flush=True)
        keep, drop, rho = robust_subset(df, feats, thr=0.35)
        folds = list(group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                        seed=cfg["splits"]["random_state"]))
        p_full = pred
        p_rob = run_p2(df, keep, cfg, y, cars, folds)
        m_full = regression_metrics(y, p_full)
        m_rob = regression_metrics(y, p_rob)
        rob = {
            "threshold_abs_rho": 0.35,
            "n_features_full": len(feats), "n_features_robust": len(keep),
            "dropped": sorted(drop)[:40], "n_dropped": len(drop),
            "full": {**m_full, "by_temp": temp_strata(df, p_full, y)},
            "robust": {**m_rob, "by_temp": temp_strata(df, p_rob, y)},
            "note": ("鲁棒子集剔除与 i_abs_mean 或 t_mean 的 |Spearman| ≥ 0.35 的特征；"
                     "full 用 05 的 P2 OOF，robust 用同一折划分重训，可直接比较。"),
        }
        print(f"  全量 {len(feats)} 特征 RMSE={m_full['rmse']:.4f}；"
              f"鲁棒 {len(keep)} 特征 RMSE={m_rob['rmse']:.4f}"
              f"（{(m_rob['rmse'] / m_full['rmse'] - 1) * 100:+.1f}%）", flush=True)
        for k, v in rob["full"]["by_temp"].items():
            print(f"    [全量] {k:12s} MAE {v['mae']:.3f} Ah")
        for k, v in rob["robust"]["by_temp"].items():
            print(f"    [鲁棒] {k:12s} MAE {v['mae']:.3f} Ah")
        save_json(rob, "robust_features")
    n_full, n_rob = rob["n_features_full"], rob["n_features_robust"]

    # ---------------- 图 23：个性化策略 + 质量门控 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(9.9, 3.9))
    styles = [("chrono_bias", "bias only", C_GRAY, "o"),
              ("chrono_affine", "affine a·p+b", C_ACC, "s"),
              ("chrono_ridge", "ridge head", C_PURPLE, "^"),
              ("chrono_active", "bias + pick best k", C_MAIN, "D")]
    Y_LO, Y_HI = -12.0, 12.0
    for key, label, col, mk in styles:
        c = pers["curves"][key]
        base = c[0]["car_mae_mean"]
        ks = np.array([r["k"] for r in c], dtype=float)
        ys = np.array([(base - r["car_mae_mean"]) / base * 100 for r in c])
        vis = ys >= Y_LO
        axes[0].plot(ks[vis], ys[vis], marker=mk, ms=4.5, lw=1.6, color=col, label=label)
        if (~vis).any():                       # 出界点只在首个出界 k 处画一个空心标记
            axes[0].plot(ks[~vis][:1], np.full(1, Y_LO + 0.7), marker=mk, ms=4.5,
                         color=col, ls="none", mfc="white")
        axes[0].annotate(f"{ys[-1]:+.1f}%", (ks[-1], ys[-1]), textcoords="offset points",
                         xytext=(6, 0), fontsize=8.5, color=col, va="center")
    axes[0].axhline(0, color=C_GRAY, ls="--", lw=1)
    axes[0].set_ylim(Y_LO, Y_HI)
    axes[0].text(0.98, 0.04,
                 "extra parameters need data:\naffine −45% / ridge −47% at k=1 (off-scale)",
                 transform=axes[0].transAxes, fontsize=7.4, color=C_GRAY,
                 style="italic", ha="right", va="bottom")
    axes[0].set_xlabel("Labelled sessions used per car (k)")
    axes[0].set_ylabel("MAE reduction vs zero-shot (%)")
    axes[0].set_title("(a) Personalization strategies, time moves forward", pad=8)
    axes[0].set_xlim(-0.8, 23.5)
    axes[0].legend(loc="upper left", fontsize=7.6, framealpha=0.92, ncol=2,
                   columnspacing=1.1, handlelength=1.6, borderpad=0.4)

    rows = gate["rows"]
    xs = [r["retained_frac"] * 100 for r in rows]
    ys = [r["mae"] for r in rows]
    axes[1].plot(xs, ys, marker="o", ms=5, lw=1.7, color=C_MAIN)
    for i, (x, yy, r) in enumerate(zip(xs, ys, rows)):
        if i in (1, 2):        # 100% 与 99.7% 挤在一起，这两档不标
            continue
        label = "no gate" if r["threshold"] == 0.0 else f"≥{r['threshold']:.2f}"
        axes[1].annotate(label, (x, yy), textcoords="offset points",
                         xytext=(-4, 9), ha="center", fontsize=8, color=C_GRAY)
    axes[1].set_xlabel("Sessions retained (%)")
    axes[1].set_ylabel("MAE (Ah)")
    axes[1].set_title("(b) Rejecting low-quality snippets", pad=8)
    axes[1].set_ylim(min(ys) - 0.012, max(ys) + 0.018)
    axes[1].invert_xaxis()

    fig.suptitle("D4/D5: what actually improves deployment", fontsize=10.5, y=1.03)
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig23_个性化策略与质量门控.png"))

    # ---------------- 图 24：鲁棒特征的代价 ----------------
    strata = list(rob["full"]["by_temp"].keys())
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    xs = np.arange(len(strata))
    w = 0.36
    v1 = [rob["full"]["by_temp"][s]["mae"] for s in strata]
    v2 = [rob["robust"]["by_temp"][s]["mae"] for s in strata]
    b1 = ax.bar(xs - w / 2, v1, width=w, color=C_GRAY, label=f"All {n_full} features")
    b2 = ax.bar(xs + w / 2, v2, width=w, color=C_MAIN, label=f"Robust {n_rob} features")
    for bars, vals in ((b1, v1), (b2, v2)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(xs, strata)
    ax.set_ylabel("MAE (Ah)")
    ax.set_title("Dropping C-rate / T-sensitive features costs accuracy in every stratum", pad=8)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig24_鲁棒特征代价.png"))
    print("完成")


if __name__ == "__main__":
    main()
