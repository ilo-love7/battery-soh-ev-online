# -*- coding: utf-8 -*-
"""08 冷启动 + 个性化：误差随该车可用片段数的收敛曲线。

两种评测口径（都要求"只用 k 段做校正、在固定评估集上算误差"）：
  chrono（时序切分，实际部署口径）：每车按里程排序，前 50% 做校正池，后 50% 做评估集。
        新车先零样本上线，之后每积累 k 段就重估一次车级偏差，用于预测"之后"的片段。
  random（同分布切分，个性化潜力上限）：校正池与评估集随机切分，衡量"如果偏差不随时间漂移，
        个性化最多能带来多少收益"。

两者的差距本身就是结论：车级偏差会随老化漂移，所以"越用越准"的幅度受限于偏差的时变性。

输出：results/metrics/coldstart.json + fig14（主图）/ fig15
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.personalize import online_curve  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, plt, savefig  # noqa: E402


def main() -> None:
    cfg = load_config()
    oof = pd.read_parquet(os.path.join(cfg["paths"]["metrics_dir"], "oof_predictions.parquet"))
    oof = oof.dropna(subset=["pred_xgb", "capacity"]).reset_index(drop=True)
    oof = oof.sort_values(["car", "mileage"]).reset_index(drop=True)

    y = oof["capacity"].values.astype(float)
    cars = oof["car"].values
    ref_mean = float(oof["bol_ref"].mean())
    k_list = cfg["coldstart"]["k_list"]
    order = np.arange(len(oof))
    print(f"样本 {len(oof)} 会话 / {oof['car'].nunique()} 车；k 网格 {k_list}")

    variants = [
        ("random_bias", "bias", 1.0, "random", "同分布切分 · 偏差校正（潜力上限）", C_ACC),
        ("chrono_bias", "bias", 5.0, "chrono", "时序切分 · 偏差校正 λ=5（实际部署）", C_MAIN),
        ("chrono_bias_light", "bias", 1.0, "chrono", "时序切分 · 偏差校正 λ=1", C_GRAY),
    ]
    curves, details = {}, {}
    for tag, corr, lam, split, label, _c in variants:
        c, d = online_curve(oof["pred_xgb"].values, y, cars, order=order, k_list=k_list,
                            correction=corr, lam=lam, eval_frac=0.5, min_sessions=20, split=split)
        curves[tag] = c
        details[tag] = d
        print(f"\n[{tag}] {label}")
        for p in c:
            print(f"  k={p['k']:2d}  车均 MAE={p['car_mae_mean']:.3f} Ah "
                  f"→ {p['car_mae_mean']/ref_mean*100:.2f} pp   (n={p['n_cars']})")

    z = curves["chrono_bias"][0]
    res = {
        "n_sessions": int(len(oof)), "n_cars": int(oof["car"].nunique()),
        "ref_capacity_mean_Ah": ref_mean,
        "protocol": {
            "chrono": "每车按里程排序，前 50% 校正池 / 后 50% 固定评估集",
            "random": "每车随机切 50%/50%，衡量个性化潜力上限",
        },
        "zero_shot": {"mae_Ah": z["car_mae_mean"], "mae_soh_pp": z["car_mae_mean"] / ref_mean * 100},
        "curves": curves,
        "summary": {},
    }
    for tag, _corr, _lam, _sp, label, _c in variants:
        c = curves[tag]
        res["summary"][tag] = {
            "label": label,
            "mae_k0": c[0]["car_mae_mean"],
            "mae_k5": next(p["car_mae_mean"] for p in c if p["k"] == 5),
            "mae_kmax": c[-1]["car_mae_mean"], "k_max": c[-1]["k"],
            "gain_pct": 100 * (1 - c[-1]["car_mae_mean"] / c[0]["car_mae_mean"]),
        }
        print(f"\n{tag}: k=0 {c[0]['car_mae_mean']:.3f} → k={c[-1]['k']} "
              f"{c[-1]['car_mae_mean']:.3f} Ah（{res['summary'][tag]['gain_pct']:+.1f}%）")

    # ---------------- 图 14（主图） ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    ax = axes[0]
    for tag, _corr, _lam, _sp, label, color in variants:
        c = curves[tag]
        ax.plot([p["k"] for p in c], [p["car_mae_mean"] for p in c], "o-", color=color,
                label=label, lw=1.8 if tag == "chrono_bias" else 1.2)
    ax.set_xlabel("该车已积累的带标签片段数 k")
    ax.set_ylabel("车均 MAE（Ah）")
    ax.set_title("(a) 误差随片段数收敛（Ah）")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for tag, color, label in [("random_bias", C_ACC, "同分布切分（潜力上限）"),
                              ("chrono_bias", C_MAIN, "时序切分（实际部署）")]:
        c = curves[tag]
        ks = [p["k"] for p in c]
        vals = [p["car_mae_mean"] / ref_mean * 100 for p in c]
        ax.plot(ks, vals, "o-", color=color, label=label)
        ax.fill_between(ks, [p["car_mae_median"] / ref_mean * 100 for p in c],
                        [p["car_mae_p90"] / ref_mean * 100 for p in c],
                        color=color, alpha=0.10)
    for p in curves["chrono_bias"]:
        ax.annotate(f"{p['car_mae_mean']/ref_mean*100:.2f}",
                    (p["k"], p["car_mae_mean"] / ref_mean * 100),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color=C_MAIN)
    ax.set_xlabel("该车已积累的带标签片段数 k")
    ax.set_ylabel("SOH 误差（百分点）")
    ax.set_title("(b) 换算到 SOH 百分点（阴影为 P50–P90 车间分布）")
    ax.legend(fontsize=7.5)
    fig.suptitle("图 14　冷启动 → 个性化：零样本可用、越用越准，但收益受偏差时变性限制", fontsize=12)
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig14_个性化收敛曲线.png"))

    # ---------------- 图 15：逐车改善（同分布口径） ----------------
    det = details["random_bias"]
    dc = pd.DataFrame([{"car": c, **v} for c, v in det.items()]).dropna()
    dc["abs_bias"] = dc["zero_shot_bias"].abs()
    big = dc[dc["abs_bias"] > 0.3]
    small = dc[dc["abs_bias"] <= 0.3]

    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    ax.scatter(small["mae_k0"], small["mae_k5"], s=18, alpha=0.7, color=C_MAIN, edgecolors="none",
               label=f"|零样本偏差| ≤ 0.3 Ah（{len(small)} 辆）")
    ax.scatter(big["mae_k0"], big["mae_k5"], s=26, alpha=0.9, color=C_ACC, edgecolors="none",
               label=f"|零样本偏差| > 0.3 Ah（{len(big)} 辆）")
    lim = [0, max(dc["mae_k0"].max(), dc["mae_k5"].max()) * 1.05]
    ax.plot(lim, lim, "--", color=C_GRAY, lw=1, label="无改善")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("零样本车均 MAE（Ah）")
    ax.set_ylabel("k=5 个性化后车均 MAE（Ah）")
    n_imp = int((dc["mae_k5"] < dc["mae_k0"]).sum())
    ax.set_title(f"图 15　逐车改善（同分布口径）：{n_imp}/{len(dc)} 辆改善，偏差越大收益越大")
    ax.legend(fontsize=8)
    savefig(fig, os.path.join(cfg["paths"]["figures_dir"], "fig15_逐车改善.png"))

    def _grp(g):
        return {"n_cars": int(len(g)), "mae_k0": float(g["mae_k0"].mean()),
                "mae_k5": float(g["mae_k5"].mean()),
                "gain": float((g["mae_k0"] - g["mae_k5"]).mean())}

    res["subgroup_random_split"] = {"all": _grp(dc), "small_bias": _grp(small), "large_bias": _grp(big)}
    print(f"\n同分布口径逐车：{n_imp}/{len(dc)} 辆改善")
    print(f"  小偏差组 {len(small)} 辆: {res['subgroup_random_split']['small_bias']['mae_k0']:.3f} → "
          f"{res['subgroup_random_split']['small_bias']['mae_k5']:.3f} Ah")
    print(f"  大偏差组 {len(big)} 辆: {res['subgroup_random_split']['large_bias']['mae_k0']:.3f} → "
          f"{res['subgroup_random_split']['large_bias']['mae_k5']:.3f} Ah")

    p = save_json(res, "coldstart")
    print(f"写出 {p}")


if __name__ == "__main__":
    main()
