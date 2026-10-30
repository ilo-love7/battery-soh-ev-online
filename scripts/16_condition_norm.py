# -*- coding: utf-8 -*-
"""16 工况归一化：把温度/倍率的影响从"电池状态"里剥出来。

背景（reports §13.6 / §15.5）：
  0–10℃ 片段 MAE 0.94 Ah，30–40℃ 只有 0.41 Ah，差 2.3 倍。
  §15.5 已经证明"删掉与温度/倍率强相关的特征"这条路走不通 —— 敏感性在电池物理里，
  不在某几个特征的锅。报告把下一步写成两句话：
    ① 按温度/倍率把特征或标签修正到参考工况再估容量；
    ② 在训练目标里显式加温度不变性约束。
  本脚本把这两句话各实现一遍，并加上一个更朴素的"残差后校正"作为对照。

四种做法（全部在 P2 按车分组 5 折内完成，任何统计量都不跨折）：
  baseline     原始 XGBoost，工况只作为普通特征（现状）
  resid_ti     残差后校正：折内 3 折交叉拟合得到训练残差，用浅 GBM 拟合
               r ~ g(t_mean, i_abs)，测试时 pred - g(·)。低容量、只捕获大尺度趋势。
  decorr_all   特征去温度化：每个特征对 [1, t, t², i, t·i] 做 ridge 回归，
               用残差替换原特征（t_mean / i_abs 本身保留，作为显式工况输入）
  decorr_shape 只对"电压域形状"组做去温度化，库仑计类保持原样
               （绝对电量是容量信号，不该被抹平）
  target_invar MLP 骨干 + 梯度反转层（GRL），主任务回归容量、对抗任务预测温度分箱。
               目标是让骨干学到的表示"猜不出温度"，即对温度不变。

输出
  results/metrics/condition_norm.json
  results/figures/fig27_工况归一化分层验收.png
  results/figures/fig28_残差工况结构.png
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.evaluate import (group_kfold_by_car, group_metrics,  # noqa: E402
                            regression_metrics)
from evsoh.features import FEATURE_GROUPS, active_features, feature_columns  # noqa: E402
from evsoh.models import build_model  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_PURPLE, C_RED, plt, savefig  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402

# 温度分箱（℃）：与 14 脚本的 cold/mid/hot 三分层兼容，但更细，用于验收
T_EDGES = [-10.0, 10.0, 18.0, 24.0, 30.0, 36.0, 60.0]
T_LABELS = ["<10", "10-18", "18-24", "24-30", "30-36", ">36"]
# 倍率分箱（平均 |I|，A）
I_EDGES = [0.0, 15.0, 17.0, 19.0, 100.0]
I_LABELS = ["<15", "15-17", "17-19", ">19"]

# 对抗训练用的温度粗分箱（类别少一些，梯度更稳）
T_ADV_EDGES = [-10.0, 15.0, 22.0, 28.0, 60.0]


def _labels(edges: list, labels: list, x: np.ndarray) -> np.ndarray:
    """分箱结果转成字符串数组。

    ``Categorical.astype(str)`` 在 pandas 2.x 返回 ndarray、3.x 返回 Index，
    两者都有 ``.to_numpy`` 的只有后者，所以统一走 ``np.asarray``。
    """
    return np.asarray(pd.cut(x, edges, labels=labels).astype(str), dtype=object)


def temp_bin(t: np.ndarray) -> np.ndarray:
    return _labels(T_EDGES, T_LABELS, t)


def cur_bin(i: np.ndarray) -> np.ndarray:
    return _labels(I_EDGES, I_LABELS, i)


def strata_report(y, pred, t, i, min_n: int = 80) -> dict:
    """按温度/倍率分层报 MAE，并给出"最差/最好"的倍数（核心验收指标）。"""
    out = {}
    for name, keys in (("by_temp", temp_bin(t)), ("by_current", cur_bin(i))):
        rows = {}
        for k in sorted(set(keys)):
            m = keys == k
            if m.sum() < min_n:
                continue
            mm = regression_metrics(y[m], pred[m])
            rows[k] = {"n": int(m.sum()), "mae": mm["mae"], "rmse": mm["rmse"],
                       "bias": mm["bias"]}
        vals = [v["mae"] for v in rows.values()]
        out[name] = {
            "bins": rows,
            "mae_min": float(min(vals)) if vals else float("nan"),
            "mae_max": float(max(vals)) if vals else float("nan"),
            "spread_ratio": float(max(vals) / min(vals)) if vals and min(vals) > 0 else float("nan"),
        }
    return out


# ---------------------------------------------------------------- 残差校正
def _resid_model():
    """低容量残差模型：浅树 + 大叶样本量，只捕获大尺度趋势，避免拟合噪声。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_depth=3, max_leaf_nodes=8, min_samples_leaf=300,
        learning_rate=0.05, max_iter=120, l2_regularization=1.0, random_state=42)


def fit_resid_correction(X, y, t, i, cars, cfg, n_inner: int = 3, seed: int = 7):
    """折内交叉拟合残差 → 拟合 g(t, i)。返回 (g, 诊断残差)。"""
    inner = list(group_kfold_by_car(cars, n_splits=n_inner, seed=seed))
    oof = np.full(len(y), np.nan)
    for tr, te in inner:
        m = build_model("xgb", cfg["models"], n_jobs=4)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    resid = y - oof
    Z = np.column_stack([t, i])
    g = _resid_model()
    g.fit(Z, resid)
    return g, resid, oof


# ---------------------------------------------------------------- 特征去温度化
def fit_decorr(X: pd.DataFrame, t: np.ndarray, i: np.ndarray, cols: list):
    """对 cols 中每列拟合 ~ [1, t, t², i, t·i] 的 ridge，返回系数（用于 transform）。

    只用训练折拟合；测试折用同一组系数做减除，因此不泄漏。
    """
    from sklearn.linear_model import Ridge

    Z = np.column_stack([t, t ** 2, i, t * i])
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    sd[sd == 0] = 1.0
    Zs = (Z - mu) / sd
    coefs, intercepts = {}, {}
    for c in cols:
        v = X[c].to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 200:
            coefs[c], intercepts[c] = None, None
            continue
        r = Ridge(alpha=10.0).fit(Zs[ok], v[ok])
        coefs[c], intercepts[c] = r.coef_.copy(), float(r.intercept_)
    return {"mu": mu, "sd": sd, "coefs": coefs, "intercepts": intercepts}


def apply_decorr(X: pd.DataFrame, t: np.ndarray, i: np.ndarray, cols: list, fit: dict):
    """返回新 DataFrame：cols 换成对工况回归的残差，其余列原样。"""
    Z = np.column_stack([t, t ** 2, i, t * i])
    Zs = (Z - fit["mu"]) / fit["sd"]
    out = X.copy()
    for c in cols:
        b = fit["coefs"].get(c)
        if b is None:
            continue
        v = out[c].to_numpy(dtype=float)
        pred = Zs @ b + fit["intercepts"][c]
        out[c] = v - pred          # NaN 会保持 NaN，插补交给 Pipeline
    return out


# ---------------------------------------------------------------- GRL 不变性
def fit_invariant_mlp(Xtr, ytr, ttr, Xte, cfg, seed: int = 42, epochs: int = 60,
                      lam: float = 0.30, use_adv: bool = True, verbose: bool = False):
    """MLP 骨干 + 梯度反转：主头回归容量，对抗头分类温度分箱。

    GRL 在反向传播时把梯度乘 -lam，于是骨干被推向"温度不可分"的表示。
    温度分箱用 4 类（<15 / 15-22 / 22-28 / >28），类别太少会让对抗太容易，
    太多则噪声大，4 类是折中的经验值。

    use_adv=False 时退化为同架构的普通 MLP —— **这个对照是必需的**，
    否则"不变性约束有效"与"MLP 本来就比 XGBoost 好"无法区分。
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 中位数插补 + winsorize + 标准化（统计量只用训练折）
    # winsorize 是必需的：部分库仑计特征有极端离群值，不裁剪会让 MLP 梯度爆炸
    # （冒烟测试里曾出现 RMSE ~1e11）。
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(Xtr, axis=0)
        lo_q = np.nanpercentile(Xtr, 0.5, axis=0)
        hi_q = np.nanpercentile(Xtr, 99.5, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    lo_q = np.where(np.isfinite(lo_q), lo_q, -np.inf)
    hi_q = np.where(np.isfinite(hi_q), hi_q, np.inf)
    A = np.clip(np.where(np.isfinite(Xtr), Xtr, med), lo_q, hi_q)
    B = np.clip(np.where(np.isfinite(Xte), Xte, med), lo_q, hi_q)
    mu, sd = A.mean(axis=0), A.std(axis=0)
    sd[sd == 0] = 1.0
    A = (A - mu) / sd
    B = (B - mu) / sd

    tbin = pd.cut(ttr, T_ADV_EDGES, labels=False).astype(int)
    n_cls = int(tbin.max()) + 1
    ymu, ysd = float(ytr.mean()), float(ytr.std())

    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.alpha * g, None

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(A.shape[1], 128), nn.ReLU(), nn.Dropout(0.20),
                nn.Linear(128, 64), nn.ReLU())
            self.reg = nn.Linear(64, 1)
            self.adv = nn.Linear(64, n_cls)

        def forward(self, x, alpha=0.0):
            h = self.body(x)
            return self.reg(h).squeeze(-1), self.adv(GRL.apply(h, alpha))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1.0e-3, weight_decay=1e-4)
    Xt = torch.tensor(A, dtype=torch.float32)
    yt = torch.tensor((ytr - ymu) / ysd, dtype=torch.float32)
    tt = torch.tensor(tbin, dtype=torch.long)
    n = len(yt)
    bs = 1024
    for ep in range(epochs):
        perm = torch.randperm(n)
        # 前 1/3 轮只做回归，让骨干先站稳，再打开对抗
        alpha = 0.0
        if use_adv:
            alpha = lam * min(1.0, max(0.0, (ep - epochs * 0.30) / (epochs * 0.25)))
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            if idx.numel() < 8:
                continue
            pr, pa = net(Xt[idx], alpha)
            loss = F.mse_loss(pr, yt[idx])
            if use_adv:
                loss = loss + 0.5 * F.cross_entropy(pa, tt[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if verbose and (ep + 1) % 20 == 0:
            print(f"      epoch {ep + 1}/{epochs} loss={loss.item():.4f} alpha={alpha:.2f}",
                  flush=True)

    net.eval()
    with torch.no_grad():
        pred, _ = net(torch.tensor(B, dtype=torch.float32), 0.0)
    return pred.numpy() * ysd + ymu


# ---------------------------------------------------------------- 主流程
def main() -> None:
    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, lab = add_soh_labels(ses, cfg)
    feats = active_features(df, feature_columns())
    df = df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)
    print(f"样本 {len(df)} 会话 / {df['car'].nunique()} 车；特征 {len(feats)} 个", flush=True)

    X = df[feats].astype(float)
    y = df["capacity"].to_numpy(dtype=float)
    cars = df["car"].to_numpy()
    t = df["t_mean"].to_numpy(dtype=float)
    i = df["i_abs_mean"].to_numpy(dtype=float)
    ref_mean = float(df["bol_ref"].mean())

    shape_cols = [c for c in FEATURE_GROUPS["shape"] if c in feats]

    folds = list(group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                    seed=cfg["splits"]["random_state"]))
    if os.environ.get("SMOKE"):          # 冒烟：只跑第一折，验证接口
        folds = folds[:1]
        print("SMOKE 模式：只跑第 1 折", flush=True)
    methods = ["baseline", "resid_ti", "decorr_all", "decorr_shape",
               "mlp_plain", "target_invar"]
    preds = {m: np.full(len(df), np.nan) for m in methods}
    diag = {"resid_gain": [], "decorr_nan": []}
    t0 = time.time()

    for fi, (tr, te) in enumerate(folds):
        print(f"[折 {fi + 1}/5] 训练 {len(tr)} / 测试 {len(te)} 会话", flush=True)

        # ---- baseline ----
        m0 = build_model("xgb", cfg["models"])
        m0.fit(X.iloc[tr], y[tr])
        preds["baseline"][te] = m0.predict(X.iloc[te])

        # ---- resid_ti：折内交叉拟合残差 → 校正 ----
        # 注意符号：resid = y − pred，所以校正应为 pred + g(·)，不是减。
        g, resid, inner_oof = fit_resid_correction(
            X.iloc[tr], y[tr], t[tr], i[tr], cars[tr], cfg)
        corr_te = g.predict(np.column_stack([t[te], i[te]]))
        preds["resid_ti"][te] = preds["baseline"][te] + corr_te
        corr_tr = g.predict(np.column_stack([t[tr], i[tr]]))
        diag["resid_gain"].append({
            "fold": fi,
            "train_mae_before": float(np.mean(np.abs(resid))),
            "train_mae_after": float(np.mean(np.abs(resid + corr_tr))),
            "corr_abs_mean": float(np.mean(np.abs(corr_tr))),
        })
        print(f"    resid_ti: 折内残差 MAE {np.mean(np.abs(resid)):.4f} → "
              f"{np.mean(np.abs(resid + corr_tr)):.4f}（校正量均值 "
              f"{np.mean(np.abs(corr_tr)):.4f} Ah）", flush=True)

        # ---- decorr_all / decorr_shape ----
        for key, cols in (("decorr_all", feats), ("decorr_shape", shape_cols)):
            fit = fit_decorr(X.iloc[tr], t[tr], i[tr], cols)
            Xtr_d = apply_decorr(X.iloc[tr], t[tr], i[tr], cols, fit)
            Xte_d = apply_decorr(X.iloc[te], t[te], i[te], cols, fit)
            md = build_model("xgb", cfg["models"])
            md.fit(Xtr_d, y[tr])
            preds[key][te] = md.predict(Xte_d)

        # ---- mlp_plain / target_invar：同架构，唯一差别是 GRL 是否打开 ----
        # 每个配置用 3 个种子平均，避免把种子噪声误读成"不变性约束的效应"。
        A_tr = X.iloc[tr].to_numpy(dtype=float)
        A_te = X.iloc[te].to_numpy(dtype=float)
        for key, use_adv in (("mlp_plain", False), ("target_invar", True)):
            ps = [fit_invariant_mlp(A_tr, y[tr], t[tr], A_te, cfg,
                                    seed=42 + fi * 10 + s, use_adv=use_adv)
                  for s in range(3)]
            preds[key][te] = np.mean(ps, axis=0)
            print(f"    {key} done（3 seeds, {time.time() - t0:.0f}s）", flush=True)

    # ---------------- 汇总 ----------------
    res = {"n_sessions": int(len(df)), "n_cars": int(df["car"].nunique()),
           "n_features": len(feats), "n_shape_features": len(shape_cols),
           "label": lab, "methods": {}, "resid_diag": diag,
           "temp_edges": T_EDGES, "current_edges": I_EDGES,
           "note": ("全部方法在 P2 按车分组 5 折内完成；resid_ti 的残差模型用折内 3 折交叉拟合，"
                    "decorr 的 ridge 系数与 target_invar 的插补/标准化统计量都只用训练折估计。")}

    base_rmse = None
    for m in methods:
        p = preds[m]
        mm = regression_metrics(y, p)
        gm = group_metrics(pd.DataFrame({"capacity": y, "p": p, "car": cars}),
                           "capacity", "p")
        st = strata_report(y, p, t, i)
        if base_rmse is None:
            base_rmse = mm["rmse"]
        res["methods"][m] = {
            **mm, **gm,
            "soh_rmse_pp": mm["rmse"] / ref_mean * 100.0,
            "delta_vs_baseline_pct": (mm["rmse"] / base_rmse - 1.0) * 100.0,
            **st,
        }
        print(f"  {m:13s} RMSE={mm['rmse']:.4f}  MAE={mm['mae']:.4f}  "
              f"R²={mm['r2']:.4f}  温度 MAE 极差={st['by_temp']['spread_ratio']:.2f}×  "
              f"({st['by_temp']['mae_min']:.3f}–{st['by_temp']['mae_max']:.3f})", flush=True)

    save_json(res, "condition_norm")
    pd.DataFrame({"car": cars, "t_mean": t, "i_abs_mean": i, "capacity": y,
                  **{f"pred_{m}": preds[m] for m in methods}}).to_parquet(
        os.path.join(cfg["paths"]["cache_dir"], "condition_norm_oof.parquet"), index=False)

    # ---------------- 图 27：分层验收 ----------------
    fig_strata(res, methods, os.path.join(cfg["paths"]["figures_dir"],
                                          "fig27_工况归一化分层验收.png"))
    # ---------------- 图 28：残差工况结构 ----------------
    fig_resid(y, preds["baseline"], t, i,
              os.path.join(cfg["paths"]["figures_dir"], "fig28_残差工况结构.png"))

    print(f"\n完成，用时 {time.time() - t0:.0f}s", flush=True)


def fig_strata(res: dict, methods: list, path: str) -> None:
    """图 27：温度分层与倍率分层的 MAE 对比（只画有信息量的三档方法）。"""
    show = [("baseline", "XGBoost baseline", C_GRAY), ("resid_ti", "Residual corr.", C_ACC),
            ("decorr_all", "Decorrelate all", C_PURPLE),
            ("mlp_plain", "MLP, no GRL", C_RED), ("target_invar", "MLP + GRL", C_MAIN)]
    bins = [b for b in T_LABELS if b in res["methods"]["baseline"]["by_temp"]["bins"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    w = 0.16
    xs = np.arange(len(bins))
    for k, (key, label, col) in enumerate(show):
        vals = [res["methods"][key]["by_temp"]["bins"][b]["mae"] for b in bins]
        axes[0].bar(xs + (k - 2.0) * w, vals, width=w, color=col, label=label)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([f"{b}°C" for b in bins], fontsize=8.5)
    axes[0].set_xlabel("Mean temperature of the snippet")
    axes[0].set_ylabel("MAE (Ah)")
    axes[0].set_title("(a) Error by temperature", pad=8)
    axes[0].set_ylim(0, max(
        res["methods"][k]["by_temp"]["bins"][b]["mae"]
        for k, _l, _c in show for b in bins) * 1.42)
    axes[0].legend(loc="upper right", fontsize=7.6, framealpha=0.95, ncol=2,
                   columnspacing=0.9, handlelength=1.4, borderpad=0.35)
    axes[0].grid(axis="x", visible=False)
    axes[0].set_axisbelow(True)

    # 右图：把"温度极差倍数"作为单一指标，越低越接近工况无关
    labels, ratios, cols = [], [], []
    for key, label, col in show:
        labels.append(label)
        ratios.append(res["methods"][key]["by_temp"]["spread_ratio"])
        cols.append(col)
    bars = axes[1].barh(range(len(ratios)), ratios, color=cols, height=0.58)
    axes[1].set_yticks(range(len(ratios)))
    axes[1].set_yticklabels(labels, fontsize=8.5)
    axes[1].invert_yaxis()
    axes[1].axvline(1.0, color=C_RED, ls="--", lw=1.1)
    axes[1].set_xlabel("Worst-to-best temperature MAE ratio (1.0 = condition-free)")
    axes[1].set_title("(b) Condition sensitivity", pad=8)
    axes[1].set_xlim(0, max(ratios) * 1.22)
    for b, v in zip(bars, ratios):
        axes[1].text(v + max(ratios) * 0.02, b.get_y() + b.get_height() / 2,
                     f"{v:.2f}×", va="center", fontsize=8.5)
    axes[1].grid(visible=False)
    fig.tight_layout()
    savefig(fig, path)


def fig_resid(y, pred, t, i, path: str) -> None:
    """图 28：残差随温度/倍率的结构 —— 判断"后校正"是否有可学的信号。"""
    r = pred - y
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))

    # 左：温度分箱的残差中位数与 IQR
    tb = temp_bin(t)
    bins = [b for b in T_LABELS if (tb == b).sum() >= 80]
    med = [np.median(r[tb == b]) for b in bins]
    q1 = [np.percentile(r[tb == b], 25) for b in bins]
    q3 = [np.percentile(r[tb == b], 75) for b in bins]
    xs = np.arange(len(bins))
    axes[0].errorbar(xs, med, yerr=[np.array(med) - np.array(q1), np.array(q3) - np.array(med)],
                     marker="o", ms=5, lw=1.5, capsize=3, color=C_MAIN)
    axes[0].axhline(0, color=C_GRAY, ls="--", lw=1)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([f"{b}°C" for b in bins], fontsize=8.5)
    axes[0].set_xlabel("Mean temperature")
    axes[0].set_ylabel("Residual (pred − true, Ah)")
    axes[0].set_title("(a) Bias drifts with temperature", pad=8)
    axes[0].grid(axis="x", visible=False)

    # 右：倍率分箱
    ib = cur_bin(i)
    bins2 = [b for b in I_LABELS if (ib == b).sum() >= 80]
    med2 = [np.median(r[ib == b]) for b in bins2]
    q12 = [np.percentile(r[ib == b], 25) for b in bins2]
    q32 = [np.percentile(r[ib == b], 75) for b in bins2]
    xs2 = np.arange(len(bins2))
    axes[1].errorbar(xs2, med2, yerr=[np.array(med2) - np.array(q12),
                                      np.array(q32) - np.array(med2)],
                     marker="s", ms=5, lw=1.5, capsize=3, color=C_ACC)
    axes[1].axhline(0, color=C_GRAY, ls="--", lw=1)
    axes[1].set_xticks(xs2)
    axes[1].set_xticklabels([f"{b} A" for b in bins2], fontsize=8.5)
    axes[1].set_xlabel("Mean charge current")
    axes[1].set_ylabel("Residual (Ah)")
    axes[1].set_title("(b) C-rate has a weaker, noisier signature", pad=8)
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, path)


if __name__ == "__main__":
    if "--replot" in sys.argv:          # 只重画图，复用已落盘的结果
        import json

        _cfg = load_config()
        with open(os.path.join(_cfg["paths"]["metrics_dir"], "condition_norm.json"),
                  encoding="utf-8") as f:
            _res = json.load(f)
        _oof = pd.read_parquet(os.path.join(_cfg["paths"]["cache_dir"],
                                            "condition_norm_oof.parquet"))
        fig_strata(_res, list(_res["methods"].keys()),
                   os.path.join(_cfg["paths"]["figures_dir"], "fig27_工况归一化分层验收.png"))
        fig_resid(_oof["capacity"].to_numpy(float), _oof["pred_baseline"].to_numpy(float),
                  _oof["t_mean"].to_numpy(float), _oof["i_abs_mean"].to_numpy(float),
                  os.path.join(_cfg["paths"]["figures_dir"], "fig28_残差工况结构.png"))
        print("重绘完成")
    else:
        main()
