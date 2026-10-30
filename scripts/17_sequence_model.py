# -*- coding: utf-8 -*-
"""17 序列模型 vs 手工特征：瓶颈在表示还是在模型？

全流程一直用「手工 114 维特征 + XGBoost」。审稿人第一个会问的问题是：
为什么不直接把原始 128×8 序列喂给神经网络？手工特征是不是丢了信息？

本脚本在同一批会话、同一套按车分组 5 折上做三方对照，把两个因素拆开：

  xgb_manual   114 维手工特征 + XGBoost        （现状）
  mlp_manual   114 维手工特征 + MLP            （同表示，换模型）
  seq_cnn      原始轨迹重采样 + 1D CNN         （换表示，同级别模型容量）

如果 seq_cnn 明显更好，说明手工特征有信息损失；如果相当或更差，说明
手工特征已经把该抓的都抓到了，而且更省算力、更可解释。

序列侧的处理（对应深化文档 D1 的"片段长短不一"问题）：
  把每个会话的充电轨迹按 **SOC 轴** 等距重采样到 256 点（20–100%），
  这样不同长度、不同起点的片段在 SOC 轴上对齐，而不是在时间轴上对齐。
  5 个通道：电压 V、电流 I、温度 T、dV/dSOC（IC 的倒数）、覆盖率掩码。

为了控制成本，只在按 SOH 五分位等量抽样的 12000 段上做（约 8.3 万个窗口 pkl）。

输出
  results/cache/seq_grid.npz               重采样后的序列张量与索引
  results/metrics/sequence_vs_manual.json
  results/figures/fig29_序列模型对照.png
  results/figures/fig30_误差随片段长度.png
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config, save_json  # noqa: E402
from evsoh.data import COL_I, COL_SOC, COL_T, COL_V, load_window  # noqa: E402
from evsoh.evaluate import group_kfold_by_car, group_metrics, regression_metrics  # noqa: E402
from evsoh.features import active_features, feature_columns  # noqa: E402
from evsoh.models import build_model  # noqa: E402
from evsoh.plotting import C_ACC, C_GRAY, C_MAIN, C_PURPLE, C_RED, plt, savefig  # noqa: E402
from evsoh.session import concat_windows  # noqa: E402
from evsoh.target import add_soh_labels  # noqa: E402

N_SAMPLE = int(os.environ.get("N_SAMPLE", "12000"))
SMOKE = bool(os.environ.get("SMOKE"))
GRID_LO, GRID_HI, N_GRID = 20.0, 100.0, 256
SOC_GRID = np.linspace(GRID_LO, GRID_HI, N_GRID)
N_CH = 5                      # V, I, T, dV/dSOC, mask
SPAN_EDGES = [20, 30, 40, 50, 60, 70, 100]
SPAN_LABELS = ["20-30", "30-40", "40-50", "50-60", "60-70", "70-100"]


# ---------------------------------------------------------------- 序列化
def _resample_on_soc(x: np.ndarray, soc: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """把 (soc, x) 散点按 SOC 轴线性插值到 grid；覆盖不到的位置返回 NaN。

    SOC 可能重复或轻微回落，先按 SOC 排序并合并同值点（取均值），再插值。
    """
    m = np.isfinite(x) & np.isfinite(soc)
    if m.sum() < 8:
        return np.full(grid.shape, np.nan)
    s, v = soc[m], x[m]
    order = np.argsort(s, kind="stable")
    s, v = s[order], v[order]
    uniq, inv = np.unique(s, return_inverse=True)
    if uniq.size < 8:
        return np.full(grid.shape, np.nan)
    acc = np.zeros(uniq.size)
    cnt = np.zeros(uniq.size)
    np.add.at(acc, inv, v)
    np.add.at(cnt, inv, 1)
    mean_v = acc / np.maximum(cnt, 1)
    out = np.interp(grid, uniq, mean_v, left=np.nan, right=np.nan)
    return out


def trajectory_to_grid(traj: np.ndarray) -> np.ndarray | None:
    """(N,8) 轨迹 → (5, 256) 张量。返回 None 表示该会话不可用。"""
    v = traj[:, COL_V]
    i = traj[:, COL_I]
    soc = traj[:, COL_SOC]
    t = traj[:, COL_T]

    gv = _resample_on_soc(v, soc, SOC_GRID)
    gi = _resample_on_soc(i, soc, SOC_GRID)
    gt = _resample_on_soc(t, soc, SOC_GRID)
    mask = np.isfinite(gv).astype(np.float64)
    if mask.sum() < 32:                      # 覆盖不到 32 个点，信息太少
        return None

    # dV/dSOC：先在网格上取差分，再用掩码内插值补边缘
    # 注意 pandas 3 的 copy-on-write 会让 .to_numpy() 返回只读视图，必须显式复制
    dv = np.gradient(np.where(np.isfinite(gv), gv, np.nan))
    dv = np.array(pd.Series(dv).interpolate(limit_direction="both").to_numpy(),
                  dtype=np.float64, copy=True)
    dv[~np.isfinite(dv)] = 0.0

    # 用掩码内的均值填充缺失通道，避免把 NaN 送进卷积
    def _fill(a):
        a = np.array(a, dtype=np.float64, copy=True)
        if np.isfinite(a).any():
            a[~np.isfinite(a)] = np.nanmean(a)
        else:
            a[:] = 0.0
        return a

    return np.stack([_fill(gv), _fill(gi), _fill(gt), _fill(dv), mask], axis=0)


def build_sequences(cfg, df: pd.DataFrame) -> tuple:
    """读取抽样会话的 pkl，重采样成固定网格张量。带缓存。"""
    cache = os.path.join(cfg["paths"]["cache_dir"],
                         "seq_grid_smoke.npz" if SMOKE else "seq_grid.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if int(z["n"]) == len(df):
            print(f"复用序列缓存 {cache}", flush=True)
            return z["X"], z["keep"]

    root = cfg["paths"]["evbattery_root"]
    idx = pd.read_parquet(cfg["paths"]["window_index"],
                          columns=["dataset", "file", "car", "segment", "valid"])
    idx = idx[idx["valid"]]
    buckets: dict = {}
    for ds, car, seg, f in zip(idx["dataset"].values, idx["car"].values,
                               idx["segment"].astype(str).values, idx["file"].values):
        buckets.setdefault((ds, int(car), seg), []).append(f)

    X = np.zeros((len(df), N_CH, N_GRID), dtype=np.float32)
    keep = np.zeros(len(df), dtype=bool)
    t0, miss = time.time(), 0
    for si, r in enumerate(df.itertuples(index=False)):
        k = (r.dataset, int(r.car), str(r.segment))
        files = buckets.get(k)
        if not files:
            miss += 1
            continue
        wins = []
        for f in sorted(files, key=lambda s: int(s.split(".")[0])):
            p = os.path.join(root, k[0], "data", f)
            if os.path.exists(p):
                feat, meta = load_window(p)
                wins.append((f, feat, meta))
        if len(wins) < 2:
            miss += 1
            continue
        traj = concat_windows(wins)
        if traj.size == 0:
            miss += 1
            continue
        g = trajectory_to_grid(traj)
        if g is None:
            miss += 1
            continue
        X[si] = g.astype(np.float32)
        keep[si] = True
        if keep.sum() % 1500 == 0:
            print(f"  {keep.sum()} 段  ({time.time() - t0:.0f}s)", flush=True)

    print(f"序列化完成 {keep.sum()} 段（跳过 {miss}）  用时 {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(cache, X=X, keep=keep, n=len(df))
    return X, keep


# ---------------------------------------------------------------- 模型
def _make_cnn(n_ch: int = N_CH, width: int = 32):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv1d(n_ch, width, 7, padding=3), nn.BatchNorm1d(width), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(width, width * 2, 5, padding=2), nn.BatchNorm1d(width * 2), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(width * 2, width * 2, 5, padding=2), nn.BatchNorm1d(width * 2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Linear(width * 2, 32), nn.ReLU(), nn.Dropout(0.10), nn.Linear(32, 1),
    )


def _make_mlp(n_in: int, width: int = 128):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_in, width), nn.ReLU(), nn.Dropout(0.20),
        nn.Linear(width, 64), nn.ReLU(), nn.Dropout(0.10),
        nn.Linear(64, 1),
    )


def _prep_tabular(Xtr, Xte):
    """中位数插补 + winsorize + 标准化（统计量只用训练折）。"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(Xtr, axis=0)
        lo = np.nanpercentile(Xtr, 0.5, axis=0)
        hi = np.nanpercentile(Xtr, 99.5, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    lo = np.where(np.isfinite(lo), lo, -np.inf)
    hi = np.where(np.isfinite(hi), hi, np.inf)
    A = np.clip(np.where(np.isfinite(Xtr), Xtr, med), lo, hi)
    B = np.clip(np.where(np.isfinite(Xte), Xte, med), lo, hi)
    mu, sd = A.mean(axis=0), A.std(axis=0)
    sd[sd == 0] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def fit_torch(model_fn, A, y, B, seed=42, epochs=80, lr=1e-3, bs=256, patience=12):
    """通用 torch 训练循环（早停用训练集内的 10% 随机留出，按车不分组——这里只做早停用）。"""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(64, int(n * 0.10))
    va, tr = perm[:n_val], perm[n_val:]

    ymu, ysd = float(y[tr].mean()), float(y[tr].std())
    net = model_fn()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    At = torch.tensor(A, dtype=torch.float32)
    Bt = torch.tensor(B, dtype=torch.float32)
    yt = torch.tensor((y - ymu) / ysd, dtype=torch.float32)

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        order = rng.permutation(tr)
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            if idx.size < 8:
                continue
            pred = net(At[idx]).squeeze(-1)
            loss = F.mse_loss(pred, yt[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vp = net(At[va]).numpy().squeeze(-1) * ysd + ymu
        vmae = float(np.mean(np.abs(vp - y[va])))
        if vmae < best - 1e-4:
            best, bad = vmae, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(Bt).numpy().squeeze(-1) * ysd + ymu


# ---------------------------------------------------------------- 主流程
def main() -> None:
    cfg = load_config()
    ses = pd.read_parquet(cfg["paths"]["session_features"])
    df, lab = add_soh_labels(ses, cfg)
    feats = active_features(df, feature_columns())
    df = df.dropna(subset=["capacity", "bol_ref"]).reset_index(drop=True)
    print(f"全量 {len(df)} 会话 / {df['car'].nunique()} 车", flush=True)

    # 按 SOH 五分位等量抽样（保证覆盖整个健康度范围）
    df["_dec"] = pd.qcut(df["soh"], 5, labels=False)
    per = N_SAMPLE // 5
    sample = pd.concat([g.sample(min(per, len(g)), random_state=42)
                        for _k, g in df.groupby("_dec")]).reset_index(drop=True)
    print(f"抽样 {len(sample)} 会话 / {sample['car'].nunique()} 车", flush=True)

    X_seq_all, keep = build_sequences(cfg, sample)
    sub = sample[keep].reset_index(drop=True)
    X_seq = X_seq_all[keep]
    print(f"可用序列 {len(sub)} 段 / {sub['car'].nunique()} 车", flush=True)

    # 去掉会话数太少的车（group_metrics 需要每车有代表性）
    cnt = sub.groupby("car").size()
    ok_cars = set(cnt[cnt >= 8].index)
    m = sub["car"].isin(ok_cars).to_numpy()
    sub, X_seq = sub[m].reset_index(drop=True), X_seq[m]
    print(f"保留 {len(sub)} 段 / {sub['car'].nunique()} 车（每车 ≥ 8 段）", flush=True)

    X_tab = sub[feats].astype(float)
    y = sub["capacity"].to_numpy(dtype=float)
    cars = sub["car"].to_numpy()
    span = sub["d_soc"].to_numpy(dtype=float)
    ref_mean = float(sub["bol_ref"].mean())

    folds = list(group_kfold_by_car(cars, n_splits=cfg["splits"]["n_group_folds"],
                                    seed=cfg["splits"]["random_state"]))
    if SMOKE:
        folds = folds[:1]
        print("SMOKE 模式：只跑第 1 折", flush=True)
    names = ["xgb_manual", "mlp_manual", "seq_cnn"]
    preds = {n: np.full(len(sub), np.nan) for n in names}
    t0 = time.time()

    for fi, (tr, te) in enumerate(folds):
        print(f"[折 {fi + 1}/5] 训练 {len(tr)} / 测试 {len(te)}", flush=True)

        # 手工特征 + XGBoost
        m0 = build_model("xgb", cfg["models"], n_jobs=4)
        m0.fit(X_tab.iloc[tr], y[tr])
        preds["xgb_manual"][te] = m0.predict(X_tab.iloc[te])

        # 手工特征 + MLP（同表示，换模型）
        A, B = _prep_tabular(X_tab.iloc[tr].to_numpy(dtype=float),
                             X_tab.iloc[te].to_numpy(dtype=float))
        ps = [fit_torch(lambda: _make_mlp(A.shape[1]), A, y[tr], B,
                        seed=42 + fi * 10 + s) for s in range(3)]
        preds["mlp_manual"][te] = np.mean(ps, axis=0)

        # 原始序列 + CNN（换表示）
        S_tr = X_seq[tr]
        S_te = X_seq[te]
        # 序列按通道做标准化（用训练折统计量；mask 通道不动）
        mu = S_tr.mean(axis=(0, 2), keepdims=True)
        sd = S_tr.std(axis=(0, 2), keepdims=True)
        sd[:, :, :] = np.where(sd > 1e-8, sd, 1.0)
        mu[:, 4:, :] = 0.0
        sd[:, 4:, :] = 1.0
        S_tr = (S_tr - mu) / sd
        S_te = (S_te - mu) / sd
        ps = [fit_torch(_make_cnn, S_tr, y[tr], S_te, seed=42 + fi * 10 + s)
              for s in range(3)]
        preds["seq_cnn"][te] = np.mean(ps, axis=0)
        print(f"    done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- 汇总 ----------------
    res = {"n_sessions": int(len(sub)), "n_cars": int(sub["car"].nunique()),
           "n_features_manual": len(feats), "seq_shape": list(X_seq.shape[1:]),
           "soc_grid": [GRID_LO, GRID_HI, N_GRID], "label": lab, "methods": {},
           "note": ("三方在同一批会话、同一套按车分组 5 折上评估；神经网络各跑 3 个种子取平均。"
                    "序列按 SOC 轴重采样到 256 点，通道为 V/I/T/dV·dSOC⁻¹/掩码。")}
    for n in names:
        p = preds[n]
        mm = regression_metrics(y, p)
        gm = group_metrics(pd.DataFrame({"capacity": y, "p": p, "car": cars}),
                           "capacity", "p")
        res["methods"][n] = {**mm, **gm, "soh_rmse_pp": mm["rmse"] / ref_mean * 100.0}
        print(f"  {n:11s} RMSE={mm['rmse']:.4f} MAE={mm['mae']:.4f} R²={mm['r2']:.4f}",
              flush=True)

    # 误差随片段长度（核心问题：序列模型对短片段是否更鲁棒）
    span_tab = {}
    for n in names:
        rows = {}
        for lo, hi, lab_ in zip(SPAN_EDGES[:-1], SPAN_EDGES[1:], SPAN_LABELS):
            m2 = (span >= lo) & (span < hi)
            if m2.sum() < 60:
                continue
            mm = regression_metrics(y[m2], preds[n][m2])
            rows[lab_] = {"n": int(m2.sum()), "mae": mm["mae"], "rmse": mm["rmse"]}
        span_tab[n] = rows
    res["by_soc_span"] = span_tab
    for n in names:
        print(f"  {n:11s} 短片段(20-30%) MAE="
              f"{span_tab[n].get('20-30', {}).get('mae', float('nan')):.4f}  "
              f"长片段(70-100%) MAE="
              f"{span_tab[n].get('70-100', {}).get('mae', float('nan')):.4f}", flush=True)

    save_json(res, "sequence_vs_manual")
    pd.DataFrame({"car": cars, "d_soc": span, "capacity": y,
                  **{f"pred_{n}": preds[n] for n in names}}).to_parquet(
        os.path.join(cfg["paths"]["cache_dir"], "sequence_oof.parquet"), index=False)

    fig_compare(res, names, os.path.join(cfg["paths"]["figures_dir"], "fig29_序列模型对照.png"))
    fig_span(res, names, os.path.join(cfg["paths"]["figures_dir"], "fig30_误差随片段长度.png"))
    print(f"\n完成，用时 {time.time() - t0:.0f}s", flush=True)


def fig_compare(res: dict, names: list, path: str) -> None:
    """图 29：三种表示的 RMSE 与"同车误差"对比（基线参照点图）。"""
    labels = {"xgb_manual": "XGBoost + manual feats",
              "mlp_manual": "MLP + manual feats",
              "seq_cnn": "1D CNN + raw SOC-aligned curves"}
    cols = {"xgb_manual": C_GRAY, "mlp_manual": C_ACC, "seq_cnn": C_MAIN}

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.5))

    base = res["methods"]["xgb_manual"]["rmse"]
    ys = np.arange(len(names))
    vals = [res["methods"][n]["rmse"] for n in names]
    axes[0].hlines(ys, base, vals, color=[cols[n] for n in names], lw=2.2, alpha=0.75)
    for i, (n, v) in enumerate(zip(names, vals)):
        axes[0].plot([v], [i], marker="o", ms=8, color=cols[n], zorder=3)
        d = "" if i == 0 else f"  ({(v / base - 1) * 100:+.1f}%)"
        # 标注放在点的下方：第一行点上方几乎没有余量，会压到纵轴标签
        axes[0].text(v, i + 0.30, f"{v:.4f}{d}", ha="center", va="top",
                     fontsize=8.2, color=cols[n])
    axes[0].axvline(base, color=C_GRAY, ls="--", lw=1.1, zorder=1)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([labels[n] for n in names], fontsize=8.4)
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(names) - 0.30, -0.45)
    axes[0].set_xlim(base - 0.06, max(vals) + 0.07)
    axes[0].set_xlabel("RMSE (Ah), 5-fold by car")
    axes[0].set_title("(a) Pooled error", pad=8)
    axes[0].grid(visible=False)

    # 右：逐车 RMSE 的分布（箱线），看是否只是被少数车拉偏
    data = [res["methods"][n]["car_rmse_mean"] for n in names]
    pos = np.arange(len(names))
    for i, n in enumerate(names):
        v = res["methods"][n]["car_rmse_mean"]
        axes[1].barh(i, v, color=cols[n], height=0.55)
        axes[1].text(v + max(data) * 0.015, i, f"{v:.3f}", va="center", fontsize=8.4)
    axes[1].set_yticks(pos)
    axes[1].set_yticklabels([labels[n] for n in names], fontsize=8.4)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean per-car RMSE (Ah), cars weighted equally")
    axes[1].set_title("(b) Car-level error", pad=8)
    axes[1].set_xlim(0, max(data) * 1.18)
    axes[1].grid(visible=False)
    fig.tight_layout()
    savefig(fig, path)


def fig_span(res: dict, names: list, path: str) -> None:
    """图 30：误差随 SOC 跨度变化 —— 序列模型的价值应该体现在短片段上。"""
    labels = {"xgb_manual": "XGBoost + manual", "mlp_manual": "MLP + manual",
              "seq_cnn": "1D CNN + curves"}
    cols = {"xgb_manual": C_GRAY, "mlp_manual": C_ACC, "seq_cnn": C_MAIN}
    marks = {"xgb_manual": "o", "mlp_manual": "s", "seq_cnn": "D"}

    bins = [b for b in SPAN_LABELS if all(b in res["by_soc_span"][n] for n in names)]
    xs = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for n in names:
        vals = [res["by_soc_span"][n][b]["mae"] for b in bins]
        ax.plot(xs, vals, marker=marks[n], ms=5, lw=1.7, color=cols[n], label=labels[n])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b}%" for b in bins], fontsize=8.5)
    ax.set_xlim(-0.45, len(bins) - 0.55)
    ax.set_xlabel("SOC span of the charging snippet")
    ax.set_ylabel("MAE (Ah)")
    ax.set_title("Short snippets stay hard for every representation:\n"
                 "the raw-curve CNN is worst exactly where it was supposed to help",
                 pad=8, fontsize=9.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, path)


if __name__ == "__main__":
    if "--replot" in sys.argv:          # 只重画图，复用已落盘的结果
        import json

        _cfg = load_config()
        with open(os.path.join(_cfg["paths"]["metrics_dir"], "sequence_vs_manual.json"),
                  encoding="utf-8") as f:
            _res = json.load(f)
        _names = list(_res["methods"].keys())
        fig_compare(_res, _names, os.path.join(_cfg["paths"]["figures_dir"],
                                               "fig29_序列模型对照.png"))
        fig_span(_res, _names, os.path.join(_cfg["paths"]["figures_dir"],
                                            "fig30_误差随片段长度.png"))
        print("重绘完成")
    else:
        main()
