# -*- coding: utf-8 -*-
"""小型神经网络工具：表格/序列两用的训练循环与预处理。

刻意保持极简（无 Lightning、无 dataloader），因为这个项目里网络只承担两个角色：
  1. 与 XGBoost 做"同表示换模型"的对照；
  2. 提供多初始化集成分位，用来做区间估计。
两者都不需要复杂训练框架，反而需要"完全可复现、种子可枚举"。

数值稳定性有两处是必需的，都是被真实故障逼出来的：
  - winsorize（0.5%/99.5%）：库仑计特征有极端离群值，不裁剪会让 MLP 梯度爆炸
    （曾出现 RMSE ~1e11）；
  - 梯度裁剪 clip_grad_norm_(1.0)：同上。
"""
from __future__ import annotations

import warnings

import numpy as np


def prep_tabular(Xtr: np.ndarray, Xte: np.ndarray):
    """中位数插补 + winsorize + 标准化。统计量只用训练集估计，避免泄漏。"""
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


def make_mlp(n_in: int, width: int = 128):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_in, width), nn.ReLU(), nn.Dropout(0.20),
        nn.Linear(width, 64), nn.ReLU(), nn.Dropout(0.10),
        nn.Linear(64, 1),
    )


def fit_torch(model_fn, A, y, B, seed: int = 42, epochs: int = 80,
              lr: float = 1e-3, bs: int = 256, patience: int = 12,
              val_frac: float = 0.10):
    """训练并返回 B 上的预测（原始尺度）。

    早停用训练集内随机留出的 val_frac（这里不按车分组——早停只看收敛，不看泛化）。
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(64, int(n * val_frac)) if n > 200 else max(16, int(n * val_frac))
    va, tr = perm[:n_val], perm[n_val:]

    ymu, ysd = float(y[tr].mean()), float(y[tr].std())
    net = model_fn()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    At = torch.tensor(A, dtype=torch.float32)
    Bt = torch.tensor(B, dtype=torch.float32)
    yt = torch.tensor((y - ymu) / ysd, dtype=torch.float32)

    best, best_state, bad = np.inf, None, 0
    for _ep in range(epochs):
        net.train()
        order = rng.permutation(tr)
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            if idx.size < 8:
                continue
            loss = F.mse_loss(net(At[idx]).squeeze(-1), yt[idx])
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
