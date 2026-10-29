# -*- coding: utf-8 -*-
"""统一绘图风格（中文字体 + 浅色主题 + 保存到 results/figures）。"""
from __future__ import annotations

import os
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.8,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
})

# 色板（与 A1 项目风格一致：主色蓝、强调橙、警示红）
C_MAIN = "#2f6fd0"
C_ACC = "#e08b2e"
C_RED = "#c0392b"
C_GREEN = "#2e8b57"
C_GRAY = "#8a8a8a"
C_PURPLE = "#7a5ea8"
PALETTE = [C_MAIN, C_ACC, C_RED, C_GREEN, C_PURPLE, C_GRAY]


def savefig(fig, path: str, tight: bool = True) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def annotate_bars(ax, bars, fmt: str = "{:.3f}", dy: float = 0.0) -> None:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=8)
