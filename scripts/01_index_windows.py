# -*- coding: utf-8 -*-
"""01 全量窗口索引：扫描 EVBattery 三个分块的全部窗口 pkl，落盘窗口级统计表。

输出：results/cache/window_index.parquet（约 127.8 万行）
      每行 = 一个 128 步窗口的元数据（车/会话/标签/里程/容量）+ 统计量 + 可用性标记。

用法：
    python scripts/01_index_windows.py            # 全量
    python scripts/01_index_windows.py --limit 20000 --workers 8   # 冒烟测试
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config  # noqa: E402
from evsoh.data import list_windows, load_window, window_stats  # noqa: E402

_CFG = None
_QC = None


def _init_worker(root: str, dataset: str, qc: dict):
    global _CFG, _QC
    _CFG = (root, dataset)
    _QC = qc


def _work(fname: str):
    root, dataset = _CFG
    path = os.path.join(root, dataset, "data", fname)
    try:
        feat, meta = load_window(path)
        return window_stats(feat, meta, dataset, fname, _QC)
    except Exception as exc:  # 损坏文件不阻断整体扫描
        return {"dataset": dataset, "file": fname, "error": repr(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每个分块只处理前 N 个文件（0=全量）")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_config()
    root = cfg["paths"]["evbattery_root"]
    qc = cfg["window_qc"]
    out = args.out or cfg["paths"]["window_index"]

    import multiprocessing as mp

    t_all = time.time()
    frames = []
    for dataset in cfg["paths"]["datasets"]:
        names = list_windows(root, dataset)
        if args.limit:
            names = names[: args.limit]
        t0 = time.time()
        rows = []
        if args.workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.workers, initializer=_init_worker,
                          initargs=(root, dataset, qc)) as pool:
                for k, r in enumerate(pool.imap_unordered(_work, names, chunksize=256), 1):
                    rows.append(r)
                    if k % 20000 == 0:
                        el = time.time() - t0
                        print(f"  {dataset}: {k}/{len(names)}  {k/el:.0f} files/s  "
                              f"eta {(len(names)-k)/max(k/el,1e-9)/60:.1f} min", flush=True)
        else:
            _init_worker(root, dataset, qc)
            for k, n in enumerate(names, 1):
                rows.append(_work(n))
                if k % 20000 == 0:
                    print(f"  {dataset}: {k}/{len(names)}", flush=True)
        df = pd.DataFrame(rows)
        df["dataset"] = dataset
        frames.append(df)
        print(f"{dataset}: {len(df)} windows in {time.time()-t0:.1f}s", flush=True)

    full = pd.concat(frames, ignore_index=True)
    if "error" in full.columns:
        bad = full["error"].notna().sum()
        print(f"读取失败窗口: {bad}")
        full = full[full["error"].isna()].drop(columns=["error"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    full.to_parquet(out, index=False)
    print(f"\n写出 {out}  {full.shape}  {os.path.getsize(out)/1e6:.1f} MB")
    print(f"总耗时 {(time.time()-t_all)/60:.1f} min")
    print("\n分块统计：")
    print(full.groupby("dataset").agg(
        windows=("file", "size"), cars=("car", "nunique"),
        segments=("segment", "nunique"),
        valid_rate=("valid", "mean"),
        cap_labeled=("capacity", lambda s: (s > 0).mean()),
    ).to_string())


if __name__ == "__main__":
    main()
