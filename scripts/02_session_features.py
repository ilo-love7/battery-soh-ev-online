# -*- coding: utf-8 -*-
"""02 会话级特征提取：对第二遍读取（只读有容量标签的会话）拼装轨迹并算特征。

输入：results/cache/window_index.parquet（脚本 01 产出）
输出：results/cache/session_features.parquet（约 4.4 万行 × 约 110 列）

用法：
    python scripts/02_session_features.py                # 默认只做有容量标签的会话
    python scripts/02_session_features.py --all          # 全部会话（推理侧演示用）
    python scripts/02_session_features.py --limit 500    # 冒烟测试
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config  # noqa: E402
from evsoh.session import build_from_files, init_worker  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="不限容量标签")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_config()
    root = cfg["paths"]["evbattery_root"]
    sq = cfg["session_qc"]
    out = args.out or cfg["paths"]["session_features"]

    idx = pd.read_parquet(cfg["paths"]["window_index"])
    print(f"窗口索引 {idx.shape}")
    idx = idx[idx["valid"]].copy()

    if not args.all:
        idx = idx[idx["capacity"] > 0]
    print(f"候选窗口（valid{' + 有容量标签' if not args.all else ''}）: {len(idx)}")

    idx = idx.copy()
    idx["fnum"] = idx["file"].str.replace(".pkl", "", regex=False).astype(int)
    idx = idx.sort_values(["dataset", "car", "segment", "fnum"])
    tasks = []
    for (ds, car, seg), g in idx.groupby(["dataset", "car", "segment"], sort=False):
        files = g["file"].tolist()
        # 会话级粗筛：SOC 跨度、窗口数
        soc_span = float(g["soc_end"].iloc[-1] - g["soc_start"].iloc[0])
        if len(files) < 2:
            continue
        if soc_span < sq["min_soc_span"] and not args.all:
            continue
        tasks.append((ds, f"{car}|{seg}", files))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"待处理会话: {len(tasks)}")

    import multiprocessing as mp

    t0 = time.time()
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=init_worker, initargs=(root,)) as pool:
        for k, r in enumerate(pool.imap_unordered(build_from_files, tasks, chunksize=16), 1):
            rows.append(r)
            if k % 2000 == 0:
                el = time.time() - t0
                print(f"  {k}/{len(tasks)}  {k/el:.1f} sess/s  eta {(len(tasks)-k)/max(k/el,1e-9)/60:.1f} min",
                      flush=True)

    df = pd.DataFrame(rows)
    if "__invalid__" in df.columns:
        n_bad = int(df["__invalid__"].sum())
        print(f"无效会话: {n_bad}")
        df = df[~df["__invalid__"].astype(bool)].drop(columns=["__invalid__"])
    df["car"] = df["car"].astype(int)
    df.to_parquet(out, index=False)
    print(f"\n写出 {out}  {df.shape}  {os.path.getsize(out)/1e6:.1f} MB")
    print(f"耗时 {(time.time()-t0)/60:.1f} min")
    if "capacity" in df.columns:
        print("\n按分块：")
        print(df.groupby("dataset").agg(sessions=("car", "size"), cars=("car", "nunique"),
                                        d_soc_med=("d_soc", "median"),
                                        quality_med=("quality", "median")).to_string())


if __name__ == "__main__":
    main()
