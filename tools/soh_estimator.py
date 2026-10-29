# -*- coding: utf-8 -*-
"""SOH 在线估计小工具（CLI）。

输入一批 EVBattery 充电片段 pkl（同一个充电会话的多个窗口文件），自动按
(car, charge_segment) 归组 → 提取会话特征 → 输出容量估计与 SOH。

用法：
    # 1) 演示：从数据集里抽 3 辆车的最新片段跑一遍
    python tools/soh_estimator.py --demo

    # 2) 指定一个目录（放若干 pkl）
    python tools/soh_estimator.py --dir D:/some/session_pkls --ref-capacity 41.0

    # 3) 指定某车某片段的窗口文件
    python tools/soh_estimator.py --files 470521.pkl 470522.pkl --data-root D:/项目/.../battery_dataset1

输出：终端表格 + 可选 --json 落盘。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evsoh.config import load_config  # noqa: E402
from evsoh.data import load_window  # noqa: E402
from evsoh.session import build_session, session_feature_names  # noqa: E402

MODEL_NAME = "xgb_capacity.joblib"


def load_bundle(path: str) -> Dict:
    import joblib

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"模型不存在：{path}\n请先运行 scripts/05_train_models.py 生成模型。")
    return joblib.load(path)


def group_files(paths: List[str]) -> Dict[str, List]:
    """按 (car, segment) 归组，返回 {key: [(path, feat, meta), ...]}。"""
    groups: Dict[str, List] = {}
    for p in paths:
        feat, meta = load_window(p)
        key = f"{meta['car']}|{meta['charge_segment']}"
        groups.setdefault(key, []).append((os.path.basename(p), feat, meta))
    return groups


def estimate(groups: Dict[str, List], bundle: Dict,
             ref_capacity=None) -> pd.DataFrame:
    """ref_capacity: None=用训练集车型均值；标量=统一参考；dict=按车给 BOL 参考容量。"""
    feats = bundle["features"]
    model = bundle["model"]
    ref_default = float(bundle.get("bol_ref_mean", 41.0))
    rows = []
    for key, wins in sorted(groups.items()):
        f = build_session(wins)
        if f.get("__invalid__"):
            rows.append({"car": key.split("|")[0], "segment": key.split("|")[1],
                         "status": "invalid snippet (too few windows / bad data)"})
            continue
        x = pd.DataFrame([{c: f.get(c, np.nan) for c in feats}])[feats].astype(float)
        cap = float(model.predict(x)[0])
        if isinstance(ref_capacity, dict):
            ref = float(ref_capacity.get(int(f["car"]), ref_default))
        elif ref_capacity:
            ref = float(ref_capacity)
        else:
            ref = ref_default
        rows.append({
            "car": f["car"], "segment": f["segment"],
            "d_soc": round(float(f["d_soc"]), 1),
            "duration_h": round(float(f["dur_s"]) / 3600.0, 2),
            "n_windows": int(f["n_windows"]),
            "quality": round(float(f["quality"]), 3),
            "capacity_Ah": round(cap, 2),
            "ref_Ah": round(ref, 2),
            "soh_pct": round(cap / ref * 100, 1),
            "status": "ok",
        })
    return pd.DataFrame(rows)


def demo(cfg: Dict, bundle: Dict, n_cars: int = 3) -> pd.DataFrame:
    """从数据集中挑几辆高里程车，各取一个有容量标签的会话做端到端演示。

    demo 里用该车自身的 BOL 参考容量（取历史容量最高 10% 的中位数）算 SOH，
    这样输出的 SOH 是"相对该车自身健康状态"，与报告口径一致。
    """
    root = cfg["paths"]["evbattery_root"]
    idx = pd.read_parquet(cfg["paths"]["window_index"])
    cand = idx[(idx["valid"]) & (idx["capacity"] > 0)]
    cand = cand.sort_values("mileage", ascending=False)
    cars = cand["car"].drop_duplicates().head(n_cars).tolist()
    paths = []
    for c in cars:
        g = cand[cand["car"] == c]
        seg = g["segment"].iloc[0]
        sub = g[g["segment"] == seg]
        ds = sub["dataset"].iloc[0]
        for fn in sub["file"]:
            paths.append(os.path.join(root, ds, "data", fn))

    # 该车 BOL 参考容量（来自历史容量标签）
    refs = {}
    try:
        ses = pd.read_parquet(cfg["paths"]["session_features"])
        from evsoh.target import bol_reference
        for c in cars:
            caps = ses.loc[(ses["car"] == c) & (ses["capacity"] > 0), "capacity"].values
            if caps.size:
                refs[c] = bol_reference(caps)
    except Exception:
        pass
    return estimate(group_files(paths), bundle, refs or None)


def parse_ref_capacity(spec: str):
    """'42.0' → 42.0；'18=42.8,62=41.4' → {18: 42.8, 62: 41.4}；'' → None。"""
    spec = (spec or "").strip()
    if not spec:
        return None
    if "=" not in spec:
        return float(spec)
    refs = {}
    for kv in spec.split(","):
        k, v = kv.split("=")
        refs[int(k.strip())] = float(v)
    return refs


def main() -> None:
    ap = argparse.ArgumentParser(description="EVBattery 充电片段 → 容量 / SOH 估计")
    ap.add_argument("--dir", default="", help="包含 pkl 文件的目录")
    ap.add_argument("--files", nargs="*", default=[], help="pkl 文件名（需配合 --data-root）")
    ap.add_argument("--data-root", default="", help="--files 所在的数据分块目录")
    ap.add_argument("--ref-capacity", default="",
                    help="BOL 参考容量（Ah）：标量 \"42.0\"，或按车 \"18=42.8,62=41.4\"；"
                         "不填则用训练集均值")
    ap.add_argument("--demo", action="store_true", help="从数据集抽样演示")
    ap.add_argument("--json", default="", help="结果另存为 JSON")
    args = ap.parse_args()
    ref = parse_ref_capacity(args.ref_capacity)

    cfg = load_config()
    model_path = os.path.join(cfg["paths"]["results_dir"], "models", MODEL_NAME)
    bundle = load_bundle(model_path)
    print(f"Model: {os.path.relpath(model_path).replace(os.sep, '/')}  "
          f"({bundle['n_train']:,} sessions / {bundle['n_cars']} cars, "
          f"leave-cars-out RMSE {bundle['train_metrics']['rmse']:.2f} Ah)\n")

    if args.demo:
        out = demo(cfg, bundle)
    elif args.dir:
        paths = sorted(glob.glob(os.path.join(args.dir, "*.pkl")))
        if not paths:
            sys.exit(f"No .pkl files found under {args.dir}")
        out = estimate(group_files(paths), bundle, ref)
    elif args.files:
        if not args.data_root:
            sys.exit("--files requires --data-root")
        paths = [os.path.join(args.data_root, "data", f) if not os.path.isabs(f) else f
                 for f in args.files]
        out = estimate(group_files(paths), bundle, ref)
    else:
        ap.print_help()
        return

    print(out.sort_values(["car", "segment"]).to_string(index=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
