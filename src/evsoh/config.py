# -*- coding: utf-8 -*-
"""配置加载与路径工具。"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_PATH = os.path.join(_ROOT, "configs", "config.yaml")

_CACHE: Dict[str, Any] = {}


def project_root() -> str:
    return _ROOT


def load_config(path: str | None = None) -> Dict[str, Any]:
    """读 configs/config.yaml，并把相对路径解析成项目内绝对路径。"""
    path = path or _CONFIG_PATH
    if path in _CACHE:
        return _CACHE[path]
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    # 数据集根目录：环境变量 EVBATTERY_ROOT 优先，其次是 config.yaml
    env_root = os.environ.get("EVBATTERY_ROOT")
    if env_root:
        p["evbattery_root"] = env_root
    if "evbattery_root" in p and not os.path.isabs(p["evbattery_root"]):
        p["evbattery_root"] = os.path.join(_ROOT, p["evbattery_root"])
    for key in ("results_dir", "cache_dir", "figures_dir", "metrics_dir", "reports_dir"):
        if key in p and not os.path.isabs(p[key]):
            p[key] = os.path.join(_ROOT, p[key])
    for key in ("window_index", "session_features"):
        if key in p and not os.path.isabs(p[key]):
            p[key] = os.path.join(_ROOT, p[key])
    for d in (p["cache_dir"], p["figures_dir"], p["metrics_dir"], p["reports_dir"]):
        os.makedirs(d, exist_ok=True)
    _CACHE[path] = cfg
    return cfg


def save_json(obj: Any, name: str) -> str:
    """把指标字典写入 results/metrics/<name>.json。"""
    import json

    cfg = load_config()
    out = os.path.join(cfg["paths"]["metrics_dir"], f"{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
    return out
