"""
运行日志：把 stdout/stderr 同时写入 experiments/ 下的文本文件。

main.py 入口调用 setup_run_logging() 后，所有 print() 会：
  1. 照常显示在终端
  2. 实时 flush 写入 <experiments>/results/<tag>.txt（绝对路径，不依赖 cwd）

目录布局（默认 initial-src/experiments/）：
  results/           — main.py 日志、pkl、report
  compare_ef_result/ — compare_ef_sddp.py 日志

可用环境变量 VRP_EXPERIMENTS_ROOT 覆盖 experiments 根目录；LRP_ 同名别名优先。
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

_SRC_INI_DIR = Path(__file__).resolve().parent.parent
_INITIAL_SRC_DIR = _SRC_INI_DIR.parent


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for f in self._streams:
            f.write(s)
            f.flush()
        return len(s)

    def flush(self) -> None:
        for f in self._streams:
            f.flush()

    def isatty(self) -> bool:
        return self._streams[0].isatty()


def experiments_root() -> Path:
    """Absolute path to experiments output root."""
    override = os.environ.get("LRP_EXPERIMENTS_ROOT",
                              os.environ.get("VRP_EXPERIMENTS_ROOT", "")).strip()
    root = Path(override) if override else (_INITIAL_SRC_DIR / "experiments")
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def results_root() -> Path:
    """main.py: console logs, pkl, reports."""
    out = experiments_root() / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def compare_ef_results_root() -> Path:
    """compare_ef_sddp.py logs."""
    out = experiments_root() / "compare_ef_result"
    out.mkdir(parents=True, exist_ok=True)
    return out


def economic_scale_tag(cfg) -> str:
    """Stable filename component preventing purchase-scale run collisions."""
    value = float(getattr(cfg, "purchase_cost_scale", 1.0))
    text = format(value, ".12g").replace("-", "m").replace(".", "p")
    return f"PScale{text}"


def result_tag(cfg) -> str:
    """Backward-readable instance tag with collision dimensions appended."""
    if getattr(cfg, "data_source", None) == "lrp":
        return getattr(cfg, "_output_tag", cfg.instance_tag())
    return f"{cfg.instance_tag()}_T{cfg.T}_{economic_scale_tag(cfg)}"


def console_log_path(cfg) -> Path:
    if getattr(cfg, "data_source", None) == "lrp":
        root = Path(cfg.out) if cfg.out is not None else results_root()
        return root / (f"{result_tag(cfg)}.txt" if getattr(cfg, "_legacy_output", True) else "console.log")
    nc = getattr(cfg, "_resolved_num_customers", None) or cfg.num_customers
    nv = getattr(cfg, "_resolved_num_vehicles", None) or cfg.num_vehicles
    if getattr(cfg, "data_source", "csv") == "hfvrp":
        if nc is None or not nv:
            # Resolve log label size from HFVRP file when unset
            from core.hfvrp_loader import parse_hfvrp_file

            meta = parse_hfvrp_file(cfg.hfvrp_file)
            nc = nc or meta["num_customers"]
            nv = nv or len(meta["vehicles"])
        stem = Path(getattr(cfg, "hfvrp_file", "hfvrp")).stem
        tag = (
            f"hfvrp-{stem}_C{nc}_T{cfg.T}_Sce{cfg.num_scenarios}_Veh{nv}_"
            f"{economic_scale_tag(cfg)}"
        )
    else:
        nc = nc or "auto"
        nv = nv or "auto"
        tag = (
            f"customer{nc}_T{cfg.T}_Sce{cfg.num_scenarios}_Veh{nv}_"
            f"{economic_scale_tag(cfg)}"
        )
    return results_root() / f"{tag}.txt"


@contextmanager
def setup_run_logging(log_path: Path) -> Iterator[Path]:
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, log_file)
    sys.stderr = _Tee(old_err, log_file)
    try:
        yield log_path
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_file.close()
