"""子问题求解器 (Gurobi / BPC / ESP / Concorde) 的统一日志标签.

格式: ``[phase1 backward stage2, gurobi]`` / ``[phase2 backward stage3, esp]`` 等.
Phase1(SBC) 与 Phase2(Level Set) 后向均使用同一前缀; 字段见各阶段日志行.
通过环境变量 ``VRP_CURRENT_PHASE`` (phase1/phase2) 自动加 phase 前缀.
"""
from __future__ import annotations

import os


def set_current_phase(phase: str | int | None) -> None:
    """设置当前外层 phase (SDDP_SBC.solve / SDDLP.solve 入口调用)."""
    if phase is None:
        os.environ.pop("VRP_CURRENT_PHASE", None)
        return
    p = _normalize_phase(str(phase))
    if p:
        os.environ["VRP_CURRENT_PHASE"] = p
    else:
        os.environ.pop("VRP_CURRENT_PHASE", None)


def _normalize_phase(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in ("1", "phase1", "sbc", "p1"):
        return "phase1"
    if s in ("2", "phase2", "sddlp", "p2"):
        return "phase2"
    if s.startswith("phase") and s[5:].isdigit():
        return s
    return ""


def _normalize_direction(direction: str) -> str:
    d = direction.strip().lower()
    if d in ("back", "backward", "bwd"):
        return "backward"
    if d in ("fwd", "forward"):
        return "forward"
    return d


def _normalize_backend(backend: str) -> str:
    b = backend.strip().lower()
    if b in ("eps", "espp", "esppcp", "espp-rc"):
        return "esp"
    if b in ("bp", "bpc", "stage2-bpc", "stage2_bpc"):
        return "bpc"
    if b in ("concord", "concorde-tsp"):
        return "concorde"
    if b in ("gurobi", "grb", "mip"):
        return "gurobi"
    return b


def tag(direction: str, stage: int, backend: str, phase: str | None = None) -> str:
    """构建标准标签, 例如 ``[phase1 forward stage3, concorde]``."""
    d = _normalize_direction(direction)
    b = _normalize_backend(backend)
    p = _normalize_phase(phase if phase is not None else os.environ.get("VRP_CURRENT_PHASE", ""))
    if p:
        return f"[{p} {d} stage{int(stage)}, {b}]"
    return f"[{d} stage{int(stage)}, {b}]"


def log(
    direction: str,
    stage: int,
    backend: str,
    message: str = "",
    *,
    phase: str | None = None,
    indent: int = 4,
    flush: bool = True,
) -> None:
    """打印带标准前缀的一行日志."""
    prefix = " " * indent + tag(direction, stage, backend, phase=phase)
    if message:
        print(f"{prefix} {message}", flush=flush)
    else:
        print(prefix, flush=flush)


def log_fallback(
    direction: str,
    stage: int,
    backend: str,
    reason: str,
    fallback: str = "gurobi",
    *,
    phase: str | None = None,
    indent: int = 4,
) -> None:
    """Exact 求解失败、回退到其它后端时的提示."""
    log(
        direction, stage, backend,
        f"fallback → {_normalize_backend(fallback)} ({reason})",
        phase=phase,
        indent=indent,
    )
