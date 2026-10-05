"""Taillard HFVRP 算例发现与派生实例默认值。

用法:
    from core.taillard_instances import make_taillard_hvrp, list_taillard_hvrp

    cfg = make_taillard_hvrp("c50_13hvrp", T=5, num_scenarios=5)
    cfg = make_taillard_hvrp("c100_20hvrp", T=7, num_scenarios=6)

    for name, meta in list_taillard_hvrp().items():
        print(name, meta)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from core.instance import VRPConfig

# 原始客户、车辆、车型、容量和成本只从数据文件解析，不在源码重复登记。
_TAILLARD_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "hfvrp" / "taillard"
)

# 默认源算例。Main 可在入口顶部改选其他 Taillard HVRP，并始终使用完整数据；
# Compare 使用此默认算例并显式传入 C/V 前缀做小规模公式对拍。
DEFAULT_TAILLARD_HVRP = "c50_14hvrp"


# Taillard 原文件 fixed_cost 为 routing benchmark 量级 (20~400)。这里仅保存
# 构造长期车队投资实例所需的经济参数；BPC/ESP/Concorde 等求解策略属于
# setup.solver_config，而不是数据集属性。
TAILLARD_ECONOMIC_DEFAULTS: Dict[str, float] = {
    "c_out": 250.0,
    "purchase_cost_scale": 1.0,
    "ops_per_year": 10.0,
}

# 容量缩放会改变可行域，不是经济参数。Taillard helper 默认忠实保留原始
# capacity；需要合成压力实例的调用方必须显式传入非 1 倍率并明确标注。
DEFAULT_TAILLARD_CAPACITY_SCALE = 1.0


def taillard_economic_defaults() -> Dict[str, float]:
    """返回 Taillard 长期投资实例的默认经济参数副本。"""
    return dict(TAILLARD_ECONOMIC_DEFAULTS)


def list_taillard_hvrp() -> Dict[str, Dict[str, Any]]:
    """直接从 Taillard 文件解析并返回原始规模摘要。"""
    from core.hfvrp_loader import parse_hfvrp_file

    instances: Dict[str, Dict[str, Any]] = {}
    for name in taillard_hvrp_names():
        parsed = parse_hfvrp_file(name)
        vehicles = parsed["vehicles"]
        type_specs = {
            (
                vehicle["capacity"],
                vehicle["fixed_cost"],
                vehicle["unit_cost"],
            )
            for vehicle in vehicles
        }
        instances[name] = {
            "customers": parsed["num_customers"],
            "vehicles": len(vehicles),
            "types": len(type_specs),
            "caps": sorted({vehicle["capacity"] for vehicle in vehicles}),
        }
    return instances


def taillard_hvrp_names() -> list[str]:
    """以数据目录为唯一真源，返回可用的 Taillard 算例名。"""
    return sorted(path.stem for path in _TAILLARD_DATA_DIR.glob("*hvrp.txt"))


def is_taillard_hvrp(name: str) -> bool:
    """Return whether ``name`` resolves to a file in the Taillard data set."""
    return name.removesuffix(".txt") in taillard_hvrp_names()


# Entry points may pass None to keep TAILLARD_ECONOMIC_DEFAULTS.
_ECONOMIC_OPTIONAL_KEYS = ("c_out", "purchase_cost_scale", "ops_per_year")


def make_taillard_hvrp(
    name: str,
    *,
    T: int,
    num_scenarios: int,
    **overrides,
) -> VRPConfig:
    """按数据文件名构建 VRPConfig，并要求显式给出实验 T/S。

    ``name`` 可带或不带 ``.txt`` 后缀.
    经济参数若传入 ``None``，保留 ``TAILLARD_ECONOMIC_DEFAULTS``。
    """
    stem = name.removesuffix(".txt")
    if not is_taillard_hvrp(stem):
        known = ", ".join(taillard_hvrp_names())
        raise KeyError(f"未知 Taillard HVRP 算例 {name!r}; 可选: {known}")

    econ = taillard_economic_defaults()
    kwargs: Dict[str, Any] = {
        "data_source": "hfvrp",
        "hfvrp_file": stem,
        "num_customers": None,
        "T": T,
        "num_scenarios": num_scenarios,
        "scenario_seed": 42,
        "demand_dist": "lognormal",
        "demand_cv": 0.3,
        "demand_active_prob": 1.0,
        "c_out": econ["c_out"],
        "B_t0": None,
        "ops_per_year": econ["ops_per_year"],
        "purchase_cost_scale": econ["purchase_cost_scale"],
        "capacity_scale": DEFAULT_TAILLARD_CAPACITY_SCALE,
    }
    kwargs.update(
        {
            key: value
            for key, value in overrides.items()
            if not (key in _ECONOMIC_OPTIONAL_KEYS and value is None)
        }
    )
    return VRPConfig(**kwargs)
