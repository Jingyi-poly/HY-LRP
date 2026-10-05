"""HFVRP benchmark 文件解析 (VRPLIB 风格 .vrp + Taillard .txt).

返回的 dict 字段统一为:
    {
      'name': str, 'source_path': Path,
      'num_customers': int,           # 不含 depot
      'coords': np.ndarray (n_all, 2) # 所有节点 (customers + 2 depots)
      'demand_nominal': np.ndarray (num_customers,)  # 名义需求 (depot 不含)
      'vehicles': list[dict] [
          {'capacity': float, 'fixed_cost': float, 'unit_cost': float}, ...
      ],
    }

仓库处理: HFVRP 单仓库 (depot id=1). 我们的模型要 numDepots=2 (start, end),
所以把 depot 坐标复制为两个节点 (numCustomers, numCustomers+1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# Section keywords we know
_VRPLIB_SECTIONS = {
    "NODE_COORD_SECTION",
    "DEMAND_SECTION",
    "CAPACITY_SECTION",
    "VEHICLES_FIXED_COST_SECTION",
    "VEHICLES_UNIT_DISTANCE_COST_SECTION",
    "DEPOT_SECTION",
    "EOF",
}


def resolve_hfvrp_path(name_or_path: str, data_root: Optional[Path] = None) -> Path:
    """支持 (1) 绝对/相对路径, (2) 文件名 (eg "X115-HVRP" / "c50_13hvrp")."""
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()

    if data_root is None:
        data_root = Path(__file__).resolve().parent.parent.parent / "data" / "hfvrp"
    data_root = Path(data_root)

    candidates: List[Path] = []
    for sub in ("X", "mini", "taillard"):
        d = data_root / sub
        if not d.is_dir():
            continue
        for ext in (".vrp", ".txt"):
            f = d / f"{name_or_path}{ext}"
            if f.exists():
                candidates.append(f)
        for f in d.glob(f"{name_or_path}*"):
            if f.is_file():
                candidates.append(f)
    if not candidates:
        raise FileNotFoundError(
            f"找不到 HFVRP 算例 {name_or_path!r}; 搜索路径={data_root} (子目录 X/mini/taillard)"
        )
    return candidates[0].resolve()


def _detect_format(path: Path) -> str:
    """返回 'vrplib' 或 'taillard'."""
    if path.suffix.lower() == ".vrp":
        return "vrplib"
    with open(path, "r") as fh:
        first = fh.readline().strip()
        if not first:
            return "vrplib"
        try:
            int(first.split()[0])
            return "taillard"
        except ValueError:
            return "vrplib"


def parse_hfvrp_file(name_or_path: str,
                     data_root: Optional[Path] = None) -> Dict[str, Any]:
    """统一入口."""
    path = resolve_hfvrp_path(name_or_path, data_root)
    fmt = _detect_format(path)
    if fmt == "vrplib":
        raw = _parse_vrplib(path)
    else:
        raw = _parse_taillard(path)
    return _normalize(raw, path)


# ---------------------------------------------------------------- VRPLIB ----
def _parse_vrplib(path: Path) -> Dict[str, Any]:
    header: Dict[str, str] = {}
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    with open(path, "r") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n").strip()
            if not line or line == "EOF":
                if line == "EOF":
                    current = None
                continue
            if line in _VRPLIB_SECTIONS:
                current = line
                sections.setdefault(current, [])
                continue
            if current is None:
                if ":" in line:
                    k, _, v = line.partition(":")
                    header[k.strip().upper()] = v.strip()
                continue
            sections[current].append(line)

    dim = int(header.get("DIMENSION", "0"))
    num_vehicles = int(header.get("VEHICLES", header.get("VEHICLE", "0")))
    if dim <= 0 or num_vehicles <= 0:
        raise ValueError(f"{path.name}: DIMENSION 或 VEHICLES 缺失/非法")

    coords = np.zeros((dim, 2), dtype=np.float64)
    for ln in sections.get("NODE_COORD_SECTION", []):
        parts = ln.split()
        if len(parts) < 3:
            continue
        idx = int(parts[0]) - 1
        coords[idx, 0] = float(parts[1])
        coords[idx, 1] = float(parts[2])

    demand = np.zeros(dim, dtype=np.float64)
    for ln in sections.get("DEMAND_SECTION", []):
        parts = ln.split()
        if len(parts) < 2:
            continue
        idx = int(parts[0]) - 1
        demand[idx] = float(parts[1])

    depot_ids: List[int] = []
    for ln in sections.get("DEPOT_SECTION", []):
        for tok in ln.split():
            try:
                v = int(tok)
            except ValueError:
                continue
            if v <= 0:
                continue
            depot_ids.append(v - 1)
    if not depot_ids:
        depot_ids = [int(np.argmin(demand))]

    def _read_per_vehicle(section_name: str, default: Optional[float]) -> np.ndarray:
        rows = sections.get(section_name, [])
        arr = np.full(num_vehicles, np.nan, dtype=np.float64)
        for ln in rows:
            parts = ln.split()
            if len(parts) < 2:
                continue
            idx = int(parts[0]) - 1
            if 0 <= idx < num_vehicles:
                arr[idx] = float(parts[1])
        if np.isnan(arr).any():
            if default is None:
                missing = int(np.isnan(arr).sum())
                raise ValueError(
                    f"{path.name}: {section_name} 缺 {missing} 条且无默认值"
                )
            arr = np.where(np.isnan(arr), default, arr)
        return arr

    capacity = _read_per_vehicle("CAPACITY_SECTION", None)
    fixed_cost = _read_per_vehicle("VEHICLES_FIXED_COST_SECTION", 0.0)
    unit_cost = _read_per_vehicle("VEHICLES_UNIT_DISTANCE_COST_SECTION", 1.0)

    return {
        "name": header.get("NAME", path.stem),
        "dim": dim,
        "coords": coords,
        "demand_with_depot": demand,
        "depot_ids": depot_ids,
        "capacity": capacity,
        "fixed_cost": fixed_cost,
        "unit_cost": unit_cost,
    }


# -------------------------------------------------------------- Taillard ----
def _parse_taillard(path: Path) -> Dict[str, Any]:
    """格式: line1=N_cust; line2..N+2: id x y demand (id=0 = depot);
    之后一行 = N_types; 每个 type 一行: cap fix unit ?? avail."""
    with open(path, "r") as fh:
        raw_lines = [ln.strip() for ln in fh if ln.strip() != ""]
    if not raw_lines:
        raise ValueError(f"{path.name}: 空文件")

    n_cust = int(raw_lines[0].split()[0])
    nodes_block = raw_lines[1: 1 + n_cust + 1]
    if len(nodes_block) != n_cust + 1:
        raise ValueError(
            f"{path.name}: 头行 n_cust={n_cust} 但节点行数={len(nodes_block)} (期望 {n_cust + 1})"
        )

    dim = n_cust + 1
    coords = np.zeros((dim, 2), dtype=np.float64)
    demand = np.zeros(dim, dtype=np.float64)
    depot_id: Optional[int] = None
    for ln in nodes_block:
        parts = ln.split()
        idx = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        d = float(parts[3]) if len(parts) >= 4 else 0.0
        coords[idx, 0] = x
        coords[idx, 1] = y
        demand[idx] = d
        if d == 0.0 and depot_id is None:
            depot_id = idx
    if depot_id is None:
        depot_id = 0
    veh_block_start = 1 + n_cust + 1
    if veh_block_start >= len(raw_lines):
        raise ValueError(f"{path.name}: 缺车辆类型段")
    n_types = int(raw_lines[veh_block_start].split()[0])
    type_rows = raw_lines[veh_block_start + 1: veh_block_start + 1 + n_types]
    if len(type_rows) != n_types:
        raise ValueError(
            f"{path.name}: 车型行数 {len(type_rows)} ≠ N_types={n_types}"
        )

    caps: List[float] = []
    fixs: List[float] = []
    units: List[float] = []
    for ln in type_rows:
        parts = ln.split()
        if len(parts) < 5:
            raise ValueError(
                f"{path.name}: 车型行字段 < 5: {ln!r} (期望 cap fix unit ?? avail)"
            )
        cap = float(parts[0])
        fix = float(parts[1])
        unit = float(parts[2])
        avail = int(float(parts[4]))
        for _ in range(avail):
            caps.append(cap)
            fixs.append(fix)
            units.append(unit)

    return {
        "name": path.stem,
        "dim": dim,
        "coords": coords,
        "demand_with_depot": demand,
        "depot_ids": [depot_id],
        "capacity": np.asarray(caps, dtype=np.float64),
        "fixed_cost": np.asarray(fixs, dtype=np.float64),
        "unit_cost": np.asarray(units, dtype=np.float64),
    }


# -------------------------------------------------------- Normalization ----
def _normalize(raw: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """把 HFVRP 单仓库扩展为 (numCustomers, numCustomers+1) 两个 depot 节点."""
    coords_in = raw["coords"]
    demand_in = raw["demand_with_depot"]
    depot_ids = raw["depot_ids"]
    if len(depot_ids) != 1:
        depot_ids = depot_ids[:1] or [int(np.argmin(demand_in))]
    depot = depot_ids[0]

    cust_mask = np.ones(coords_in.shape[0], dtype=bool)
    cust_mask[depot] = False
    customer_idx = np.where(cust_mask)[0]
    num_customers = int(customer_idx.size)

    new_coords = np.zeros((num_customers + 2, 2), dtype=np.float64)
    new_coords[:num_customers] = coords_in[customer_idx]
    new_coords[num_customers] = coords_in[depot]
    new_coords[num_customers + 1] = coords_in[depot]

    demand_nominal = demand_in[customer_idx].astype(np.float64)

    vehicles: List[Dict[str, float]] = []
    n_v = int(raw["capacity"].shape[0])
    for v in range(n_v):
        vehicles.append({
            "capacity": float(raw["capacity"][v]),
            "fixed_cost": float(raw["fixed_cost"][v]),
            "unit_cost": float(raw["unit_cost"][v]),
        })

    return {
        "name": raw["name"],
        "source_path": path,
        "num_customers": num_customers,
        "coords": new_coords,
        "demand_nominal": demand_nominal,
        "vehicles": vehicles,
    }


def euc_2d_distance(coords: np.ndarray) -> np.ndarray:
    """全对称欧氏距离, 不四舍五入 (我们的 routing 模型支持实数 cost)."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def group_identical_vehicles(vehicles: List[Dict[str, float]]) -> Dict[int, List[int]]:
    """按完全相同的 (capacity, fixed_cost, unit_cost) 分组。"""
    type_of: Dict[tuple, int] = {}
    out: Dict[int, List[int]] = {}
    for v, spec in enumerate(vehicles):
        key = (
            spec["capacity"],
            spec["fixed_cost"],
            spec["unit_cost"],
        )
        if key not in type_of:
            type_of[key] = len(type_of)
            out[type_of[key]] = []
        out[type_of[key]].append(v)
    return out
