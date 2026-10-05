import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from core.backend_telemetry import backend_call, record_backend_event


def _concorde_threshold() -> int:
    """运行时读取 env, 以便 main 在 import 之后设置 Taillard 默认值."""
    return int(os.environ.get("VRP_FORWARD_S3_CONCORDE_THRESHOLD", "15"))


def _bool_env(name: str, default_val: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default_val)
    return raw not in ("0", "false", "False")


def is_forward_s3_concorde_enabled() -> bool:
    """Forward stage-3 Concorde 开关 (env ``VRP_FORWARD_S3_USE_CONCORDE``, 默认关).

    兼容旧 env: ``VRP_FORWARD_S3_SOLVER=concorde`` 也会启用 Concorde.
    """
    if _bool_env("VRP_FORWARD_S3_USE_CONCORDE", False):
        return True
    legacy = os.environ.get("VRP_FORWARD_S3_SOLVER", "gurobi").strip().lower()
    return legacy == "concorde"


def get_forward_stage3_solver(n_assigned: Optional[int] = None) -> str:
    """返回 ``"concorde"`` 或 ``"gurobi"``.

    规则:
    1) 显式开关: ``VRP_FORWARD_S3_USE_CONCORDE=1`` 或 legacy ``VRP_FORWARD_S3_SOLVER=concorde`` ⇒ 总是 concorde.
    2) 自动模式(默认): 当 ``n_assigned > VRP_FORWARD_S3_CONCORDE_THRESHOLD`` 时用 concorde, 否则 gurobi.
    """
    if is_forward_s3_concorde_enabled():
        return "concorde"
    if n_assigned is None:
        return "gurobi"
    return "concorde" if int(n_assigned) > _concorde_threshold() else "gurobi"


def _resolve_concorde_bin() -> str | None:
    bin_name = os.environ.get("VRP_CONCORDE_BIN", "concorde")
    if os.path.isabs(bin_name):
        if os.path.isfile(bin_name) and os.access(bin_name, os.X_OK):
            return bin_name
        return None

    # 1) Respect PATH first (for system-installed concorde).
    in_path = shutil.which(bin_name)
    if in_path:
        return in_path

    # 2) If user supplied a relative path in env, resolve from cwd.
    if bin_name != "concorde":
        rel = Path(bin_name).expanduser()
        if rel.is_file() and os.access(rel, os.X_OK):
            return str(rel.resolve())
        return None

    # 3) Project-local fallback candidates (common in this repo).
    repo_root = Path(__file__).resolve().parents[1]
    local_candidates = [
        repo_root / "tools" / "concorde" / "concorde",
        repo_root / "tools" / "concorde_build" / "concorde-bld" / "TSP" / "concorde",
    ]
    for cand in local_candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def solve_stage3_with_concorde(prob_data, node, x_prev) -> Dict:
    """Record route acceptance separately from actual CLI invocations."""
    result = _solve_stage3_with_concorde(prob_data, node, x_prev)
    reason = result.get("reason")
    operation = "accepted" if result.get("ok", False) else "rejected"
    if reason == "empty_route" or str(reason).startswith("concorde_not_found:"):
        operation = "skip"
    record_backend_event(
        "concorde", operation, reason or "route_certified",
        stage=3, vehicle=node.info,
    )
    return result


def _solve_stage3_with_concorde(prob_data, node, x_prev) -> Dict:
    """Forward Stage-3 TSP via Concorde CLI."""
    v = node.info
    depot_start = prob_data.numAllnodes - 2
    depot_end = prob_data.numAllnodes - 1
    assigned_j = [
        j for j in prob_data.J
        if round(x_prev.get(f"alpha[{j},{v}]", 0.0)) == 1 and node.active[j] == 1
    ]

    if not assigned_j:
        return {
            "ok": True,
            "reason": "empty_route",
            "obj": 0.0,
            "stage_cost": 0.0,
            "x_star_dict": {},
            "route_nodes": [],
        }

    use_n = assigned_j + [depot_start, depot_end]
    n = len(use_n)

    bin_path = _resolve_concorde_bin()
    if bin_path is None:
        bin_name = os.environ.get("VRP_CONCORDE_BIN", "concorde")
        return {"ok": False, "reason": f"concorde_not_found:{bin_name}"}

    c = prob_data.c_routing[v]
    sub = np.asarray(c, dtype=np.float64)[np.ix_(use_n, use_n)]
    if not np.all(np.isfinite(sub)):
        return {"ok": False, "reason": "nonfinite_cost_matrix"}

    eps = float(os.environ.get("VRP_CONCORDE_SYM_EPS", "1e-8"))
    if np.max(np.abs(sub - sub.T)) > eps:
        return {"ok": False, "reason": "asymmetric_cost_matrix"}

    scale_env = os.environ.get("VRP_CONCORDE_SCALE")
    scale = max(1, int(scale_env)) if scale_env not in (None, "") else 1

    int_mat_np = np.round(sub * scale).astype(np.int64)
    np.fill_diagonal(int_mat_np, 0)

    s_idx = n - 2
    e_idx = n - 1
    int_mat_np[s_idx, e_idx] = 0
    int_mat_np[e_idx, s_idx] = 0

    timeout_s = float(os.environ.get("VRP_CONCORDE_TIMEOUT_S", "30"))

    with tempfile.TemporaryDirectory(prefix="vrp_concorde_") as td:
        tsp_path = os.path.join(td, "stage3.tsp")
        sol_path = os.path.join(td, "stage3.sol")
        with open(tsp_path, "w", encoding="ascii") as f:
            f.write("NAME: stage3\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {n}\n")
            f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in int_mat_np.tolist():
                f.write(" ".join(str(vv) for vv in row))
                f.write("\n")
            f.write("EOF\n")

        try:
            with backend_call(
                "concorde", "subprocess", stage=3,
                time_limit=timeout_s, vehicle=v, must_visit=len(assigned_j),
            ) as telemetry:
                try:
                    cp = subprocess.run(
                        [bin_path, "-o", sol_path, tsp_path],
                        cwd=td,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=timeout_s,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    telemetry["reason"] = "concorde_timeout"
                    raise
                telemetry["status"] = cp.returncode
                telemetry["reason"] = (
                    "process_completed" if cp.returncode == 0
                    else f"concorde_exit_{cp.returncode}"
                )
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "concorde_timeout"}
        except Exception as exc:
            return {"ok": False, "reason": f"concorde_exception:{type(exc).__name__}"}

        if cp.returncode != 0:
            tail = (cp.stdout or "").strip().splitlines()[-3:]
            return {
                "ok": False,
                "reason": f"concorde_exit_{cp.returncode}: {' | '.join(tail)}",
            }
        if not os.path.exists(sol_path):
            return {"ok": False, "reason": "concorde_no_solution_file"}

        try:
            tour = _read_concorde_tour(sol_path)
        except (OSError, UnicodeError, ValueError) as exc:
            return {
                "ok": False,
                "reason": f"invalid_solution_file:{type(exc).__name__}",
            }
        if len(tour) != n:
            return {"ok": False, "reason": "invalid_tour_length"}

    path_local, path_reason = _cycle_to_open_path(tour, s_idx, e_idx)
    if path_reason is not None:
        return {"ok": False, "reason": path_reason}  # need depot-copy zero edge

    route_nodes = [use_n[k] for k in path_local]
    if (
        route_nodes[0] != depot_start
        or route_nodes[-1] != depot_end
        or len(route_nodes) != n
        or len(set(route_nodes)) != n
        or set(route_nodes[1:-1]) != set(assigned_j)
    ):
        return {"ok": False, "reason": "invalid_open_route_coverage"}
    try:
        stage_cost = _path_cost(route_nodes, c)
    except (OverflowError, ValueError):
        return {"ok": False, "reason": "invalid_open_route_cost"}
    if not math.isfinite(stage_cost):
        return {"ok": False, "reason": "invalid_open_route_cost"}

    x_star_dict = {
        f"x[{a},{b}]": 1.0
        for a, b in zip(route_nodes[:-1], route_nodes[1:])
    }

    return {
        "ok": True,
        "reason": None,
        "obj": stage_cost,
        "stage_cost": stage_cost,
        "x_star_dict": x_star_dict,
        "route_nodes": route_nodes,
    }


def _read_concorde_tour(sol_path: str) -> List[int]:
    with open(sol_path, "r", encoding="ascii") as f:
        tokens = [int(tok) for tok in f.read().strip().split()]
    if not tokens:
        return []
    n = tokens[0]
    return tokens[1: 1 + n]


def _cycle_to_open_path(
    tour: List[int], start: int, end: int
) -> tuple[List[int], Optional[str]]:
    """Remove the artificial depot edge from a Hamiltonian cycle.

    Local Concorde node IDs must be the exact permutation ``0..n-1``.  The
    two depot copies must be adjacent in the returned cycle; otherwise no
    single start-to-end arc contains every customer and the cycle cannot be
    interpreted as a feasible open route.
    """
    n = len(tour)
    if n < 2:
        return [], "invalid_tour_length"
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start == end
        or start < 0
        or end < 0
        or start >= n
        or end >= n
    ):
        return [], "invalid_depot_indices"
    if any(isinstance(node, bool) or not isinstance(node, int) for node in tour):
        return [], "invalid_tour_permutation"
    if set(tour) != set(range(n)) or len(set(tour)) != n:
        return [], "invalid_tour_permutation"

    try:
        ps = tour.index(start)
        pe = tour.index(end)
    except ValueError:
        return [], "invalid_tour_permutation"

    if (ps - pe) % n not in (1, n - 1):
        return [], "nonadjacent_depots"
    if n == 2:
        return [start, end], None

    def _walk(forward: bool) -> List[int]:
        path = [start]
        cur = ps
        while cur != pe:
            cur = (cur + 1) % n if forward else (cur - 1 + n) % n
            path.append(tour[cur])
            if len(path) > n + 1:
                return []
        return path

    path_fwd = _walk(forward=True)
    path_bwd = _walk(forward=False)
    candidates = [path for path in (path_fwd, path_bwd) if len(path) == n]
    if len(candidates) != 1:
        return [], "cannot_extract_open_path"
    path = candidates[0]
    if (
        path[0] != start
        or path[-1] != end
        or len(set(path)) != n
        or set(path) != set(range(n))
        or start in path[1:]
        or end in path[:-1]
    ):
        return [], "invalid_open_path"
    return path, None


def _path_cost(path: List[int], c) -> float:
    return math.fsum(float(c[i, j]) for i, j in zip(path[:-1], path[1:]))
