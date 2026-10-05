"""Branch-and-price backend for fixed-fleet Stage-2 pieces beyond the DP ceiling.

The exact subset DP (``subset_dp.py``) enumerates ``2^n`` customer sets and is
limited to about 20 active customers by memory.  Larger nodes used to fall
back to the compact Gurobi model, whose LP relaxation only sees the convex
envelope of the Stage-3 cut maxima and therefore branches for minutes on C50.

This module drives the C++ kernel ``cpp/s2_bp_kernel.cpp``: a set-partitioning
master over exact-cost vehicle columns, priced by an exact min-max knapsack
search, with the canonical ``assignment_order`` rows kept in the master.  It
solves the *same* model as the DP and the Gurobi builder (checked by
``tests/test_s2_bp_solver.py`` against both), so the result is certified and
scored through the shared policy certifier exactly like a DP result:

* the incumbent assignment is feasibility-certified and archive-rescored
  (``cost_star_value`` is the exact directed-up policy value, never the
  kernel's own objective);
* the kernel's lower bound is a Lagrangian bound with explicit floating-point
  allowances; adding the exact outsourcing cost of inactive customers gives a
  rigorous lower endpoint for the piece value ``C(n)``.

Environment:
  VRP_S2_BP_KERNEL        auto|off      use the kernel when importable (default auto)
  VRP_S2_BP_REL_GAP       float         B&B relative gap target (default 0 = close)
  VRP_S2_BP_ABS_GAP       float         B&B absolute gap target (default 1e-7)
  VRP_S2_BP_ACCEPT_REL_GAP float        accept a time-limited interval up to this
                                        relative width (default 1e-4, Gurobi's MIPGap)
  VRP_S2_BP_TIME_LIMIT    float         default per-solve wall limit when the caller
                                        gives none (default 120 s)
  VRP_S2_BP_VERBOSE       int           kernel logging level (default 0)
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from core.backend_telemetry import backend_call, record_backend_event

from cuts import exact_subroutines as exact_sub
from solvers.forward_policy_certification import certify_stage2_forward_policy
from solvers.forward_stage2_policy import score_forward_stage2_policy

from .fleet_pieces import Counts, FleetLayout, float_down
from .forward_stage2_dp import (
    ForwardStage2DPResult,
    _FORWARD_CERTIFIED_GAP_ABS_TOL,
    _SELF_CHECK_ABS_TOL,
    _SELF_CHECK_REL_TOL,
    dense_stage2_x_dict,
)
from .mip_start import exact_forward_policy_score
from .purpose import (
    PHASE2_FORWARD,
    ForwardStage2Purpose,
    normalize_forward_stage2_purpose,
)
from .subset_dp import SubsetDPNotApplicable, _dyadic_int64

STATUS = "s2_bp"
KERNEL_ENV = "VRP_S2_BP_KERNEL"
REL_GAP_ENV = "VRP_S2_BP_REL_GAP"
ABS_GAP_ENV = "VRP_S2_BP_ABS_GAP"
ACCEPT_REL_GAP_ENV = "VRP_S2_BP_ACCEPT_REL_GAP"
TIME_LIMIT_ENV = "VRP_S2_BP_TIME_LIMIT"
VERBOSE_ENV = "VRP_S2_BP_VERBOSE"
DEFAULT_REL_GAP = 0.0
DEFAULT_ABS_GAP = 1e-7
DEFAULT_ACCEPT_REL_GAP = 1e-4
DEFAULT_TIME_LIMIT = 120.0


class BPNotApplicable(RuntimeError):
    """The kernel cannot represent this node (caller falls back)."""


def _load_kernel():
    if os.environ.get(KERNEL_ENV, "auto").strip().lower() in ("0", "off", "false", "no"):
        return None
    kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp")
    if os.path.isdir(kernel_dir) and kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    try:
        import s2_bp_kernel  # type: ignore
    except ImportError:
        return None
    return s2_bp_kernel


_KERNEL = _load_kernel()


def kernel_available() -> bool:
    return _KERNEL is not None


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


# --------------------------------------------------------------------------
# piece compilation
# --------------------------------------------------------------------------


@dataclass
class BPPiece:
    """One fixed-fleet Stage-2 piece compiled for the kernel."""

    customers: List[Any]
    vehicles_all: List[Any]
    active_idx: List[int]              # positions in ``customers``
    fleet: List[Any]                   # leading-run vehicles (kernel order)
    counts: Counts
    c_out: List[float]                 # active customers
    q_int: List[int]
    w_int: List[int]
    caps_int: List[int]
    types: List[int]
    empty_theta: List[float]
    cut_a: List[List[float]]
    cut_p: List[List[List[float]]]
    inactive_outsourcing: Fraction
    off_fleet_theta: Fraction          # sum of theta(empty) over vehicles with z = 0
    n_cuts: int

    @property
    def fixed_offset(self) -> Fraction:
        """Constant part of ``C(counts)`` outside the kernel objective."""
        return self.inactive_outsourcing + self.off_fleet_theta

    @property
    def n(self) -> int:
        return len(self.active_idx)

    @property
    def m(self) -> int:
        return len(self.fleet)

    def assign_to_x_dict(self, assign: Sequence[int]) -> Dict[str, float]:
        """Sparse ``alpha[j,v]``/``y[v]`` dictionary of a kernel assignment."""
        out: Dict[str, float] = {}
        used = set()
        for pos, k in zip(self.active_idx, assign):
            if k < 0:
                continue
            vehicle = self.fleet[int(k)]
            out[f"alpha[{self.customers[pos]},{vehicle}]"] = 1.0
            used.add(vehicle)
        for vehicle in used:
            out[f"y[{vehicle}]"] = 1.0
        return out

    def hint_from_x_dict(self, x_dict: Optional[Mapping[str, float]]) -> List[int]:
        hint = [-1] * self.n
        if not x_dict:
            return hint
        fleet_pos = {vehicle: k for k, vehicle in enumerate(self.fleet)}
        for i, pos in enumerate(self.active_idx):
            customer = self.customers[pos]
            for vehicle, k in fleet_pos.items():
                if float(x_dict.get(f"alpha[{customer},{vehicle}]", 0.0)) > 0.5:
                    hint[i] = k
                    break
        return hint


def _check_cut_locality(cuts_payload, m: int, n: int) -> None:
    for index, cut in enumerate(cuts_payload):
        succ = int(cut["succ"])
        if succ < 0 or succ >= m:
            raise BPNotApplicable(f"cut {index} has successor {succ} out of range")
        pi_y = list(cut["piY"])
        pi_alpha = list(cut["piAlpha"])
        if len(pi_y) != m or len(pi_alpha) != m:
            raise BPNotApplicable(f"cut {index} has a bad shape")
        for pos in range(m):
            if pos == succ:
                continue
            if float(pi_y[pos]) != 0.0 or any(float(c) != 0.0 for c in pi_alpha[pos]):
                raise BPNotApplicable(
                    f"cut {index} of successor {succ} references vehicle position {pos}"
                )
        if len(pi_alpha[succ]) != n:
            raise BPNotApplicable(f"cut {index} has a bad shape")


def compile_piece(prob_data, node, cuts_payload, layout: FleetLayout, counts: Counts) -> BPPiece:
    """Compile ``C(counts)`` for the kernel (exact integer volumes/scores)."""
    if _KERNEL is None:
        raise BPNotApplicable("s2_bp_kernel is not built (customized-subprob/s2forward/cpp/build_bp.sh)")
    customers = list(prob_data.J)
    vehicles_all = list(prob_data.V)
    if len(node.successor) != len(vehicles_all):
        raise BPNotApplicable("one Stage-3 successor per vehicle required")
    counts = layout.check_counts(counts)
    active_idx = [pos for pos, j in enumerate(customers) if int(node.active[j]) == 1]
    n = len(active_idx)
    if n > int(_KERNEL.MAX_CUSTOMERS):
        raise BPNotApplicable(f"{n} active customers > kernel limit {_KERNEL.MAX_CUSTOMERS}")
    cuts_payload = list(cuts_payload)
    _check_cut_locality(cuts_payload, len(vehicles_all), len(customers))

    fleet = layout.prefix_vehicles(counts)
    vehicle_pos = {v: pos for pos, v in enumerate(vehicles_all)}
    group_of = {v: g for g, group in enumerate(layout.groups) for v in group}

    active_customers = [customers[pos] for pos in active_idx]
    c_out = [float(node.c_out[j]) for j in active_customers]
    inactive = sum(
        (Fraction(float(node.c_out[j])) for pos, j in enumerate(customers) if pos not in set(active_idx)),
        Fraction(0),
    )
    volumes = [float(node.volume[j]) for j in active_customers]
    caps = [float(prob_data.Qv[v]) for v in fleet]
    try:
        scaled = _dyadic_int64(volumes + caps)
    except SubsetDPNotApplicable as exc:
        raise BPNotApplicable(str(exc)) from exc
    q_int = [int(x) for x in scaled[:n]]
    caps_int = [int(x) for x in scaled[n:]]
    weights = [float(np.round(np.log(j + 2), 4)) for j in active_customers]
    try:
        w_int = [int(x) for x in _dyadic_int64(weights)]
    except SubsetDPNotApplicable as exc:
        raise BPNotApplicable(str(exc)) from exc
    for g in layout.groups:
        cap_set = {float(prob_data.Qv[v]) for v in g}
        if len(cap_set) > 1:
            raise BPNotApplicable("vehicles of one type must share their capacity")

    types = [group_of[v] for v in fleet]
    empty_theta: List[float] = []
    cut_a: List[List[float]] = []
    cut_p: List[List[List[float]]] = []
    n_cuts = 0
    fleet_set = set(fleet)
    # Vehicles with z = 0 still carry theta_v >= max(0, beta_c) (y = alpha = 0).
    off_fleet_theta = Fraction(0)
    for v in vehicles_all:
        pos = vehicle_pos[v]
        betas = [float(cut["beta"]) for cut in cuts_payload if int(cut["succ"]) == pos]
        theta_empty = max(0.0, max(betas, default=0.0))
        if v not in fleet_set:
            off_fleet_theta += Fraction(theta_empty)
            continue
        a_list: List[float] = []
        p_list: List[List[float]] = []
        for cut in cuts_payload:
            if int(cut["succ"]) != pos:
                continue
            a_list.append(float(cut["beta"]) + float(cut["piY"][pos]))
            row = cut["piAlpha"][pos]
            p_list.append([float(row[j]) for j in active_idx])
        n_cuts += len(a_list)
        empty_theta.append(theta_empty)
        cut_a.append(a_list)
        cut_p.append(p_list)
    return BPPiece(
        customers=customers,
        vehicles_all=vehicles_all,
        active_idx=active_idx,
        fleet=list(fleet),
        counts=counts,
        c_out=c_out,
        q_int=q_int,
        w_int=w_int,
        caps_int=caps_int,
        types=types,
        empty_theta=empty_theta,
        cut_a=cut_a,
        cut_p=cut_p,
        inactive_outsourcing=inactive,
        off_fleet_theta=off_fleet_theta,
        n_cuts=n_cuts,
    )


# --------------------------------------------------------------------------
# kernel call
# --------------------------------------------------------------------------


@dataclass
class BPSolve:
    """Raw kernel outcome for one piece (before policy certification)."""

    x_dict_sparse: Dict[str, float]
    assign: List[int]
    kernel_ub: float                   # active-customer objective as the kernel sees it
    kernel_lb: float
    lower_bound: float                 # rigorous lower endpoint of C(counts) (float_down)
    upper_bound_kernel: float          # kernel_ub + inactive outsourcing (informational)
    status: str
    optimal: bool
    has_solution: bool
    seconds: float
    stats: Dict[str, Any] = field(default_factory=dict)


def solve_piece_bp(
    piece: BPPiece,
    *,
    time_limit: Optional[float] = None,
    rel_gap: Optional[float] = None,
    abs_gap: Optional[float] = None,
    bound_stop: Optional[float] = None,
    hint_x_dict: Optional[Mapping[str, float]] = None,
    verbose: Optional[int] = None,
) -> BPSolve:
    """Run the kernel on a compiled piece and return its certified interval."""
    if _KERNEL is None:
        record_backend_event("branch_price", "skip", "kernel_not_built", stage=2)
        raise BPNotApplicable("s2_bp_kernel is not built")
    params = {
        "time_limit": float(
            _float_env(TIME_LIMIT_ENV, DEFAULT_TIME_LIMIT) if time_limit is None else time_limit
        ),
        "rel_gap": float(_float_env(REL_GAP_ENV, DEFAULT_REL_GAP) if rel_gap is None else rel_gap),
        "abs_gap": float(_float_env(ABS_GAP_ENV, DEFAULT_ABS_GAP) if abs_gap is None else abs_gap),
        "verbose": int(_int_env(VERBOSE_ENV, 0) if verbose is None else verbose),
    }
    if not (params["time_limit"] > 0.0):
        params["time_limit"] = 1e-3
    if bound_stop is not None and math.isfinite(float(bound_stop)):
        # the kernel works on the active-customer objective
        params["bound_stop"] = float(Fraction(float(bound_stop)) - piece.fixed_offset)
    hint = piece.hint_from_x_dict(hint_x_dict)
    started = time.perf_counter()
    with backend_call("branch_price", "solve", stage=2,
                      time_limit=params["time_limit"], active_customers=piece.n,
                      fleet_size=piece.m) as telemetry:
        raw = _KERNEL.solve(
            piece.c_out, piece.q_int, piece.w_int, piece.caps_int, piece.types,
            piece.empty_theta, piece.cut_a, piece.cut_p, hint, params,
        )
        telemetry.update(status=raw.get("status"), reason=raw.get("error"),
                         proven=bool(raw.get("optimal", False)),
                         has_solution=bool(raw.get("has_solution", False)),
                         lower_bound=raw.get("lb"), upper_bound=raw.get("ub"))
    seconds = time.perf_counter() - started
    if raw.get("error"):
        raise BPNotApplicable(f"kernel error: {raw['error']}")
    has_solution = bool(raw.get("has_solution", False))
    assign = [int(k) for k in raw.get("assign", [])] if has_solution else []
    kernel_lb = float(raw["lb"])
    kernel_ub = float(raw["ub"]) if has_solution else math.inf
    if math.isfinite(kernel_lb):
        lower = float_down(Fraction(kernel_lb) + piece.fixed_offset)
    else:
        lower = kernel_lb  # -inf (no bound) or +inf (infeasible)
    upper = (
        float(Fraction(kernel_ub) + piece.fixed_offset) if math.isfinite(kernel_ub) else math.inf
    )
    stats = {
        "nodes": int(raw.get("nodes", 0)),
        "cg_iterations": int(raw.get("cg_iterations", 0)),
        "pricing_nodes": int(raw.get("pricing_nodes", 0)),
        "columns": int(raw.get("columns", 0)),
        "root_lb": float(raw.get("root_lb", -math.inf)),
        "kernel_seconds": float(raw.get("seconds", seconds)),
        "n_active": piece.n,
        "fleet_size": piece.m,
        "cuts": piece.n_cuts,
    }
    return BPSolve(
        x_dict_sparse=piece.assign_to_x_dict(assign) if has_solution else {},
        assign=assign,
        kernel_ub=kernel_ub,
        kernel_lb=kernel_lb,
        lower_bound=lower,
        upper_bound_kernel=upper,
        status=str(raw.get("status", "unknown")),
        optimal=bool(raw.get("optimal", False)),
        has_solution=has_solution,
        seconds=seconds,
        stats=stats,
    )


# --------------------------------------------------------------------------
# forward Stage-2 entry (mirrors solve_forward_stage2_by_dp)
# --------------------------------------------------------------------------


def _model_stats(piece: BPPiece, solve: BPSolve, seconds: float) -> dict:
    """Same schema as ``models.learned_cut_dispatch.collect_s2_model_stats``."""
    return {
        "models": 1,
        "backend_gurobi": 0,
        "backend_dp": 0,
        "backend_probe": 0,
        "backend_bp": 1,
        "cuts": int(piece.n_cuts),
        "explicit": int(piece.n_cuts),
        "lazy": 0,
        "callback_calls": 0,
        "callback_checks": 0,
        "callback_added": 0,
        "callback_add_calls": 0,
        "build_seconds": 0.0,
        "solve_seconds": max(0.0, float(seconds)),
        "epochs": 1,
        "bp_nodes": solve.stats.get("nodes", 0),
        "bp_cg_iterations": solve.stats.get("cg_iterations", 0),
        "bp_pricing_nodes": solve.stats.get("pricing_nodes", 0),
        "bp_columns": solve.stats.get("columns", 0),
    }


def accept_rel_gap() -> float:
    return _float_env(ACCEPT_REL_GAP_ENV, DEFAULT_ACCEPT_REL_GAP)


def bp_time_limit() -> float:
    """Kernel wall-clock budget per piece (``VRP_S2_BP_TIME_LIMIT``)."""
    value = _float_env(TIME_LIMIT_ENV, DEFAULT_TIME_LIMIT)
    return value if math.isfinite(value) and value > 0.0 else DEFAULT_TIME_LIMIT


@dataclass
class BPForwardOutcome:
    """Outcome of one forward branch-and-price attempt.

    ``result`` is set only when the interval satisfies the caller's contract.
    ``warm_start`` carries the certified incumbent (if any) so a Gurobi
    fallback can start from it.  ``reason`` explains a ``None`` result.
    """

    result: Optional[ForwardStage2DPResult]
    warm_start: Optional[Dict[str, float]]
    reason: str


def solve_forward_stage2_by_bp(
    prob_data,
    node,
    cut_lag,
    x_prev,
    *,
    node_idx: Any = None,
    log=None,
    purpose: ForwardStage2Purpose | str = PHASE2_FORWARD,
    time_limit: Optional[float] = None,
    hint_x_dict: Optional[Mapping[str, float]] = None,
) -> BPForwardOutcome:
    """Solve a fixed-fleet forward Stage-2 node by branch-and-price.

    ``result`` is ``None`` when the kernel is unavailable, structurally
    rejects the node, or ends with an interval too wide for the caller's
    contract; the caller then continues with Gurobi.  The result object reuses
    the DP result dataclass because the downstream certificate handling is
    identical: ``cost_star_value`` is the exact rescored feasible policy and
    ``objective_lower_bound`` is rigorous.
    """
    purpose = normalize_forward_stage2_purpose(purpose)
    started = time.time()
    layout = FleetLayout(prob_data)
    period = int(node.info[1])
    flags = {v: x_prev.get(f"z[{v},{period}]", 0.0) for v in layout.vehicles}
    counts = layout.counts_from_vehicle_flags(flags)
    succ_to_pos = {succ: pos for pos, succ in enumerate(node.successor)}
    payload = exact_sub.build_s2_bp_cuts(prob_data, node, cut_lag, succ_to_pos)
    try:
        piece = compile_piece(prob_data, node, payload, layout, counts)
        record_backend_event("branch_price", "attempt", "fixed_fleet",
                             node=node_idx, purpose=purpose.value)
        solve = solve_piece_bp(piece, time_limit=time_limit, hint_x_dict=hint_x_dict)
    except BPNotApplicable as exc:
        reason = f"branch-and-price not applicable ({exc})"
        if log is not None:
            log(f"node={node_idx}: {reason}; Gurobi")
        return BPForwardOutcome(None, None, reason)
    if not solve.has_solution:
        reason = f"branch-and-price returned no policy ({solve.status})"
        if log is not None:
            log(f"node={node_idx}: {reason}; Gurobi")
        return BPForwardOutcome(None, None, reason)

    x_dict = dense_stage2_x_dict(prob_data, solve.x_dict_sparse)
    certified, _stage_cost = certify_stage2_forward_policy(prob_data, node, x_prev, x_dict)
    score = score_forward_stage2_policy(prob_data, node, certified, payload)
    cost_star = float(score["cost_star_value"])
    objective_exact = exact_forward_policy_score(prob_data, node, cut_lag, certified)
    kernel_total = solve.upper_bound_kernel
    if abs(cost_star - kernel_total) > _SELF_CHECK_ABS_TOL + _SELF_CHECK_REL_TOL * abs(cost_star):
        reason = (
            f"branch-and-price value {kernel_total!r} != exact rescoring "
            f"{cost_star!r} of its assignment"
        )
        if log is not None:
            log(f"node={node_idx}: {reason}; Gurobi")
        return BPForwardOutcome(None, dict(certified), reason)
    lower = solve.lower_bound
    lower_exact = Fraction(lower) if math.isfinite(lower) else None
    if lower_exact is not None and lower_exact > objective_exact:
        # The kernel bound is binary64 with explicit rounding allowances; it
        # may still land a few ulps above the exact cost of its own incumbent
        # (whose cost is a rigorous upper bound).  Anything beyond the
        # self-check band would mean an invalid bound: never certify it.
        excess = lower_exact - objective_exact
        if excess > Fraction(_SELF_CHECK_ABS_TOL) + Fraction(_SELF_CHECK_REL_TOL) * abs(objective_exact):
            reason = (
                f"branch-and-price lower bound {lower!r} exceeds the exact incumbent "
                f"cost {float(objective_exact)!r}"
            )
            if log is not None:
                log(f"node={node_idx}: {reason}; Gurobi")
            return BPForwardOutcome(None, dict(certified), reason)
        lower_exact = objective_exact
        lower = float_down(objective_exact)
    lb_certified = lower_exact is not None
    certified_gap = max(0.0, cost_star - lower) if lb_certified else math.inf
    # A search-exhausted run closes to the B&B tolerance (default 1e-7 plus
    # the explicit rounding allowances).  A time-limited run is accepted under
    # the same relative-gap contract as the Gurobi fallback it replaces.
    # Phase 1 only needs a certified feasible policy.
    tight_tol = max(_FORWARD_CERTIFIED_GAP_ABS_TOL, 10.0 * DEFAULT_ABS_GAP)
    accepted = bool(
        (not purpose.requires_tight_dp_gap)
        or (lb_certified and solve.optimal and certified_gap <= tight_tol)
        or (lb_certified and certified_gap <= accept_rel_gap() * max(1.0, abs(cost_star)))
    )
    seconds = time.time() - started
    result = ForwardStage2DPResult(
        x_dict=dict(certified),
        cost_star_value=cost_star,
        stage_cost_value=float(score["stage_cost_value"]),
        theta_by_succ=dict(score["theta_by_succ"]),
        counts=tuple(counts),
        n_active=int(piece.n),
        dp_value=float(kernel_total),
        seconds=seconds,
        model_stats=_model_stats(piece, solve, seconds),
        objective_lower_bound=lower if lb_certified else float("-inf"),
        lb_certified=lb_certified,
        optimality_certified=bool(lb_certified and lower_exact == objective_exact),
        certified_gap=certified_gap,
        certificate_source=f"{STATUS}:{solve.status}",
    )
    if log is not None:
        st = solve.stats
        log(
            f"node={node_idx} n_active={piece.n} fleet={counts} cuts={piece.n_cuts} "
            f"cost_star={cost_star:,.6f} lb={lower:,.6f} gap={certified_gap:.3e} "
            f"status={solve.status} nodes={st.get('nodes')} cg={st.get('cg_iterations')} "
            f"pricing_nodes={st.get('pricing_nodes')} cols={st.get('columns')} "
            f"accepted={int(accepted)} in {seconds:.2f}s"
        )
    if not accepted:
        return BPForwardOutcome(
            None, dict(certified),
            f"branch-and-price interval too wide (gap={certified_gap:.3e}, status={solve.status})",
        )
    record_backend_event("branch_price", "accepted", "certified_policy",
                         node=node_idx, certified_gap=certified_gap,
                         proven=result.optimality_certified)
    return BPForwardOutcome(result, dict(certified), "accepted")


__all__ = [
    "BPForwardOutcome",
    "BPNotApplicable",
    "BPPiece",
    "BPSolve",
    "STATUS",
    "accept_rel_gap",
    "bp_time_limit",
    "compile_piece",
    "kernel_available",
    "solve_forward_stage2_by_bp",
    "solve_piece_bp",
]
