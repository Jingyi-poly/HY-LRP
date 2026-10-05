"""Independent fixed-fleet physical CVRP/outsource reference for diagnostics.

The current EF supplies the routing formulation and discrete policy checker.
Only shallow reference-owned copies have zero purchase costs/budgets and unit
scenario weight. Fixing the same fleet throughout that artificial investment
horizon leaves exactly the requested node's physical recourse problem.

No learned theta cuts, assignment-order restrictions, or production caches are
used. A bound from this model belongs to true recourse Q, not to the existing
assignment/cut-surrogate piece table C_A. Active demands must be positive:
the EF's load-MTZ rows need that condition to exclude customer-only cycles.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping

from gurobipy import GRB, GurobiError

from core.solver_bounds import certified_gurobi_minimization_lower_bound
from models.extensive_model_builder import ExtensiveModelBuilder
from solvers.forward_policy_certification import InvalidForwardPolicy


def _finite(value, label):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _fleet_flags(prob_data, node, trial_z):
    if not isinstance(trial_z, Mapping):
        raise ValueError("trial_z must map current-period z names or vehicle ids to bits")
    period = int(node.info[1])
    named = all(f"z[{v},{period}]" in trial_z for v in prob_data.V)
    flags = {}
    for vehicle in prob_data.V:
        key = f"z[{vehicle},{period}]" if named else vehicle
        if key not in trial_z:
            raise ValueError(f"trial_z is missing {key!r}")
        value = _finite(trial_z[key], f"trial_z[{key}]")
        if value not in (0.0, 1.0):
            raise ValueError("fixed-fleet reference requires exact binary purchase flags")
        flags[vehicle] = int(value)
    return flags


def _reference_data(prob_data, node):
    if int(prob_data.T) <= 0 or not 0 <= int(node.info[1]) < int(prob_data.T):
        raise ValueError("node period must belong to the investment horizon")
    for customer in prob_data.J:
        active = _finite(node.active[customer], f"active[{customer}]")
        demand = _finite(node.volume[customer], f"volume[{customer}]")
        outsource = _finite(node.c_out[customer], f"c_out[{customer}]")
        if active not in (0.0, 1.0) or demand < 0.0 or outsource < 0.0:
            raise ValueError("invalid physical demand/outsource data")
        if active and demand <= 0.0:
            raise ValueError("EF load-MTZ reference requires positive active demands")
        if not active and outsource != 0.0:
            raise ValueError("inactive customers must have zero outsourcing cost")
    start, end = int(prob_data.numAllnodes) - 2, int(prob_data.numAllnodes) - 1
    arcs = ([(start, j) for j in prob_data.J]
            + [(j, end) for j in prob_data.J]
            + [(i, j) for i in prob_data.J for j in prob_data.J if i != j])
    for vehicle in prob_data.V:
        if _finite(prob_data.Qv[vehicle], f"capacity[{vehicle}]") < 0.0:
            raise ValueError("negative vehicle capacity")
        for tail, head in arcs:
            if _finite(prob_data.c_routing[vehicle][tail, head], "routing cost") < 0.0:
                raise ValueError("physical reference requires nonnegative routing costs")
    reference_pd = copy.copy(prob_data)
    reference_pd.cost_purchase = {v: 0.0 for v in prob_data.V}
    reference_pd.B_t0 = [0.0] * int(prob_data.T)
    reference_node = copy.copy(node)
    reference_node.multi_coeff = 1.0
    return reference_pd, reference_node


def _attribute(model, name):
    try:
        value = float(getattr(model, name))
    except (AttributeError, TypeError, ValueError, OverflowError, GurobiError):
        return None
    return value if math.isfinite(value) and abs(value) < 0.5 * GRB.INFINITY else None


def _certified_result(model, builder, node):
    """Keep solver dual bounds independent from incumbent availability."""
    raw_bound = _attribute(model, "ObjBound")
    lower = certified_gurobi_minimization_lower_bound(model)
    # This reference reports only the original solver bound; a feasible
    # incumbent must never be promoted into a lower-bound certificate.
    if lower != raw_bound:
        lower = None
    upper, certificate, error = None, None, None
    if int(model.SolCount) > 0:
        try:
            certificate = builder.certify_rounded_incumbent(model, [node])
            upper = float(certificate["objective_ub"])
        except InvalidForwardPolicy as exc:
            error = str(exc)
    consistent = lower is None or upper is None or lower <= upper
    if not consistent:
        lower = None
    gap = None if lower is None or upper is None else upper - lower
    return {
        "lb": lower, "ub": upper,
        "lb_certified": lower is not None,
        "ub_certified": upper is not None,
        "raw_obj_bound": raw_bound,
        "raw_obj_val": _attribute(model, "ObjVal") if model.SolCount else None,
        "status": int(model.Status), "sol_count": int(model.SolCount),
        "absolute_gap": gap, "bound_consistent": consistent,
        "optimality_proven": bool(
            model.Status == GRB.OPTIMAL and gap is not None and 0.0 <= gap <= 1e-6
        ),
        "policy_error": error,
    }


def _policy_payload(model, prob_data, node, flags):
    """Extract a compact route payload only after complete EF certification."""
    index, period = int(node.index), int(node.info[1])
    start, end = int(prob_data.numAllnodes) - 2, int(prob_data.numAllnodes) - 1
    bit = lambda name: ExtensiveModelBuilder._incumbent_bit(model, name)
    stage2, routes = {}, {}
    for vehicle in prob_data.V:
        stage2[f"y[{vehicle}]"] = bit(f"y_{index}[{vehicle}]")
        next_node, arcs = {}, {}
        for tail in [start] + list(prob_data.J):
            for head in list(prob_data.J) + [end]:
                if tail == head or (tail == start and head == end):
                    continue
                if bit(f"x_{index}[{vehicle},{tail},{head}]"):
                    next_node[tail] = head
                    arcs[f"x[{tail},{head}]"] = 1.0
        visited = set(next_node.values()) - {end}
        for customer in prob_data.J:
            stage2[f"alpha[{customer},{vehicle}]"] = int(customer in visited)
        path = []
        if next_node:
            path = [start]
            while path[-1] != end:
                path.append(next_node[path[-1]])
        routes[vehicle] = {"path": path, "arcs": arcs}
    return {
        "z": {f"z[{v},{period}]": float(flags[v]) for v in prob_data.V},
        "stage2": stage2, "routes": routes,
        "outsourced": [j for j in prob_data.J if bit(f"s_{index}[{j}]")],
    }


def solve_merged_route_reference(
    prob_data, node, trial_z, *, time_limit_s=10.0, threads=1, mip_gap=0.0,
):
    """Return physical Q(trial) bounds and a separately validated route policy.

    ``time_limit_s`` limits optimization; ``seconds`` also includes model
    construction and independent policy checks. TIME_LIMIT does not imply
    optimality, even if a useful ObjBound or incumbent was obtained.
    """
    seconds = _finite(time_limit_s, "time_limit_s")
    relative_gap = _finite(mip_gap, "mip_gap")
    if seconds <= 0.0 or relative_gap < 0.0:
        raise ValueError("time_limit_s must be positive and mip_gap nonnegative")
    if isinstance(threads, bool) or int(threads) != threads or threads <= 0:
        raise ValueError("threads must be a positive integer")
    started = time.monotonic()
    flags = _fleet_flags(prob_data, node, trial_z)
    reference_pd, reference_node = _reference_data(prob_data, node)
    builder = ExtensiveModelBuilder(reference_pd)
    model = builder.build([reference_node])
    try:
        model.Params.OutputFlag = 0
        model.Params.Threads = int(threads)
        model.Params.TimeLimit = seconds
        model.Params.MIPGap = relative_gap
        model.Params.MIPGapAbs = 0.0
        for vehicle, available in flags.items():
            for period in range(int(prob_data.T)):
                var = model.getVarByName(f"z[{vehicle},{period}]")
                var.LB = var.UB = available
        model.update()
        build_seconds = time.monotonic() - started
        model.optimize()
        result = _certified_result(model, builder, reference_node)
        result["policy"] = (
            _policy_payload(model, reference_pd, reference_node, flags)
            if result["ub_certified"] else None
        )
        result.update(
            fleet=flags, node=int(node.index), threads=int(threads),
            time_limit_s=seconds, build_seconds=build_seconds,
            solve_seconds=float(model.Runtime), seconds=time.monotonic() - started,
            num_vars=int(model.NumVars), num_constraints=int(model.NumConstrs),
        )
        return result
    finally:
        model.dispose()


__all__ = ["solve_merged_route_reference"]
