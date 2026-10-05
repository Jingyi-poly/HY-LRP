"""Small full-column physical routing references, preserving binary64 inputs."""
from __future__ import annotations

from fractions import Fraction
import math
import time

import gurobipy as gp
from gurobipy import GRB


def _fraction(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return Fraction(value)


def _down(value):
    result = float(value)
    if Fraction(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


def _flags(pd, node, flags):
    result = {}
    for vehicle in pd.V:
        name = f"z[{vehicle},{node.info[1]}]"
        value = flags[vehicle] if vehicle in flags else flags[name]
        if float(value) not in (0.0, 1.0):
            raise ValueError("fleet flags must be exact binary values")
        result[vehicle] = int(value)
    return result


def enumerate_route_columns(pd, node):
    """One cheapest elementary path per nonempty feasible subset and vehicle.

    Routes serving the same subset on the same vehicle have identical column
    coefficients; retaining only their cheapest ordering is lossless for the
    LP and integer master. Physical types are never inferred from labels.
    """
    jobs = tuple(j for j in pd.J if node.active[j] == 1)
    if len(jobs) > 10:
        raise ValueError("full-column reference is limited to ten active customers")
    for j in pd.J:
        if node.active[j] not in (0, 1):
            raise ValueError("customer activity must be binary")
        if _fraction(node.volume[j], "demand") < 0 or _fraction(node.c_out[j], "outsourcing") < 0:
            raise ValueError("negative demand or outsourcing cost")
        if not node.active[j] and node.c_out[j] != 0:
            raise ValueError("inactive customers must have zero outsourcing cost")
    start, end = int(pd.numAllnodes) - 2, int(pd.numAllnodes) - 1
    full = (1 << len(jobs)) - 1
    loads = {0: Fraction()}
    for mask in range(1, full + 1):
        bit = mask & -mask
        loads[mask] = loads[mask ^ bit] + _fraction(node.volume[jobs[bit.bit_length() - 1]], "demand")
    columns = []
    for vehicle in pd.V:
        capacity = _fraction(pd.Qv[vehicle], "capacity")
        if capacity < 0:
            raise ValueError("negative vehicle capacity")
        matrix = pd.c_routing[vehicle]
        costs = {}
        for tail in (start,) + jobs:
            for head in jobs + (end,):
                if tail == head or (tail == start and head == end):
                    continue
                value = _fraction(matrix[tail, head], "routing cost")
                if value < 0:
                    raise ValueError("negative physical routing cost")
                costs[tail, head] = value
        labels = {(1 << k, k): (costs[start, j], (start, j))
                  for k, j in enumerate(jobs) if loads[1 << k] <= capacity}
        for mask in range(1, full + 1):
            if loads[mask] > capacity:
                continue
            best = None
            for last, customer in enumerate(jobs):
                label = labels.get((mask, last))
                if label is None:
                    continue
                finished = (label[0] + costs[customer, end], label[1] + (end,))
                if best is None or finished < best:
                    best = finished
                for nxt, next_customer in enumerate(jobs):
                    next_mask = mask | (1 << nxt)
                    if mask & (1 << nxt) or loads[next_mask] > capacity:
                        continue
                    key = next_mask, nxt
                    value = (label[0] + costs[customer, next_customer], label[1] + (next_customer,))
                    if key not in labels or value < labels[key]:
                        labels[key] = value
            if best is not None:
                columns.append({
                    "vehicle": vehicle, "mask": mask,
                    "customers": tuple(j for k, j in enumerate(jobs) if mask & (1 << k)),
                    "path": best[1], "cost": float(best[0]), "exact_cost": best[0],
                    "exact_load": loads[mask],
                })
    return jobs, columns


def _integer_reference(jobs, columns, node, flags):
    values = {0: Fraction()}
    for vehicle, available in flags.items():
        if not available:
            continue
        routes = [(0, Fraction())] + [(r["mask"], r["exact_cost"]) for r in columns if r["vehicle"] == vehicle]
        updated = {}
        for covered, base in values.items():
            for mask, cost in routes:
                if covered & mask:
                    continue
                union, candidate = covered | mask, base + cost
                if union not in updated or candidate < updated[union]:
                    updated[union] = candidate
        values = updated
    return min(value + sum((_fraction(node.c_out[j], "outsourcing")
                            for k, j in enumerate(jobs) if not covered & (1 << k)), Fraction())
               for covered, value in values.items())


def solve_full_route_lp(pd, node, flags, *, time_limit_s=5.0, solve_integer=True):
    """Solve a complete route/outsource SPP LP and optional integer reference.

    The certified ``lb`` is the dual objective after exact checks against all
    original-cost columns. ``obj`` is Gurobi's optimum of the nearest-binary64
    route-cost LP. No purchase costs or node probability weights enter either.
    """
    if not math.isfinite(float(time_limit_s)) or time_limit_s <= 0:
        raise ValueError("time_limit_s must be finite and positive")
    started = time.monotonic()
    availability = _flags(pd, node, flags)
    jobs, columns = enumerate_route_columns(pd, node)
    enumerated = time.monotonic()
    model = gp.Model("full_physical_route_reference")
    try:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = float(time_limit_s)
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        route = model.addVars(len(columns), obj=[r["cost"] for r in columns], name="route")
        outsource = model.addVars(jobs, obj={j: float(node.c_out[j]) for j in jobs}, name="outsource")
        coverage = {j: model.addConstr(
            gp.quicksum(route[r] for r, column in enumerate(columns) if j in column["customers"])
            + outsource[j] == 1, name=f"cover[{j}]") for j in jobs}
        fleet = {v: model.addConstr(
            gp.quicksum(route[r] for r, column in enumerate(columns) if column["vehicle"] == v)
            <= availability[v], name=f"fleet[{v}]") for v in pd.V}
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"full-column LP did not close: status={model.Status}")
        customer_dual = {j: float(coverage[j].Pi) for j in jobs}
        vehicle_dual = {v: float(fleet[v].Pi) for v in pd.V}
        certified_customer = {j: min(Fraction(customer_dual[j]), _fraction(node.c_out[j], "outsourcing")) for j in jobs}
        certified_vehicle = {}
        for v in pd.V:
            candidates = [Fraction(), Fraction(vehicle_dual[v])]
            candidates += [column["exact_cost"] - sum((certified_customer[j] for j in column["customers"]), Fraction())
                           for column in columns if column["vehicle"] == v]
            certified_vehicle[v] = Fraction(_down(min(candidates)))
        exact_lb = sum(certified_customer.values(), Fraction()) + sum(
            (availability[v] * certified_vehicle[v] for v in pd.V), Fraction())
        result = {
            "lb": _down(exact_lb), "obj": float(model.ObjVal), "lb_certified": True,
            "duals": {"customer": customer_dual, "vehicle": vehicle_dual},
            "certified_duals": {"customer": {j: float(x) for j, x in certified_customer.items()},
                                "vehicle": {v: float(x) for v, x in certified_vehicle.items()}},
            "routes": columns, "route_values": [float(route[r].X) for r in range(len(columns))],
            "outsource_values": {j: float(outsource[j].X) for j in jobs},
            "flags": availability, "customers": jobs, "status": int(model.Status),
            "lp_seconds": float(model.Runtime), "enumeration_seconds": enumerated - started,
        }
        if solve_integer:
            expected = _integer_reference(jobs, columns, node, availability)
            for variable in model.getVars():
                variable.VType = GRB.BINARY
            model.Params.IntFeasTol = 1e-9
            model.Params.MIPGap = 0
            model.Params.MIPGapAbs = 0
            model.optimize()
            result.update(
                integer_obj=float(model.ObjVal) if model.SolCount else None,
                integer_lb=float(model.ObjBound), integer_status=int(model.Status),
                integer_seconds=float(model.Runtime), exact_integer_obj=float(expected),
                exact_integer_fraction=expected,
            )
            if model.Status == GRB.OPTIMAL and abs(model.ObjVal - float(expected)) > 1e-6:
                raise AssertionError("integer route master disagrees with exact fleet-subset DP")
            if Fraction(result["lb"]) > expected:
                raise AssertionError("certified LP lower bound exceeds exact integer optimum")
        result["seconds"] = time.monotonic() - started
        return result
    finally:
        model.dispose()


__all__ = ["enumerate_route_columns", "solve_full_route_lp"]
