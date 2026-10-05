"""Standalone physical route/outsource LP with RouteOpt root pricing.

The restricted-master objective is not a lower bound. Only a dual checked by
complete pricing (or an elementary-route bound) is reported as a lower bound.
No learned theta cuts or production solver state enter this experiment.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math
import time

import gurobipy as gp
from gurobipy import GRB
from core.backend_telemetry import backend_call, backend_scope
from .capacity_price_bound import CapacityPriceBound
from .physical_identity import physical_fingerprint_from_groups


def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _directed(value, upward=False):
    number = float(value)
    if (Fraction(number) < value if upward else Fraction(number) > value):
        number = math.nextafter(number, math.inf if upward else -math.inf)
    return number


def _ceil(value):
    return -((-value.numerator) // value.denominator)


def _physical_groups(pd, node, flags):
    jobs = tuple(j for j in pd.J if node.active[j] == 1)
    demands, outsourcing = [], []
    for j in pd.J:
        if node.active[j] not in (0, 1):
            raise ValueError("activity must be binary")
        demand = _finite(node.volume[j], "demand")
        out = _finite(node.c_out[j], "outsourcing cost")
        if demand < 0 or out < 0 or (not node.active[j] and out != 0):
            raise ValueError("invalid physical demand/outsourcing data")
        if node.active[j]:
            demands.append(Fraction(demand))
            outsourcing.append(Fraction(out))
    start, end = int(pd.numAllnodes) - 2, int(pd.numAllnodes) - 1
    groups, index = [], {}
    for v in pd.V:
        named = f"z[{v},{node.info[1]}]"
        flag = flags[v] if v in flags else flags[named]
        if float(flag) not in (0., 1.):
            raise ValueError("fleet flags must be exact binary values")
        capacity = _finite(pd.Qv[v], "capacity")
        if capacity < 0:
            raise ValueError("negative capacity")
        source = pd.c_routing[v]
        matrix = [[0.] * (len(jobs) + 1) for _ in range(len(jobs) + 1)]
        for a, i in enumerate(jobs, 1):
            matrix[0][a] = _finite(source[start, i], "depot cost")
            matrix[a][0] = _finite(source[i, end], "return cost")
            for b, j in enumerate(jobs, 1):
                if a != b:
                    matrix[a][b] = _finite(source[i, j], "routing cost")
        for a in range(len(matrix)):
            for b in range(len(matrix)):
                if matrix[a][b] < 0:
                    raise ValueError("physical route costs must be nonnegative")
                if matrix[a][b] != matrix[b][a]:
                    raise ValueError("this RouteOpt adapter requires a symmetric cost matrix")
        key = (capacity.hex(), tuple(x.hex() for row in matrix for x in row))
        if key not in index:
            index[key] = len(groups)
            groups.append(dict(vehicles=[], available=0, capacity=Fraction(capacity),
                               costs=matrix))
        group = groups[index[key]]
        group["vehicles"].append(v)
        group["available"] += int(flag)
    return jobs, demands, outsourcing, groups


def _route_data(route, group, demands, relaxed_columns=False, grid_demands=None):
    route = tuple(route)
    if not route or any(type(j) is not int or not 1 <= j <= len(demands) for j in route):
        return None
    physical = (len(set(route)) == len(route)
                and sum((demands[j - 1] for j in route), Fraction()) <= group["capacity"])
    if not physical:
        if not relaxed_columns or sum(grid_demands[j - 1] for j in route) > group["scaled_capacity"]:
            return None
    path = (0,) + route + (0,)
    cost = sum((Fraction(group["costs"][a][b]) for a, b in zip(path, path[1:])), Fraction())
    return tuple(sorted(Counter(route).items())), cost, physical


def _seed_routes(group, demands, outsourcing):
    """Cheap feasible columns only; no bound or exactness claim uses this seed."""
    eligible = [j for j, d in enumerate(demands, 1) if d <= group["capacity"]]
    for first in eligible:
        yield (first,)
        for mode in (0, 1):
            route, load, unused = [first], demands[first - 1], set(eligible) - {first}
            while unused:
                feasible = [j for j in unused if load + demands[j - 1] <= group["capacity"]]
                if not feasible:
                    break
                last = route[-1]
                nxt = min(feasible, key=lambda j: (
                    group["costs"][last][j] - (float(outsourcing[j - 1]) if mode else 0), j))
                route.append(nxt)
                unused.remove(nxt)
                load += demands[nxt - 1]
                yield tuple(route)


def _dual_certificate(u, lower_prices, groups):
    beta = [Fraction(_directed(min(Fraction(), value))) for value in lower_prices]
    intercept = Fraction(_directed(sum(u, Fraction())))
    exact_bound = intercept + sum((g["available"] * b for g, b in zip(groups, beta)), Fraction())
    return dict(lb=_directed(exact_bound), intercept=float(intercept),
                dual_u=[float(x) for x in u], dual_beta=[float(x) for x in beta])


def _certificate_slopes(certificate, groups):
    if "eta_slope_by_vehicle" in certificate:
        return {int(v): Fraction(value)
                for v, value in certificate["eta_slope_by_vehicle"].items()}
    return {v: Fraction(beta) for group, beta in zip(groups, certificate["dual_beta"])
            for v in group["vehicles"]}


def _certificate_update_reason(candidate, incumbent, groups):
    """Prefer anchor gain, then a coefficientwise dominant same-intercept cut."""
    if candidate["lb"] > incumbent["lb"]:
        return "anchor_gain"
    if (candidate["lb"] != incumbent["lb"]
            or candidate["intercept"] != incumbent["intercept"]):
        return None
    new, old = _certificate_slopes(candidate, groups), _certificate_slopes(incumbent, groups)
    if (new.keys() == old.keys() and all(new[v] >= old[v] for v in old)
            and any(new[v] > old[v] for v in old)):
        return "slope_dominance"
    return None


@backend_scope(phase="1.5", path="physical_seed", stage=2)
def solve_routeopt_root_lp(
    pd, node, flags, *, time_limit_s=5., threads=1, max_iterations=100,
    cost_scale=1_000_000, resource_scale=1000, max_routes=1000, ng_size=0,
    relaxed_columns=False, progress=None, route_consumer=None, package_cover=False,
):
    """Return a certified physical-recourse LB, not an integer-CVRP solution.

    ``route_consumer`` may retain newly inserted/improved elementary routes
    for a separate primal heuristic. It receives (vehicles, original customer
    sequence, exact cost); relaxed walks are never exported as physical routes.

    ``package_cover=True`` adds a certified integer-package outsourcing floor
    when the original data support it. Unsupported data use the ordinary CG
    in the same call. Its multiplier caps customer prices but never changes
    the pricing objective or route domain. Only complete exported certificates
    compete for the best bound; RMP objectives remain diagnostic values.

    Pricing alone uses a relaxation: arc costs round down, customer prices up,
    demands down and capacity up. The actual restricted master and all inserted
    routes retain their original costs. A completed relaxed pricing can certify
    a physical-route lower bound even when its minimizing route is infeasible
    under the original demands. With ``relaxed_columns=False`` that route is
    not inserted. With ``True``, the master is itself a relaxation: repeated
    visits have repeated coverage coefficients, and resource-grid-feasible
    columns are allowed. Neither mode returns an integer feasible policy.

    ``root_lp_gap`` is available only for an original-route master. For a
    relaxed-column master its objective is neither a bound on the physical
    root LP nor an integer policy cost. Neither mode marks routing optimal.
    """
    started = time.monotonic()
    budget = _finite(time_limit_s, "time limit")
    if budget <= 0 or any(type(x) is not int or x <= 0 for x in
                          (threads, max_iterations, cost_scale, resource_scale, max_routes)):
        raise ValueError("time budget and integer configuration values must be positive")
    if type(ng_size) is not int or ng_size < 0:
        raise ValueError("ng_size must be a nonnegative integer (zero means elementary)")
    if type(relaxed_columns) is not bool:
        raise ValueError("relaxed_columns must be boolean")
    if type(package_cover) is not bool:
        raise ValueError("package_cover must be boolean")
    deadline = started + budget
    jobs, demands, outsourcing, groups = _physical_groups(pd, node, flags)
    fingerprint = physical_fingerprint_from_groups(jobs, demands, outsourcing, groups)
    capacity_bounds = [CapacityPriceBound.from_group(demands, group) for group in groups]
    initial_u = outsourcing if not any(g["available"] for g in groups) else [Fraction()] * len(jobs)
    initial_prices = [bound.lower_bound(initial_u) for bound in capacity_bounds]
    best = _dual_certificate(initial_u, initial_prices, groups)
    best["certificate_source"] = "physical_capacity_dual" if any(initial_u) else "nonnegative_route_bound"
    best["certificate_method"] = "fractional_knapsack" if any(initial_u) else "nonnegative_routes"
    package = None
    if package_cover:
        from .package_cover import (
            make_package_cover, initial_package_certificate,
            project_package_duals, export_package_certificate,
        )
        package = make_package_cover(pd, node, flags)
        if package is not None:
            candidate = initial_package_certificate(package)
            if _certificate_update_reason(candidate, best, groups):
                best = candidate
    result = dict(best)
    result.update(lb_certified=True,
                  certificate_source=best.get("certificate_source", "nonnegative_route_bound"),
                  rmp_obj=None, root_lp_gap=None, iterations=0, pricing_seconds=0.,
                  route_count=0, pricing_calls=0, completed_pricing_calls=0,
                  relaxed_route_count=0, relaxed_columns=relaxed_columns,
                  rejected_routes=0, improved_columns=0, history=[],
                  cost_scale=cost_scale, resource_scale=resource_scale,
                  ng_size=ng_size,
                  physical_fingerprint=fingerprint,
                  certificate_domain="physical_elementary_routes",
                  replay_data=dict(jobs=list(jobs), demands=[float(d) for d in demands],
                                   outsourcing=[float(c) for c in outsourcing],
                                   incoming_minima=[[float(c) for c in bound.incoming]
                                                    for bound in capacity_bounds]),
                  physical_optimality_proven=False,
                  groups=[dict(vehicles=g["vehicles"], available=g["available"],
                               capacity=float(g["capacity"])) for g in groups])
    if package_cover:
        result.update(package_cover_requested=True, package_cover_used=package is not None)
    if not jobs or not any(g["available"] for g in groups):
        upper = _directed(sum(outsourcing, Fraction()), upward=True)
        result.update(rmp_obj=upper,
                      root_lp_gap=None if relaxed_columns else upper - result["lb"],
                      rmp_minus_physical_lb=upper - result["lb"], status="trivial",
                      seconds=time.monotonic() - started)
        return result
    if package_cover and time.monotonic() >= deadline:
        result.update(status="preparation_time_limit", seconds=time.monotonic() - started)
        return result
    from .pricing import price_routes

    quantized_demands = [(d * resource_scale).__floor__() for d in demands]
    result["replay_data"]["quantized_demands"] = quantized_demands
    if min(quantized_demands) <= 0:
        # Native resource labels require positive integers. The physical
        # certificate does not: retain a capacity-aware analytic candidate.
        prices = [bound.lower_bound(outsourcing) for bound in capacity_bounds]
        candidate = _dual_certificate(outsourcing, prices, groups)
        if _certificate_update_reason(candidate, result, groups):
            for field in ("eta_slope_by_vehicle", "dual_gamma", "package_cover"):
                result.pop(field, None)
            result.update(candidate, certificate_source="physical_capacity_dual",
                          certificate_method="fractional_knapsack")
        result.update(status="unsupported_resource_grid", seconds=time.monotonic() - started)
        return result
    for group in groups:
        group["scaled_costs"] = [[(Fraction(c) * cost_scale).__floor__() for c in row]
                                  for row in group["costs"]]
        group["scaled_capacity"] = _ceil(group["capacity"] * resource_scale)
    model = gp.Model("physical_routeopt_root_lp")
    status = "iteration_limit"
    try:
        model.Params.OutputFlag = 0
        model.Params.Threads = threads
        model.Params.Method = 1
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        cover = [model.addConstr(gp.LinExpr() == 1, name=f"cover[{j}]") for j in jobs]
        fleet = [model.addConstr(gp.LinExpr() <= g["available"], name=f"fleet[{k}]")
                 for k, g in enumerate(groups)]
        package_row = None
        if package is not None:
            package_row = model.addConstr(
                gp.LinExpr() >= package.profile.lower_packages, name="package_cover")
        for i, cost in enumerate(outsourcing):
            rows, coefficients = [cover[i]], [1.]
            if package_row is not None:
                rows.append(package_row)
                coefficients.append(float(package.profile.counts[i]))
            model.addVar(obj=float(cost), column=gp.Column(coefficients, rows), name=f"out[{i}]")
        columns = {}

        def add_route(k, route):
            checked = _route_data(route, groups[k], demands, relaxed_columns, quantized_demands)
            if checked is None:
                result["rejected_routes"] += 1
                return False
            counts, exact_cost, physical = checked
            key = k, counts
            grid_cost = (exact_cost * cost_scale).__floor__()
            if key in columns:
                var, previous, _ = columns[key]
                if exact_cost >= previous:
                    return False
                var.Obj = _directed(exact_cost, upward=True)
                columns[key] = var, exact_cost, grid_cost
                result["improved_columns"] += 1
            else:
                rows = [cover[j - 1] for j, count in counts] + [fleet[k]]
                coefficients = [float(count) for j, count in counts] + [1.]
                var = model.addVar(obj=_directed(exact_cost, upward=True),
                                   column=gp.Column(coefficients, rows),
                                   name=f"route[{len(columns)}]")
                columns[key] = var, exact_cost, grid_cost
                result["relaxed_route_count"] += int(not physical)
            if physical and route_consumer is not None:
                route_consumer(
                    tuple(groups[k]["vehicles"]),
                    tuple(jobs[j - 1] for j in route), exact_cost,
                )
            return True

        for k, group in enumerate(groups):
            for route in _seed_routes(group, demands, outsourcing):
                if time.monotonic() >= deadline:
                    break
                add_route(k, route)
            if time.monotonic() >= deadline:
                break
        for iteration in range(max_iterations):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "time_limit"
                break
            model.Params.TimeLimit = remaining
            with backend_call("gurobi", "route_restricted_master_lp", model=model):
                model.optimize()
            if model.Status != GRB.OPTIMAL:
                status = f"rmp_status_{model.Status}"
                break
            result["rmp_obj"] = float(model.ObjVal)
            if package is None:
                u = [min(Fraction(row.Pi), out) for row, out in zip(cover, outsourcing)]
            else:
                u, gamma = project_package_duals(
                    package, [row.Pi for row in cover], package_row.Pi)
            scaled_prices = [_ceil(value * cost_scale) for value in u]
            scaled_fleet = [(min(Fraction(), Fraction(row.Pi)) * cost_scale).__floor__()
                            for row in fleet]
            elementary_floor = -sum((max(Fraction(), value) for value in u), Fraction())
            lower_prices = [bound.lower_bound(u) for bound in capacity_bounds]
            changes, complete = 0, 0
            calls = [dict(group=k, status="budget_not_priced", complete=False,
                          solver_call=False, seconds=0., routes=0, min_rc=None,
                          elementary_lb=_directed(elementary_floor),
                          capacity_price_lb=_directed(value), native_price_lb=None,
                          scaled_fleet_dual=scaled_fleet[k],
                          physical_bound_source="fractional_knapsack")
                     for k, value in enumerate(lower_prices)]
            for k, group in enumerate(groups):
                call = calls[k]
                if not group["available"]:
                    # Its slope matters off-anchor even though no pricing
                    # solve is needed at the current fleet.
                    call.update(status="unused_type_analytic_bound")
                    continue
                if all(d > group["capacity"] for d in demands):
                    lower_prices[k] = Fraction()
                    complete += 1
                    call.update(status="no_feasible_nonempty_route", complete=True,
                                min_rc=0, physical_bound_source="empty_route_domain")
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "time_limit"
                    break
                pending = sum(g["available"] > 0 for g in groups[k:])
                call_budget = max(0.001, remaining / pending)
                priced = price_routes(group["scaled_costs"], scaled_prices,
                                      quantized_demands, group["scaled_capacity"],
                                      fleet_dual=scaled_fleet[k], time_limit_s=call_budget,
                                      max_routes=max_routes, ng_size=ng_size)
                result["pricing_calls"] += 1
                result["pricing_seconds"] += float(priced["seconds"])
                certified = bool(priced.get("pricing_complete") and priced.get("lb_certified"))
                if certified:
                    value = priced["min_rc"]
                    if type(value) is not int:
                        raise ValueError("certified native pricing bound must be integer ticks")
                    # Pricing subtracts the fleet dual once at the depot.
                    # Restore it once to bound c(route) - sum customer duals.
                    native_bound = Fraction(value + scaled_fleet[k], cost_scale)
                    call["native_price_lb"] = _directed(native_bound)
                    if native_bound > lower_prices[k]:
                        lower_prices[k] = native_bound
                        call["physical_bound_source"] = "complete_native_pricing"
                    elif native_bound == lower_prices[k]:
                        call["physical_bound_source"] = "fractional_knapsack_and_native"
                    complete += 1
                for route in priced.get("routes", []):
                    changes += int(add_route(k, route))
                call.update(status=priced["status"], complete=certified, solver_call=True,
                            seconds=priced["seconds"], routes=len(priced.get("routes", [])),
                            min_rc=priced.get("min_rc") if certified else None,
                            diagnostic=priced.get("stderr", "").strip() if not certified else "")
            if relaxed_columns:
                # Diagnostic only: a physical-route certificate need not be
                # dual-feasible for repeated-customer/grid-relaxed columns.
                column_floor = [None] * len(groups)
                for (k, counts), (var, exact_cost, grid_cost) in columns.items():
                    rc = grid_cost - sum(scaled_prices[j - 1] * count for j, count in counts)
                    if column_floor[k] is None or rc < column_floor[k]:
                        column_floor[k] = rc
                for k, value in enumerate(column_floor):
                    calls[k]["retained_column_min_price"] = (
                        None if value is None else _directed(Fraction(value, cost_scale)))
            for call, value in zip(calls, lower_prices):
                call["physical_price_lb"] = _directed(value)
            certificate = (_dual_certificate(u, lower_prices, groups) if package is None
                           else export_package_certificate(package, u, lower_prices, gamma))
            saved_reason = _certificate_update_reason(certificate, result, groups)
            if saved_reason:
                result.update(certificate, certificate_source=certificate.get(
                    "certificate_source", "routeopt_pricing_dual"),
                    certificate_method="physical_price_bounds", certificate_iteration=iteration)
            result["iterations"] = iteration + 1
            result["completed_pricing_calls"] += sum(
                c["complete"] and c.get("solver_call", False) for c in calls)
            row = dict(iteration=iteration, rmp_obj=result["rmp_obj"], lb=result["lb"],
                       current_dual_lb=certificate["lb"], added=changes, complete=complete,
                       seconds=time.monotonic() - started, pricing=calls,
                       dual_u=[float(value) for value in u], scaled_prices=scaled_prices,
                       current_intercept=certificate["intercept"],
                       current_dual_beta=certificate["dual_beta"], saved_reason=saved_reason)
            if package is not None:
                row["package_gamma"] = float(gamma)
            result["history"].append(row)
            if progress is not None:
                progress(row)
            if changes == 0:
                status = ("priced_stationary" if complete == sum(g["available"] > 0 for g in groups)
                          else "incomplete_pricing")
                break
        result.update(status=status, route_count=len(columns), seconds=time.monotonic() - started)
        if result["rmp_obj"] is not None:
            difference = result["rmp_obj"] - result["lb"]
            result["rmp_minus_physical_lb"] = difference
            result["root_lp_gap"] = None if relaxed_columns else difference
            if not relaxed_columns and difference < -1e-6:
                raise ArithmeticError("pricing certificate exceeds restricted-master objective")
        return result
    finally:
        model.dispose()


__all__ = ["solve_routeopt_root_lp"]
