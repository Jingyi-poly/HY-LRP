"""LRP physical-domain adapter for the unchanged Final policy-search core.

Only a separately certified incumbent is returned. No surrogate theta, cut,
forward trial, target or oracle support enters the objective or is mutated.
The existing shared refinement deadline includes preparation; certification is
cooperative, as in Final, and can finish slightly after that deadline.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as F
import math
import time

from core import customized_subprob  # exposes the unchanged original module
from core.solver_bounds import minimization_bounds_inverted
from models.stage_builder import _instance, _node_context
from solvers.forward_policy_certification import (
    _normalize_lrp_forward_policy, certify_policy, certify_stage1_forward_policy,
    certify_stage2_forward_policy, certify_stage3_forward_policy,
)
from s2backward.routeopt import physical_policy_search as original


def _normalize(data, tree, policy):
    """Normalize and fully audit once on a fresh, private data snapshot."""
    result, certificate = _normalize_lrp_forward_policy(data, tree, policy)
    return result, float(certificate['feasible_upper_bound'])


def _routes(tree, node, policy):
    result = {}
    for r in node.successor:
        i = int(tree[3][r].info)
        arcs = [tuple(map(int, name[2:-1].split(',')))[1:]
                for name, bit in policy[3][r].items() if bit]
        if not arcs:
            result[i] = ()
            continue
        outgoing = dict(arcs)
        route, v = [], outgoing[0]
        while v:
            route.append(v)
            v = outgoing[v]
        result[i] = tuple(route)
    return result


def _exact_node_cost(data, tree, node, policy):
    ctx = _node_context(data, node, stage=2)
    total = sum((F(float(ctx.outsourcing[j])) * int(policy[2][node.index][f'e[{j}]'])
                 for j in range(ctx.n)), F())
    for r in node.successor:
        i = int(tree[3][r].info)
        for name, bit in policy[3][r].items():
            if bit:
                _, v, w = map(int, name[2:-1].split(','))
                total += F(float(ctx.route_cost[i, v, w]))
    return total


def _merge(data, tree, policy, incumbent):
    current, upper = _normalize(data, tree, policy)
    reused = []
    if incumbent is None:
        return current, upper, reused
    other, _ = _normalize(data, tree, incumbent)
    for q in tree[1][0].successor:
        node = tree[2][q]
        ctx = _node_context(data, node, stage=2)
        names = [f'A[{i},{ctx.interval}]' for i in range(ctx.m)]
        # Same raw exact fleet at THIS interval; other intervals may differ.
        if not all(policy[1][0][key] in (0., 1.) and incumbent[1][0][key] in (0., 1.)
                   and float(policy[1][0][key]).hex() == float(incumbent[1][0][key]).hex()
                   for key in names):
            continue
        if _exact_node_cost(data, tree, node, other) < _exact_node_cost(data, tree, node, current):
            current[2][q] = deepcopy(other[2][q])
            for r in node.successor:
                current[3][r] = deepcopy(other[3][r])
            reused.append(q)
    return current, float(certify_policy(data, tree, current)['feasible_upper_bound']), reused


def _improve_pass(data, tree, policy, deadline, *, package_moves):
    current, before_upper = _normalize(data, tree, policy)
    ordered = []
    for q in tree[1][0].successor:
        node = tree[2][q]
        ctx = _node_context(data, node, stage=2)
        outsourced = sum((F(float(ctx.outsourcing[j])) * int(current[2][q][f'e[{j}]'])
                          for j in range(ctx.n)), F())
        route_cost = _exact_node_cost(data, tree, node, current) - outsourced
        probability = F(float(data.arrays['scenario_prob'][ctx.scenario]))
        ordered.append((-probability * route_cost, q))
    ordered.sort()
    records = []
    for position, (_, q) in enumerate(ordered):
        started = time.monotonic()
        if started >= deadline:
            break
        node = tree[2][q]
        ctx = _node_context(data, node, stage=2)
        vehicles = tuple(i for i in range(ctx.m)
                         if current[1][0][f'A[{i},{ctx.interval}]'] == 1.)
        if not vehicles:
            continue
        # No heterogeneous-facility canonicalization and no cross-period reuse.
        costs = {i: {(v, w): F(float(ctx.route_cost[i, v, w]))
                     for v in range(ctx.n + 1) for w in range(ctx.n + 1)}
                 for i in vehicles}
        for i in vehicles:
            costs[i][0, 0] = F()  # idle is no tour, not a depot self arc
        demand = {j + 1: F(float(ctx.demand[j])) for j in range(ctx.n)}
        capacity = {i: F(float(ctx.capacity[i])) for i in vehicles}
        charges = {j + 1: F(float(ctx.outsourcing[j])) for j in range(ctx.n)}
        active = tuple(j + 1 for j in range(ctx.n) if ctx.active[j])
        original_routes = _routes(tree, node, current)
        routes = {i: original_routes[i] for i in vehicles}
        original._validate_routes(routes, vehicles, active, demand, capacity)
        before = original._objective(routes, vehicles, active, charges, costs, 0, 0)
        # Preparation consumes the same allowance; residual allocation follows Final.
        now = time.monotonic()
        if now >= deadline:
            break
        local_deadline = min(deadline, now + (deadline - now) / (len(ordered) - position))
        candidate, trace = original._search(routes, vehicles, demand, capacity, costs, 0, 0,
            local_deadline, active=active, charges=charges, package_moves=package_moves)
        original._validate_routes(candidate, vehicles, active, demand, capacity)
        after = original._objective(candidate, vehicles, active, charges, costs, 0, 0)
        if after > before:
            raise AssertionError('Final move core increased exact LRP physical cost')
        if after < before:
            selected = {j for route in candidate.values() for j in route}
            current[2][q] = {f'alpha[{i},{j}]': float(j + 1 in candidate.get(i, ()))
                             for i in range(ctx.m) for j in range(ctx.n)}
            current[2][q].update({f'u[{i}]': float(bool(candidate.get(i))) for i in range(ctx.m)})
            current[2][q].update({f'e[{j}]': float(bool(ctx.active[j]) and j + 1 not in selected)
                                  for j in range(ctx.n)})
            for r in node.successor:
                i = int(tree[3][r].info)
                route = candidate.get(i, ())
                path = (0,) + route + (0,) if route else ()
                current[3][r] = {f'r[{i},{v},{w}]': 1. for v, w in zip(path, path[1:])}
        records.append(dict(node=q, context=ctx.key, package_moves=package_moves,
            before_exact=str(before), after_exact=str(after), trace=trace,
            seconds=time.monotonic() - started, deadline=local_deadline))
    normalized, upper = _normalize(data, tree, current)
    if normalized[1] != policy[1] or upper > before_upper:
        raise AssertionError('LRP polish changed availability or worsened physical cost')
    return normalized, upper, records


def improve_inner_policy(prob_data, tree, inner_policy, incumbent_policy, *,
                         certified_lb, budget, max_seconds=5.):
    """One Final-style ordinary pass, then a package pass only in leftover time."""
    max_seconds = float(max_seconds)
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError('max_seconds must be finite and nonnegative')
    started = time.monotonic()
    if inner_policy is None or budget is None or max_seconds == 0:
        return None
    remaining = float(budget.remaining())
    if not math.isfinite(remaining) or remaining <= 0:
        return None
    deadline = started + min(max_seconds, remaining)
    lower = None if certified_lb is None else float(certified_lb)
    if lower is not None and not math.isfinite(lower):
        raise ValueError('certified_lb must be finite or None')
    data = _instance(prob_data)
    input_upper = float(certify_policy(data, tree, inner_policy['x_star'])['feasible_upper_bound'])
    reported = float(inner_policy['ub'])
    if not math.isfinite(reported) or abs(reported - input_upper) > 1e-6:
        raise RuntimeError('inner policy UB differs from its original-array certificate')
    current, merged_upper, reused = _merge(data, tree, inner_policy['x_star'], incumbent_policy)
    if lower is not None and minimization_bounds_inverted(lower, merged_upper):
        raise RuntimeError('global LB exceeds the merged physical policy UB')
    remaining = float(budget.remaining())
    if not math.isfinite(remaining) or remaining <= 0 or time.monotonic() >= deadline:
        return None
    root = deepcopy(current[1])
    records, passes = [], []
    upper = merged_upper
    for package_moves in (False, True):
        if time.monotonic() >= deadline:
            break
        before = upper
        current, upper, rows = _improve_pass(data, tree, current, deadline,
                                           package_moves=package_moves)
        records.extend(rows)
        passes.append(dict(name='package' if package_moves else 'ordinary',
                           initial_ub=before, final_ub=upper))
    current, upper = _normalize(data, tree, current)
    if current[1] != root or upper > merged_upper:
        raise RuntimeError('polish violated immutable availability or physical cost')
    if lower is not None and minimization_bounds_inverted(lower, upper):
        raise RuntimeError('global LB exceeds the polished physical policy UB')
    return dict(policy=current, ub=upper, certified=True, stats=dict(
        input_ub=input_upper, merged_ub=merged_upper, reused_nodes=reused,
        seconds=time.monotonic()-started, budget_seconds=deadline-started,
        deadline_reached=time.monotonic() >= deadline, passes=passes, rows=records,
        moves=sum(len(row['trace']['moves']) for row in records),
        evaluations=sum(row['trace']['evaluations'] for row in records),
        original_move_core=original.__file__, no_solver_or_native=True))
