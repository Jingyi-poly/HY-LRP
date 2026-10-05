"""Improve complete physical policies using exact-cost local route moves.

Fleet purchases remain unchanged; outsourcing may change through exact-cost
customer exchanges. Every move uses exact Fraction sums of original binary64
costs, demands and charges. Ordinary moves run first; larger package exchanges
use only leftover time. The shared allowance is cooperative: final complete-
policy certification can finish just after it.
Only a complete independently certified policy is returned; no solver, oracle
bound, cut archive or current-trial state is changed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import math
import time


def _fraction(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('physical data must be finite')
    return Fraction(value)


def _route_cost(route, vehicle, costs, start, end):
    if not route:
        return Fraction()
    path = (start,) + tuple(route) + (end,)
    return sum((costs[vehicle][a, b] for a, b in zip(path, path[1:])), Fraction())


def _objective(routes, vehicles, active, charges, costs, start, end):
    served = {j for v in vehicles for j in routes[v]}
    return (sum((_route_cost(routes[v], v, costs, start, end) for v in vehicles), Fraction())
            + sum((charges[j] for j in active if j not in served), Fraction()))


def _validate_routes(routes, vehicles, active, demands, capacity):
    if set(routes) != set(vehicles):
        raise AssertionError('local search changed the purchased vehicle domain')
    served = [j for v in vehicles for j in routes[v]]
    if len(set(served)) != len(served) or not set(served) <= set(active):
        raise AssertionError('duplicate or inactive served customer')
    if any(sum((demands[j] for j in routes[v]), Fraction()) > capacity[v] for v in vehicles):
        raise AssertionError('local move exceeded exact physical capacity')


def _outsource_swaps(routes, vehicles, active, demands, capacity, charges,
                     costs, start, end, *, deadline=None):
    served = {j for route in routes.values() for j in route}
    outsourced = tuple(j for j in active if j not in served)
    for v in vehicles:
        route = routes[v]
        load = sum((demands[j] for j in route), Fraction())
        for position, customer in enumerate(route):
            for replacement in outsourced:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                if load - demands[customer] + demands[replacement] > capacity[v]:
                    continue
                delta = (_replace_delta(route, v, position, 1, (replacement,), costs, start, end)
                         + charges[customer] - charges[replacement])
                yield delta, 'outsource_swap', v, v, (position, replacement)


def _package_exchanges(routes, vehicles, active, demands, capacity, charges,
                       costs, start, end, *, deadline=None):
    served = {j for route in routes.values() for j in route}
    outsourced = tuple(j for j in active if j not in served)
    for v in vehicles:
        route = routes[v]
        load = sum((demands[j] for j in route), Fraction())
        before = _route_cost(route, v, costs, start, end)
        for position, removed in enumerate(route):
            for a, b in combinations(outsourced, 2):
                if deadline is not None and time.monotonic() >= deadline:
                    return
                if load - demands[removed] + demands[a] + demands[b] > capacity[v]:
                    continue
                for first, second in ((a, b), (b, a)):
                    candidate = route[:position] + (first, second) + route[position + 1:]
                    delta = (_route_cost(candidate, v, costs, start, end) - before
                             + charges[removed] - charges[first] - charges[second])
                    yield delta, 'outsource_replace_1_with_2', v, v, (position, first, second)
        # Remove any two visits, inserting the replacement at the earlier one.
        for i, k in combinations(range(len(route)), 2):
            for customer in outsourced:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                if load - demands[route[i]] - demands[route[k]] + demands[customer] > capacity[v]:
                    continue
                candidate = route[:i] + (customer,) + route[i + 1:k] + route[k + 1:]
                delta = (_route_cost(candidate, v, costs, start, end) - before
                         + charges[route[i]] + charges[route[k]] - charges[customer])
                yield delta, 'outsource_replace_2_with_1', v, v, (i, k, customer)


def _base_moves(routes, vehicles, demands, capacity, costs, start, end):
    """Yield relocate, directed 2-opt, swap and suffix-exchange deltas."""
    loads = {v: sum((demands[j] for j in routes[v]), Fraction()) for v in vehicles}
    old_costs = {v: _route_cost(routes[v], v, costs, start, end) for v in vehicles}
    for a in vehicles:
        ra, ca = routes[a], costs[a]
        for i, customer in enumerate(ra):
            left = start if i == 0 else ra[i-1]
            right = end if i + 1 == len(ra) else ra[i+1]
            remove = ca[left, right] - ca[left, customer] - ca[customer, right]
            for b in vehicles:
                if b == a or loads[b] + demands[customer] > capacity[b]:
                    continue
                rb, cb = routes[b], costs[b]
                for j in range(len(rb) + 1):
                    before = start if j == 0 else rb[j-1]
                    after = end if j == len(rb) else rb[j]
                    delta = remove + cb[before, customer] + cb[customer, after] - cb[before, after]
                    yield delta, 'relocate', a, b, (i, j)
        # Directed 2-opt: account for reversed internal arcs, not just endpoints.
        reversal = [Fraction()]
        for left, right in zip(ra, ra[1:]):
            reversal.append(reversal[-1] + ca[right, left] - ca[left, right])
        for i in range(len(ra)):
            before = start if i == 0 else ra[i-1]
            for j in range(i+1, len(ra)):
                after = end if j+1 == len(ra) else ra[j+1]
                delta = (ca[before, ra[j]] + ca[ra[i], after]
                         - ca[before, ra[i]] - ca[ra[j], after]
                         + reversal[j] - reversal[i])
                yield delta, '2opt', a, a, (i, j)
    for ai, a in enumerate(vehicles):
        ra, ca = routes[a], costs[a]
        for b in vehicles[ai+1:]:
            rb, cb = routes[b], costs[b]
            for i, ja in enumerate(ra):
                al = start if i == 0 else ra[i-1]
                ar = end if i+1 == len(ra) else ra[i+1]
                for j, jb in enumerate(rb):
                    if loads[a] - demands[ja] + demands[jb] > capacity[a] or loads[b] - demands[jb] + demands[ja] > capacity[b]:
                        continue
                    bl = start if j == 0 else rb[j-1]
                    br = end if j+1 == len(rb) else rb[j+1]
                    delta = (ca[al, jb] + ca[jb, ar] - ca[al, ja] - ca[ja, ar]
                             + cb[bl, ja] + cb[ja, br] - cb[bl, jb] - cb[jb, br])
                    yield delta, 'swap', a, b, (i, j)
            # Exchange suffixes. Different vehicle costs are evaluated in full.
            prefix_a, prefix_b = [Fraction()], [Fraction()]
            for j in ra:
                prefix_a.append(prefix_a[-1] + demands[j])
            for j in rb:
                prefix_b.append(prefix_b[-1] + demands[j])
            for i in range(len(ra)+1):
                for j in range(len(rb)+1):
                    if i == len(ra) and j == len(rb):
                        continue
                    if prefix_a[i] + loads[b] - prefix_b[j] > capacity[a] or prefix_b[j] + loads[a] - prefix_a[i] > capacity[b]:
                        continue
                    new_a, new_b = ra[:i] + rb[j:], rb[:j] + ra[i:]
                    delta = (_route_cost(new_a, a, costs, start, end)
                             + _route_cost(new_b, b, costs, start, end)
                             - old_costs[a] - old_costs[b])
                    yield delta, '2opt_star', a, b, (i, j)


def _replace_delta(route, vehicle, index, length, replacement, costs, start, end):
    before = start if index == 0 else route[index-1]
    after = end if index+length == len(route) else route[index+length]
    old_path = (before,) + route[index:index+length] + (after,)
    new_path = (before,) + replacement + (after,)
    matrix = costs[vehicle]
    return (sum((matrix[a, b] for a, b in zip(new_path, new_path[1:])), Fraction())
            - sum((matrix[a, b] for a, b in zip(old_path, old_path[1:])), Fraction()))


def _segment_relocations(routes, vehicles, demands, capacity, costs, start, end):
    loads = {v: sum((demands[j] for j in routes[v]), Fraction()) for v in vehicles}
    for a in vehicles:
        ra = routes[a]
        for size in (2, 3):
            for i in range(len(ra)-size+1):
                segment = ra[i:i+size]
                segment_load = sum((demands[j] for j in segment), Fraction())
                reduced = ra[:i] + ra[i+size:]
                remove = _replace_delta(ra, a, i, size, (), costs, start, end)
                for b in vehicles:
                    if b != a and loads[b] + segment_load > capacity[b]:
                        continue
                    destination = reduced if a == b else routes[b]
                    for reverse in (False, True):
                        placed = tuple(reversed(segment)) if reverse else segment
                        for j in range(len(destination)+1):
                            if a == b and destination[:j] + placed + destination[j:] == ra:
                                continue
                            delta = remove + _replace_delta(destination, b, j, 0, placed, costs, start, end)
                            yield delta, 'segment_relocate', a, b, (i, size, j, reverse)


def _segment_exchanges(routes, vehicles, demands, capacity, costs, start, end, sizes):
    loads = {v: sum((demands[j] for j in routes[v]), Fraction()) for v in vehicles}
    for ai, a in enumerate(vehicles):
        ra = routes[a]
        for b in vehicles[ai+1:]:
            rb = routes[b]
            for size_a, size_b in sizes:
                for i in range(len(ra)-size_a+1):
                    segment_a = ra[i:i+size_a]
                    load_a = sum((demands[j] for j in segment_a), Fraction())
                    for j in range(len(rb)-size_b+1):
                        segment_b = rb[j:j+size_b]
                        load_b = sum((demands[k] for k in segment_b), Fraction())
                        if loads[a]-load_a+load_b > capacity[a] or loads[b]-load_b+load_a > capacity[b]:
                            continue
                        for reverse_a in ((False, True) if size_a > 1 else (False,)):
                            placed_a = tuple(reversed(segment_a)) if reverse_a else segment_a
                            for reverse_b in ((False, True) if size_b > 1 else (False,)):
                                placed_b = tuple(reversed(segment_b)) if reverse_b else segment_b
                                delta = (_replace_delta(ra, a, i, size_a, placed_b, costs, start, end)
                                         + _replace_delta(rb, b, j, size_b, placed_a, costs, start, end))
                                yield delta, 'segment_exchange', a, b, (i, size_a, j, size_b, reverse_a, reverse_b)


def _neighbors(routes, vehicles, demands, capacity, costs, start, end):
    """Stable order also determines which equal-cost improving move is chosen."""
    yield from _base_moves(routes, vehicles, demands, capacity, costs, start, end)
    yield from _segment_relocations(routes, vehicles, demands, capacity, costs, start, end)
    # Keep these passes separate across all vehicle pairs: merging the lists
    # inside each pair would change deterministic tie breaking.
    yield from _segment_exchanges(routes, vehicles, demands, capacity, costs, start, end,
                                   ((2, 1), (1, 2), (2, 2)))
    yield from _segment_exchanges(routes, vehicles, demands, capacity, costs, start, end,
                                   ((3, 1), (1, 3), (3, 2), (2, 3), (3, 3)))


def _apply(routes, move):
    _, kind, a, b, indices = move
    result = dict(routes)
    ra, rb = routes[a], routes[b]
    if kind == 'outsource_swap':
        i, replacement = indices
        result[a] = ra[:i] + (replacement,) + ra[i+1:]
    elif kind == 'outsource_replace_1_with_2':
        i, first, second = indices
        result[a] = ra[:i] + (first, second) + ra[i+1:]
    elif kind == 'outsource_replace_2_with_1':
        i, k, replacement = indices
        result[a] = ra[:i] + (replacement,) + ra[i+1:k] + ra[k+1:]
    elif kind == 'segment_relocate':
        i, size, j, reverse = indices
        segment = tuple(reversed(ra[i:i+size])) if reverse else ra[i:i+size]
        result[a] = ra[:i] + ra[i+size:]
        destination = result[a] if a == b else rb
        result[b] = destination[:j] + segment + destination[j:]
    elif kind == 'segment_exchange':
        i, size_a, j, size_b, reverse_a, reverse_b = indices
        segment_a, segment_b = ra[i:i+size_a], rb[j:j+size_b]
        if reverse_a:
            segment_a = tuple(reversed(segment_a))
        if reverse_b:
            segment_b = tuple(reversed(segment_b))
        result[a] = ra[:i] + segment_b + ra[i+size_a:]
        result[b] = rb[:j] + segment_a + rb[j+size_b:]
    elif kind == 'relocate':
        i, j = indices
        result[a] = ra[:i] + ra[i+1:]
        result[b] = rb[:j] + (ra[i],) + rb[j:]
    elif kind == 'swap':
        i, j = indices
        result[a] = ra[:i] + (rb[j],) + ra[i+1:]
        result[b] = rb[:j] + (ra[i],) + rb[j+1:]
    elif kind == '2opt':
        i, j = indices
        result[a] = ra[:i] + tuple(reversed(ra[i:j+1])) + ra[j+1:]
    elif kind == '2opt_star':
        i, j = indices
        result[a], result[b] = ra[:i] + rb[j:], rb[:j] + ra[i:]
    else:
        raise ValueError('unknown local move')
    return result


def _search(routes, vehicles, demands, capacity, costs, start, end, deadline, *,
            active, charges, package_moves=False):
    _validate_routes(routes, vehicles, active, demands, capacity)
    current = dict(routes)
    moves, checks, exhausted = [], 0, False
    while time.monotonic() < deadline:
        best = None
        def neighbors():
            yield from _outsource_swaps(current, vehicles, active, demands, capacity,
                                         charges, costs, start, end, deadline=deadline)
            if package_moves:
                yield from _package_exchanges(current, vehicles, active, demands, capacity,
                                              charges, costs, start, end, deadline=deadline)
            yield from _neighbors(current, vehicles, demands, capacity, costs, start, end)
        for move in neighbors():
            checks += 1
            if time.monotonic() >= deadline:
                exhausted = True
                break
            if move[0] < 0 and (best is None or move[0] < best[0]):
                best = move
        if best is None:
            break
        updated = _apply(current, best)
        _validate_routes(updated, vehicles, active, demands, capacity)
        before = _objective(current, vehicles, active, charges, costs, start, end)
        after = _objective(updated, vehicles, active, charges, costs, start, end)
        if not after < before or after - before != best[0]:
            raise AssertionError('local delta is not a strict exact true-cost improvement')
        current = updated
        moves.append(dict(kind=best[1], exact_delta=str(best[0]), delta=float(best[0])))
        if exhausted:
            break
    return current, dict(moves=moves, move_counts=dict(Counter(m['kind'] for m in moves)),
                         evaluations=checks, deadline_reached=exhausted or time.monotonic() >= deadline)


def _canonical_routes(pd, routes, vehicles, signatures, costs, start, end):
    from models.stage2_symmetry import canonicalize_assignment_rows
    for k in getattr(pd, 'K', ()):
        group = [v for v in pd.V_k[k] if v in vehicles]
        if group and any(signatures[v] != signatures[group[0]] for v in group):
            raise ValueError('declared interchangeable vehicles differ in original capacity/cost matrix')
    rows = [[int(j in routes[v]) for j in pd.J] for v in vehicles]
    ranked, _, _ = canonicalize_assignment_rows(pd, vehicles, rows, [int(bool(routes[v])) for v in vehicles])
    by_set = {frozenset(route): route for route in routes.values() if route}
    result = {v: by_set.get(frozenset(j for j, bit in zip(pd.J, row) if bit), ())
              for v, row in zip(vehicles, ranked)}
    before = sum((_route_cost(routes[v], v, costs, start, end) for v in vehicles), Fraction())
    after = sum((_route_cost(result[v], v, costs, start, end) for v in vehicles), Fraction())
    if after != before:
        raise AssertionError('canonicalization changed physical route cost')
    return result


def _improve_pass(policy, pd, tree, budget_s, *, package_moves=False):
    from s2backward.routeopt.restricted_master import certify_complete_policy, _certify_node, _path
    budget_s = float(budget_s)
    if not math.isfinite(budget_s) or budget_s < 0:
        raise ValueError('budget_s must be finite and nonnegative')
    started = time.monotonic()
    deadline = started + budget_s
    # The original model/checker charges all unassigned j and requires h_j=0
    # for inactive j. Validate that contract before using an active-only domain.
    for node in tree[2]:
        if any(float(node.active[j]) == 0. and _fraction(node.c_out[j]) != 0 for j in pd.J):
            raise ValueError('inactive customer has nonzero outsourcing charge')
    current, initial_ub = certify_complete_policy(pd, tree, policy)
    initial = deepcopy(current)
    # Original exact fleet identity is required for grouping, not rounded bits.
    for key, bit in policy[1][0].items():
        if key.startswith('z[') and float(bit) not in (0., 1.):
            raise ValueError('local route grouping requires original exact binary fleet')
    start, end = int(pd.numAllnodes)-2, int(pd.numAllnodes)-1
    costs, capacity, signatures = {}, {}, {}
    for v in pd.V:
        capacity[v] = _fraction(pd.Qv[v])
        costs[v] = {(i, j): _fraction(pd.c_routing[v][i, j])
                    for i in (start, *pd.J) for j in (*pd.J, end)
                    if i != j and (i, j) != (start, end)}
        if capacity[v] < 0 or any(cost < 0 for cost in costs[v].values()):
            raise ValueError('negative original physical capacity or route cost')
        # An unused vehicle has no route, not a depot-to-depot arc.
        costs[v][start, end] = Fraction()
        signatures[v] = (capacity[v], tuple(costs[v].items()))
    groups = {}
    for node in tree[2]:
        fleet = tuple(v for v in pd.V if current[1][0][f'z[{v},{node.info[1]}]'] == 1.)
        routes = {int(tree[3][s].info): _path(current[3][s], start, end) for s in node.successor}
        served = frozenset(j for route in routes.values() for j in route)
        key = (tuple((float(node.active[j]).hex(), float(node.volume[j]).hex(), float(node.c_out[j]).hex())
                     for j in pd.J), fleet, served)
        route_cost = sum((_route_cost(routes[v], v, costs, start, end) for v in pd.V), Fraction())
        groups.setdefault(key, []).append((node, routes, route_cost))
    ordered = sorted(groups.values(), key=lambda members: (
        -sum((_fraction(n.multi_coeff) * cost for n, _, cost in members), Fraction()),
        min(n.index for n, _, _ in members)))
    stats = dict(budget_s=budget_s, groups=len(ordered), searched_groups=0, improved_nodes=[],
                 rows=[], moves=0, evaluations=0)
    for group_index, members in enumerate(ordered):
        now = time.monotonic()
        if now >= deadline:
            break
        node, source_routes, old_route_cost = min(members, key=lambda item: (item[2], item[0].index))
        vehicles = tuple(v for v in pd.V if current[1][0][f'z[{v},{node.info[1]}]'] == 1.)
        if not vehicles:
            continue
        demands = {j: _fraction(node.volume[j]) for j in pd.J}
        charges = {j: _fraction(node.c_out[j]) for j in pd.J}
        active = tuple(j for j in pd.J if float(node.active[j]) == 1.)
        if any(value < 0 for value in (*demands.values(), *charges.values())):
            raise ValueError('negative demand or outsourcing charge')
        original = {v: source_routes[v] for v in vehicles}
        before_cost = _objective(original, vehicles, active, charges, costs, start, end)
        # Fair residual allocation prevents one hard scenario consuming every period.
        group_deadline = min(deadline, now + (deadline-now)/(len(ordered)-group_index))
        candidate, trace = _search(original, vehicles, demands, capacity, costs, start, end,
                                   group_deadline, active=active, charges=charges, package_moves=package_moves)
        candidate = _canonical_routes(pd, candidate, vehicles, signatures, costs, start, end)
        _validate_routes(candidate, vehicles, active, demands, capacity)
        new_route_cost = sum((_route_cost(candidate[v], v, costs, start, end) for v in vehicles), Fraction())
        after_cost = _objective(candidate, vehicles, active, charges, costs, start, end)
        if after_cost > before_cost:
            raise AssertionError('route search increased exact physical cost')
        updates = []
        for target, _, previous_cost in members:
            if after_cost >= previous_cost + before_cost - old_route_cost:
                continue
            assignment = {f'y[{v}]': float(bool(candidate.get(v))) for v in pd.V}
            assignment.update({f'alpha[{j},{v}]': float(j in candidate.get(v, ())) for j in pd.J for v in pd.V})
            routes = {}
            for third in target.successor:
                v = int(tree[3][third].info)
                route = candidate.get(v, ())
                path = (start,) + route + (end,) if route else ()
                routes[third] = {f'x[{a},{b}]': 1. for a, b in zip(path, path[1:])}
            trial = {1: current[1], 2: {target.index: assignment}, 3: routes}
            checked, checked_routes, _ = _certify_node(pd, tree, target.index, current[1][0], trial)
            updates.append((target.index, checked, checked_routes))
        for target, checked, checked_routes in updates:
            current[2][target] = checked
            current[3].update(checked_routes)
            stats['improved_nodes'].append(int(target))
        stats['searched_groups'] += 1
        stats['moves'] += len(trace['moves'])
        stats['evaluations'] += trace['evaluations']
        stats['rows'].append(dict(source_node=int(node.index), members=[int(n.index) for n, _, _ in members],
                                  fleet=list(vehicles), before_route_cost=float(old_route_cost),
                                  after_route_cost=float(new_route_cost), exact_route_delta=str(new_route_cost-old_route_cost),
                                  before_cost=float(before_cost), after_cost=float(after_cost), exact_delta=str(after_cost-before_cost),
                                  elapsed_seconds=time.monotonic()-now, trace=trace))
    normalized, upper = certify_complete_policy(pd, tree, current)
    if normalized[1] != initial[1]:
        raise AssertionError('local route search changed the investment trajectory')
    if upper >= initial_ub:
        normalized, upper = initial, initial_ub
        stats['improved_nodes'] = []
    stats.update(seconds=time.monotonic()-started, budget_exhausted=time.monotonic() >= deadline,
                 move_counts=dict(Counter(m['kind'] for row in stats['rows'] for m in row['trace']['moves'])),
                 neighborhood='ordinary routes + outsourcing swap' + (' + package exchanges' if package_moves else ''))
    return dict(policy=normalized, ub=upper, certified=True, initial_ub=initial_ub,
                improvement=initial_ub-upper, improved_nodes=stats['improved_nodes'], stats=stats)


def improve(policy, pd, tree, budget_s):
    """Certify an ordinary pass, then use only leftover time for larger exchanges."""
    started = time.monotonic()
    first = _improve_pass(policy, pd, tree, budget_s)
    deadline = started + float(budget_s)
    remaining = deadline - time.monotonic()
    first_pass = dict(name='ordinary', seconds=first['stats']['seconds'],
                      initial_ub=first['initial_ub'], final_ub=first['ub'])
    if remaining <= 0:
        first['stats']['passes'] = [first_pass]
        return first
    second = _improve_pass(first['policy'], pd, tree, remaining, package_moves=True)
    if second['ub'] > first['ub']:
        raise AssertionError('package pass worsened the certified ordinary policy')
    result = dict(second, initial_ub=first['initial_ub'], improvement=first['initial_ub']-second['ub'])
    rows = [dict(row, pass_name=name) for name, part in (('ordinary', first), ('package', second))
            for row in part['stats']['rows']]
    counts = Counter(move['kind'] for row in rows for move in row['trace']['moves'])
    improved = sorted(set(first['improved_nodes']) | set(second['improved_nodes']))
    result['improved_nodes'] = improved
    result['stats'] = dict(second['stats'], budget_s=float(budget_s), rows=rows,
        groups=first['stats']['groups']+second['stats']['groups'],
        searched_groups=first['stats']['searched_groups']+second['stats']['searched_groups'],
        moves=sum(counts.values()), evaluations=first['stats']['evaluations']+second['stats']['evaluations'],
        move_counts=dict(counts), improved_nodes=improved, seconds=time.monotonic()-started,
        budget_exhausted=time.monotonic() >= deadline,
        neighborhood='ordinary first; larger package exchanges only within remaining allowance',
        passes=[first_pass, dict(name='package', allowance_seconds=remaining,
                seconds=second['stats']['seconds'], initial_ub=first['ub'], final_ub=second['ub'],
                ub_improvement=first['ub']-second['ub'])])
    return result


def improve_inner_policy(pd, tree, inner_policy, incumbent_policy, *,
                         certified_lb, budget, max_seconds=5.):
    """Polish one complete policy within an existing refinement allowance.

    The caller decides when to attempt this and whether its returned feasible
    UB improves the global incumbent. No callback or algorithm state is touched.
    ``certified_lb`` is an existing finite lower bound or None, never inferred
    from a target/gap. ``budget.remaining()`` supplies the shared time left.
    Preparation consumes the same allowance as the search. Final independent
    certification is cooperative and may finish slightly after the deadline.
    """
    from core.solver_bounds import minimization_bounds_inverted
    from s2backward.routeopt.restricted_master import (
        certify_complete_policy, merge_same_fleet_policy,
    )
    max_seconds = float(max_seconds)
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError('max_seconds must be finite and nonnegative')
    lower = None if certified_lb is None else float(certified_lb)
    if lower is not None and not math.isfinite(lower):
        raise ValueError('certified_lb must be finite or None')
    if inner_policy is None or budget is None or max_seconds == 0:
        return None
    started = time.monotonic()
    remaining = float(budget.remaining())
    if not math.isfinite(remaining) or remaining <= 0:
        return None
    allowance = min(max_seconds, remaining)
    deadline = started + allowance
    _, input_ub = certify_complete_policy(pd, tree, inner_policy['x_star'])
    reported = float(inner_policy['ub'])
    if not math.isfinite(reported) or abs(reported-input_ub) > 1e-6:
        raise RuntimeError('inner policy cost does not match its complete-policy certificate')
    merged = merge_same_fleet_policy(pd, tree, inner_policy['x_star'], incumbent_policy)
    if lower is not None and minimization_bounds_inverted(lower, merged['ub']):
        raise RuntimeError('global LB exceeds the complete feasible policy UB')
    prepared = time.monotonic()
    remaining = float(budget.remaining())
    if not math.isfinite(remaining):
        return None
    search_seconds = min(deadline-prepared, remaining)
    if search_seconds <= 0:
        return None
    result = improve(merged['policy'], pd, tree, search_seconds)
    if result.get('certified') is not True:
        raise RuntimeError('route search returned an uncertified feasible policy')
    normalized, upper = certify_complete_policy(pd, tree, result['policy'])
    reported = float(result['ub'])
    if not math.isfinite(reported) or abs(reported-upper) > 1e-6:
        raise RuntimeError('route search cost does not match its complete-policy certificate')
    if lower is not None and minimization_bounds_inverted(lower, upper):
        raise RuntimeError('route search feasible policy falls below the certified global LB')
    if normalized[1] != merged['policy'][1] or upper > merged['ub']:
        raise RuntimeError('route search changed purchases or increased physical cost')
    finished = time.monotonic()
    stats = dict(result['stats'])
    stats.update(budget_seconds=allowance, search_seconds=search_seconds,
                 preparation_seconds=prepared-started,
                 search_elapsed_seconds=result['stats']['seconds'],
                 seconds=finished-started, budget_exhausted=finished >= deadline,
                 input_ub=input_ub, merged_ub=merged['ub'],
                 reused_nodes=list(merged['reused_nodes']))
    return dict(policy=normalized, ub=upper, certified=True, stats=stats)
