"""Audit LRP forward policies against the original data, with upward costs."""
from __future__ import annotations

import math
import re
from fractions import Fraction

import numpy as np

from models.stage_builder import _instance, _node_context
from models.stage_model_core import NodeContext, audit_tour, certify_node, certify_policy as _certify_policy, exact_capacity, facility_cost
from solvers.forward_ub import round_fraction_up


class InvalidForwardPolicy(RuntimeError):
    """An incumbent is not a feasible policy of the original LRP."""


def _bit(values, name):
    if name not in values:
        raise InvalidForwardPolicy(f"missing binary decision {name}")
    value = float(values[name])
    if not math.isfinite(value) or min(abs(value), abs(value - 1)) > 2e-6:
        raise InvalidForwardPolicy(f"nonbinary decision {name}={value}")
    return int(round(value))


def _exact(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise InvalidForwardPolicy(f"invalid nonnegative cost {value}")
    return Fraction.from_float(value)


def _facility_exact(data, A):
    total = Fraction(0)
    for i, row in enumerate(A):
        previous = 0
        for k, current in enumerate(row):
            if current and not previous:
                total += _exact(data.arrays['opening_cost'][i, k])
            elif current and previous:
                total += _exact(data.arrays['continuation_cost'][i, k])
            elif previous and not current:
                total += _exact(data.arrays['closing_cost'][i, k])
            previous = current
    return total


def certify_stage1_forward_policy(prob_data, decisions):
    """Certify on a fresh original-data snapshot at this public boundary."""
    return _certify_stage1_data(_instance(prob_data), decisions)


def _certify_stage1_data(data, decisions):
    m, _, _, L, _ = data.shape
    A = [[_bit(decisions, f'A[{i},{k}]') for k in range(L)] for i in range(m)]
    try:
        facility_cost(data, A)
    except ValueError as exc:
        raise InvalidForwardPolicy(str(exc)) from exc
    normalized = {}
    for i in range(m):
        previous = 0
        for k, current in enumerate(A[i]):
            events = {'A': current, 'o': current * (1 - previous),
                      'h': current * previous, 'b': previous * (1 - current)}
            for group, bit in events.items():
                name = f'{group}[{i},{k}]'
                if name in decisions and _bit(decisions, name) != bit:
                    raise InvalidForwardPolicy(f'wrong facility transition {name}')
                normalized[name] = float(bit)
            previous = current
    return normalized, round_fraction_up(_facility_exact(data, A))


def certify_stage2_forward_policy(prob_data, node, x_prev, decisions):
    """Certify assignment/outsourcing/capacity on a fresh public snapshot."""
    ctx = _node_context(_instance(prob_data), node, stage=2)
    return _certify_stage2_context(ctx, x_prev, decisions)


def _certify_stage2_context(ctx, x_prev, decisions):
    A = [_bit(x_prev, f'A[{i},{ctx.interval}]') for i in range(ctx.m)]
    alpha = [[_bit(decisions, f'alpha[{i},{j}]') for j in range(ctx.n)] for i in range(ctx.m)]
    u = [_bit(decisions, f'u[{i}]') for i in range(ctx.m)]
    e = [_bit(decisions, f'e[{j}]') for j in range(ctx.n)]
    for j in range(ctx.n):
        if sum(row[j] for row in alpha) + e[j] != int(ctx.active[j]):
            raise InvalidForwardPolicy(f'customer {j} is not served exactly once if active')
    for i in range(ctx.m):
        if u[i] > A[i] or u[i] != int(any(alpha[i])):
            raise InvalidForwardPolicy(f'unavailable or empty dispatch at facility {i}')
        if not exact_capacity(ctx.demand, alpha[i], float(ctx.capacity[i]) * u[i]):
            raise InvalidForwardPolicy(f'capacity violation at facility {i}')
    normalized = {f'alpha[{i},{j}]': float(alpha[i][j]) for i in range(ctx.m) for j in range(ctx.n)}
    normalized.update({f'u[{i}]': float(u[i]) for i in range(ctx.m)})
    normalized.update({f'e[{j}]': float(e[j]) for j in range(ctx.n)})
    total = sum((_exact(ctx.outsourcing[j]) * e[j] for j in range(ctx.n)), Fraction(0))
    return normalized, round_fraction_up(total)


def certify_stage3_forward_policy(prob_data, node, x_prev, decisions):
    """Certify a directed cycle using a fresh original-data snapshot."""
    ctx = _node_context(_instance(prob_data), node, stage=3)
    return _certify_stage3_context(ctx, int(node.info), x_prev, decisions)


def _certify_stage3_context(ctx, i, x_prev, decisions):
    alpha = [_bit(x_prev, f'alpha[{i},{j}]') for j in range(ctx.n)]
    u = _bit(x_prev, f'u[{i}]')
    if not exact_capacity(ctx.demand, alpha, float(ctx.capacity[i]) * u):
        raise InvalidForwardPolicy(f'capacity violation at facility {i}')
    if any(alpha[j] > int(ctx.active[j]) for j in range(ctx.n)):
        raise InvalidForwardPolicy('inactive customer assigned to a route')
    arcs = []
    for name in decisions:
        if not str(name).startswith('r['):
            continue
        match = re.fullmatch(r'r\[(\d+),(\d+),(\d+)\]', name)
        if match is None:
            raise InvalidForwardPolicy(f'invalid route variable {name}')
        fi, v, w = map(int, match.groups())
        selected = _bit(decisions, name)
        if selected and fi != i:
            raise InvalidForwardPolicy('route belongs to a different facility')
        if selected:
            arcs.append((v, w))
    try:
        audit_tour(ctx, i, alpha, u, arcs)
    except ValueError as exc:
        raise InvalidForwardPolicy(str(exc)) from exc
    normalized = {f'r[{i},{v},{w}]': 1.0 for v, w in sorted(arcs)}
    total = sum((_exact(ctx.route_cost[i, v, w]) for v, w in arcs), Fraction(0))
    return normalized, round_fraction_up(total)


def _validated_route_context(data, node, parent_context):
    """Reuse only the same freshly reconstructed context within this audit."""
    supplied = node if isinstance(node, NodeContext) else getattr(node, 'context', None)
    if (isinstance(supplied, NodeContext)
            and type(supplied.period) is int and type(supplied.scenario) is int
            and supplied.period == parent_context.period
            and supplied.scenario == parent_context.scenario):
        if supplied.key != parent_context.key:
            raise ValueError('NodeContext does not match builder instance')
        return parent_context
    # Preserve the original validation/resolution for absent, foreign-period,
    # noncanonical or unsupported context representations.
    return _node_context(data, node, stage=3)


def certify_lrp_forward_policy(prob_data, x_star, scen_tree):
    """Reaudit every node on ONE private fresh original-data snapshot."""
    return _certify_lrp_snapshot(_instance(prob_data), x_star, scen_tree)[1]


def _normalize_lrp_forward_policy(prob_data, scen_tree, x_star):
    """Private combined normalization/certification, without a trusted cache."""
    return _certify_lrp_snapshot(_instance(prob_data), x_star, scen_tree)


def _certify_lrp_snapshot(data, x_star, scen_tree):
    m, _, _, L, _ = data.shape
    root, _ = _certify_stage1_data(data, x_star[1][0])
    normalized = {1: {0: root}, 2: {}, 3: {}}
    A = np.array([[root[f'A[{i},{k}]'] for k in range(L)] for i in range(m)], dtype=int)
    nodes, total = {}, _facility_exact(data, A)
    for second_ind in scen_tree[1][0].successor:
        node = scen_tree[2][second_ind]
        ctx = _node_context(data, node, stage=2)
        state, _ = _certify_stage2_context(ctx, root, x_star[2][second_ind])
        normalized[2][second_ind] = state
        tours = {}
        node_exact = sum((_exact(ctx.outsourcing[j]) * int(state[f'e[{j}]']) for j in range(ctx.n)), Fraction(0))
        facilities = set()
        for third_ind in node.successor:
            third = scen_tree[3][third_ind]
            i = int(third.info)
            if i in facilities:
                raise InvalidForwardPolicy('duplicate facility route node')
            facilities.add(i)
            route_ctx = _validated_route_context(data, third, ctx)
            route, _ = _certify_stage3_context(route_ctx, i, state, x_star[3][third_ind])
            normalized[3][third_ind] = route
            arcs = [tuple(map(int, name[2:-1].split(',')))[1:] for name in route]
            tours[i] = {'arcs': arcs}
            node_exact += sum((_exact(ctx.route_cost[i, v, w]) for v, w in arcs), Fraction(0))
        if facilities != set(range(m)):
            raise InvalidForwardPolicy('policy is missing facility routes')
        record = certify_node(ctx, A[:, ctx.interval],
            [[state[f'alpha[{i},{j}]'] for j in range(ctx.n)] for i in range(m)],
            [state[f'e[{j}]'] for j in range(ctx.n)], [state[f'u[{i}]'] for i in range(m)], tours)
        key = (ctx.period, ctx.scenario)
        if key in nodes:
            raise InvalidForwardPolicy('duplicate period/scenario node')
        nodes[key] = record
        total += _exact(data.arrays['scenario_prob'][ctx.scenario]) * node_exact
    certificate = _certify_policy(data, A, nodes)
    certificate['feasible_upper_bound'] = round_fraction_up(total)
    certificate['cost_rounding'] = 'exact binary64 rational sum, rounded upward once'
    certificate['all_original_nodes_audited'] = True
    certificate['nodes'] = {f'{t},{s}': record for (t, s), record in nodes.items()}
    return normalized, certificate


__all__ = ['InvalidForwardPolicy', 'certify_stage1_forward_policy',
           'certify_stage2_forward_policy', 'certify_stage3_forward_policy',
           'certify_lrp_forward_policy', 'certify_policy']


def certify_policy(prob_data, scen_tree, x_best):
    """Public comparison API; no solver is called while auditing a policy."""
    return certify_lrp_forward_policy(prob_data, x_best, scen_tree)
