"""Exact capacity/incoming-cost bound for original elementary LRP pricing.

This is not equation (3)'s total-positive-prize bound. For every original
own-root elementary tour, its cost is at least the sum of the minimum incoming
arc costs of its visited customers. A fractional knapsack therefore gives an
upper bound on its possible net prize. Negating that exact upper bound gives
a lower bound on idle-inclusive physical pricing, for every availability A.

Only original active customers may be predecessors; self arcs and other
facility roots are excluded. Directed/nonmetric costs and zero demand are
supported. Individually oversized customers are ineligible, as in Investment's
CapacityPriceBound. No native, LP or MIP solver is called.
"""
from __future__ import annotations

from fractions import Fraction as F
import math
from numbers import Integral, Real

import numpy as np

from models.stage_model_core import NodeContext


FORMULA = 'capacity_incoming_fractional_knapsack_v1'


def _exact(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{label} must be an ordinary finite real number')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be an ordinary finite real number') from exc
    if not math.isfinite(number) or abs(number) >= 1e100:
        raise ValueError(f'{label} must be an ordinary finite real number')
    if isinstance(value, F):
        return value
    if isinstance(value, Integral):
        return F(int(value))
    return F(number)


def _ratio(value):
    return value.numerator, value.denominator


def _down(value):
    answer = float(value)
    if F(answer) > value:
        answer = math.nextafter(answer, -math.inf)
    if not math.isfinite(answer) or abs(answer) >= 1e100:
        raise ValueError('physical pricing lower endpoint is not ordinary finite')
    return answer


def capacity_price_lower_bound(ctx, facility_id, lambda_vector):
    """Return an exact fractional-knapsack pricing lower end and its proof.

    ``ctx`` is the validated original NodeContext; prices cover its complete
    customer vector. Inactive prices are immaterial (but still must be finite).
    No availability mask or current S2 assignment restricts this domain.

    With E={j: active_j and demand_j<=capacity_i}, incoming l_j the minimum
    original arc from the own root or another active customer, and
    p_j=max(lambda_j-l_j,0), this helper returns

        h_i(lambda) >= -max{sum(p_j*x_j): sum(d_j*x_j)<=capacity_i,
                           0<=x_j<=1 for j in E}.

    Numerator/denominator pairs preserve all exact arithmetic and the optimal
    fractional vector. The knapsack dual threshold supplies an independent
    equality certificate: U=capacity*mu+sum(max(p_j-mu*d_j,0)). Only safe_lower
    is rounded, toward minus infinity. This is a per-facility pricing bound,
    not a global LRP LB and not a route incumbent.
    """
    if not isinstance(ctx, NodeContext):
        raise TypeError('a validated original NodeContext is required')
    if (isinstance(facility_id, bool) or not isinstance(facility_id, Integral)
            or not 0 <= facility_id < ctx.m):
        raise ValueError('invalid physical facility index')
    i = int(facility_id)
    prices = tuple(_exact(v, 'customer price') for v in lambda_vector)
    if len(prices) != ctx.n:
        raise ValueError('prices must cover the complete original customer domain')
    active = np.asarray(ctx.active)
    demands = np.asarray(ctx.demand)
    matrix = np.asarray(ctx.route_cost[i])
    if (active.shape != (ctx.n,) or demands.shape != (ctx.n,)
            or matrix.shape != (ctx.n+1, ctx.n+1)
            or not np.isin(active, (0, 1)).all()):
        raise ValueError('malformed original physical domain')
    if (not np.isfinite(demands).all() or np.any(demands < 0)
            or np.any(demands[active == 0] != 0)
            or not np.isfinite(matrix).all() or np.any(matrix < 0)
            or np.any(np.diag(matrix) != 0)):
        raise ValueError('original demands/costs must be finite and nonnegative')
    capacity = _exact(ctx.capacity[i], 'original facility capacity')
    if capacity < 0:
        raise ValueError('original facility capacity must be nonnegative')
    demand = tuple(_exact(v, 'original demand') for v in demands)
    active_ids = tuple(int(j) for j in np.flatnonzero(active))
    eligible = tuple(j for j in active_ids if demand[j] <= capacity)
    oversized = tuple(j for j in active_ids if demand[j] > capacity)

    # Binary64 comparisons merely choose an existing input value: they do not
    # round a sum/product. Convert only the selected n minima, not n^2 arcs.
    incoming = [None] * ctx.n
    predecessors = [None] * ctx.n
    if active_ids:
        vertices = np.asarray((0,) + tuple(j+1 for j in active_ids), dtype=int)
        columns = np.asarray(tuple(j+1 for j in active_ids), dtype=int)
        costs = np.array(matrix[np.ix_(vertices, columns)], copy=True)
        costs[vertices[:, None] == columns[None, :]] = math.inf
        argmin = np.argmin(costs, axis=0)
        for position, j in enumerate(active_ids):
            pred = int(vertices[argmin[position]])
            predecessors[j] = pred
            incoming[j] = _exact(matrix[pred, j+1], 'original incoming arc minimum')
    profits = [F()] * ctx.n
    fractions = [F()] * ctx.n
    ratios = [None] * ctx.n
    positive_demand_items = []
    free = []
    for j in eligible:
        profit = profits[j] = max(F(), prices[j] - incoming[j])
        if profit <= 0:
            continue
        if demand[j] == 0:
            fractions[j] = F(1)
            free.append(j)
        else:
            ratios[j] = profit / demand[j]
            positive_demand_items.append(j)
    order = tuple(sorted(positive_demand_items, key=lambda j: (-ratios[j], j)))
    remaining = capacity
    for j in order:
        if remaining <= 0:
            break
        fractions[j] = min(F(1), remaining / demand[j])
        remaining -= fractions[j] * demand[j]
    upper = sum((profits[j] * fractions[j] for j in eligible), F())
    threshold = next((ratios[j] for j in order if fractions[j] < 1), F())
    dual_upper = capacity * threshold + sum(
        (max(F(), profits[j]-threshold*demand[j]) for j in eligible), F())
    used = sum((demand[j]*fractions[j] for j in eligible), F())
    if upper != dual_upper or used > capacity or any(not 0 <= x <= 1 for x in fractions):
        raise AssertionError('exact fractional-knapsack primal/dual certificate failed')
    old = -sum((max(F(), prices[j]) for j in active_ids), F())
    lower = -upper
    if lower < old:
        raise AssertionError('nonnegative incoming/capacity bound weakened equation (3)')
    safe = _down(lower)
    return dict(
        formula=FORMULA, bound_kind='original_elementary_route_pricing_lower',
        safe_lower=safe, lower_exact=_ratio(lower),
        positive_profit_upper_exact=_ratio(upper), eq3_lower_exact=_ratio(old),
        improvement_over_eq3_exact=_ratio(lower-old),
        facility_id=i, context_key=ctx.key, route_key=ctx.route_key(i),
        objective_kind='WITH_IDLE', domain='EXACT_PHYSICAL',
        independent_of_availability=True,
        active_customers=active_ids, eligible_customers=eligible,
        oversized_customers=oversized, zero_demand_positive_customers=tuple(free),
        lambda_exact=tuple(map(_ratio, prices)), demand_exact=tuple(map(_ratio, demand)),
        capacity_exact=_ratio(capacity),
        incoming_exact=tuple(None if v is None else _ratio(v) for v in incoming),
        incoming_predecessor_vertices=tuple(predecessors),
        profit_exact=tuple(map(_ratio, profits)),
        profit_demand_ratio_exact=tuple(None if v is None else _ratio(v) for v in ratios),
        positive_demand_order=order, fractional_selection_exact=tuple(map(_ratio, fractions)),
        used_capacity_exact=_ratio(used), dual_threshold_exact=_ratio(threshold),
        dual_upper_exact=_ratio(dual_upper),
        downward_rounding_slack_exact=_ratio(lower-F(safe)),
        solver_called=False, physical_global_lb_claimed=False,
    )


__all__ = ['FORMULA', 'capacity_price_lower_bound']
