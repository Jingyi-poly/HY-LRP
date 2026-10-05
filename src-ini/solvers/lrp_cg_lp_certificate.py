"""Repair a restricted-route LP primal and certify a full-route-LP upper end.

This pure arithmetic helper never calls a solver. It returns a feasible point
of the complete continuous route master, not an integer policy, a physical
LRP upper bound, or any lower bound. The restricted solver's ObjVal and its
feasibility tolerances are not trusted as certificates.
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction as F
import math
from numbers import Integral, Real

from models.stage_model_core import NodeContext
from solvers.lrp_physical_policy_pool import _route
from solvers.lrp_physical_types import AuditedRoute


def _ordinary(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be an ordinary finite real number")
    try:
        answer = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an ordinary finite real number") from exc
    # Match the physical-certificate convention and reject solver infinity
    # sentinels instead of producing a formally finite but unusable bound.
    if not math.isfinite(answer) or abs(answer) >= 1e100:
        raise ValueError(f"{label} must be an ordinary finite real number")
    return answer


def _ratio(value):
    return value.numerator, value.denominator


def _up(value):
    result = _ordinary(value, "repaired LP objective")
    if F(result) < value:
        result = math.nextafter(result, math.inf)
    return _ordinary(result, "repaired LP upper endpoint")


def certify_route_lp_upper(ctx, A, master, *, expected_node_signature, include_primal=False):
    """Return an exact-feasible continuous route point's upward objective.

    ``master`` is the existing dictionary with ``routes``, ``x``, ``e`` and
    ``objective``. The caller must supply the node signature derived from the
    original instance; NodeContext alone does not contain physical-ID maps or
    model flags. All snapshots receive basic identity/index checks. Only routes
    with a positive repaired weight are rebuilt from ``ctx`` and compared with
    their original AuditedRoute, including audit identity and exact cost. Zero,
    negative-clipped and closed-facility weights contribute no route to the
    feasible point; their unselected costs/orders are not claimed as audited.

    Clip negative route weights to zero and discard closed-facility weights.
    Uniformly shrink the remaining vector until each facility usage <= A and
    each customer coverage <= active. Reconstruct every outsourcing weight as
    active minus coverage, using exact Fractions throughout. No route is
    shortened or changed, including for asymmetric/nonmetric/zero-demand data.

    ``upper`` is for the full continuous route LP only. The optional rational
    primal contains integer numerator/denominator pairs and is JSON/pickle
    friendly. Input objects are not mutated.
    """
    if not isinstance(ctx, NodeContext):
        raise TypeError("a validated original NodeContext is required")
    if not isinstance(include_primal, bool):
        raise TypeError("include_primal must be boolean")
    if (not isinstance(expected_node_signature, str) or len(expected_node_signature) != 64
            or any(c not in '0123456789abcdef' for c in expected_node_signature)):
        raise ValueError("expected original physical node signature is required")
    if not isinstance(master, Mapping) or not {'routes', 'x', 'e', 'objective'} <= set(master):
        raise ValueError("restricted master must provide routes, x, e and objective")
    mask = tuple(A)
    if len(mask) != ctx.m or any(isinstance(v, bool) or not isinstance(v, Real)
                               or v not in (0, 1) for v in mask):
        raise ValueError("availability mask must contain one binary value per facility")
    mask = tuple(int(v) for v in mask)
    active = tuple(int(v) for v in ctx.active)
    if len(active) != ctx.n or any(float(v) != b or b not in (0, 1)
                                   for v, b in zip(ctx.active, active)):
        raise ValueError("original customer activity must be binary and complete")
    outsource_costs = tuple(F(_ordinary(v, "original outsourcing cost")) for v in ctx.outsourcing)
    if len(outsource_costs) != ctx.n or any(c < 0 for c in outsource_costs):
        raise ValueError("original outsourcing costs must be complete and nonnegative")
    routes, raw_x = tuple(master['routes']), tuple(master['x'])
    if len(routes) != len(raw_x):
        raise ValueError("route vector and route snapshot lengths differ")
    raw_x = tuple(F(_ordinary(v, "route LP weight")) for v in raw_x)
    raw_e = master['e']
    if (not isinstance(raw_e, Mapping) or any(type(j) is not int for j in raw_e)
            or set(raw_e) != {j for j, bit in enumerate(active) if bit}):
        raise ValueError("raw outsourcing must map every active original customer exactly once")
    raw_e = {j: F(_ordinary(value, "raw outsourcing LP weight")) for j, value in raw_e.items()}
    raw_objective = F(_ordinary(master['objective'], "raw LP objective"))
    for route in routes:
        if not isinstance(route, AuditedRoute) or route.node_signature != expected_node_signature:
            raise ValueError("route lacks the expected original audited node identity")
        if (isinstance(route.facility_id,bool) or not isinstance(route.facility_id,Integral)
                or not 0 <= route.facility_id < ctx.m):
            raise ValueError("route facility index is invalid")
        order = route.customers_in_order
        if (not isinstance(order,tuple) or not order or len(set(order)) != len(order)
                or any(isinstance(j,bool) or not isinstance(j,Integral) or not 0 <= j < ctx.n for j in order)):
            raise ValueError("route customer indices are invalid")
        if (not isinstance(route.audit_signature,str) or len(route.audit_signature) != 64
                or any(c not in '0123456789abcdef' for c in route.audit_signature)):
            raise ValueError("route audit identity is malformed")
    weights = tuple(max(F(), x) if mask[r.facility_id] else F()
                    for x, r in zip(raw_x, routes))
    positive_indices = tuple(k for k,weight in enumerate(weights) if weight)
    for k in positive_indices:
        route = routes[k]
        if _ordinary(route.cost, "audited route cost") < 0:
            raise ValueError("audited route cost must be nonnegative")
        # _route invokes ctx.check_route_state and independently recomputes the
        # exact capacity, elementary own-root directed arcs, costs and hash.
        fresh = _route(ctx, expected_node_signature, route.facility_id,
                       route.customers_in_order, route.source, route.generation_id)
        if fresh != route:
            raise ValueError("stored route differs from original-array audit")
    usage, coverage = [F() for _ in mask], [F() for _ in active]
    for route, weight in zip(routes, weights):
        if not weight:
            continue
        usage[route.facility_id] += weight
        for j in route.customers_in_order:
            coverage[j] += weight
    scale = min([F(1)] + [F(mask[i])/used for i, used in enumerate(usage) if used > mask[i]]
                + [F(active[j])/covered for j, covered in enumerate(coverage) if covered > active[j]])
    repaired_x = tuple(scale * value for value in weights)
    repaired_usage = tuple(scale * value for value in usage)
    repaired_coverage = tuple(scale * value for value in coverage)
    repaired_e = tuple(F(bit)-value for bit, value in zip(active, repaired_coverage))
    if (any(value < 0 for value in repaired_x) or any(value < 0 for value in repaired_e)
            or any(value > bit for value, bit in zip(repaired_usage, mask))
            or any(c+e != a for c,e,a in zip(repaired_coverage, repaired_e, active))):
        raise AssertionError("exact LP primal repair did not restore feasibility")
    exact = sum((F(*r.cost_exact)*weight for r, weight in zip(routes, repaired_x) if weight), F())
    exact += sum((cost*value for cost, value in zip(outsource_costs, repaired_e)), F())
    upper = _up(exact)
    negative_count = sum(v < 0 for v in raw_x)
    closed_count = sum(v > 0 and not mask[r.facility_id] for v,r in zip(raw_x,routes))
    diagnostic = dict(
        route_count=len(routes), audited_route_count=len(positive_indices),
        audited_positive_routes=len(positive_indices), audited_positive_route_indices=positive_indices,
        original_route_audit_passed=True, original_route_audit_scope='positive_repaired_support_only',
        negative_route_weights_clipped=negative_count,
        closed_facility_route_weights_discarded=closed_count,
        uniform_scale_exact=_ratio(scale), uniform_scaling_used=scale < 1,
        facility_excess_before_scaling_exact=_ratio(max([F()]+[u-a for u,a in zip(usage,mask)])),
        customer_excess_before_scaling_exact=_ratio(max([F()]+[c-a for c,a in zip(coverage,active)])),
        outsourcing_reconstructed=True,
        changed_outsourcing_weights=sum(raw_e.get(j,F()) != value for j,value in enumerate(repaired_e)),
        raw_objective=float(raw_objective), raw_objective_used_as_bound=False,
        repaired_minus_raw_objective_exact=_ratio(exact-raw_objective),
        upward_rounding_slack_exact=_ratio(F(upper)-exact), exact_primal_feasible=True,
        integer_policy_certified=False, physical_lower_bound_claimed=False,
    )
    result = dict(upper=upper, cost_exact=_ratio(exact),
        bound_kind='complete_route_lp_feasible_upper', node_signature=expected_node_signature,
        context_key=ctx.key, availability_mask=mask, diagnostics=diagnostic)
    if include_primal:
        result['primal'] = dict(x=tuple(map(_ratio,repaired_x)), e=tuple(map(_ratio,repaired_e)),
            facility_usage=tuple(map(_ratio,repaired_usage)),
            customer_coverage=tuple(map(_ratio,repaired_coverage)),
            route_audit_ids=tuple(r.audit_signature for r in routes))
    return result
