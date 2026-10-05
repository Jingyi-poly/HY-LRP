"""Static valid inequalities shared by the decomposed VRP models.

This module contains only inequalities that are independent of a dynamically
chosen Benders/Lagrangian multiplier.  They should be installed once when a
model is built and then reused by every solve of that model.

The three assignment/routing capacity helpers and the two objective lower
bounds live together here so every decomposed backend uses the same formulas:

``Stage-1 route-aware CapCut``
    For one scenario-period node, let ``c[j]`` be the outsourcing cost,
    ``d[j]`` the demand and ``ell[j]`` a lower bound on the routing cost paid
    whenever customer ``j`` is served.  Serving ``j`` can therefore save at
    most ``r[j] = max(0, c[j] - ell[j])`` from the all-outsourced baseline
    ``C0 = sum(c[j])``.  The purchased capacity is
    ``C(z) = sum(Q[v] * z[v])``.  Fractional relaxation gives, for every
    nonnegative ``lambda``,

    ``eta >= b[lambda] - lambda*C(z)``, where

    ``b[lambda] = min(C0, C0 - sum(max(0, r[j]-lambda*d[j])) + rho)``

    and ``rho`` is the cheapest capacity-compatible customer-to-end-depot
    arc.  The incoming terms account for every route arc except its return to
    the end depot.  An all-outsourced solution costs ``C0``; every nonempty
    routed solution pays at least one such return arc, which proves the cap.

    Rows at ``lambda=0`` and the positive breakpoints ``r[j]/d[j]`` form the
    complete envelope.  ``ell[j]`` is the cheapest feasible incoming arc over
    vehicles that can carry ``j``; a customer predecessor is considered only
    when the pair fits that vehicle.  This dominates the former envelope that
    silently set every routing lower bound to zero.  Terms in the closed
    absolute ``1e-6`` zero band are removed only with a conservative intercept
    reduction.  Costs and capacity are *per operation*; probabilities and
    annual operating multipliers belong only in the Stage-1 objective.

``Stage-3 RouteCut``
    If an assigned customer must have one incoming route arc, its contribution
    is at least the cheapest allowed non-self incoming arc.  Hence

    ``theta >= sum(min_incoming_cost[j] * alpha[j])``.

    Incoming-arc bound only (no return-arc term, avoids double-counting depot
    return).  Nonnegative terms at most ``1e-6`` are omitted.

The C++ BPC/ESP backends do not need the capacity rows below: their columns and
labels already encode the corresponding capacity feasibility.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

import gurobipy as gp


# Static rows treat represented coefficients in the closed band
# ``[-1e-6, 1e-6]`` as zero.  Every coefficient in this module is generated
# from nonnegative physical/economic data; row-specific code below either
# drops such a term only in a weakening direction or compensates the
# RHS/intercept conservatively.  This is a modelling sparsification policy,
# not a replacement for the directed rounding used to preserve validity.
STATIC_CUT_ZERO_TOL = 1e-6
_CAPACITY_TOL = STATIC_CUT_ZERO_TOL
EXTENDED_COVER_MAX_CUTS = 16

__all__ = [
    "STATIC_CUT_ZERO_TOL",
    "EXTENDED_COVER_MAX_CUTS",
    "add_extended_cover_cuts",
    "add_half_capacity_clique_cut",
    "add_vehicle_half_capacity_cuts",
    "compute_stage1_service_lower_bounds",
    "compute_stage1_return_lower_bound",
    "route_saving_lambda_breakpoints",
    "compute_stage1_capcut_rows",
    "add_stage1_capacity_cuts",
    "compute_min_incoming_costs",
    "add_route_cost_lower_bound",
    "add_stage2_route_cost_lower_bounds",
]


def _value(values: Mapping[Any, float] | Sequence[float], key: Any) -> float:
    return float(values[key])


def _arc_value(values: Any, tail: Any, head: Any) -> float:
    """Read an arc cost from a tuple-keyed or two-dimensional container."""
    try:
        return float(values[tail, head])
    except (KeyError, IndexError, TypeError):
        try:
            return float(values[tail][head])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"routing cost is missing for allowed arc ({tail!r}, {head!r})"
            ) from exc


def _name(prefix: str, base: str) -> str:
    return f"{prefix}_{base}" if prefix else base


def _finite_nonnegative(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative, got {value!r}")
    return result


def _exact_difference_floor(
    minuend: float | Fraction,
    subtrahends: Iterable[float],
    *,
    label: str,
) -> tuple[Fraction, float]:
    """Subtract represented nonnegative values and round toward ``-inf``."""
    exact = (
        minuend if isinstance(minuend, Fraction)
        else Fraction.from_float(float(minuend))
    )
    for value in subtrahends:
        exact -= Fraction.from_float(
            _finite_nonnegative(value, label=label)
        )
    if exact <= 0:
        return exact, 0.0
    try:
        result = float(exact)
    except OverflowError:
        return exact, sys.float_info.max
    if Fraction.from_float(result) > exact:
        result = math.nextafter(result, -math.inf)
    return exact, result


def _exact_fraction_floor(value: Fraction, *, label: str) -> float:
    """Convert a nonnegative exact value to finite binary64 toward ``-inf``."""
    if value < 0:
        raise ValueError(f"{label} must be nonnegative, got {value!r}")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds finite binary64") from exc
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _exact_nonnegative_product_ceiling(
    left: float,
    right: float,
    *,
    label: str,
) -> float:
    """Round a product of represented binary64 inputs toward ``+infinity``."""
    left_value = _finite_nonnegative(left, label=label)
    right_value = _finite_nonnegative(right, label=label)
    exact = Fraction.from_float(left_value) * Fraction.from_float(right_value)
    try:
        result = float(exact)
    except OverflowError as exc:
        raise ValueError(f"{label} exact product exceeds finite binary64") from exc
    if Fraction.from_float(result) < exact:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise ValueError(f"{label} upward-rounded product is non-finite")
    return result


def _active_customers(
    customers: Iterable[Any],
    active: Mapping[Any, float] | Sequence[float] | None,
) -> list[Any]:
    customer_list = list(customers)
    if active is None:
        return customer_list

    selected = []
    for customer in customer_list:
        try:
            active_value = float(active[customer])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"active is missing customer {customer!r}"
            ) from exc
        if not math.isfinite(active_value):
            raise ValueError(
                f"active[{customer!r}] must be finite, got {active_value!r}"
            )
        if active_value < 0.0 or active_value > 1.0:
            raise ValueError(
                f"active[{customer!r}] must lie in [0, 1], got {active_value!r}"
            )
        if active_value > 0.5:
            selected.append(customer)
    return selected


def add_extended_cover_cuts(
    model,
    *,
    customers: Iterable[Any],
    volumes: Mapping[Any, float] | Sequence[float],
    capacity: float,
    alpha,
    prefix: str = "",
    max_cuts: int = EXTENDED_COVER_MAX_CUTS,
    tol: float = _CAPACITY_TOL,
) -> int:
    """Add extended three-customer cover inequalities.

    For a three-item cover ``{a,b,c}``, each pair remains within capacity but
    the triple exceeds it.  With items sorted by non-increasing demand, the
    cover is extended with selected dominating items preceding ``a``::

        sum(alpha[j] : j in extension) <= 2.

    Two-item covers omitted (already implied by the knapsack).
    """
    if max_cuts <= 0:
        return 0
    tol = _finite_nonnegative(tol, label="tol")

    cap = _finite_nonnegative(capacity, label="capacity")
    customer_list = list(customers)
    volume_by_customer = {
        customer: _finite_nonnegative(
            _value(volumes, customer), label=f"volumes[{customer!r}]"
        )
        for customer in customer_list
    }
    items = sorted(
        (
            (j, volume_by_customer[j])
            for j in customer_list
            if volume_by_customer[j] > tol
        ),
        key=lambda item: -item[1],
    )
    if len(items) < 3:
        return 0

    added = 0
    for a in range(len(items)):
        volume_a = items[a][1]
        for b in range(a + 1, len(items)):
            volume_b = items[b][1]
            if volume_a + volume_b > cap + tol:
                continue

            for c in range(b + 1, len(items)):
                if volume_a + volume_b + items[c][1] <= cap + tol:
                    # Items are sorted descending, so no later c can be a cover.
                    break

                extension_positions = list(range(a + 1)) + [b, c]
                extension = [items[pos][0] for pos in extension_positions]
                model.addConstr(
                    gp.quicksum(alpha[j] for j in extension) <= 2,
                    name=_name(
                        prefix,
                        f"cover3[{items[a][0]},{items[b][0]},{items[c][0]}]",
                    ),
                )
                added += 1
                if added >= max_cuts:
                    return added
                # Keep one representative c per (a,b) to cap build cost.
                break

    return added


def add_half_capacity_clique_cut(
    model,
    *,
    customers: Iterable[Any],
    vehicles: Iterable[Any],
    volumes: Mapping[Any, float] | Sequence[float],
    capacities: Mapping[Any, float] | Sequence[float],
    outsource,
    active: Mapping[Any, float] | Sequence[float] | None = None,
    y=None,
    prefix: str = "",
    tol: float = _CAPACITY_TOL,
) -> int:
    """Add the global half-capacity incompatibility inequality.

    Customers with demand greater than half the largest vehicle capacity are
    pairwise incompatible on every vehicle.  Hence at most one of them can be
    served by each activated vehicle::

        sum(outsource[j] for j in clique) + sum(y[v]) >= len(clique)

    If no activation variables are supplied, ``len(vehicles)`` is used as the
    (weaker) service limit.  Only active customers participate.
    """
    customer_list = _active_customers(customers, active)
    vehicle_list = list(vehicles)
    if not customer_list or not vehicle_list:
        return 0
    tol = _finite_nonnegative(tol, label="tol")
    q_max = max(
        _finite_nonnegative(_value(capacities, v), label=f"capacities[{v!r}]")
        for v in vehicle_list
    )
    clique = [
        j for j in customer_list
        if _finite_nonnegative(_value(volumes, j), label=f"volumes[{j!r}]")
        > 0.5 * q_max + tol
    ]
    if len(clique) <= 1:
        return 0
    lhs = gp.quicksum(outsource[j] for j in clique)
    if y is not None:
        lhs += gp.quicksum(y[v] for v in vehicle_list)
        rhs = len(clique)
    else:
        rhs = len(clique) - len(vehicle_list)
        if rhs <= 0:
            return 0
    model.addConstr(
        lhs >= rhs,
        name=_name(prefix, "half_capacity_clique"),
    )
    return 1


def add_vehicle_half_capacity_cuts(
    model,
    *,
    customers: Iterable[Any],
    vehicles: Iterable[Any],
    volumes: Mapping[Any, float] | Sequence[float],
    capacities: Mapping[Any, float] | Sequence[float],
    alpha,
    y,
    active: Mapping[Any, float] | Sequence[float] | None = None,
    prefix: str = "",
    tol: float = _CAPACITY_TOL,
) -> int:
    """Add one heterogeneous half-capacity clique row per vehicle.

    A vehicle can contain at most one customer whose demand exceeds half of
    that vehicle's own capacity.  Scaling by activation strengthens fractional
    ``y`` while preserving the binary feasible set.
    """
    customer_list = _active_customers(customers, active)
    tol = _finite_nonnegative(tol, label="tol")
    added = 0
    for vehicle in vehicles:
        capacity = _finite_nonnegative(
            _value(capacities, vehicle), label=f"capacities[{vehicle!r}]"
        )
        clique = [
            customer
            for customer in customer_list
            if _finite_nonnegative(
                _value(volumes, customer),
                label=f"volumes[{customer!r}]",
            )
            > 0.5 * capacity + tol
        ]
        if len(clique) <= 1:
            continue
        model.addConstr(
            gp.quicksum(alpha[customer, vehicle] for customer in clique)
            <= y[vehicle],
            name=_name(prefix, f"vehicle_half_capacity[{vehicle}]"),
        )
        added += 1
    return added


# ---------------------------------------------------------------------------
# Stage-1 route-aware aggregate-capacity / fractional-knapsack cuts
# ---------------------------------------------------------------------------


def compute_stage1_service_lower_bounds(
    *,
    customers: Iterable[Any],
    vehicles: Iterable[Any],
    capacities: Mapping[Any, float] | Sequence[float],
    volumes: Mapping[Any, float] | Sequence[float],
    routing_costs: Mapping[Any, Any] | Sequence[Any],
    depot_start: Any,
    active: Mapping[Any, float] | Sequence[float] | None = None,
) -> dict[Any, float | None]:
    """Return the cheapest capacity-compatible incoming cost for each customer.

    ``None`` means that no vehicle can carry the customer, so it is forced to
    be outsourced and has zero possible service saving.  Capacity comparisons
    use exact arithmetic over the represented binary64 inputs.  In particular,
    excluding a predecessor due to rounded ``d[i] + d[j]`` could otherwise
    make a lower bound infinitesimally too large.
    """
    customer_list = _active_customers(customers, active)
    vehicle_list = list(vehicles)
    demand = {
        customer: _finite_nonnegative(
            _value(volumes, customer), label=f"volumes[{customer!r}]"
        )
        for customer in customer_list
    }
    demand_exact = {
        customer: Fraction.from_float(value)
        for customer, value in demand.items()
    }
    capacity = {
        vehicle: _finite_nonnegative(
            _value(capacities, vehicle), label=f"capacities[{vehicle!r}]"
        )
        for vehicle in vehicle_list
    }
    capacity_exact = {
        vehicle: Fraction.from_float(value)
        for vehicle, value in capacity.items()
    }

    result: dict[Any, float | None] = {}
    for customer in customer_list:
        vehicle_bounds = []
        for vehicle in vehicle_list:
            if demand_exact[customer] > capacity_exact[vehicle]:
                continue
            try:
                vehicle_costs = routing_costs[vehicle]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    f"routing_costs is missing vehicle {vehicle!r}"
                ) from exc
            predecessors = [depot_start]
            predecessors.extend(
                predecessor
                for predecessor in customer_list
                if predecessor != customer
                and demand_exact[predecessor] + demand_exact[customer]
                <= capacity_exact[vehicle]
            )
            vehicle_bounds.append(min(
                _finite_nonnegative(
                    _arc_value(vehicle_costs, predecessor, customer),
                    label=(
                        f"routing_costs[{vehicle!r}]"
                        f"[{predecessor!r},{customer!r}]"
                    ),
                )
                for predecessor in predecessors
            ))
        result[customer] = min(vehicle_bounds) if vehicle_bounds else None
    return result


def compute_stage1_return_lower_bound(
    *,
    customers: Iterable[Any],
    vehicles: Iterable[Any],
    capacities: Mapping[Any, float] | Sequence[float],
    volumes: Mapping[Any, float] | Sequence[float],
    routing_costs: Mapping[Any, Any] | Sequence[Any],
    depot_end: Any,
    active: Mapping[Any, float] | Sequence[float] | None = None,
) -> float | None:
    """Return the cheapest feasible last-customer-to-end-depot arc.

    ``None`` means that no active customer can be served by any vehicle.  The
    all-outsourced cost already gives the exact capacity-independent bound in
    that case.
    """
    customer_list = _active_customers(customers, active)
    vehicle_list = list(vehicles)
    demand_exact = {
        customer: Fraction.from_float(
            _finite_nonnegative(
                _value(volumes, customer),
                label=f"volumes[{customer!r}]",
            )
        )
        for customer in customer_list
    }
    capacity_exact = {
        vehicle: Fraction.from_float(
            _finite_nonnegative(
                _value(capacities, vehicle),
                label=f"capacities[{vehicle!r}]",
            )
        )
        for vehicle in vehicle_list
    }

    candidates = []
    for vehicle in vehicle_list:
        try:
            vehicle_costs = routing_costs[vehicle]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"routing_costs is missing vehicle {vehicle!r}"
            ) from exc
        for customer in customer_list:
            if demand_exact[customer] > capacity_exact[vehicle]:
                continue
            candidates.append(
                _finite_nonnegative(
                    _arc_value(vehicle_costs, customer, depot_end),
                    label=(
                        f"routing_costs[{vehicle!r}]"
                        f"[{customer!r},{depot_end!r}]"
                    ),
                )
            )
    return min(candidates) if candidates else None


def route_saving_lambda_breakpoints(
    *,
    customers: Iterable[Any],
    volumes: Mapping[Any, float] | Sequence[float],
    outsource_costs: Mapping[Any, float] | Sequence[float],
    service_lower_bounds: Mapping[Any, float | None] | Sequence[float | None],
    active: Mapping[Any, float] | Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Return ``0`` and distinct positive ``(c[j]-ell[j])_+ / d[j]`` points.

    A zero-demand customer has no positive finite breakpoint.  The zero point
    is retained because it gives the nontrivial unconditional lower bound
    ``sum(min(c[j], ell[j]))``.
    """
    ratios = [0.0]
    for customer in _active_customers(customers, active):
        demand = _finite_nonnegative(
            _value(volumes, customer), label=f"volumes[{customer!r}]"
        )
        cost = _finite_nonnegative(
            _value(outsource_costs, customer),
            label=f"outsource_costs[{customer!r}]",
        )
        try:
            raw_service_lb = service_lower_bounds[customer]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"service_lower_bounds is missing customer {customer!r}"
            ) from exc
        if raw_service_lb is None:
            saving_exact = Fraction(0)
        else:
            service_lb = _finite_nonnegative(
                raw_service_lb,
                label=f"service_lower_bounds[{customer!r}]",
            )
            saving_exact = max(
                Fraction(0),
                Fraction.from_float(cost) - Fraction.from_float(service_lb),
            )
        if demand == 0.0 or saving_exact == 0:
            continue
        ratio_exact = saving_exact / Fraction.from_float(demand)
        try:
            ratio = float(ratio_exact)
        except OverflowError as exc:
            raise ValueError(
                f"route saving / volume for customer {customer!r} "
                "exceeds finite binary64"
            ) from exc
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError(
                f"route saving / volumes[{customer!r}] "
                f"must be finite and positive, got {ratio!r}"
            )
        ratios.append(ratio)

    # Distinct represented densities; every nonnegative lambda yields a valid
    # row, so nearest binary64 conversion cannot strengthen the inequality.
    return tuple(sorted(set(ratios)))


def compute_stage1_capcut_rows(
    *,
    customers: Iterable[Any],
    volumes: Mapping[Any, float] | Sequence[float],
    outsource_costs: Mapping[Any, float] | Sequence[float],
    service_lower_bounds: Mapping[Any, float | None] | Sequence[float | None],
    return_lower_bound: float | None,
    active: Mapping[Any, float] | Sequence[float] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Compute ``(lambda, intercept)`` rows for the Stage-1 CapCut.

    Each returned pair represents

    ``eta >= intercept - lambda * sum(Q[v] * z[v])``

    with ``intercept = min(C0, C0 - sum_j max(0,
    r[j]-lambda*d[j]) + rho)`` over active customers,
    ``r[j]=(c[j]-ell[j])_+`` and ``rho`` the cheapest feasible route-return
    arc.  A ``None`` service bound marks a customer that no vehicle can carry;
    its saving is zero.  ``return_lower_bound=None`` means no customer can be
    served, so the unstrengthened intercept is already ``C0``.  Before the
    documented absolute-zero cleanup, these rows exactly represent the
    aggregate fractional-knapsack lower bound plus one mandatory return arc.
    Rows whose intercept is at most ``1e-6`` are dominated by the model's
    ``eta >= 0`` bound and are omitted.
    """
    customer_list = _active_customers(customers, active)
    if return_lower_bound is None:
        return_exact = Fraction(0)
    else:
        return_value = _finite_nonnegative(
            return_lower_bound, label="return_lower_bound"
        )
        return_exact = (
            Fraction(0)
            if return_value <= STATIC_CUT_ZERO_TOL
            else Fraction.from_float(return_value)
        )

    # Validate once and retain normalized values so that both breakpoint and
    # intercept calculations use exactly the same active customer set.
    demand_by_customer = {}
    cost_by_customer = {}
    saving_exact_by_customer = {}
    for customer in customer_list:
        demand_by_customer[customer] = _finite_nonnegative(
            _value(volumes, customer), label=f"volumes[{customer!r}]"
        )
        cost_by_customer[customer] = _finite_nonnegative(
            _value(outsource_costs, customer),
            label=f"outsource_costs[{customer!r}]",
        )
        try:
            raw_service_lb = service_lower_bounds[customer]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"service_lower_bounds is missing customer {customer!r}"
            ) from exc
        if raw_service_lb is None:
            saving_exact_by_customer[customer] = Fraction(0)
        else:
            service_lb = _finite_nonnegative(
                raw_service_lb,
                label=f"service_lower_bounds[{customer!r}]",
            )
            saving_exact_by_customer[customer] = max(
                Fraction(0),
                Fraction.from_float(cost_by_customer[customer])
                - Fraction.from_float(service_lb),
            )

    lambdas = route_saving_lambda_breakpoints(
        customers=customer_list,
        volumes=demand_by_customer,
        outsource_costs=cost_by_customer,
        service_lower_bounds=service_lower_bounds,
        active=None,
    )
    all_outsource_exact = sum(
        (
            Fraction.from_float(cost_by_customer[customer])
            for customer in customer_list
        ),
        Fraction(0),
    )
    rows = []
    for lambda_value in lambdas:
        lambda_exact = Fraction.from_float(lambda_value)
        base_intercept_exact = all_outsource_exact - sum(
            (
                max(
                    Fraction(0),
                    saving_exact_by_customer[customer]
                    - lambda_exact
                    * Fraction.from_float(demand_by_customer[customer]),
                )
                for customer in customer_list
            ),
            Fraction(0),
        )
        intercept_exact = min(
            all_outsource_exact,
            base_intercept_exact + return_exact,
        )
        intercept = _exact_fraction_floor(
            intercept_exact, label="CapCut intercept"
        )
        # An intercept inside the static zero band is dominated by eta >= 0.
        if intercept > STATIC_CUT_ZERO_TOL:
            rows.append((lambda_value, intercept))
    return tuple(rows)


def add_stage1_capacity_cuts(
    model,
    *,
    eta,
    z,
    vehicles: Iterable[Any],
    customers: Iterable[Any],
    capacities: Mapping[Any, float] | Sequence[float],
    volumes: Mapping[Any, float] | Sequence[float],
    outsource_costs: Mapping[Any, float] | Sequence[float],
    routing_costs: Mapping[Any, Any] | Sequence[Any],
    depot_start: Any,
    depot_end: Any,
    active: Mapping[Any, float] | Sequence[float] | None = None,
    period: Any | None = None,
    prefix: str = "s1",
    enabled: bool = True,
) -> int:
    """Add the route-aware fractional-knapsack CapCut envelope.

    ``eta`` is for one scenario-period node.  With ``period=None``, ``z[v]``
    must already refer to the matching-period slice; otherwise uses
    ``z[v, period]``.  No scenario probabilities or operating frequencies.
    Unique ``prefix`` required when adding several nodes to one model.

    Penalty coefficients in the closed absolute ``1e-6`` zero band are
    omitted only after subtracting their binary upper-bound contribution from
    the intercept.  Returns the number of nontrivial linear rows added.
    """
    if not enabled:
        return 0

    vehicle_list = list(vehicles)
    capacity_by_vehicle = {
        vehicle: _finite_nonnegative(
            _value(capacities, vehicle), label=f"capacities[{vehicle!r}]"
        )
        for vehicle in vehicle_list
    }
    service_lower_bounds = compute_stage1_service_lower_bounds(
        customers=customers,
        vehicles=vehicle_list,
        capacities=capacity_by_vehicle,
        volumes=volumes,
        routing_costs=routing_costs,
        depot_start=depot_start,
        active=active,
    )
    return_lower_bound = compute_stage1_return_lower_bound(
        customers=customers,
        vehicles=vehicle_list,
        capacities=capacity_by_vehicle,
        volumes=volumes,
        routing_costs=routing_costs,
        depot_end=depot_end,
        active=active,
    )
    rows = compute_stage1_capcut_rows(
        customers=customers,
        volumes=volumes,
        outsource_costs=outsource_costs,
        service_lower_bounds=service_lower_bounds,
        return_lower_bound=return_lower_bound,
        active=active,
    )

    def z_var(vehicle):
        return z[vehicle] if period is None else z[vehicle, period]

    added = 0
    for row_index, (lambda_value, intercept) in enumerate(rows):
        # The penalty is subtracted from the lower-bound RHS, so every
        # lambda*Q coefficient is rounded upward.  A nearest product rounded
        # downward would make the cut infinitesimally too strong at z=1.
        penalties = {
            vehicle: _exact_nonnegative_product_ceiling(
                lambda_value,
                capacity_by_vehicle[vehicle],
                label=f"CapCut lambda*capacity[{vehicle!r}]",
            )
            for vehicle in vehicle_list
        }
        dropped_penalties = [
            penalty
            for penalty in penalties.values()
            if penalty <= STATIC_CUT_ZERO_TOL
        ]
        adjusted_exact, adjusted_intercept = _exact_difference_floor(
            intercept,
            dropped_penalties,
            label="CapCut dropped penalty",
        )
        if adjusted_exact <= Fraction.from_float(STATIC_CUT_ZERO_TOL):
            continue
        total_capacity_penalty = gp.quicksum(
            penalty * z_var(vehicle)
            for vehicle, penalty in penalties.items()
            if penalty > STATIC_CUT_ZERO_TOL
        )
        model.addConstr(
            eta >= adjusted_intercept - total_capacity_penalty,
            name=_name(prefix, f"fractional_cap[{row_index}]"),
        )
        added += 1
    return added


# ---------------------------------------------------------------------------
# Stage-3 incoming-arc route-cost lower bound
# ---------------------------------------------------------------------------


def compute_min_incoming_costs(
    *,
    customers: Iterable[Any],
    routing_costs: Any,
    allowed_arcs: Iterable[tuple[Any, Any]] | None = None,
    allowed_predecessors: Mapping[Any, Iterable[Any]] | None = None,
    active: Mapping[Any, float] | Sequence[float] | None = None,
) -> dict[Any, float]:
    """Return the cheapest allowed non-self incoming cost for each active customer.

    Exactly one of ``allowed_arcs`` and ``allowed_predecessors`` must be given.
    ``routing_costs`` may be a tuple-keyed mapping, a NumPy-like matrix, or a
    nested mapping/sequence.  Every inspected cost must be finite and
    nonnegative.  Self-loops are always ignored, even if supplied by the caller.

    Raises ``ValueError`` if an active customer has no allowed non-self incoming
    arc; silently assigning a zero coefficient in that case would make a model
    construction error look like a valid RouteCut.
    """
    if (allowed_arcs is None) == (allowed_predecessors is None):
        raise ValueError(
            "provide exactly one of allowed_arcs or allowed_predecessors"
        )

    active_customers = _active_customers(customers, active)
    active_set = set(active_customers)
    best: dict[Any, float] = {}

    if allowed_arcs is not None:
        for arc in allowed_arcs:
            try:
                tail, head = arc
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"allowed arc must be a (tail, head) pair, got {arc!r}"
                ) from exc
            if head not in active_set or tail == head:
                continue
            cost = _finite_nonnegative(
                _arc_value(routing_costs, tail, head),
                label=f"routing_costs[{tail!r},{head!r}]",
            )
            if head not in best or cost < best[head]:
                best[head] = cost
    else:
        assert allowed_predecessors is not None
        for head in active_customers:
            try:
                predecessors = allowed_predecessors[head]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    f"active customer {head!r} has no allowed predecessors"
                ) from exc
            for tail in predecessors:
                if tail == head:
                    continue
                cost = _finite_nonnegative(
                    _arc_value(routing_costs, tail, head),
                    label=f"routing_costs[{tail!r},{head!r}]",
                )
                if head not in best or cost < best[head]:
                    best[head] = cost

    missing = [customer for customer in active_customers if customer not in best]
    if missing:
        raise ValueError(
            "active customers have no allowed non-self incoming arc: "
            + ", ".join(repr(customer) for customer in missing)
        )
    return best


def add_route_cost_lower_bound(
    model,
    *,
    theta,
    alpha,
    customers: Iterable[Any],
    active: Mapping[Any, float] | Sequence[float] | None = None,
    vehicle: Any | None = None,
    incoming_costs: Mapping[Any, float] | Sequence[float] | None = None,
    routing_costs: Any | None = None,
    allowed_arcs: Iterable[tuple[Any, Any]] | None = None,
    allowed_predecessors: Mapping[Any, Iterable[Any]] | None = None,
    prefix: str = "s3",
    enabled: bool = True,
    tol: float = _CAPACITY_TOL,
) -> int:
    """Add ``theta >= sum_j ell[j] * alpha[j, vehicle]`` for active customers.

    Pass either precomputed ``incoming_costs`` or ``routing_costs`` together
    with exactly one allowed-arc representation.  If ``vehicle`` is ``None``,
    ``alpha`` is treated as a one-dimensional ``alpha[j]`` container; otherwise
    ``alpha[j, vehicle]`` is used.

    Only the minimum incoming arc is charged; no return/depot term.  Returns
    one when a nontrivial row is added and zero when disabled or when all
    valid coefficients lie in the absolute zero band.
    """
    if not enabled:
        return 0
    if not math.isfinite(float(tol)) or tol < 0.0:
        raise ValueError(f"tol must be finite and nonnegative, got {tol!r}")

    customer_list = list(customers)
    active_customers = _active_customers(customer_list, active)
    if incoming_costs is None:
        if routing_costs is None:
            raise ValueError(
                "routing_costs is required when incoming_costs is not supplied"
            )
        coefficients = compute_min_incoming_costs(
            customers=customer_list,
            routing_costs=routing_costs,
            allowed_arcs=allowed_arcs,
            allowed_predecessors=allowed_predecessors,
            active=active,
        )
    else:
        if (
            routing_costs is not None
            or allowed_arcs is not None
            or allowed_predecessors is not None
        ):
            raise ValueError(
                "pass incoming_costs or routing_costs/allowed arcs, not both"
            )
        coefficients = {}
        for customer in active_customers:
            try:
                raw_cost = incoming_costs[customer]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    f"incoming_costs is missing active customer {customer!r}"
                ) from exc
            coefficients[customer] = _finite_nonnegative(
                raw_cost, label=f"incoming_costs[{customer!r}]"
            )

    zero_tol = max(float(tol), STATIC_CUT_ZERO_TOL)
    positive_customers = [
        customer
        for customer in active_customers
        if coefficients[customer] > zero_tol
    ]
    if not positive_customers:
        return 0

    def alpha_var(customer):
        return alpha[customer] if vehicle is None else alpha[customer, vehicle]

    vehicle_suffix = "" if vehicle is None else f"[{vehicle}]"
    model.addConstr(
        theta
        >= gp.quicksum(
            coefficients[customer] * alpha_var(customer)
            for customer in positive_customers
        ),
        name=_name(prefix, f"route_cost_lb{vehicle_suffix}"),
    )
    return 1


def add_stage2_route_cost_lower_bounds(
    model,
    *,
    theta,
    alpha,
    vehicles: Iterable[Any],
    successors: Iterable[Any],
    customers: Iterable[Any],
    active: Mapping[Any, float] | Sequence[float] | None,
    routing_costs: Mapping[Any, Any] | Sequence[Any],
    allowed_predecessors: Mapping[Any, Iterable[Any]] | None = None,
    volumes: Mapping[Any, float] | Sequence[float] | None = None,
    capacities: Mapping[Any, float] | Sequence[float] | None = None,
    depot_start: Any | None = None,
    prefix: str = "s2",
    enabled: bool = True,
    tol: float = _CAPACITY_TOL,
) -> int:
    """Install one shared RouteCut for every vehicle/successor pair.

    A Stage-2 node must have exactly one Stage-3 successor per vehicle, in the
    same order.  When volumes, capacities and the start depot are supplied,
    each vehicle excludes customer predecessors whose demand pair cannot fit;
    otherwise the caller must supply a shared predecessor mapping.
    """
    if not enabled:
        return 0

    vehicle_list = list(vehicles)
    successor_list = list(successors)
    customer_list = list(customers)
    if len(successor_list) != len(vehicle_list):
        raise ValueError(
            "Stage-2 successors must contain exactly one entry per vehicle: "
            f"got {len(successor_list)} successors for {len(vehicle_list)} vehicles"
        )
    capacity_aware = (
        volumes is not None or capacities is not None or depot_start is not None
    )
    if capacity_aware and (
        volumes is None or capacities is None or depot_start is None
    ):
        raise ValueError(
            "volumes, capacities and depot_start must be supplied together"
        )
    if not capacity_aware and allowed_predecessors is None:
        raise ValueError(
            "allowed_predecessors is required without capacity-aware inputs"
        )
    active_customers = _active_customers(customer_list, active)

    added = 0
    for vehicle, successor in zip(vehicle_list, successor_list):
        vehicle_predecessors = allowed_predecessors
        if capacity_aware:
            assert volumes is not None and capacities is not None
            cap = Fraction.from_float(
                _finite_nonnegative(
                    _value(capacities, vehicle),
                    label=f"capacities[{vehicle!r}]",
                )
            )
            demand = {
                customer: Fraction.from_float(
                    _finite_nonnegative(
                        _value(volumes, customer),
                        label=f"volumes[{customer!r}]",
                    )
                )
                for customer in active_customers
            }
            vehicle_predecessors = {
                head: [depot_start]
                + [
                    tail
                    for tail in active_customers
                    if tail != head and demand[tail] + demand[head] <= cap
                ]
                for head in active_customers
            }
        added += add_route_cost_lower_bound(
            model,
            theta=theta[successor],
            alpha=alpha,
            customers=customer_list,
            active=active,
            vehicle=vehicle,
            routing_costs=routing_costs[vehicle],
            allowed_predecessors=vehicle_predecessors,
            prefix=prefix,
            tol=tol,
        )
    return added
