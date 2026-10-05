"""Physical-facility adapter for the investment algorithm's static bounds.

Each selected customer has one entering and one leaving arc.  Minimizing each
entering arc separately, plus the cheapest return arc, gives a route bound;
the reversed construction gives another.  Pair-incompatible predecessors are
impossible by the *facility* capacity.  This proof needs neither symmetry nor
the triangle inequality, and zero-demand active customers remain included.

The existing fractional-knapsack CapCut then uses the cheapest of these
customer entering costs across eligible facilities.  It only aggregates
available handling capacity; it never exchanges physical facilities.
"""
from fractions import Fraction
import heapq
import math

from cuts.benders_cuts import add_unique_cut
from cuts.static_valid_inequalities import STATIC_CUT_ZERO_TOL, compute_stage1_capcut_rows
from models.stage_model_core import AffineCut, route_degree_bounds


def _floor(value):
    result = float(value)
    return math.nextafter(result, -math.inf) if Fraction.from_float(result) > value else result


def basic_route_cuts(ctx, facility):
    """Investment's always-explicit incoming-only RouteCut at a physical depot.

    This is part of the base S2 model, not the learned archive or the optional
    radial strengthening. Each selected customer pays its cheapest feasible
    incoming arc. There is deliberately no return or dispatch coefficient,
    matching the original RouteCut. Removing tiny positive terms only weakens
    this row because assignment variables are nonnegative.
    """
    degree = route_degree_bounds(ctx, facility)
    incoming = tuple(value if value > STATIC_CUT_ZERO_TOL else 0.
                     for value in degree.incoming)
    if not any(incoming):
        return []
    return [AffineCut(
        "route", ctx.route_key(facility), 0., (*incoming, 0.), "parent",
        {"source": "investment_basic_incoming_route_bound", "static": True},
    )]


def basic_node_cuts(ctx):
    """Investment's route-aware CapCut, mapped to physical A[i,k(t)].

    The shared original generator produces the aggregate fractional-knapsack
    rows. Incoming minima are taken across physically eligible facilities;
    no interchangeability is assumed. The cheapest feasible return arc is
    charged at most once. All capacity products are rounded downward in the
    affine RHS, so conversion to binary64 cannot strengthen a valid cut.
    """
    incoming_by_customer = {j: [] for j in range(ctx.n)}
    returns = []
    for i in range(ctx.m):
        degree = route_degree_bounds(ctx, i)
        for j in degree.eligible:
            incoming_by_customer[j].append(degree.incoming[j])
        if degree.eligible:
            returns.append(degree.return_cost)
    service_lb = {j: min(values) if values else None
                  for j, values in incoming_by_customer.items()}
    rows = compute_stage1_capcut_rows(
        customers=range(ctx.n), volumes=ctx.demand,
        outsource_costs=ctx.outsourcing, service_lower_bounds=service_lb,
        return_lower_bound=min(returns) if returns else None, active=ctx.active,
    )
    cuts = []
    for multiplier, intercept in rows:
        coefficients = tuple(_floor(-Fraction.from_float(multiplier)
                             * Fraction.from_float(float(capacity)))
                             for capacity in ctx.capacity)
        # Match add_stage1_capacity_cuts' absolute zero-band treatment: a
        # removed negative coefficient must be charged at its binary upper
        # bound in the intercept. Dropping it alone would strengthen the cut
        # (and very tiny terms can also be dropped by Gurobi itself).
        adjusted = Fraction.from_float(intercept) + sum(
            (Fraction.from_float(value) for value in coefficients
             if -value <= STATIC_CUT_ZERO_TOL), Fraction())
        if adjusted <= Fraction.from_float(STATIC_CUT_ZERO_TOL):
            continue
        cuts.append(AffineCut(
            "node", ctx.key, _floor(adjusted),
            tuple(value if -value > STATIC_CUT_ZERO_TOL else 0.
                  for value in coefficients), "availability_box",
            {"source": "investment_basic_route_aware_capcut", "static": True,
             "capacity_multiplier": multiplier},
        ))
    return cuts


def _residual_root_distances(ctx, facility, eligible, demands, cap, incoming, return_cost):
    """Exact shortest-path LOWER bounds in an auxiliary relaxation graph.

    Original physical arc costs are never replaced. For any actual tour, its
    cost minus one incoming minimum per visited vertex is a nonnegative tour
    in this auxiliary graph. Splitting it at j proves residual tour >= d(0,j)
    + d(j,0). Auxiliary paths may use extra customers; this only weakens the
    bound and does not authorize transit in the physical routing model.
    """
    vertices = [0]+[j+1 for j in eligible]
    shifts = {j+1: Fraction.from_float(incoming[j]) for j in eligible}
    shifts[0] = Fraction.from_float(return_cost)
    graph = {v: [] for v in vertices}
    reverse = {v: [] for v in vertices}
    for v in vertices:
        for w in vertices:
            if v == w or (v and w and demands[v-1]+demands[w-1] > cap):
                continue
            cost = Fraction.from_float(float(ctx.route_cost[facility,v,w]))-shifts[w]
            if cost < 0:
                raise AssertionError('Incoming minimum exceeds an allowed arc')
            graph[v].append((w,cost)); reverse[w].append((v,cost))
    def distances(edges):
        distance = {0: Fraction()}; queue = [(Fraction(),0)]
        while queue:
            value, v = heapq.heappop(queue)
            if value != distance[v]:
                continue
            for w, cost in edges[v]:
                candidate = value+cost
                if w not in distance or candidate < distance[w]:
                    distance[w] = candidate
                    heapq.heappush(queue,(candidate,w))
        if len(distance) != len(vertices):
            raise AssertionError('Every eligible customer has direct physical root arcs')
        return distance
    outgoing, incoming_dist = distances(graph), distances(reverse)
    return {j: outgoing[j+1]+incoming_dist[j+1] for j in eligible}


def seed_lrp_static_cuts(prob_data, scen_tree, archive):
    """Opt in to additional degree/radial rows in the ordinary cut archive.

    Base model CapCuts/RouteCuts are installed independently of this switch.
    Existing explicit-True archives keep their historical enhanced supports.
    """
    from models.stage_builder import _instance, _node_context
    instance = _instance(prob_data)
    counts = {2: 0, 3: 0}
    for node in scen_tree[2]:
        ctx = _node_context(instance, node, stage=2)
        demands = [Fraction.from_float(float(d)) for d in ctx.demand]
        incoming_by_customer = {j: [] for j in range(ctx.n)}
        returns = []
        for i, third_ind in enumerate(node.successor):
            cap = Fraction.from_float(float(ctx.capacity[i]))
            degree = route_degree_bounds(ctx, i)
            eligible = degree.eligible
            if not eligible:
                continue
            incoming = {j: degree.incoming[j] for j in eligible}
            outgoing = {j: degree.outgoing[j] for j in eligible}
            for j in eligible:
                incoming_by_customer[j].append(incoming[j])
            return_cost = degree.return_cost
            depart_cost = degree.depart_cost
            returns.append(return_cost)
            for customer_costs, root_cost in ((incoming, return_cost), (outgoing, depart_cost)):
                pi = {f'alpha[{i},{j}]': v for j, v in customer_costs.items() if v}
                if root_cost:
                    pi[f'u[{i}]'] = root_cost
                counts[3] += add_unique_cut(archive.setdefault(3, {}).setdefault(third_ind, []), pi, 0.)
            # Retain the whole incoming-degree bound AND the root connection
            # still required after paying those minimum arcs. Each j yields
            # a globally valid affine cut, also for asymmetric/nonmetric data.
            radial = _residual_root_distances(ctx, i, eligible, demands, cap, incoming, return_cost)
            base = {f'alpha[{i},{j}]': value for j, value in incoming.items() if value}
            if return_cost:
                base[f'u[{i}]'] = return_cost
            for j, distance in radial.items():
                if not distance:
                    continue
                pi = dict(base)
                pi[f'alpha[{i},{j}]'] = _floor(Fraction.from_float(incoming[j])+distance)
                counts[3] += add_unique_cut(archive[3][third_ind], pi, 0.)
        service_lb = {j: min(values) if values else None for j, values in incoming_by_customer.items()}
        rows = compute_stage1_capcut_rows(customers=range(ctx.n), volumes=ctx.demand,
            outsource_costs=ctx.outsourcing, service_lower_bounds=service_lb,
            return_lower_bound=min(returns) if returns else None, active=ctx.active)
        for multiplier, intercept in rows:
            # Negative capacity slopes round downward on the binary A box.
            pi = {f'A[{i},{ctx.interval}]': _floor(-Fraction.from_float(multiplier)
                  * Fraction.from_float(float(ctx.capacity[i]))) for i in range(ctx.m)}
            counts[2] += add_unique_cut(archive.setdefault(2, {}).setdefault(node.index, []), pi, intercept)
    return counts
