"""Improve a physical policy using elementary routes from one seed call.

The restricted integer master supplies feasible policies only. Its objective
bound is never a lower bound on the original routing problem.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math
from numbers import Integral
import re
import time

import gurobipy as gp
from gurobipy import GRB

from core.solver_bounds import minimization_bounds_inverted, minimization_gap_percent
from models.stage2_symmetry import canonicalize_assignment_rows
from solvers.forward_policy_certification import (
    InvalidForwardPolicy, certify_stage1_forward_policy,
    certify_stage2_forward_policy, certify_stage3_forward_policy,
)
from solvers.forward_ub import ForwardUBAccumulator, round_fraction_up
from s2backward.routeopt.root_lp import _physical_groups


_ARC = re.compile(r"^x\[(-?\d+),(-?\d+)\]$")


def _certify_node(pd, tree, index, fleet, policy):
    node = tree[2][index]
    assignment, outsourcing = certify_stage2_forward_policy(pd, node, fleet, policy[2][index])
    routes, total = {}, ForwardUBAccumulator(outsourcing)
    if {int(tree[3][s].info) for s in node.successor} != set(pd.V):
        raise ValueError("Stage-3 successors must contain every vehicle exactly once")
    if len(node.successor) != len(pd.V):
        raise ValueError("duplicate Stage-3 vehicle")
    for third in node.successor:
        route_node = tree[3][third]
        if route_node.multi_coeff != node.multi_coeff:
            raise ValueError("Stage-2 and Stage-3 operating weights differ")
        routes[third], cost = certify_stage3_forward_policy(
            pd, route_node, assignment, policy[3][third])
        total.add_weighted(1., cost)
    return assignment, routes, total.upper_bound()


def certify_complete_policy(pd, tree, policy):
    """Normalize and independently certify every physical decision and cost."""
    normalized = {1: {}, 2: {}, 3: {}}
    fleet, purchase = certify_stage1_forward_policy(pd, policy[1][0])
    normalized[1][0] = fleet
    total = ForwardUBAccumulator(purchase)
    for index in tree[1][0].successor:
        assignment, routes, value = _certify_node(pd, tree, index, fleet, policy)
        normalized[2][index] = assignment
        normalized[3].update(routes)
        total.add_weighted(tree[2][index].multi_coeff, value)
    return normalized, total.upper_bound()


def merge_same_fleet_policy(pd, tree, candidate_policy, incumbent_policy):
    """Combine independent physical recourse blocks at identical raw fleets.

    The candidate's entire investment trajectory is retained. Only a complete
    Stage-2 assignment plus all its routes may move between policies; neither
    theta estimates nor solver objective values take part in the comparison.
    A nearly binary original fleet is certified normally, but is not eligible
    for reuse through this exact-identity gate.
    """
    merged, ub = certify_complete_policy(pd, tree, candidate_policy)
    reused = []
    if incumbent_policy is not None:
        incumbent, _ = certify_complete_policy(pd, tree, incumbent_policy)
        candidate_fleet = candidate_policy[1][0]
        incumbent_fleet = incumbent_policy[1][0]
        for index in tree[1][0].successor:
            node = tree[2][index]
            names = [f"z[{v},{node.info[1]}]" for v in pd.V]
            if not all(candidate_fleet[name] in (0., 1.)
                       and incumbent_fleet[name] in (0., 1.)
                       and float(candidate_fleet[name]).hex() == float(incumbent_fleet[name]).hex()
                       for name in names):
                continue
            _, _, candidate_cost = _certify_node(pd, tree, index, merged[1][0], merged)
            assignment, routes, incumbent_cost = _certify_node(
                pd, tree, index, merged[1][0], incumbent)
            if incumbent_cost < candidate_cost:
                merged[2][index] = assignment
                merged[3].update(routes)
                reused.append(int(index))
        merged, ub = certify_complete_policy(pd, tree, merged)
    return dict(policy=merged, ub=ub, certified=True, reused_nodes=reused)


def closed_policy_gap(pd, tree, inner_policy, incumbent_policy, *, certified_lb, tolerance):
    """Return a complete feasible policy only when its global interval closes.

    The caller supplies an existing certified Stage-1/global lower bound. No
    lower bound is inferred from a gap or a restricted-master objective.
    """
    if inner_policy is None or certified_lb is None:
        return None
    lower, tolerance = float(certified_lb), float(tolerance)
    if not math.isfinite(lower) or lower <= 0:
        return None
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid global relative gap tolerance")
    _, raw_upper = certify_complete_policy(pd, tree, inner_policy["x_star"])
    reported = float(inner_policy["ub"])
    if not math.isfinite(reported) or abs(reported - raw_upper) > 1e-6:
        raise RuntimeError("complete inner policy cost does not match its certificate")
    merged = merge_same_fleet_policy(pd, tree, inner_policy["x_star"], incumbent_policy)
    upper = merged["ub"]
    if minimization_bounds_inverted(lower, upper):
        raise RuntimeError("global LB exceeds the complete feasible policy UB")
    gap = minimization_gap_percent(lower, upper)
    if gap >= tolerance * 100:
        return None
    return dict(x_star=merged["policy"], ub=upper, certified=True,
                certified_global_lb=lower, gap_percent=gap, reused_nodes=merged["reused_nodes"])


def _path(route, start, end):
    if not route:
        return ()
    following = {int(match[1]): int(match[2]) for name, value in route.items()
                 if value == 1 and (match := _ARC.fullmatch(name))}
    sequence, current = [], start
    while current != end:
        current = following[current]
        if current != end:
            sequence.append(current)
    return tuple(sequence)


class RoutePolicyImprover:
    """One seed-call owner; one actual-fleet MIP per physical scenario."""

    def __init__(self, pd, tree, initial_policy):
        self.pd, self.tree = pd, tree
        self.policy, self.initial_ub = certify_complete_policy(pd, tree, deepcopy(initial_policy))
        self._fleet = self.policy[1][0]
        self._group = None
        self._changed = False
        self.stats = dict(groups=0, solves=0, columns=0, improved_nodes=0,
                          skipped_deadline=0, rejected_incumbents=0, seconds=0., rows=[])

    def _domain(self, node):
        return (tuple(node.active[j] for j in self.pd.J),
                tuple(float(node.volume[j]).hex() for j in self.pd.J),
                tuple(float(node.c_out[j]).hex() for j in self.pd.J))

    def start_group(self, node):
        """Return a consumer for already-checked original-customer routes."""
        jobs, demands, outsourcing, groups = _physical_groups(
            self.pd, node, {v: 1 for v in self.pd.V})
        current = dict(node=node, domain=self._domain(node), jobs=jobs,
                       demands=dict(zip(jobs, demands)), outsourcing=outsourcing,
                       groups=groups, columns={})
        self._group = current
        self.stats["groups"] += 1
        lookup = {tuple(g["vehicles"]): k for k, g in enumerate(groups)}

        def consume(vehicles, route, exact_cost):
            k = lookup[tuple(vehicles)]
            route = tuple(route)
            if (not route or any(isinstance(j, bool) or not isinstance(j, Integral)
                                 or j not in current["demands"] for j in route)
                    or len(set(route)) != len(route)):
                raise ValueError("physical route must visit distinct active customers")
            if sum((current["demands"][j] for j in route), Fraction()) > groups[k]["capacity"]:
                raise ValueError("physical route exceeds original capacity")
            if not isinstance(exact_cost, Fraction) or exact_cost < 0:
                raise ValueError("route consumer requires a nonnegative exact cost")
            key = k, frozenset(route)
            previous = current["columns"].get(key)
            if previous is None or exact_cost < previous[1]:
                current["columns"][key] = route, exact_cost

        current["consume"] = consume
        return consume

    def improve_group(self, members, deadline):
        """Improve the largest purchased fleet present in this physical group.

        Availability is componentwise nondecreasing along the certified
        investment trajectory. Thus the latest matching period has the largest
        actual fleet; every period with those same bits shares one solve.
        """
        started = time.monotonic()
        current, self._group = self._group, None
        if current is None:
            raise ValueError("start_group must precede improve_group")
        if not math.isfinite(float(deadline)):
            raise ValueError("deadline must be finite monotonic time")
        members = tuple(members)
        for member in members:
            if self._domain(member) != current["domain"]:
                raise ValueError("route columns cannot cross physical scenario domains")
        if not members:
            return
        latest = max(members, key=lambda member: member.info[1])
        purchased = tuple(v for v in self.pd.V if self._fleet[f"z[{v},{latest.info[1]}]"] == 1)
        if not purchased:
            return
        targets = [member for member in members
                   if tuple(v for v in self.pd.V if self._fleet[f"z[{v},{member.info[1]}]"] == 1)
                   == purchased]
        full_fleet = len(purchased) == len(self.pd.V)
        try:
            if time.monotonic() >= deadline:
                self.stats["skipped_deadline"] += 1
                return
            vehicle_group = {v: g for g in current["groups"] for v in g["vehicles"]}
            start, end = self.pd.numAllnodes - 2, self.pd.numAllnodes - 1
            for node in targets:
                for third in node.successor:
                    vehicle = int(self.tree[3][third].info)
                    route = _path(self.policy[3][third], start, end)
                    if route:
                        path = (start,) + route + (end,)
                        cost = sum((Fraction(float(self.pd.c_routing[vehicle][a, b]))
                                    for a, b in zip(path, path[1:])), Fraction())
                        current["consume"](vehicle_group[vehicle]["vehicles"], route, cost)
            representative = targets[0]
            current["mip_seconds"] = 1.
            if not full_fleet:
                representative = min(targets, key=lambda node: (
                    _certify_node(self.pd, self.tree, node.index, self._fleet, self.policy)[2], node.index))
                groups = [dict(group, vehicles=[v for v in group["vehicles"] if v in purchased])
                          for group in current["groups"]]
                current = dict(current, groups=groups, mip_seconds=3., columns={
                    key: value for key, value in current["columns"].items()
                    if groups[key[0]]["vehicles"]})
            self.stats["columns"] += len(current["columns"])
            candidate, metadata = self._solve_group(current, representative, deadline)
            self.stats["rows"].append(metadata)
            if candidate is None:
                return
            source_by_vehicle = {int(self.tree[3][s].info): s for s in representative.successor}
            for node in targets:
                trial = {2: {node.index: candidate[0]}, 3: {
                    s: candidate[1][source_by_vehicle[int(self.tree[3][s].info)]]
                    for s in node.successor}}
                assignment, routes, value = _certify_node(
                    self.pd, self.tree, node.index, self._fleet, trial)
                _, _, previous = _certify_node(
                    self.pd, self.tree, node.index, self._fleet, self.policy)
                if value < previous:
                    self.policy[2][node.index] = assignment
                    self.policy[3].update(routes)
                    self.stats["improved_nodes"] += 1
                    self._changed = True
        finally:
            self.stats["seconds"] += time.monotonic() - started

    def _solve_group(self, current, node, deadline):
        from core.backend_telemetry import backend_call, backend_scope, record_backend_event
        model = gp.Model("restricted_physical_route_policy")
        metadata = dict(node=int(node.index), status=None, solutions=0, seconds=0.,
                        fleet=[int(self._fleet[f"z[{v},{node.info[1]}]"]) for v in self.pd.V],
                        requested_seconds=0.)
        try:
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
            model.Params.IntFeasTol = model.Params.FeasibilityTol = 1e-9
            model.Params.MIPGap = model.Params.MIPGapAbs = 0.
            items = list(current["columns"].items())
            route_vars = [model.addVar(vtype=GRB.BINARY, obj=round_fraction_up(cost))
                          for _, (_, cost) in items]
            out_vars = [model.addVar(vtype=GRB.BINARY, obj=float(cost))
                        for cost in current["outsourcing"]]
            for j, out in zip(current["jobs"], out_vars):
                model.addConstr(gp.quicksum(var for var, ((_, covered), _) in zip(route_vars, items)
                                           if j in covered) + out == 1)
            for k, group in enumerate(current["groups"]):
                model.addConstr(gp.quicksum(var for var, ((g, _), _) in zip(route_vars, items)
                                           if g == k) <= len(group["vehicles"]))
            # The certified current policy is present even if no new column helps.
            selected = set()
            for third in node.successor:
                vehicle = int(self.tree[3][third].info)
                assigned = frozenset(j for j in current["jobs"]
                                     if self.policy[2][node.index][f"alpha[{j},{vehicle}]"])
                if assigned:
                    k = next(k for k, group in enumerate(current["groups"])
                             if vehicle in group["vehicles"])
                    selected.add((k, assigned))
            for var, (key, _) in zip(route_vars, items):
                var.Start = int(key in selected)
            for var, j in zip(out_vars, current["jobs"]):
                var.Start = int(not any(j in covered for _, covered in selected))
            model.update()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stats["skipped_deadline"] += 1
                record_backend_event("gurobi", "skip", "restricted_master_deadline",
                                     phase="1.5", path="physical_seed", stage=2)
                return None, metadata
            requested = min(current.get("mip_seconds", 1.), remaining)
            metadata["requested_seconds"] = requested
            model.Params.TimeLimit = requested
            with backend_scope(phase="1.5", path="physical_seed", stage=2):
                with backend_call("gurobi", "route_restricted_master_mip", model=model):
                    model.optimize()
            self.stats["solves"] += 1
            metadata.update(status=int(model.Status), solutions=int(model.SolCount), seconds=float(model.Runtime))
            if not model.SolCount:
                return None, metadata
            try:
                chosen = []
                for var, item in zip(route_vars, items):
                    value = float(var.X)
                    bit = int(round(value))
                    if bit not in (0, 1) or abs(value - bit) > 1e-6:
                        raise InvalidForwardPolicy("nonbinary restricted route incumbent")
                    if bit:
                        chosen.append(item)
                return self._certify_columns(current, node, chosen), metadata
            except InvalidForwardPolicy:
                self.stats["rejected_incumbents"] += 1
                return None, metadata
        finally:
            model.dispose()

    def _certify_columns(self, current, node, chosen):
        pd = self.pd
        rows = {v: [0] * len(pd.J) for v in pd.V}
        sequences = {}
        for k, group in enumerate(current["groups"]):
            selected = [(covered, route) for (g, covered), (route, _cost) in chosen if g == k]
            if len(selected) > len(group["vehicles"]):
                raise InvalidForwardPolicy("restricted incumbent exceeds available fleet")
            for vehicle, (covered, route) in zip(group["vehicles"], selected):
                row = [int(j in covered) for j in pd.J]
                rows[vehicle] = row
                sequences[tuple(row)] = route
        available = [v for v in pd.V if self._fleet[f"z[{v},{node.info[1]}]"] == 1]
        ordered, active, _ = canonicalize_assignment_rows(
            pd, available, [rows[v] for v in available], [int(any(rows[v])) for v in available])
        by_vehicle = dict(zip(available, zip(ordered, active)))
        assignment, routes = {}, {}
        third_by_vehicle = {int(self.tree[3][s].info): s for s in node.successor}
        for vehicle in pd.V:
            row, used = by_vehicle.get(vehicle, (rows[vehicle], 0))
            assignment[f"y[{vehicle}]"] = float(used)
            assignment.update({f"alpha[{j},{vehicle}]": float(value) for j, value in zip(pd.J, row)})
            sequence = (pd.numAllnodes - 2,) + sequences[tuple(row)] + (pd.numAllnodes - 1,) if used else ()
            routes[third_by_vehicle[vehicle]] = {f"x[{a},{b}]": 1. for a, b in zip(sequence, sequence[1:])}
        assignment, routes, _ = _certify_node(
            pd, self.tree, node.index, self._fleet, {2: {node.index: assignment}, 3: routes})
        return assignment, routes

    def report_candidate(self):
        if not self._changed:
            return None
        policy, ub = certify_complete_policy(self.pd, self.tree, self.policy)
        if ub >= self.initial_ub:
            return None
        return dict(policy=policy, ub=ub, certified=True)
