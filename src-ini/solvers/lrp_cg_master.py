"""Incremental restricted route LP; its objective is never a physical lower bound.

The caller supplies a validated physical context, facility mask and audited
routes. This session retains its original columns across external pool pruning.
Gurobi model state (including its basis) stays alive until ``close``.
"""
from __future__ import annotations

from fractions import Fraction
import math
import time

import gurobipy as gp


class IncrementalRouteMaster:
    def __init__(self, ctx, routes, A):
        self.ctx = ctx
        self.A = tuple(A)
        if len(self.A) != ctx.m or any(v not in (0, 1) for v in self.A):
            raise ValueError("invalid facility mask")
        self._model = None
        self._closed = False
        self._routes = []
        self._variables = []
        self._positions = {}
        self._scope = None
        self._latest_keys = set()
        self._added = self._replaced = self._solves = 0
        try:
            self._model = gp.Model("lrp_incremental_node_route_master")
            model = self._model
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
            model.Params.Seed = 42
            model.Params.Method = 1
            self._cover = {j: model.addConstr(gp.LinExpr() == 1., name=f"cover[{j}]")
                           for j in range(ctx.n) if ctx.active[j]}
            self._fleet = {i: model.addConstr(gp.LinExpr() <= self.A[i], name=f"facility[{i}]")
                           for i in range(ctx.m)}
            self._outsource = {j: model.addVar(lb=0., obj=float(ctx.outsourcing[j]),
                column=gp.Column([1.], [row]), name=f"e[{j}]")
                for j, row in self._cover.items()}
            model.ModelSense = gp.GRB.MINIMIZE
            self.sync(routes)
        except BaseException:
            self.close()
            raise

    def _require_open(self):
        if self._closed:
            raise RuntimeError("incremental route master is closed")

    @staticmethod
    def _key(route):
        return route.facility_id, frozenset(route.customers_in_order)

    def sync(self, routes):
        """Append/reprice audited columns, retaining omitted session columns.

        Validation precedes mutation. The component does not replace the
        original-data route audit performed by the physical pool/adapter.
        """
        self._require_open()
        routes = tuple(routes)
        scope = self._scope
        for route in routes:
            if scope is None:
                scope = route.node_signature
            if route.node_signature != scope:
                raise ValueError("DOMAIN_MISMATCH: mixed route node signatures")
            customers = tuple(route.customers_in_order)
            if (route.facility_id not in self._fleet or not customers
                    or len(set(customers)) != len(customers)
                    or any(j not in self._cover for j in customers)):
                raise ValueError("route does not belong to the restricted master domain")
            if not math.isfinite(float(route.cost)) or route.cost < 0:
                raise ValueError("invalid route cost")
        self._scope = scope
        added = replaced = 0
        self._latest_keys = {self._key(route) for route in routes}
        for route in routes:
            key = self._key(route)
            position = self._positions.get(key)
            if position is not None:
                old = self._routes[position]
                if Fraction(*route.cost_exact) < Fraction(*old.cost_exact):
                    self._variables[position].Obj = float(route.cost)
                    self._routes[position] = route
                    replaced += 1
                continue
            rows = [self._cover[j] for j in route.customers_in_order]
            rows.append(self._fleet[route.facility_id])
            position = len(self._routes)
            variable = self._model.addVar(lb=0., obj=float(route.cost),
                column=gp.Column([1.] * len(rows), rows), name=f"x[{position}]")
            self._positions[key] = position
            self._variables.append(variable)
            self._routes.append(route)
            added += 1
        self._added += added
        self._replaced += replaced
        return dict(added=added, replaced=replaced, **self.diagnostics)

    @property
    def routes(self):
        """Immutable audited session columns, including globally pruned ones."""
        self._require_open()
        return tuple(self._routes)

    @property
    def diagnostics(self):
        return dict(columns=len(self._routes), retained=sum(
            key not in self._latest_keys for key in self._positions),
            added_total=self._added, replaced_total=self._replaced,
            lp_solves=self._solves, objective_is_physical_lower_bound=False)

    def solve(self, deadline, integer=False):
        self._require_open()
        if integer:
            raise ValueError("IncrementalRouteMaster supports LP only; audit integer policies separately")
        if deadline is None or not math.isfinite(float(deadline)):
            raise ValueError("restricted master requires a finite absolute deadline")
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            return None
        self._model.Params.TimeLimit = remaining
        self._model.optimize()
        self._solves += 1
        result = dict(status=int(self._model.Status), routes=tuple(self._routes),
                      master_diagnostics=self.diagnostics)
        if self._model.Status != gp.GRB.OPTIMAL:
            return result
        result.update(objective=float(self._model.ObjVal),
            x=tuple(float(v.X) for v in self._variables),
            e={j: float(v.X) for j, v in self._outsource.items()},
            lambda_vector=tuple(float(self._cover[j].Pi) if j in self._cover else 0.
                                for j in range(self.ctx.n)),
            beta=tuple(float(self._fleet[i].Pi) for i in range(self.ctx.m)))
        return result

    def close(self):
        if not self._closed:
            self._closed = True
            if self._model is not None:
                self._model.dispose()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
