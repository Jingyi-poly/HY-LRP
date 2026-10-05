"""Audited physical routes and a restricted, global-A-variable policy MIP.

The model is a primal heuristic.  Its bound is deliberately confined to a
diagnostic dictionary: only an original-array audited complete policy is an UB.
No eta/theta, cut, LevelSet target, or global lower bound is produced here.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as F
import hashlib
import math
import time
from types import SimpleNamespace

import numpy as np

from core.solve_deadline import SolveDeadlineReached
from models.stage_builder import _instance, _node_context
from models.stage_model_core import (LinearMILP, audit_tour, facility_block,
                                     matrix_primal_check, audit_gurobi_matrix)
from solvers.forward_policy_certification import (
    certify_policy, certify_stage1_forward_policy, certify_stage2_forward_policy,
    certify_stage3_forward_policy,
)
from solvers.forward_ub import round_fraction_up
from solvers.lrp_physical_types import (
    AuditedRoute, PoolPolicyResult, global_policy_signature, node_signature,
)


def _remaining(deadline):
    if deadline is None:
        return math.inf
    if math.isnan(float(deadline)):
        raise ValueError("deadline must not be NaN")
    left = float(deadline) - time.monotonic()
    if left <= 0:
        raise SolveDeadlineReached("physical pool deadline reached")
    return left


def _route(ctx, signature, facility, customers, source, generation):
    if isinstance(facility, (bool, np.bool_)) or not isinstance(facility, (int, np.integer)):
        raise ValueError("facility must be its integer physical array position")
    i = int(facility)
    if not 0 <= i < ctx.m:
        raise ValueError("wrong physical facility")
    values = tuple(customers)
    if any(isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)) for j in values):
        raise ValueError("route customer positions must be integers")
    order = tuple(int(j) for j in values)
    if not order or len(set(order)) != len(order) or any(not 0 <= j < ctx.n for j in order):
        raise ValueError("pool routes must be nonempty, unique original customers")
    if type(generation) is not int or generation < 0:
        raise ValueError("generation_id must be a nonnegative integer")
    alpha = tuple(int(j in order) for j in range(ctx.n))
    ctx.check_route_state(i, alpha, 1)  # exact per-facility capacity, including zero demand
    path = (0,) + tuple(j + 1 for j in order) + (0,)
    arcs = tuple(zip(path, path[1:]))
    audit_tour(ctx, i, alpha, 1, arcs)
    cost = sum((F(float(ctx.route_cost[i, v, w])) for v, w in arcs), F())
    demand = sum((F(float(ctx.demand[j])) for j in order), F())
    payload = repr((signature, i, order, cost.numerator, cost.denominator,
                    demand.numerator, demand.denominator)).encode()
    return AuditedRoute(signature, i, order, round_fraction_up(cost),
                        round_fraction_up(demand), hashlib.sha256(payload).hexdigest(),
                        str(source), generation, (cost.numerator, cost.denominator),
                        (demand.numerator, demand.denominator))


def audit_route(prob_data, node, facility, customers_in_order, *, source="physical_policy",
                generation_id=0):
    """Recompute one original directed own-root tour; never infer a shortcut."""
    data = _instance(prob_data)
    ctx = _node_context(data, node, stage=2)
    return _route(ctx, node_signature(data, ctx), facility, customers_in_order,
                  source, generation_id)


def audit_node_policy(prob_data, tree, node, A_mask, policy):
    """Return (original node snippet, exact-upward unweighted cost, audit).

    A fixed node may have all facilities closed even if that plan is globally
    infeasible.  Thus only its availability vector is used, not a fake S1 plan.
    """
    data = _instance(prob_data)
    ctx = _node_context(data, node, stage=2)
    from models.stage_model_core import binary_vector, certify_node
    mask = binary_vector(A_mask, ctx.m, "availability")
    root = {f"A[{i},{ctx.interval}]": float(mask[i]) for i in range(ctx.m)}
    state, _ = certify_stage2_forward_policy(data, node, root, policy[2][node.index])
    result = {2: {node.index: state}, 3: {}}
    tours, facilities = {}, set()
    cost = sum((F(float(ctx.outsourcing[j])) * int(state[f"e[{j}]"])
                for j in range(ctx.n)), F())
    for rid in node.successor:
        third = tree[3][rid]
        i = int(third.info)
        if i in facilities or not 0 <= i < ctx.m:
            raise ValueError("duplicate/wrong physical facility node")
        facilities.add(i)
        route, _ = certify_stage3_forward_policy(data, third, state, policy[3][rid])
        result[3][rid] = route
        arcs = [tuple(map(int, name[2:-1].split(",")))[1:] for name in route]
        tours[i] = {"arcs": arcs}
        cost += sum((F(float(ctx.route_cost[i, v, w])) for v, w in arcs), F())
    if facilities != set(range(ctx.m)):
        raise ValueError("node policy must contain all physical facilities")
    report = certify_node(ctx, mask,
        [[state[f"alpha[{i},{j}]"] for j in range(ctx.n)] for i in range(ctx.m)],
        [state[f"e[{j}]"] for j in range(ctx.n)],
        [state[f"u[{i}]"] for i in range(ctx.m)], tours)
    report.update(node_signature=node_signature(data, ctx),
                  exact_cost=(cost.numerator, cost.denominator),
                  feasible_upper_bound=round_fraction_up(cost))
    return result, round_fraction_up(cost), report


def normalize_policy(prob_data, tree, policy):
    """Strip approximate theta/model fields and audit every original node."""
    data = _instance(prob_data)
    root, _ = certify_stage1_forward_policy(data, policy[1][0])
    result = {1: {0: root}, 2: {}, 3: {}}
    for q in tree[1][0].successor:
        node = tree[2][q]
        ctx = _node_context(data, node, stage=2)
        part, _, _ = audit_node_policy(data, tree, node,
            [root[f"A[{i},{ctx.interval}]"] for i in range(ctx.m)], policy)
        result[2].update(part[2]); result[3].update(part[3])
    audit = certify_policy(data, tree, result)
    return result, float(audit["feasible_upper_bound"]), audit


def all_outsourcing_policy(prob_data, tree, current_root):
    """Use the supplied feasible *full* facility schedule, never all-closed by fiat."""
    data = _instance(prob_data)
    root, _ = certify_stage1_forward_policy(data, current_root)
    policy = {1: {0: root}, 2: {}, 3: {}}
    for q in tree[1][0].successor:
        node = tree[2][q]
        ctx = _node_context(data, node, stage=2)
        state = {f"alpha[{i},{j}]": 0. for i in range(ctx.m) for j in range(ctx.n)}
        state.update({f"u[{i}]": 0. for i in range(ctx.m)})
        state.update({f"e[{j}]": float(ctx.active[j]) for j in range(ctx.n)})
        policy[2][q] = state
        policy[3].update({rid: {} for rid in node.successor})
    return normalize_policy(data, tree, policy)


def _key(route):
    return (route.node_signature, route.facility_id, frozenset(route.customers_in_order))


class PhysicalRoutePool:
    """Parent-owned pool; immutable route snapshots outlive column pruning."""
    def __init__(self, prob_data, tree, *, max_routes_per_node_facility=200):
        if type(max_routes_per_node_facility) is not int or max_routes_per_node_facility < 1:
            raise ValueError("route soft limit must be a positive integer")
        self.data = _instance(prob_data)
        self.tree = deepcopy(tree)
        self.signature = global_policy_signature(self.data)
        # Use the existing tree validator when a ProblemData wrapper is needed.
        from core.problem_data import ProblemData
        from core.scenario_tree import validate_operating_weights
        validate_operating_weights(ProblemData(self.data), self.tree)
        self.contexts = {q: _node_context(self.data, self.tree[2][q], stage=2)
                         for q in self.tree[1][0].successor}
        self.signatures = {q: node_signature(self.data, ctx) for q, ctx in self.contexts.items()}
        self.max_routes_per_node_facility = max_routes_per_node_facility
        self.version = 0
        self._routes = {}
        self._pins = {}
        self._snapshots = {}
        self.diagnostics = {"evicted": 0, "pinned_over_limit": 0}

    def _q(self, node):
        q = int(node) if isinstance(node, (int, np.integer)) else int(node.index)
        if q not in self.contexts:
            raise ValueError("unknown physical node")
        if not isinstance(node, (int, np.integer)):
            if node_signature(self.data, node) != self.signatures[q]:
                raise ValueError("DOMAIN_MISMATCH: route node")
        return q

    def routes(self, node=None, facility=None):
        signature = None if node is None else self.signatures[self._q(node)]
        return tuple(sorted((r for r in self._routes.values()
            if (signature is None or r.node_signature == signature)
            and (facility is None or r.facility_id == facility)), key=lambda r: r.audit_signature))

    @property
    def count(self):
        return len(self._routes)

    def _prune(self):
        pinned = set().union(*self._pins.values()) if self._pins else set()
        buckets = {}
        for key, route in self._routes.items():
            buckets.setdefault(key[:2], []).append((key, route))
        over = 0
        for rows in buckets.values():
            excess = max(0, len(rows) - self.max_routes_per_node_facility)
            deletable = sorted(((k, r) for k, r in rows if k not in pinned),
                               key=lambda kr: (kr[1].generation_id, kr[1].audit_signature))
            for key, _ in deletable[:excess]:
                del self._routes[key]; self.version += 1; self.diagnostics["evicted"] += 1
            over += max(0, len(rows) - min(excess, len(deletable)) - self.max_routes_per_node_facility)
        self.diagnostics["pinned_over_limit"] = over

    def add_route(self, node, facility, customers_in_order, *, source="physical_policy",
                  generation_id=0, pin=None):
        q = self._q(node)
        route = _route(self.contexts[q], self.signatures[q], facility, customers_in_order,
                       source, generation_id)
        return self._store_audited_route(route,pin=pin)

    def _store_audited_route(self,route,*,pin=None):
        key = _key(route)
        old = self._routes.get(key)
        if old is None or F(*route.cost_exact) < F(*old.cost_exact):
            self._routes[key] = route; self.version += 1
        if pin is not None:
            self._pins.setdefault(str(pin), set()).add(key)
        self._prune()
        return self._routes.get(key, route)

    def add_audited_route(self, node, route, *, pin=None, deadline=None):
        if deadline is not None and time.monotonic() >= deadline:
            return None
        if not isinstance(route, AuditedRoute):
            raise TypeError("expected an AuditedRoute")
        q = self._q(node)
        fresh = _route(self.contexts[q], self.signatures[q], route.facility_id,
                       route.customers_in_order, route.source, route.generation_id)
        if fresh != route:
            raise ValueError("DOMAIN_MISMATCH or altered route audit")
        if deadline is not None and time.monotonic() >= deadline:
            return None
        return self._store_audited_route(fresh,pin=pin)

    def _policy_routes(self, policy, source, generation):
        output = []
        for q, ctx in self.contexts.items():
            for rid in self.tree[2][q].successor:
                i = int(self.tree[3][rid].info)
                arcs = [tuple(map(int, name[2:-1].split(",")))[1:] for name in policy[3][rid]]
                if not arcs:
                    continue
                successor = dict(arcs); order = []; v = successor[0]
                while v:
                    order.append(v - 1); v = successor[v]
                output.append((q, _route(ctx, self.signatures[q], i, order, source, generation)))
        return output

    def add_policy(self, policy, *, source="forward", generation_id=0, pin=None):
        normalized, upper, audit = normalize_policy(self.data, self.tree, policy)
        audited = self._policy_routes(normalized, source, generation_id)
        # Protect all incoming keys before admitting/pruning any of them.
        transient = "__incoming_policy__"
        self._pins[transient] = {_key(route) for _, route in audited}
        try:
            for q, route in audited:
                self.add_audited_route(q, route)
            if pin is not None:
                self._pins[str(pin)] = set(self._pins[transient])
                self._snapshots[str(pin)] = (deepcopy(normalized), upper,
                                             tuple(route for _, route in audited))
        finally:
            self._pins.pop(transient, None)
            self._prune()
        return normalized, upper, audit

    def pinned_snapshot(self, name):
        """Return an independent policy copy and immutable original route records."""
        return deepcopy(self._snapshots[str(name)])

    def replace_route_pins(self, name, node_route_pairs):
        """Atomically replace one owner's pin set after reauditing every route."""
        verified = []
        for node, route in node_route_pairs:
            if not isinstance(route, AuditedRoute):
                raise TypeError('pins require audited physical routes')
            q = self._q(node)
            fresh = _route(self.contexts[q], self.signatures[q], route.facility_id,
                           route.customers_in_order, route.source, route.generation_id)
            if route != fresh:
                raise ValueError('DOMAIN_MISMATCH or altered pinned route audit')
            verified.append((q, fresh))
        transient = '__replacement_pins__'
        self._pins[transient] = {_key(route) for _, route in verified}
        try:
            for q, route in verified:
                self.add_audited_route(q, route)
            self._pins[str(name)] = set(self._pins[transient])
        finally:
            self._pins.pop(transient, None)
            self._prune()

    def add_singletons(self, *, generation_id=0, deadline=None):
        before = self.count
        # Fixed pins and unique new keys permit one final top-K prune. An old
        # singleton in an overflowing bucket could instead be evicted and then
        # reinserted later in this loop; retain the original order in that case.
        counts, existing_singletons = {}, set()
        for key, route in self._routes.items():
            bucket = key[:2]
            counts[bucket] = counts.get(bucket, 0) + 1
            if len(route.customers_in_order) == 1:
                existing_singletons.add(bucket)
        batch = not any(
            (self.signatures[q], i) in existing_singletons
            and counts.get((self.signatures[q], i), 0) + ctx.n > self.max_routes_per_node_facility
            for q, ctx in self.contexts.items() for i in range(ctx.m))
        audited_any = False
        try:
            for q, ctx in self.contexts.items():
                for i in range(ctx.m):
                    _remaining(deadline)
                    for j in range(ctx.n):
                        _remaining(deadline)
                        if ctx.active[j] and F(float(ctx.demand[j])) <= F(float(ctx.capacity[i])):
                            if not batch:
                                self.add_route(q, i, (j,), source="singleton",
                                               generation_id=generation_id)
                                continue
                            route = _route(ctx, self.signatures[q], i, (j,),
                                           "singleton", generation_id)
                            audited_any = True
                            key = _key(route)
                            previous = self._routes.get(key)
                            if previous is None or F(*route.cost_exact) < F(*previous.cost_exact):
                                self._routes[key] = route
                                self.version += 1
        finally:
            # Preserve the completed insertion prefix on deadline/audit error,
            # and do not mutate the pool if no route reached the audit boundary.
            if batch and audited_any:
                self._prune()
        return self.count - before


def build_pool_model(pool, *, deadline=None):
    """Pure canonical matrix, all original common A/o/h/b variables are free."""
    _remaining(deadline)
    M = LinearMILP()
    facility_block(M, pool.data)
    columns = {}
    for q, ctx in pool.contexts.items():
        _remaining(deadline)
        covered = [[] for _ in range(ctx.n)]
        p = float(pool.data.arrays["scenario_prob"][ctx.scenario])
        for i in range(ctx.m):
            _remaining(deadline)
            zvars = []
            for rindex, route in enumerate(pool.routes(q, i)):
                _remaining(deadline)
                col = M.var("z", (q, i, rindex), p * route.cost)
                columns[col] = route
                zvars.append((col, 1.))
                for j in route.customers_in_order:
                    covered[j].append((col, 1.))
            M.row(f"route_available_{q}_{i}", zvars + [(M.groups["A"][i, ctx.interval], -1.)], ub=0.)
        for j in range(ctx.n):
            _remaining(deadline)
            e = M.var("e", (q, j), p * float(ctx.outsourcing[j]), ub=float(ctx.active[j]))
            M.row(f"cover_{q}_{j}", covered[j] + [(e, 1.)],
                  lb=float(ctx.active[j]), ub=float(ctx.active[j]))
    for group, entries in M.groups.items():
        for key, col in entries.items():
            M.names[col] = f"{group}[{','.join(map(str, key))}]"
    M.validate()
    return M, columns


def _build_native(M, env, deadline):
    """Existing canonical coefficients/tolerances, with cooperative build expiry."""
    import gurobipy as gp
    _remaining(deadline)
    model = gp.Model("lrp_physical_restricted_policy", env=env)
    try:
        model.Params.OutputFlag = 0
        model.Params.FeasibilityTol = 1e-8
        model.Params.IntFeasTol = 1e-8
        model.Params.OptimalityTol = 1e-8
        variables = []
        for col, name in enumerate(M.names):
            if col % 64 == 0:
                _remaining(deadline)
            variables.append(model.addVar(lb=M.lower[col], ub=M.upper[col], obj=M.cost[col],
                vtype=gp.GRB.BINARY if M.integer[col] else gp.GRB.CONTINUOUS, name=name))
        model.ModelSense = gp.GRB.MINIMIZE
        for index, (source, name, sense, rhs) in enumerate(M.expanded_rows()):
            if index % 64 == 0:
                _remaining(deadline)
            row = M.rows[source]
            expr = gp.LinExpr(list(row.values()), [variables[j] for j in row])
            if sense == "=": model.addConstr(expr == rhs, name=name)
            elif sense == ">": model.addConstr(expr >= rhs, name=name)
            else: model.addConstr(expr <= rhs, name=name)
        model.update()
        _remaining(deadline)
        audit = audit_gurobi_matrix(M, model, variables)
        return model, variables, audit
    except BaseException:
        model.dispose()
        raise


def _extract_policy(pool, M, columns, x):
    matrix_primal_check(SimpleNamespace(linear=M), x)
    bits = np.rint(np.asarray(x)).astype(int)
    root = {M.names[c]: float(bits[c]) for g in ("A", "o", "h", "b")
            for c in M.groups[g].values()}
    policy, _, _ = all_outsourcing_policy(pool.data, pool.tree, root)
    for q, ctx in pool.contexts.items():
        state = policy[2][q]
        for j in range(ctx.n):
            state[f"e[{j}]"] = float(bits[M.groups["e"][q, j]])
    for col, route in columns.items():
        if not bits[col]:
            continue
        q = next(q for q, sig in pool.signatures.items() if sig == route.node_signature)
        i = route.facility_id; state = policy[2][q]
        state[f"u[{i}]"] = 1.
        for j in route.customers_in_order:
            state[f"alpha[{i},{j}]"] = 1.
        rid = next(r for r in pool.tree[2][q].successor if int(pool.tree[3][r].info) == i)
        path = (0,) + tuple(j + 1 for j in route.customers_in_order) + (0,)
        policy[3][rid] = {f"r[{i},{v},{w}]": 1. for v, w in zip(path, path[1:])}
    return normalize_policy(pool.data, pool.tree, policy)


def _start_vector(pool, M, columns, policy):
    start = np.zeros(len(M.names))
    for g in ("A", "o", "h", "b"):
        for c in M.groups[g].values():
            start[c] = policy[1][0][M.names[c]]
    # A cheaper order for an identical subset is safe, even in a nonmetric graph.
    needed = {_key(route) for _, route in pool._policy_routes(policy, "start", 0)}
    for col, route in columns.items():
        start[col] = float(_key(route) in needed)
    for (q, j), col in M.groups["e"].items():
        start[col] = policy[2][q][f"e[{j}]"]
    matrix_primal_check(SimpleNamespace(linear=M), start)
    return start


def solve_pool_policy(pool, current_root, *, incumbent_policy=None, deadline=None,
                      time_limit=15., mip_gap=1e-2, audit_reserve=5., seed=42):
    """Return only a complete audited UB; build/audit consume the same deadline."""
    started = time.monotonic()
    if seed != 42:
        raise ValueError("V1 physical pool uses the prescribed Seed=42")
    for value, name in ((time_limit, "time_limit"), (mip_gap, "mip_gap"),
                        (audit_reserve, "audit_reserve")):
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if deadline is not None and math.isnan(float(deadline)):
        raise ValueError("deadline must not be NaN")
    stop = min(started + float(time_limit), math.inf if deadline is None else float(deadline))
    audit_started = time.monotonic()
    best, upper, _ = all_outsourcing_policy(pool.data, pool.tree, current_root)
    if incumbent_policy is not None:
        candidate, candidate_upper, _ = pool.add_policy(incumbent_policy,
            source="global_incumbent", pin="global_incumbent")
        if candidate_upper < upper:
            best, upper = candidate, candidate_upper
    audit_seconds = time.monotonic() - audit_started
    audit_completed = time.monotonic()
    build_seconds = solve_seconds = 0.
    diagnostic = {"restricted_problem": True, "original_lb_eligible": False,
                  "global_signature": pool.signature, "deadline": stop,
                  "probability_applied_once": True, "threads": 1, "seed": 42}
    env = model = None

    def close_native():
        nonlocal model, env
        if model is not None:
            model.dispose(); model = None
        if env is not None:
            env.dispose(); env = None

    def result(status):
        close_native()  # include cleanup in wall time, including budget overrun
        selected = []
        for _, audited in pool._policy_routes(best, "audited_incumbent", 0):
            stored = pool._routes.get(_key(audited))
            selected.append(stored if stored is not None and
                            stored.audit_signature == audited.audit_signature else audited)
        snapshots = tuple(selected)
        diagnostic["selected_routes"] = [dict(
            node_signature=route.node_signature, facility_id=route.facility_id,
            customers_in_order=route.customers_in_order, source=route.source,
            generation_id=route.generation_id, audit_signature=route.audit_signature)
            for route in snapshots]
        pool._pins["global_incumbent"] = {_key(route) for route in snapshots}
        pool._snapshots["global_incumbent"] = (deepcopy(best), upper, snapshots)
        pool._prune()
        returned = time.monotonic()
        diagnostic.update(policy_audit_completed_monotonic=audit_completed,
                          returned_monotonic=returned,
                          deadline_overrun_seconds=max(0., returned - stop))
        return PoolPolicyResult(deepcopy(best), upper, status, pool.version,
            returned - started, build_seconds, solve_seconds, audit_seconds,
            dict(diagnostic), snapshots)

    if time.monotonic() >= stop - audit_reserve:
        return result("NO_BUDGET")
    if pool.count == 0:
        return result("EMPTY_POOL")
    build_started = time.monotonic()
    try:
        import gurobipy as gp
        build_stop = stop - audit_reserve
        M, columns = build_pool_model(pool, deadline=build_stop)
        _remaining(build_stop)
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0); env.start()
        model, variables, diagnostic["matrix_audit"] = _build_native(M, env, build_stop)
        start = _start_vector(pool, M, columns, best)
        for var, value in zip(variables, start):
            var.Start = float(value)
        model.Params.Threads = 1; model.Params.Seed = 42
        model.Params.MIPGap = float(mip_gap)
        model.Params.TimeLimit = _remaining(build_stop)
        build_seconds = time.monotonic() - build_started
        solve_started = time.monotonic()
        model.optimize()
        solve_seconds = time.monotonic() - solve_started
        status, sol_count = int(model.Status), int(model.SolCount)
        diagnostic.update(solver_status=status, solver_sol_count=sol_count)
        # This bound is diagnostic of the restricted route pool and nothing else.
        try:
            restricted_bound = float(model.ObjBoundC)
            if math.isfinite(restricted_bound) and abs(restricted_bound) < gp.GRB.INFINITY:
                diagnostic["restricted_pool_bound_diagnostic_only"] = restricted_bound
        except (AttributeError, gp.GurobiError):
            pass
        if sol_count > 0:
            audit_started = time.monotonic()
            try:
                x = np.array([var.X for var in variables], dtype=float)
                candidate, candidate_upper, audit = _extract_policy(pool, M, columns, x)
                diagnostic["policy_audit"] = audit
                diagnostic["solver_objval_diagnostic_only"] = float(model.ObjVal)
                if candidate_upper < upper:
                    best, upper = candidate, candidate_upper
            except (ValueError, RuntimeError) as exc:
                diagnostic["rejected_primal"] = repr(exc)
            finally:
                audit_seconds += time.monotonic() - audit_started
                audit_completed = time.monotonic()
        if status == gp.GRB.INFEASIBLE:
            diagnostic["unexpected_infeasibility"] = "audited facility plan plus all outsourcing is feasible"
            return result("RESTRICTED_INFEASIBLE_DIAGNOSTIC")
        if "rejected_primal" in diagnostic:
            return result("INVALID_PRIMAL_FALLBACK")
        return result("OPTIMAL_RESTRICTED" if status == gp.GRB.OPTIMAL else
                      "TIME_LIMIT" if status == gp.GRB.TIME_LIMIT else f"SOLVER_STATUS_{status}")
    except SolveDeadlineReached:
        build_seconds = max(build_seconds, time.monotonic() - build_started)
        return result("NO_BUDGET")
    except gp.GurobiError as exc:
        diagnostic["solver_error"] = repr(exc)
        return result("SOLVER_ERROR")
    finally:
        close_native()
