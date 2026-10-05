"""Certified facility pricing over the complete physical LRP route domain.

The production adapter reuses the verified free-state native PCTSP oracle.  It
requests the idle-inclusive objective h_i(lambda); a failed or expired native
call is replaced by the nonnegative-cost bound from equation (3) of the V2
implementation specification.  Incumbents and bounds remain separate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import math
import time

from core.solver_bounds import _fraction_to_finite_float_down
from models.stage_builder import _instance, _node_context
from solvers.lrp_native_oracle import LRPNativeRouteOracle, NativeUnavailable
from solvers.lrp_physical_cg_types import PricingResult
from solvers.lrp_physical_policy_pool import audit_route
from solvers.lrp_physical_types import node_signature


def price_hash(values):
    return hashlib.sha256(repr(tuple(float(v).hex() for v in values)).encode()).hexdigest()


def trivial_idle_lower(active, prices):
    return -sum((F(float(p)) for a, p in zip(active, prices) if a and p > 0), F())


def transported_lower(old_lower, old_prices, new_prices, active):
    if len(old_prices) != len(new_prices) or len(active) != len(new_prices):
        raise ValueError("price transport dimension mismatch")
    payment = sum((max(F(), F(float(n)) - F(float(o)))
                   for a, o, n in zip(active, old_prices, new_prices) if a), F())
    return F(float(old_lower)) - payment


def _order_from_tour(tour):
    arcs = tuple(tuple(map(int, arc)) for arc in tour.get("arcs", ()))
    if not arcs:
        return ()
    successor = {}
    for v, w in arcs:
        if v in successor:
            raise ValueError("pricing incumbent has duplicate outgoing arc")
        successor[v] = w
    order, current = [], 0
    for _ in range(len(arcs)):
        current = successor[current]
        if current == 0:
            break
        order.append(current - 1)
    if current != 0 or len(order) + 1 != len(arcs):
        raise ValueError("pricing incumbent is not one own-root tour")
    return tuple(order)


def strengthen_capacity_bound(ctx, prices, result):
    """Add a freshly rebuilt analytic proof to original scoped evidence."""
    from solvers.lrp_capacity_price_bound import capacity_price_lower_bound
    analytic=capacity_price_lower_bound(ctx,result.facility_id,prices)
    if result.safe_lower is not None and result.safe_lower>=analytic['safe_lower']:
        return result
    return replace(result,safe_lower=analytic['safe_lower'],
        certificate_source='capacity_incoming_analytic',
        error_accounting={**result.error_accounting,'formula':analytic['formula'],
                          'fallback':True})


class FacilityPricingAdapter:
    def __init__(self, prob_data, tree, node, *, backend="native", top_k=1,
                 capacity_bound=False,ng_size=8):
        self.data, self.tree, self.node = _instance(prob_data), tree, node
        self.ctx = _node_context(self.data, node, stage=2)
        self.signature = node_signature(self.data, node)
        self.backend = backend
        if type(top_k) is not int or not 1 <= top_k <= 64:
            raise ValueError('top_k must be an integer in 1..64')
        self.top_k = top_k
        if not isinstance(capacity_bound,bool):
            raise TypeError('capacity_bound must be boolean')
        self.capacity_bound = capacity_bound
        if type(ng_size) is not int or not 0<=ng_size<=256:
            raise ValueError('ng_size must be an integer in 0..256')
        self.ng_size=ng_size
        if backend not in {"native", "reference"}:
            raise ValueError("pricing backend must be native or reference")

    def _third(self, facility):
        matches = [self.tree[3][r] for r in self.node.successor
                   if int(self.tree[3][r].info) == facility]
        if len(matches) != 1:
            raise ValueError("each physical facility needs exactly one S3 child")
        return matches[0]

    def price(self, facility_id, lambda_vector, *, deadline, time_limit_s=3.,
              generation_id=0):
        started = time.monotonic(); i = int(facility_id)
        lam = tuple(float(v) for v in lambda_vector)
        if len(lam) != self.ctx.n or any(not math.isfinite(v) for v in lam):
            raise ValueError("pricing requires one finite signed price per customer")
        if not 0 <= i < self.ctx.m:
            raise ValueError("invalid facility")
        h = price_hash(lam)
        fallback = _fraction_to_finite_float_down(
            trivial_idle_lower(self.ctx.active, lam), label='physical pricing fallback')
        fallback_source='nonnegative_cost_fallback'; fallback_formula='eq3'
        if self.capacity_bound:
            from solvers.lrp_capacity_price_bound import capacity_price_lower_bound
            analytic=capacity_price_lower_bound(self.ctx,i,lam)
            fallback=analytic['safe_lower']
            fallback_source='capacity_incoming_analytic'
            fallback_formula=analytic['formula']
        if deadline is None or not math.isfinite(float(deadline)):
            raise ValueError("pricing requires a finite absolute deadline")
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            return PricingResult(self.signature, i, h, "WITH_IDLE", "EXACT_PHYSICAL",
                "NO_BUDGET", fallback, fallback, None, None, None, True,
                fallback_source, {"formula": fallback_formula, "fallback": True})
        solve_started = time.monotonic(); route = None; incumbent = None
        raw_lower = safe_lower = None; status = "NATIVE_UNSUPPORTED"; source = "none"
        diagnostics = {}; additional = []
        try:
            oracle = LRPNativeRouteOracle(self.data, self._third(i))
            pi = {f"alpha[{i},{j}]": lam[j] for j in range(self.ctx.n)}
            pi[f"u[{i}]"] = 0.
            result = oracle.solve(pi, time_limit=min(float(time_limit_s), remaining),
                                  deadline=deadline, ng_size=self.ng_size,
                                  **({'top_k': self.top_k} if self.top_k > 1 else {}))
            status = result["status"]
            raw_lower = result.get("raw_bound")
            if result.get("outer_lb") is not None:
                safe_lower = min(0., float(result["outer_lb"]))
                source = "lrp_native_complete_idle_pricing"
            if result.get("inner_xcp") is not None:
                order = _order_from_tour(result["tour"])
                incumbent = float(result["inner_value"])
                if order:
                    route = audit_route(self.data, self.node, i, order,
                        source="physical_cg_pricing", generation_id=generation_id)
                    independently = float(F(*route.cost_exact) - sum((F(lam[j]) for j in order), F()))
                    if abs(independently - incumbent) > 2e-6 + 1e-10 * max(1., abs(incumbent)):
                        raise ValueError("native pricing incumbent objective mismatch")
                elif abs(incumbent) > 1e-9:
                    raise ValueError("idle pricing incumbent must have zero value")
            diagnostics = {"native_status": result.get("status"),
                           "native_executed": result.get("native_executed"),
                           "exact": result.get("exact", False),
                           "ng_size": self.ng_size}
            candidate_errors = list(result.get('candidate_rejections', ()))
            for candidate in result.get('candidate_primals', ()):
                if time.monotonic() >= deadline:
                    candidate_errors.append('candidate audit deadline'); break
                try:
                    order = _order_from_tour(candidate['tour'])
                    if not order:
                        continue
                    extra = audit_route(self.data, self.node, i, order,
                        source='physical_cg_pricing', generation_id=generation_id)
                    exact = F(*extra.cost_exact)-sum((F(lam[j]) for j in order), F())
                    if abs(float(exact)-candidate['inner_value']) > 2e-6+1e-10*max(1.,abs(float(exact))):
                        raise ValueError('additional pricing route objective mismatch')
                    if safe_lower is not None and F(safe_lower) > exact:
                        safe_lower = None
                        candidate_errors.append('additional route contradicts native lower bound')
                    if time.monotonic() >= deadline:
                        candidate_errors.append('candidate audit deadline'); break
                    if extra != route and all(extra != old for old in additional):
                        additional.append(extra)
                except (ValueError, TypeError, KeyError) as exc:
                    candidate_errors.append(str(exc))
            diagnostics.update(additional_route_count=len(additional),
                               candidate_rejections=tuple(candidate_errors))
        except (NativeUnavailable, RuntimeError, ValueError) as exc:
            diagnostics = {"native_error": repr(exc)}
            status = "NATIVE_UNSUPPORTED" if isinstance(exc, NativeUnavailable) else "INVALID"
            route = None; incumbent = None; safe_lower = None
            additional = []
        solve_elapsed = time.monotonic() - solve_started
        used_fallback = safe_lower is None
        if used_fallback:
            safe_lower = fallback; source = fallback_source
        elif self.capacity_bound and fallback > safe_lower:
            # This is a separately reconstructed analytic proof; preserve the
            # raw native bound but do not relabel this stronger number native.
            safe_lower = fallback; source = fallback_source; used_fallback = True
        if incumbent is None:
            incumbent = 0.  # idle is an audited feasible pricing candidate
        return PricingResult(self.signature, i, h, "WITH_IDLE", "EXACT_PHYSICAL",
            status, raw_lower, float(safe_lower), route,
            None if route is None else route.cost, incumbent, True, source,
            {**diagnostics, "formula": fallback_formula if used_fallback else "native_bound",
             "fallback": used_fallback}, 0., solve_elapsed,
            max(0., time.monotonic() - solve_started - solve_elapsed), tuple(additional))


def status_counts(results):
    return dict(Counter(result.status for result in results))
