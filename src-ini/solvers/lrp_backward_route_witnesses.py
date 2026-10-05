"""Bounded, primal-only transport of already solved native backward routes.

Neither a bundle support nor a native objective is a route cost. Only the
original ordered tour, already accepted by native _decode, may enter here.
No optimizer is constructed or called by this transport layer.
"""
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
import math
import time

from models.stage_builder import _instance
from solvers.lrp_physical_policy_pool import audit_route
from solvers.lrp_physical_types import AuditedRoute, node_signature


def _deadline(value):
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError('route witness deadline must be finite or None')
    return float(value)


def _before_deadline(deadline):
    return deadline is None or time.monotonic() < deadline


@dataclass(frozen=True)
class BackwardRouteWitness:
    instance_sha256: str
    node_id: tuple[int, int]
    stage3_node_index: int
    facility_id: int
    route: AuditedRoute

    def __post_init__(self):
        if (not isinstance(self.instance_sha256, str) or len(self.instance_sha256) != 64
                or not isinstance(self.node_id, tuple) or len(self.node_id) != 2
                or any(type(v) is not int or v < 0 for v in self.node_id)
                or type(self.stage3_node_index) is not int or self.stage3_node_index < 0
                or type(self.facility_id) is not int or self.facility_id < 0
                or not isinstance(self.route, AuditedRoute)
                or type(self.route.facility_id) is not int):
            raise ValueError('invalid backward route witness scope')


def validate_route_witness(prob_data, node, witness, *, instance_sha256=None):
    """Reaudit immutable evidence against the receiving node's original data."""
    if not isinstance(witness, BackwardRouteWitness):
        raise TypeError('expected BackwardRouteWitness')
    witness.__post_init__()
    data = _instance(prob_data)
    expected_hash = data.logical_hash() if instance_sha256 is None else instance_sha256
    ctx, facility = node.context, int(node.info)
    if (witness.instance_sha256 != expected_hash
            or witness.node_id != (ctx.period, ctx.scenario)
            or witness.stage3_node_index != int(node.index)
            or witness.facility_id != facility
            or not isinstance(witness.route, AuditedRoute)
            or witness.route.facility_id != facility
            or witness.route.node_signature != node_signature(data, ctx)):
        raise ValueError('backward route witness has a foreign physical scope')
    route = witness.route
    fresh = audit_route(data, ctx, facility, route.customers_in_order,
                        source=route.source, generation_id=route.generation_id)
    if fresh != route:
        raise ValueError('backward route witness differs from original-data audit')
    return witness


class BackwardRouteWitnessBuffer:
    """Keep bounded cheapest tours per scope/coverage; never cache scalar bounds."""
    def __init__(self, prob_data, *, max_routes=512, max_routes_per_scope=32):
        if any(type(v) is not int or v < 1 for v in (max_routes, max_routes_per_scope)):
            raise ValueError('route witness limits must be positive integers')
        self.data = _instance(prob_data)
        self.instance_sha256 = self.data.logical_hash()
        self.limits = dict(max_routes=max_routes, max_routes_per_scope=max_routes_per_scope)
        self._rows = OrderedDict()
        self._nodes = {}

    def __len__(self):
        return len(self._rows)

    @staticmethod
    def _key(witness):
        return (witness.node_id, witness.facility_id,
                frozenset(witness.route.customers_in_order))

    def accept(self, node, witness):
        validate_route_witness(self.data, node, witness,
                               instance_sha256=self.instance_sha256)
        key = self._key(witness)
        old = self._rows.get(key)
        if old is not None and Fraction(*old.route.cost_exact) <= Fraction(*witness.route.cost_exact):
            return False
        self._rows[key] = witness
        self._rows.move_to_end(key)
        self._nodes[key] = node
        scope = key[:2]
        scoped = [k for k in self._rows if k[:2] == scope]
        while len(scoped) > self.limits['max_routes_per_scope']:
            oldest = scoped.pop(0)
            self._rows.pop(oldest); self._nodes.pop(oldest)
        while len(self._rows) > self.limits['max_routes']:
            oldest, _ = self._rows.popitem(last=False)
            self._nodes.pop(oldest)
        return True

    def capture_native(self, node, result):
        """Use a decoded nonempty original tour, never alpha/reduced costs."""
        if (result.get('native_executed') is not True
                or result.get('incumbent_policy_certified') is not True):
            return False
        if result.get('source') not in {'lrp_native_pctsp', 'lrp_native_pctsp_residual'}:
            return False
        ctx, facility = node.context, int(node.info)
        if result.get('context') != ctx.route_key(facility):
            raise ValueError('native backward route result has a foreign context')
        tour = result.get('tour')
        if not isinstance(tour, dict) or tour.get('facility') != facility:
            raise ValueError('native backward route has a foreign facility')
        path = tuple(tour.get('local_route', ()))
        if not path:
            return False  # idle is not a route column
        if (len(path) < 3 or path[0] != 0 or path[-1] != 0
                or any(type(v) is not int for v in path)):
            raise ValueError('native backward route lacks its audited own-root order')
        if tuple(tuple(arc) for arc in tour.get('arcs', ())) != tuple(zip(path, path[1:])):
            raise ValueError('native backward ordered tour and original arcs disagree')
        route = audit_route(self.data, ctx, facility, tuple(v - 1 for v in path[1:-1]),
                            source='native_backward_primal', generation_id=0)
        if result.get('route_cost') != route.cost:
            raise ValueError('native backward route cost differs from original-data audit')
        witness = BackwardRouteWitness(self.instance_sha256, (ctx.period, ctx.scenario),
                                       int(node.index), facility, route)
        return self.accept(node, witness)

    def peek(self, max_routes=None, *, deadline=None):
        """Audit a prefix progressively; never acknowledge or return a late row.

        Hash the original data once per snapshot, then check the same absolute
        deadline before and after each route audit.  A caller may explicitly
        acknowledge a completed prefix, or defer the entire transaction.  A
        failed audit raises without changing any buffered witness.
        """
        if max_routes is not None and (type(max_routes) is not int or max_routes < 0):
            raise ValueError('peek route limit must be a nonnegative integer or None')
        deadline = _deadline(deadline)
        keys = tuple(self._rows)[:max_routes]
        if not keys or not _before_deadline(deadline):
            return ()
        # A mutation of the original data cannot silently cross a transport boundary.
        current_hash = self.data.logical_hash()
        witnesses = []
        for key in keys:
            if not _before_deadline(deadline):
                break
            witness = validate_route_witness(self.data, self._nodes[key], self._rows[key],
                                             instance_sha256=current_hash)
            if not _before_deadline(deadline):
                break
            witnesses.append(witness)
        return tuple(witnesses)

    def discard(self, witnesses):
        """Acknowledge only this exact snapshot; leave unconsumed/newer rows intact."""
        count = 0
        for witness in witnesses:
            if not isinstance(witness, BackwardRouteWitness):
                raise TypeError('expected BackwardRouteWitness acknowledgement')
            key = self._key(witness)
            if self._rows.get(key) == witness:
                self._rows.pop(key); self._nodes.pop(key); count += 1
        return count

    def drain(self, max_routes=None, *, deadline=None):
        deadline = _deadline(deadline)
        witnesses = self.peek(max_routes, deadline=deadline)
        # Unlike peek, drain commits an acknowledgement.  If the batch used
        # its deadline, leave every row available to the next transaction.
        if not _before_deadline(deadline):
            return ()
        self.discard(witnesses)
        return witnesses
