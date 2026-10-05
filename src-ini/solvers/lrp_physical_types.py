"""Physical-forward result types and domain signatures (no optimization here).

Customer/facility integers are positions in the original arrays.  Their physical
ID maps are bound into the signatures; route vertices use depot 0 and customer
position j + 1.  Node costs are unweighted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

import numpy as np

from models.stage_builder import _instance, _node_context


PHYSICAL_SCHEMA = "lrp_physical_forward_v1"
NODE_LOWER_SOURCES = frozenset({"exact_mip_bound", "network_relax_bound",
                               "existing_s2_bound", "nonnegative_cost_floor",
                               "route_lp_price_bound"})


def _hash_array(h, name, value):
    value = np.ascontiguousarray(value)
    if value.dtype.hasobject:
        raise ValueError("physical signatures require non-object arrays")
    h.update(name.encode()); h.update(value.dtype.str.encode())
    h.update(repr(value.shape).encode()); h.update(value.tobytes())


def _semantics(data):
    """Refuse declared extensions which the physical model does not implement."""
    flags = data.metadata.get("model_flags", {})
    # Instance.validate checks the established flags.  Explicit additional fees
    # or a restricted arc mask cannot silently disappear in the new models.
    unsupported = {"dispatch_cost", "route_fixed_cost", "vehicle_cost",
                   "route_usage_cost", "legal_arc_mask", "arc_allowed"}
    present = unsupported.intersection(data.arrays) | unsupported.intersection(flags)
    if present:
        raise ValueError("Unsupported physical fee/arc declaration: " + ",".join(sorted(present)))
    return json.dumps({"schema": PHYSICAL_SCHEMA, "flags": flags,
                       "arc_domain": "complete_directed_without_self_arcs",
                       "initial_state": "all_closed"}, sort_keys=True,
                      separators=(",", ":")).encode()


def node_signature(prob_data, node_or_context) -> str:
    data = _instance(prob_data)
    ctx = _node_context(data, node_or_context, stage=2)
    h = hashlib.sha256(_semantics(data))
    h.update(ctx.key.encode())  # includes actual t/s and the whole physical domain
    for key, count in (("facility_ids", ctx.m), ("customer_ids", ctx.n)):
        _hash_array(h, key, data.arrays.get(key, np.arange(count, dtype=np.int64)))
    return h.hexdigest()


def global_policy_signature(prob_data) -> str:
    data = _instance(prob_data)
    h = hashlib.sha256(_semantics(data))
    # In particular: full A/o/h/b costs, min_open, initial state, k(t), and pi.
    for key, value in sorted(data.arrays.items()):
        _hash_array(h, key, value)
    return h.hexdigest()


@dataclass(frozen=True)
class AuditedRoute:
    node_signature: str
    facility_id: int
    customers_in_order: tuple[int, ...]
    cost: float
    total_demand: float
    audit_signature: str
    source: str
    generation_id: int
    # Exact binary64 input sums permit sound cheapest-order comparisons.
    cost_exact: tuple[int, int] = (0, 1)
    demand_exact: tuple[int, int] = (0, 1)


@dataclass(frozen=True)
class NodePhysicalCertificate:
    node_id: tuple[int, int]
    node_signature: str
    A_mask: tuple[int, ...]
    q_lower: float | None
    q_upper: float | None
    policy: object | None
    lower_source: str | None
    upper_source: str | None
    status: str
    closed_within_tolerance: bool
    model_domain_complete: bool
    domain_kind: str
    envelope_version: int
    wall_seconds: float
    build_seconds: float
    solve_seconds: float
    audit_seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    audited_routes: tuple[AuditedRoute, ...] = ()

    def __post_init__(self):
        if self.domain_kind not in {"exact_mip", "network_relax", "existing_s2", "route_cg"}:
            raise ValueError("Unrecognized physical certificate domain")
        if len(self.node_id) != 2 or any(type(v) is not int or v < 0 for v in self.node_id):
            raise ValueError("Invalid physical node ID")
        if not self.A_mask or any(type(v) is not int or v not in (0, 1) for v in self.A_mask):
            raise ValueError("Invalid availability mask")
        for value in (self.q_lower, self.q_upper):
            if value is not None and (not math.isfinite(value) or abs(value) >= 1e100):
                raise ValueError("Physical bounds must be finite ordinary numbers")
        if self.q_lower is not None:
            if self.lower_source not in NODE_LOWER_SOURCES:
                raise ValueError("Uncertified or restricted-pool lower source")
            if not self.model_domain_complete:
                raise ValueError("Node lower requires the full physical assignment domain")
            expected = {"exact_mip_bound": "exact_mip", "network_relax_bound": "network_relax",
                        "existing_s2_bound": "existing_s2",
                        "route_lp_price_bound": "route_cg"}.get(self.lower_source)
            if expected is not None and self.domain_kind != expected:
                raise ValueError("Lower source does not match model domain")
        if self.q_upper is not None and (self.policy is None or not self.upper_source):
            raise ValueError("Physical upper requires an audited policy and provenance")
        if self.q_lower is not None and self.q_upper is not None:
            tol = 2e-6 + 1e-10 * max(1., abs(self.q_lower), abs(self.q_upper))
            if self.q_lower > self.q_upper + tol:
                raise ValueError("Contradictory physical L/U certificate; never clamp")


@dataclass(frozen=True)
class PoolPolicyResult:
    policy: object | None
    audited_global_ub: float | None
    status: str
    pool_version: int
    wall_seconds: float
    build_seconds: float = 0.
    solve_seconds: float = 0.
    audit_seconds: float = 0.
    diagnostics: dict[str, Any] = field(default_factory=dict)
    route_snapshot: tuple[AuditedRoute, ...] = ()


def validate_node_certificate(value, prob_data, node, A_mask):
    """Typed boundary: pool results/route bounds cannot become true node LB."""
    if not isinstance(value, NodePhysicalCertificate):
        raise TypeError("Expected NodePhysicalCertificate, never a pool/route result")
    value.__post_init__()
    ctx = _node_context(_instance(prob_data), node, stage=2)
    supplied = tuple(A_mask)
    if (value.node_signature != node_signature(prob_data, node)
            or value.node_id != (ctx.period, ctx.scenario)
            or len(supplied) != ctx.m or any(v not in (0, 1) for v in supplied)
            or value.A_mask != supplied):
        raise ValueError("DOMAIN_MISMATCH: physical node/mask/signature")
    return value
