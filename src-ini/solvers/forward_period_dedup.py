"""Exact, within-pass equivalence keys for repeated subproblems.

Stage-2 and backward Stage-3 keys fingerprint the complete effective model,
including every cut coefficient and intercept.  The forward Stage-3 key uses
the smaller exact routing identity: vehicle, assigned customers, directed
costs and solver policy.  Every reused route is independently re-certified
and re-priced for its target node.  All caches remain local to one call.
"""

from __future__ import annotations

import math
import hashlib
import os
from collections.abc import Mapping

from models.stage3_arc_domain import stage3_route_arcs


def canonical_float(value) -> str:
    """Losslessly encode a finite numeric value for an equality key."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value cannot enter a dedup key: {value!r}")
    # Normalize signed zero because it produces the same linear expression.
    if number == 0.0:
        number = 0.0
    return number.hex()


def canonical_options(options) -> tuple:
    """Convert small solver-policy/config payloads into stable tuples."""
    if options is None:
        return ()
    if isinstance(options, Mapping):
        return tuple(
            (str(key), ('float', '+inf') if str(key) in (
                'time_limit', 'sub_time_limit', 'stage2_sub_time_limit',
                'phase1_time_limit', 'phase2_time_limit') and value == math.inf
             else canonical_options(value))
            for key, value in sorted(options.items(), key=lambda item: str(item[0]))
        )
    if isinstance(options, (tuple, list)):
        return tuple(canonical_options(value) for value in options)
    if isinstance(options, float):
        return ("float", canonical_float(options))
    if isinstance(options, (str, int, bool)):
        return (type(options).__name__, options)
    return (type(options).__name__, repr(options))


def _hash_semantic_value(hasher, value) -> None:
    """Feed a deterministic, lossless supported value into ``hasher``."""
    if value is None:
        hasher.update(b"N;")
        return
    if isinstance(value, bool):
        hasher.update(b"B1;" if value else b"B0;")
        return
    if isinstance(value, (int, float)):
        token = canonical_float(value).encode("ascii")
        hasher.update(b"F" + str(len(token)).encode("ascii") + b":" + token + b";")
        return
    if isinstance(value, str):
        token = value.encode("utf-8")
        hasher.update(b"S" + str(len(token)).encode("ascii") + b":" + token + b";")
        return
    if isinstance(value, range):
        hasher.update(b"R[")
        for item in value:
            _hash_semantic_value(hasher, item)
        hasher.update(b"]")
        return
    if isinstance(value, Mapping):
        hasher.update(b"M{")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_semantic_value(hasher, key)
            _hash_semantic_value(hasher, value[key])
        hasher.update(b"}")
        return
    if isinstance(value, (tuple, list)):
        hasher.update(b"T[")
        for item in value:
            _hash_semantic_value(hasher, item)
        hasher.update(b"]")
        return
    # Duck-type arrays (no numpy import)
    if hasattr(value, "shape") and hasattr(value, "flat"):
        hasher.update(b"A")
        _hash_semantic_value(hasher, tuple(int(size) for size in value.shape))
        hasher.update(b"[")
        for item in value.flat:
            scalar = item.item() if hasattr(item, "item") else item
            _hash_semantic_value(hasher, scalar)
        hasher.update(b"]")
        return
    if hasattr(value, "item"):
        _hash_semantic_value(hasher, value.item())
        return
    raise TypeError(
        "unsupported value in Phase1/Phase2 semantic fingerprint: "
        f"{type(value).__name__}"
    )


def forward_semantic_fingerprint(prob_data, scen_tree) -> str:
    """Hash forward model data (excludes search-only policy)."""
    hasher = hashlib.sha256()
    hasher.update(b"vrp-forward-semantic-v1;")
    problem_fields = (
        "dd",
        "numCustomers",
        "numChargers",
        "numDepots",
        "numVehicles",
        "numAllnodes",
        "cost_purchase",
        "B_t0",
        "T",
        "Qv",
        "K",
        "V_k",
        "ops_per_year",
        "c_per_km",
        "c_routing",
    )
    for name in problem_fields:
        _hash_semantic_value(hasher, name)
        if hasattr(prob_data, name):
            _hash_semantic_value(hasher, getattr(prob_data, name))
        else:
            _hash_semantic_value(hasher, "<missing>")

    node_fields = (
        "time",
        "index",
        "info",
        "predecessor",
        "successor",
        "probability",
        "active",
        "n",
        "c_out",
        "volume",
        "w",
        "multi_coeff",
    )
    for stage in sorted(scen_tree, key=int):
        _hash_semantic_value(hasher, int(stage))
        nodes = scen_tree[stage]
        _hash_semantic_value(hasher, len(nodes))
        for node in nodes:
            for name in node_fields:
                _hash_semantic_value(hasher, name)
                if hasattr(node, name):
                    _hash_semantic_value(hasher, getattr(node, name))
                else:
                    _hash_semantic_value(hasher, "<missing>")
    return hasher.hexdigest()


def _config_get(config, name, default=None):
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(config, name, default)


def forward_solve_policy_fingerprint(config, phase: int) -> tuple:
    """Fingerprint search policy for telemetry, never for semantic rejection."""
    if int(phase) not in (1, 2):
        raise ValueError(f"forward policy phase must be 1 or 2, got {phase!r}")
    prefix = f"phase{int(phase)}"
    policy = {
        "version": "forward-solve-policy-v2",
        "phase": int(phase),
        "num_processes": int(_config_get(config, "num_processes", 1)),
        "mip_gap": _config_get(config, f"{prefix}_tol", None),
        "sub_time_limit": _config_get(config, f"{prefix}_time_limit", None),
        "lazy_threshold": _config_get(
            config,
            f"{prefix}_lazy_threshold",
            8 if int(phase) == 1 else 0,
        ),
        "period_dedup": bool(_config_get(
            config, f"{prefix}_forward_period_dedup", True
        )),
        "s2_capacity_rhs": "Q*y" if int(phase) == 1 else "Q",
        "s2_activation_order": int(phase) == 1,
        "backend_env": {
            name: os.environ.get(name, "<unset>")
            for name in (
                "VRP_USE_ESP_BP",
                "VRP_S2_FORWARD_SOLVER",
                "VRP_FORWARD_S3_USE_CONCORDE",
                "VRP_FORWARD_S3_SOLVER",
                "VRP_FORWARD_S3_CONCORDE_THRESHOLD",
                "VRP_CONCORDE_SCALE",
            )
        },
    }
    return canonical_options(policy)


def forward_values_are_finite(values) -> bool:
    """Return whether a handoff payload contains only finite numeric leaves."""
    def finite(value):
        if isinstance(value, Mapping):
            return all(finite(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite(item) for item in value)
        if isinstance(value, bool):
            return True
        if isinstance(value, (int, float)) or hasattr(value, "item"):
            try:
                return math.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                return False
        return False

    return finite(values)


def cut_fingerprint(cut) -> tuple:
    """Fingerprint one ``(slope_dict, intercept)`` learned value cut."""
    slope, intercept = cut
    terms = tuple(
        (str(name), canonical_float(coefficient))
        for name, coefficient in sorted(slope.items(), key=lambda item: str(item[0]))
    )
    return canonical_float(intercept), terms


def cut_pool_fingerprint(cuts) -> tuple:
    """Fingerprint the complete ordered pool.

    Preserving order is conservative: two reordered pools are mathematically
    equivalent but may lead a lazy callback through a different search path.
    """
    return tuple(cut_fingerprint(cut) for cut in cuts)


def cut_archive_fingerprint(cut_lag) -> tuple:
    """Fingerprint a complete stage/node learned-cut archive."""
    return tuple(
        (
            int(stage),
            tuple(
                (int(node), cut_pool_fingerprint(pool))
                for node, pool in sorted(nodes.items(), key=lambda item: int(item[0]))
            ),
        )
        for stage, nodes in sorted(cut_lag.items(), key=lambda item: int(item[0]))
    )


def stage2_archive_fingerprint(node, cut_lag) -> str:
    """Digest of the S3->S2 cut pools one Stage-2 node's model is built from.

    Equal digests mean the fixed-fleet piece MIPs of ``node`` are the same
    model (same successors, same ordered pools with identical coefficients);
    the piece table's stall memory is keyed by it.
    """
    pools = cut_lag.get(3, {}) if isinstance(cut_lag, Mapping) else {}
    payload = tuple(
        (int(successor), cut_pool_fingerprint(pools.get(successor, ())))
        for successor in node.successor
    )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def _node_vector(values) -> tuple:
    return tuple(canonical_float(value) for value in values)


def _rounded_binary(value) -> int:
    # Matches the builders' np.round(..., 0) semantics for finite scalar data.
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite binary state in dedup key: {value!r}")
    return int(round(number))


def _lrp_array_digest(*arrays):
    """Exact array bytes keep complete directed costs out of large tuple keys."""
    import numpy as np
    h = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        h.update(str(value.shape).encode()); h.update(str(value.dtype).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def _lrp_physical_ids(data, facility=None):
    m, n, _, _, _ = data.shape
    facilities = tuple(data.arrays.get('facility_ids', range(m)))
    customers = tuple(data.arrays.get('customer_ids', range(n)))
    return (facilities if facility is None else (facilities[facility],), customers)


def _lrp_bit(state, name):
    value = float(state[name])
    if value not in (0., 1.):
        raise ValueError(f'LRP period state must be exactly binary: {name}')
    return int(value)


def stage2_period_key(second_node, x_prev_stage1, cut_lag, prob_data,
                      scen_tree, *, model_options=None, solve_policy=None):
    """Exact within-scenario S2 model/complete subtree identity across periods.

    The interval label is replaced by its full physical availability vector.
    No facility is interchangeable: IDs, capacity and every directed route-cost
    entry are retained. Only local theta node names are normalized by facility.
    """
    from models.stage_builder import _instance, _node_context, _route_pools, _as_cut
    data = _instance(prob_data)
    ctx = _node_context(data, second_node, stage=2)
    pools, mapping = _route_pools(ctx, second_node, cut_lag)
    for facility, successor in mapping.items():
        route_node = scen_tree[3][successor]
        if route_node.info != facility or route_node.predecessor != second_node.index:
            raise ValueError('S2 successor does not identify its physical facility')
    state = tuple(_lrp_bit(x_prev_stage1, f'A[{i},{ctx.interval}]') for i in range(ctx.m))
    cuts = tuple((i, tuple((canonical_float(cut.intercept),
                          tuple(canonical_float(v) for v in cut.coefficients))
                         for item in pools[i] for cut in [_as_cut(item, ctx, i)]))
                 for i in range(ctx.m))
    return ('lrp-s2-period-v1', ctx.scenario, _lrp_physical_ids(data), state,
            _lrp_array_digest(ctx.active, ctx.demand, ctx.outsourcing,
                              ctx.capacity, ctx.route_cost), cuts,
            canonical_options(model_options), canonical_options(solve_policy))


def forward_stage3_route_key(vehicle, third_node, x_prev_stage2, prob_data, *,
                             solve_policy=None):
    """Exact LRP physical facility route identity, independently re-audited on hit.

    Retain the whole directed matrix and active/demand/capacity context, even
    for unassigned customers. This is conservative and never uses a common
    depot or homogeneous-vehicle equivalence.
    """
    from models.stage_builder import _instance, _node_context
    data = _instance(prob_data)
    ctx = _node_context(data, third_node, stage=3)
    facility = int(vehicle)
    if facility != int(third_node.info) or not 0 <= facility < ctx.m:
        raise ValueError('Wrong physical facility in Stage-3 route key')
    alpha = tuple(_lrp_bit(x_prev_stage2, f'alpha[{facility},{j}]') for j in range(ctx.n))
    u = _lrp_bit(x_prev_stage2, f'u[{facility}]')
    ctx.check_route_state(facility, alpha, u)
    return ('lrp-forward-s3-route-v1', facility,
            _lrp_physical_ids(data, facility), alpha, u,
            _lrp_array_digest(ctx.active, ctx.demand, ctx.capacity[facility:facility+1],
                              ctx.route_cost[facility]), canonical_options(solve_policy))


def stage3_period_key(scenario, vehicle, third_node, x_prev_stage2, prob_data, *,
                      solve_policy=None):
    """Original entry point with LRP state names and a conservative context key."""
    return ('lrp-s3-period-v1', int(scenario),
            forward_stage3_route_key(vehicle, third_node, x_prev_stage2, prob_data,
                                     solve_policy=solve_policy))


def remap_theta_by_vehicle(theta_by_successor, source_node, target_node, scen_tree):
    """Compatibility name: match physical facility IDs, never vehicle symmetry."""
    by_facility = {int(scen_tree[3][q].info): float(theta_by_successor[q])
                   for q in source_node.successor}
    if len(by_facility) != len(source_node.successor):
        raise ValueError('Duplicate physical facility in source subtree')
    return {q: by_facility[int(scen_tree[3][q].info)] for q in target_node.successor}


def remap_stage2_forward_values(values, source_node, target_node, scen_tree):
    """Rename only local theta IDs; physical alpha/u/e/z/route-base names stay."""
    out = dict(values)
    theta = {q: out.pop(f'theta[{q}]') for q in source_node.successor}
    out.update({f'theta[{q}]': value for q, value in
                remap_theta_by_vehicle(theta, source_node, target_node, scen_tree).items()})
    return out
