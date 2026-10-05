"""Exact, independently re-audited Phase-1 forward handoff for LRP."""
from __future__ import annotations

import copy
import hashlib
import math
import os

from algorithms.base_algorithm import cut_archive_fingerprint, forward_values_are_finite
from core.solver_settings import (configured_gurobi_threads,
                                 configured_forward_s2_backend, forward_s2_bpc_policy)
from core.solver_bounds import minimization_bounds_inverted
from models.stage_builder import _instance
from solvers.forward_policy_certification import certify_policy


def semantic_fingerprint(prob_data, tree):
    """Full arrays plus physical tree identities; never a homogeneous fleet key."""
    data = _instance(prob_data)
    nodes = tuple((int(stage), tuple((node.time, node.index, node.info,
        node.predecessor, tuple(node.successor), tuple(sorted(node.probability.items())),
        node.multi_coeff, None if node.context is None else node.context.key,
        tuple(sorted(getattr(node, 'successor_facilities', {}).items())))
        for node in layer)) for stage, layer in sorted(tree.items()))
    return hashlib.sha256(repr((data.logical_hash(), nodes)).encode()).hexdigest()


def solve_policy(forward):
    """Actual forward settings, including the native assignment policy."""
    s2_backend = configured_forward_s2_backend(forward.instance.shape[1])
    policy = {'version': 'lrp-forward-effective-v4-basic-static-bounds',
            'mip_rel_gap': float(forward.stage_builder.mip_gap),
            'mip_abs_gap': 1e-8, 'threads': configured_gurobi_threads(),
            's2_abs_tol': forward._s2_abs_tol,
            's2_rel_cap': forward._s2_rel_cap,
            'sub_time_limit': forward.sub_time_limit,
            'connectivity': forward.stage_builder.connectivity,
            'route_backend': os.environ.get('LRP_S3_BACKEND', 'native'),
            's2_backend': s2_backend,
            's2_bpc_policy': forward_s2_bpc_policy(forward.phase) if s2_backend == 'bpc' else None}
    if forward.stage2_sub_time_limit != forward.sub_time_limit:
        policy['stage2_sub_time_limit'] = forward.stage2_sub_time_limit
    if forward.stage2_mip_gap != forward.stage_builder.mip_gap:
        policy['stage2_mip_gap'] = forward.stage2_mip_gap
    return policy


def compatible_solve_policy(source, target):
    """A stricter P1 certificate can serve P2; weaker requests must re-solve.

    P1 does not schedule dynamic S2 absolute gaps. Conservatively reject any
    source that did: its per-node cap cannot be inferred from an outer scalar.
    The target clamps its scheduled absolute gap to at least baseline 1e-8.
    """
    if not isinstance(source, dict) or source.get('s2_abs_tol') is not None:
        return False
    source, target = dict(source), dict(target)
    source.setdefault('stage2_mip_gap', source.get('mip_rel_gap'))
    target.setdefault('stage2_mip_gap', target.get('mip_rel_gap'))
    if set(source) != set(target):
        return False
    fixed = set(target) - {'mip_rel_gap', 'stage2_mip_gap', 's2_abs_tol', 's2_rel_cap'}
    if any(source[k] != target[k] for k in fixed):
        return False
    try:
        source_gap, target_gap = float(source['mip_rel_gap']), float(target['mip_rel_gap'])
        source_s2_gap, target_s2_gap = float(source['stage2_mip_gap']), float(target['stage2_mip_gap'])
    except (TypeError, ValueError):
        return False
    return (math.isfinite(source_gap) and math.isfinite(target_gap)
            and 0. <= source_gap <= target_gap
            and math.isfinite(source_s2_gap) and math.isfinite(target_s2_gap)
            and 0. <= source_s2_gap <= target_s2_gap)


def make_snapshot(prob_data, tree, cuts, forward, values, seconds, iteration):
    audit = certify_policy(prob_data, tree, values[1])
    if (not forward_values_are_finite(values)
            or audit['feasible_upper_bound'] != values[2]
            or minimization_bounds_inverted(values[0], values[2])):
        raise ValueError('Cannot snapshot an uncertified LRP forward pass')
    return {'schema': 'lrp_forward_handoff_v1', 'values': copy.deepcopy(values),
            'cut_fingerprint': cut_archive_fingerprint(cuts),
            'semantic_fingerprint': semantic_fingerprint(prob_data, tree),
            'solve_policy_fingerprint': solve_policy(forward),
            'stage1_lb_certified': True, 'forward_incumbent_feasible': True,
            'forward_seconds': float(seconds), 'phase1_iteration': int(iteration),
            'forward_diagnostics': copy.deepcopy(forward.last_forward_diagnostics)}


def take_snapshot(payload, prob_data, tree, cuts, forward):
    """Return (values, reason), performing no optimizer calls on either path."""
    if not isinstance(payload, dict) or payload.get('schema') != 'lrp_forward_handoff_v1':
        return None, 'unsupported_payload'
    if payload.get('semantic_fingerprint') != semantic_fingerprint(prob_data, tree):
        return None, 'semantic_model_changed'
    if payload.get('cut_fingerprint') != cut_archive_fingerprint(cuts):
        return None, 'learned_cut_archive_changed'
    if not compatible_solve_policy(payload.get('solve_policy_fingerprint'), solve_policy(forward)):
        return None, 'solve_policy_changed'
    values = payload.get('values')
    if not isinstance(values, tuple) or len(values) != 4 or not forward_values_are_finite(values):
        return None, 'invalid_forward_values'
    if payload.get('stage1_lb_certified') is not True or payload.get('forward_incumbent_feasible') is not True:
        return None, 'uncertified_forward'
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (values[0], values[2])):
        return None, 'nonfinite_forward_bounds'
    if minimization_bounds_inverted(values[0], values[2]):
        return None, 'inverted_forward_bounds'
    try:
        audit = certify_policy(prob_data, tree, values[1])
    except (ValueError, KeyError, TypeError, RuntimeError, IndexError):
        return None, 'forward_policy_audit_failed'
    if audit['feasible_upper_bound'] != values[2]:
        return None, 'forward_policy_cost_changed'
    return copy.deepcopy(values), None
