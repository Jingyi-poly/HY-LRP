"""Physical LRP free-availability S2 oracle for SBC and Level Set.

All physical facilities remain in the native problem. Native y means dispatch
u; native z is the free local copy of A[i,k(t)], never a vehicle purchase.
The lower channel uses only the repaired root dual certificate. A separate
exact-rational audit builds the feasible upper support and its parent point.
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import math
from pathlib import Path
import hashlib
import os
import time

from core.backend_telemetry import backend_call
from cuts.lrp_static_bounds import basic_route_cuts
from models.stage_builder import _as_cut, _instance, _node_context, _route_pools, _state_keys
from solvers.forward_stage2_bpc import _load_native, NativeBPCUnavailable
from solvers.forward_policy_certification import InvalidForwardPolicy, certify_stage2_forward_policy
from solvers.forward_ub import round_fraction_up
from solvers.lrp_bpc_cut_projection import project_route_cuts


_REVIEWED_SOURCE_SHA256 = 'bcab2cf6b3356af4cb0333c037a3e6f84fa82873216c70f49d840ae565180068'


def _load_reviewed_native():
    native = _load_native()
    if getattr(native, 'backward_deadline_contract', None) != 'all_mode_pricing_deadline_v1':
        raise NativeBPCUnavailable('S2 backward extension lacks the reviewed deadline contract')
    if getattr(native, 'backward_gap_contract', None) != 'certified_root_query_gap_v1':
        raise NativeBPCUnavailable('S2 backward extension lacks the reviewed query-gap contract')
    source = Path(native.__file__).parent / 'stage2_branch_price.cpp'
    if hashlib.sha256(source.read_bytes()).hexdigest() != _REVIEWED_SOURCE_SHA256:
        raise NativeBPCUnavailable('S2 backward certificate source has not passed the v8 audit')
    return native


def _finite(value):
    if isinstance(value, (str, bytes, bool)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and abs(result) < 1.e99 else None


def _down(exact):
    value = float(exact)
    if not math.isfinite(value):
        raise ValueError('nonfinite native certificate arithmetic')
    return math.nextafter(value, -math.inf) if Fraction.from_float(value) > exact else value


def _fraction(value):
    return Fraction.from_float(float(value))


def _bit(value, label):
    # The BPC ABI exports integer arrays. Tolerant rounding could certify a
    # different primal policy than the native solver actually returned.
    if isinstance(value, (str, bytes)) or value not in (0, 1):
        raise InvalidForwardPolicy(f'nonbinary native {label}')
    return int(value)


def _policy(ctx, data, node, raw, multipliers, cuts, successors):
    alpha, dispatch = raw['alpha'], raw['y']
    if (len(alpha) != ctx.m or len(dispatch) != ctx.m
            or any(len(row) != ctx.n for row in alpha)):
        raise InvalidForwardPolicy('native assignment shape mismatch')
    decisions = {f'alpha[{i},{j}]': _bit(alpha[i][j], f'alpha[{i},{j}]')
                 for i in range(ctx.m) for j in range(ctx.n)}
    for i in range(ctx.m):
        decisions[f'u[{i}]'] = _bit(dispatch[i], f'y[{i}]')
    z = [int(bool(decisions[f'u[{i}]']) or multipliers[i] > 0.) for i in range(ctx.m)]
    # For pi>0 an idle facility optimally has z=1; pi<0 chooses z=u.
    # pi=0 has interchangeable local z values. Use the native optimal choice
    # z=u deterministically without imposing minimum-open or transition rules.
    if 'z' in raw:
        if len(raw['z']) != ctx.m:
            raise InvalidForwardPolicy('native z shape mismatch')
        native_z = [_bit(value, f'z[{i}]') for i, value in enumerate(raw['z'])]
        if any(native_z[i] != z[i] and multipliers[i] != 0. for i in range(ctx.m)):
            raise InvalidForwardPolicy('native free-availability elimination mismatch')
        if any(native_z[i] < decisions[f'u[{i}]'] for i in range(ctx.m)):
            raise InvalidForwardPolicy('native dispatch exceeds free availability')
    parent = {key: float(z[i]) for i, key in enumerate(_state_keys(ctx))}
    for j in range(ctx.n):
        decisions[f'e[{j}]'] = int(ctx.active[j]) - sum(decisions[f'alpha[{i},{j}]'] for i in range(ctx.m))
    normalized, stage_cost = certify_stage2_forward_policy(data, node, parent, decisions)
    objective = sum((_fraction(ctx.outsourcing[j]) * int(normalized[f'e[{j}]'])
                     for j in range(ctx.n)), Fraction(0))
    for i in range(ctx.m):
        state = [int(normalized[f'alpha[{i},{j}]']) for j in range(ctx.n)] + [int(normalized[f'u[{i}]'])]
        theta = max([Fraction(0)] + [
            _fraction(cut.intercept) + sum((_fraction(coefficient) * bit
                for coefficient, bit in zip(cut.coefficients, state)), Fraction(0))
            for cut in cuts[i]])
        normalized[f'theta[{successors[i]}]'] = round_fraction_up(theta)
        normalized[f'z[{i}]'] = float(z[i])
        objective += theta - _fraction(multipliers[i]) * z[i]
    normalized.update(parent)
    normalized['stage_cost'] = stage_cost
    return round_fraction_up(objective), parent, normalized


def _sparse_envelope_guard(payload, m):
    """Bound v6's internal removal of |alpha coefficient| <= 1e-15.

    A max-of-cuts envelope changes by at most the maximum discarded magnitude
    for one successor, on alpha in [0,1]. Sum these bounds across successors.
    This also covers negative tiny terms: dropping those can raise the oracle.
    No full-tree incumbent objective is ever used as a lower endpoint.
    """
    per_successor = [Fraction(0)] * m
    for cut in payload:
        omitted = sum((abs(_fraction(value)) for row in cut['piAlpha'] for value in row
                       if value and abs(value) <= 1.e-15), Fraction(0))
        per_successor[cut['succ']] = max(per_successor[cut['succ']], omitted)
    return sum(per_successor, Fraction(0))


def _empty(diagnostic, reason):
    return dict(ok=False, reason=reason, outer_lb=None, inner_value=None, xcp={},
                decisions={}, lb_certified=False, incumbent_policy_certified=False,
                exact=False, query_gap_closed=False, diagnostic=diagnostic)


def solve_s2_backward_with_bpc(prob_data, node, cut_lag, pi, *, time_limit_s,
                               deadline=None, root_bound_only=False, phase=2,
                               max_nodes=None, max_depth=None, max_colgen_iters=None,
                               max_cutting_rounds=None, backward_gap_abs=0.,
                               backward_gap_rel=0.):
    """Return certified ``outer_lb`` and independently feasible ``inner_value``.

    ``xcp`` contains exactly A[i,k(t)] and is suitable for a Level Set support.
    Incumbents and raw native objectives never supply a cut intercept. A timeout
    may retain the independently certified root LB; ``exact`` requires an equal
    lower/upper endpoint and a complete native proof without relaxed closure.
    Invalid caller data raise; unavailable/failed native solves return ok=False.
    The time budget includes preparation and obeys a shared monotonic deadline.
    """
    started = time.monotonic()
    backward_gap_abs, backward_gap_rel = float(backward_gap_abs), float(backward_gap_rel)
    if any(not math.isfinite(value) or value < 0. for value in (backward_gap_abs, backward_gap_rel)):
        raise ValueError("backward query gaps must be finite and nonnegative")
    limit = float(time_limit_s)
    if not math.isfinite(limit) or limit <= 0.:
        raise ValueError('time_limit_s must be positive and finite')
    if phase not in (1, 2):
        raise ValueError('phase must be 1 or 2')
    if deadline is not None and not math.isfinite(float(deadline)):
        raise ValueError('deadline must be finite')
    if not isinstance(root_bound_only, bool):
        raise TypeError('root_bound_only must be a bool')
    data = _instance(prob_data)
    ctx = _node_context(data, node, stage=2)
    keys = _state_keys(ctx)
    if not isinstance(pi, Mapping):
        raise TypeError('multipliers must map physical availability keys to coefficients')
    if set(pi) - set(keys):
        raise ValueError(f'unknown or investment multiplier keys: {sorted(set(pi)-set(keys))}')
    multipliers = [_finite(pi.get(key, 0.)) for key in keys]
    if any(value is None for value in multipliers):
        raise ValueError('multipliers must be finite numeric coefficients')
    pools, successors = _route_pools(ctx, node, cut_lag)
    cuts = {i: basic_route_cuts(ctx, i) + [_as_cut(raw, ctx, i) for raw in pools[i]]
            for i in range(ctx.m)}
    payload, projection = project_route_cuts(cuts, range(ctx.m), ctx.active)
    guard = _sparse_envelope_guard(payload, ctx.m)
    # v6 pricing uses exact binary64 capacity comparisons. Its optional cover
    # separator instead sums demand in ordinary double precision, so enable
    # covers only where every possible nonnegative subset sum is exact.
    exact_integer_sums = (all(float(value).is_integer() and 0. <= value <= 2.**53
                              for value in ctx.demand)
                          and sum((_fraction(value) for value in ctx.demand), Fraction(0)) <= 2**53)
    diagnostic = dict(backend='bpc', context=ctx.key, native_executed=False,
        physical_facilities=ctx.m, current_interval=ctx.interval,
        cut_projection=projection, root_bound_only=root_bound_only,
        lower_source='native_repaired_root_dual', sparse_filter_guard=round_fraction_up(guard),
        capacity_safe_cover_cuts=exact_integer_sums,
        requested_backward_gap_abs=backward_gap_abs, requested_backward_gap_rel=backward_gap_rel)
    if deadline is not None and time.monotonic() >= deadline:
        return _empty(diagnostic, 'deadline_before_native')
    def setting(suffix, default):
        return os.environ.get('LRP_' + suffix, os.environ.get('VRP_' + suffix, default))
    def boolean(suffix, default):
        raw = str(setting(suffix, int(default))).strip().lower()
        if raw in ('1', 'true', 'yes', 'on'):
            return True
        if raw in ('0', 'false', 'no', 'off'):
            return False
        raise ValueError(suffix + ' must be a boolean (0/1)')
    rounds = int(setting('PHASE1_S2_BPC_ROOT_CUT_ROUNDS', 0)
        if phase == 1 and root_bound_only else setting('S2_BPC_CUT_ROUNDS', -1))
    if max_cutting_rounds is not None:
        rounds = int(max_cutting_rounds)
    if rounds < -1:
        raise ValueError('max_cutting_rounds must be -1 or nonnegative')
    search = dict(num_threads=int(setting('S2_BPC_NUM_THREADS', 1)),
        max_nodes=int(setting('S2_BP_MAX_NODES', 1000000) if max_nodes is None else max_nodes),
        max_depth=int(setting('S2_BP_MAX_DEPTH', 100000) if max_depth is None else max_depth),
        max_colgen_iters=int(setting('S2_BP_MAX_CG', 1000) if max_colgen_iters is None else max_colgen_iters),
        pricing_top_k=int(setting('S2_BP_TOPK', 5)),
        pricing_top_k_root=int(setting('S2_BPC_PRICING_TOPK_ROOT', 30)),
        pricing_top_k_shallow=int(setting('S2_BPC_PRICING_TOPK_SHALLOW', 20)),
        pricing_top_k_deep=int(setting('S2_BPC_PRICING_TOPK_DEEP', 5)),
        rc_tol=float(setting('S2_BP_RC_TOL', 1.e-7)), int_tol=float(setting('S2_BP_INT_TOL', 1.e-6)))
    if any(search[key] < 1 for key in ('num_threads', 'max_nodes', 'max_depth', 'max_colgen_iters', 'pricing_top_k',
                                     'pricing_top_k_root', 'pricing_top_k_shallow', 'pricing_top_k_deep')):
        raise ValueError('native search counts must be positive')
    if any(not math.isfinite(search[key]) or search[key] <= 0. for key in ('rc_tol', 'int_tol')):
        raise ValueError('native tolerances must be positive and finite')
    try:
        native = _load_reviewed_native()
        remaining = limit - (time.monotonic() - started)
        if deadline is not None:
            remaining = min(remaining, deadline - time.monotonic())
        if remaining <= 0.:
            return _empty(diagnostic, 'deadline_before_native')
        with backend_call('bpc', 'solve_stage2_lag', phase=phase, path='backward', stage=2) as event:
            raw = native.solve_stage2_lag(n=ctx.n, m=ctx.m, numSucc=ctx.m,
                active=[int(value) for value in ctx.active], volume=ctx.demand.tolist(),
                cOut=[float(ctx.outsourcing[j]) if ctx.active[j] else 0. for j in range(ctx.n)],
                Qv=ctx.capacity.tolist(), piZ=multipliers, cuts=payload,
                time_limit_s=remaining, solve_mode=2, root_bound_only=root_bound_only,
                backward_gap_abs=backward_gap_abs, backward_gap_rel=backward_gap_rel,
                theta_lower_bound=0., use_vehicle_clustering=False, use_purchase_order=False,
                vehicle_types=list(range(ctx.m)), max_cutting_rounds=rounds,
                use_ryan_foster=boolean('S2_BPC_USE_RYAN_FOSTER', True),
                use_heuristic_pricing=boolean('S2_BPC_USE_HEURISTIC_PRICING', True),
                use_diving=boolean('S2_BPC_USE_DIVING', True),
                use_dual_stabilization=boolean('S2_BPC_USE_DUAL_STABILIZATION', True),
                use_restricted_mip=boolean('S2_BPC_USE_RESTRICTED_MIP', True),
                restricted_mip_time_limit=float(setting('S2_BPC_RESTRICTED_MIP_TIME_LIMIT', .5)),
                use_cut_aging=boolean('S2_BPC_USE_CUT_AGING', True),
                cull_rc_threshold=float(setting('S2_BPC_CULL_RC_THRESHOLD', 10.)),
                use_sr3_cuts=boolean('S2_BPC_USE_SR3_CUTS', True),
                use_clique_cuts=boolean('S2_BPC_USE_CLIQUE_CUTS', True),
                use_cover_cuts=exact_integer_sums and boolean('S2_BPC_USE_COVER_CUTS', True), **search)
            if not isinstance(raw, dict):
                raise TypeError('native result must be a dictionary')
            event.update(native_lb_certified=raw.get('lb_certified') is True,
                         incumbent_available=raw.get('feasible') is True)
    except (NativeBPCUnavailable, OSError, RuntimeError, ValueError, TypeError, OverflowError) as exc:
        return _empty(diagnostic, f'native_unavailable_or_failed: {exc}')
    diagnostic.update(native_executed=True, raw={key: value for key, value in raw.items()
        if key not in {'alpha', 'y', 'z', 's', 'theta'}})
    if raw.get('abort_reason'):
        return _empty(diagnostic, 'native_internal_error: ' + str(raw['abort_reason']))
    answer = _empty(diagnostic, 'native_no_usable_certificate')
    if raw.get('feasible') is True:
        try:
            value, xcp, decisions = _policy(ctx, data, node, raw, multipliers, cuts, successors)
            answer.update(inner_value=value, xcp=xcp, decisions=decisions,
                          incumbent_policy_certified=True)
        except (InvalidForwardPolicy, ValueError, TypeError, KeyError, OverflowError) as exc:
            diagnostic['incumbent_rejected'] = str(exc)
    lower, root = _finite(raw.get('lb')), _finite(raw.get('root_lp_after_cuts'))
    if raw.get('lb_certified') is True and lower is not None and root is not None:
        # Native full-tree completion may replace lb by an ordinarily rounded
        # incumbent. Keep the directed repaired dual channel independently.
        lower = _down(_fraction(min(lower, root)) - guard)
        if answer['inner_value'] is None or lower <= answer['inner_value']:
            answer.update(outer_lb=lower, lb_certified=True)
        else:
            diagnostic['lower_rejected'] = 'certified_lower_exceeds_independently_audited_policy'
    answer['exact'] = bool(answer['lb_certified'] and answer['incumbent_policy_certified']
        and raw.get('optimality_proven') is True and raw.get('tree_complete') is True
        and raw.get('proof_relaxed') is False and not raw.get('timed_out')
        and answer['outer_lb'] == answer['inner_value'])
    if answer['lb_certified'] and answer['incumbent_policy_certified']:
        exact_gap = _fraction(answer['inner_value']) - _fraction(answer['outer_lb'])
        tolerance = max(_fraction(backward_gap_abs),
                        _fraction(backward_gap_rel) * abs(_fraction(answer['inner_value'])))
        answer['query_gap_closed'] = 0 <= exact_gap <= tolerance
    answer['ok'] = answer['lb_certified'] or answer['incumbent_policy_certified']
    answer['reason'] = None if answer['ok'] else answer['reason']
    return answer


__all__ = ['solve_s2_backward_with_bpc', 'NativeBPCUnavailable']
