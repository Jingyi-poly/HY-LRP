"""LRP forward assignment adapter for the original Stage-2 C++ BPC kernel.

Native assignments cross this boundary only after policy audit. Native bounds
remain diagnostic. A separately verified full-LP reference may supply a fixed-A
envelope lower bound; it never supplies an integer optimality certificate.
Every returned policy is audited and scored against the complete original cut
archive with exact binary64 arithmetic.  Physical facilities are distinct.
"""
from __future__ import annotations

from fractions import Fraction
import importlib.util
import math
import os
from pathlib import Path
import sysconfig
import time

from core.backend_telemetry import backend_call
from core.solver_settings import forward_s2_bpc_policy
from models.stage_builder import (_as_cut, _instance, _node_context,
                                  _route_pools, _state_keys, _state_values)
from models.stage_model_core import route_degree_bounds
from solvers.forward_policy_certification import (InvalidForwardPolicy,
                                                 certify_stage2_forward_policy)
from solvers.forward_ub import round_fraction_up
from solvers.lrp_bpc_cut_projection import project_route_cuts


class NativeBPCUnavailable(RuntimeError):
    """The original native extension cannot be safely loaded in this runtime."""


_NATIVE_CACHE = {}


def _load_native():
    # Do not import Investment exact_subroutines: it initializes other backends
    # and its legacy cut payload uses a different alpha ordering and y names.
    directory = Path(os.environ.get('LRP_S2_BPC_DIR', os.environ.get('VRP_S2_BPC_DIR',
        str(Path(__file__).resolve().parents[1] / 'customized-subprob/s2backward/bpc'))))
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if not suffix:
        raise NativeBPCUnavailable('Python extension ABI suffix is unavailable')
    binary = directory / ('stage2_bp_cpp' + suffix)
    if not binary.is_file():
        raise NativeBPCUnavailable(f'missing current-ABI C++ BPC extension: {binary}')
    sources = [directory / 'stage2_branch_price.cpp', directory / 'stage2_bp_pybind.cpp',
               directory / 'build.sh', *directory.glob('*.h'), *directory.glob('*.hpp')]
    if any(not path.is_file() or path.stat().st_mtime_ns > binary.stat().st_mtime_ns
           for path in sources):
        raise NativeBPCUnavailable('C++ BPC sources are missing or newer than the extension; rebuild it')
    key = (str(binary.resolve()), binary.stat().st_mtime_ns)
    if key in _NATIVE_CACHE:
        return _NATIVE_CACHE[key]
    try:
        spec = importlib.util.spec_from_file_location('stage2_bp_cpp', binary)
        if spec is None or spec.loader is None:
            raise ImportError('no extension loader')
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
    except (ImportError, OSError, RuntimeError) as exc:
        raise NativeBPCUnavailable(f'C++ BPC import failed: {exc}') from exc
    if not callable(getattr(native, 'solve_stage2_lag', None)):
        raise NativeBPCUnavailable('C++ BPC extension lacks solve_stage2_lag')
    _NATIVE_CACHE[key] = native
    return native


def _cuts(ctx, node, cut_lag, available):
    from cuts.lrp_static_bounds import basic_route_cuts
    pools, successors = _route_pools(ctx, node, cut_lag)
    validated = {}
    for i in range(ctx.m):
        validated[i] = (basic_route_cuts(ctx, i)
                        + [_as_cut(raw, ctx, i) for raw in pools[i]])
    # Only exact epigraph equivalences under fixed-zero state are removed.
    # Keep the complete validated archive for independent policy rescoring.
    payload, projection = project_route_cuts(validated, available, ctx.active)
    return validated, payload, successors, projection


def _independent_reference_lower_bound(lower, diagnostic, upper):
    """Accept only the canonical full-LP certificate paired with this primal."""
    if (diagnostic.get('certified') is not True
            or diagnostic.get('source') != 'full_assignment_LP_exact_dual_audit'
            or diagnostic.get('matrix_source') != 'canonical_stage_specification'):
        return None
    values = (lower, diagnostic.get('lower_bound'), upper)
    if any(isinstance(v, (bool, str, bytes)) for v in values):
        return None
    try:
        bound, recorded, objective = map(float, values)
    except (TypeError, ValueError, OverflowError):
        return None
    if (not all(math.isfinite(v) for v in (bound, recorded, objective))
            or bound != recorded or bound > objective):
        return None
    return bound


def _score(prob_data, node, ctx, x_prev, decisions, cuts, successors, diagnostic,
           reference_lower_bound=None):
    normalized, stage_cost = certify_stage2_forward_policy(prob_data, node, x_prev, decisions)
    exact_cost = sum((Fraction.from_float(float(ctx.outsourcing[j]))
                      for j in range(ctx.n) if normalized[f'e[{j}]']), Fraction(0))
    # Physical audit above normalizes every state to 0/1; all AffineCut
    # coefficients are finite. Zero coordinates contribute exactly zero,
    # so avoid constructing and multiplying their rational coefficients.
    exact_theta = {}
    for i in range(ctx.m):
        state = [int(normalized[f'alpha[{i},{j}]']) for j in range(ctx.n)]
        state.append(int(normalized[f'u[{i}]']))
        exact_theta[i] = max([Fraction(0)] + [
            Fraction.from_float(float(cut.intercept)) + sum(
                (Fraction.from_float(float(c)) for c, bit in zip(cut.coefficients, state) if bit),
                Fraction(0)) for cut in cuts[i]])
    normalized.update({key: value for key, value in x_prev.items() if str(key).startswith('A[')})
    normalized.update({f'z[{i}]': float(x_prev[f'A[{i},{ctx.interval}]']) for i in range(ctx.m)})
    normalized.update({f'theta[{successors[i]}]': round_fraction_up(value)
                       for i, value in exact_theta.items()})
    # The original refresh handoff reads a complete compact-model vector,
    # including its zero-cost route_base equality auxiliaries.  These do not
    # alter the physical policy, native payload, or objective computed above.
    for i in range(ctx.m):
        degree = route_degree_bounds(ctx, i)
        coefficients = (*degree.incoming, degree.return_cost)
        if any(coefficients):
            state = [int(normalized[f'alpha[{i},{j}]']) for j in range(ctx.n)]
            state.append(int(normalized[f'u[{i}]']))
            normalized[f'route_base[{i}]'] = float(sum(
                (Fraction.from_float(value) for value, bit in zip(coefficients, state) if bit),
                Fraction(0)))
    normalized['stage_cost'] = stage_cost
    result = dict(ok=True, x=normalized, stage_cost=stage_cost,
                objective=round_fraction_up(exact_cost + sum(exact_theta.values(), Fraction(0))),
                diagnostic=dict(diagnostic, policy_certified=True,
                    complete_archive_rescored=True, optimality_claimed=False))
    reference = _independent_reference_lower_bound(reference_lower_bound,
        diagnostic.get('reference_lp', {}), result['objective'])
    if reference is not None:
        result['lower_bound'] = reference
        result['diagnostic']['lower_bound_source'] = 'independent_reference_lp'
    return result


def solve_s2_forward_with_bpc(prob_data, node, cut_lag, x_prev, *, time_limit_s,
                              phase=1, deadline=None, forward_gap=None, stage_builder=None):
    """Return an audited feasible trial, or an explicit reason for fallback.

    ``deadline`` uses ``time.monotonic()``.  Invalid input/cut domains raise;
    expected native unavailability or malformed native policies return ok=False.
    No native LB/optimality flag is exposed as an algorithm certificate.
    Native budget checks also cover pricing and primal search. The passed
    remaining budget is still a soft limit, not a hard process interruption.
    """
    started = time.monotonic()
    limit = float(time_limit_s)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('time_limit_s must be positive and finite')
    if phase not in (1, 2):
        raise ValueError('phase must be 1 or 2')
    if deadline is not None and not math.isfinite(float(deadline)):
        raise ValueError('deadline must be finite')
    policy = forward_s2_bpc_policy(phase)
    gap = policy['forward_gap'] if forward_gap is None else float(forward_gap)
    if not math.isfinite(gap) or gap < 0:
        raise ValueError('forward_gap must be finite and nonnegative')
    ctx = _node_context(_instance(prob_data), node, stage=2)
    availability = _state_values(x_prev, _state_keys(ctx), 'facility state')
    available = [i for i, bit in enumerate(availability) if bit]
    cuts, payload, successors, projection = _cuts(ctx, node, cut_lag, available)
    diagnostic = dict(backend='bpc', available_facilities=available,
                      cuts=projection['original_count'], native_cuts=len(payload),
                      cut_projection=projection,
                      physical_facilities=ctx.m, native_executed=False)
    empty = {f'alpha[{i},{j}]': 0. for i in range(ctx.m) for j in range(ctx.n)}
    empty.update({f'u[{i}]': 0. for i in range(ctx.m)})
    empty.update({f'e[{j}]': float(ctx.active[j]) for j in range(ctx.n)})
    if not available or not any(ctx.active):
        return _score(prob_data, node, ctx, x_prev, empty, cuts, successors,
                      dict(diagnostic, reason='analytic_empty_assignment'))
    try:
        native = _load_native()
    except NativeBPCUnavailable as exc:
        return dict(ok=False, reason=str(exc), diagnostic=diagnostic)
    native_options = {}
    reference = None
    if (getattr(native, 'forward_anytime_contract', None) == 'lrp_forward_anytime_v1'
            and policy['reference_lp'] and gap > 0.):
        from solvers.forward_s2_reference import forward_s2_reference_bound
        reference_deadline = started + limit
        if deadline is not None:
            reference_deadline = min(reference_deadline, deadline)
        # Leave most of the shared budget for actual primal search, even in a
        # short diagnostic. Model construction and certification are included.
        allowance = min(policy['reference_lp_seconds'],
                        max(0., reference_deadline-time.monotonic()) * .2)
        if allowance > 0:
            reference, reference_diagnostic = forward_s2_reference_bound(
                prob_data, node, cut_lag, x_prev, phase=phase,
                time_limit_s=allowance, deadline=reference_deadline,
                stage_builder=stage_builder)
            diagnostic['reference_lp'] = reference_diagnostic
            if reference is not None:
                native_options['forward_reference_lb'] = reference
    remaining = limit - (time.monotonic() - started)
    if deadline is not None:
        remaining = min(remaining, float(deadline) - time.monotonic())
    if remaining <= 0:
        return dict(ok=False, reason='deadline_before_native', diagnostic=diagnostic)
    try:
        with backend_call('bpc', 'solve_stage2_lag', phase=phase, path='forward', stage=2) as event:
            raw = native.solve_stage2_lag(
                n=ctx.n, m=len(available), numSucc=ctx.m,
                active=[int(v) for v in ctx.active], volume=ctx.demand.tolist(),
                cOut=[float(ctx.outsourcing[j]) if ctx.active[j] else 0. for j in range(ctx.n)],
                Qv=[float(ctx.capacity[i]) for i in available], piZ=[0.] * len(available),
                cuts=payload, time_limit_s=remaining, num_threads=policy['threads'], solve_mode=1,
                pricing_top_k=policy['pricing_top_k'], max_nodes=policy['max_nodes'],
                max_depth=policy['max_depth'], max_colgen_iters=policy['max_colgen_iters'],
                rc_tol=policy['rc_tol'], int_tol=policy['int_tol'],
                pricing_top_k_root=15, pricing_top_k_shallow=10, pricing_top_k_deep=5,
                cull_rc_threshold=0., use_heuristic_pricing=True, use_diving=True,
                use_dual_stabilization=True, use_restricted_mip=True,
                restricted_mip_time_limit=1., use_cut_aging=True,
                forward_gap=gap, theta_lower_bound=0.,
                use_vehicle_clustering=False, use_ryan_foster=True,
                use_purchase_order=False, vehicle_types=list(range(len(available))),
                **native_options)
            if not isinstance(raw, dict):
                raise TypeError('native BPC result must be a dictionary')
            event.update(incumbent_available=bool(raw.get('feasible')))
        diagnostic.update(native_executed=True,
            native_objective=raw.get('obj'), native_timed_out=raw.get('timed_out'),
            native_termination_reason=raw.get('termination_reason'),
            native_seconds=raw.get('t_solve'), native_nodes=raw.get('nodes_processed'))
        diagnostic['native_forward_reference_stop'] = raw.get('forward_reference_stop', False)
        diagnostic['native_cg_primal_calls'] = raw.get('forward_cg_primal_calls', 0)
        diagnostic['native_cg_primal_improvements'] = raw.get('forward_cg_primal_improvements', 0)
        # Native bounds remain diagnostics, even if native claims exactness.
        # Only the independently audited full-LP reference may cross below.
        diagnostic['native_reported_bound'] = raw.get('lb')
        diagnostic['native_reported_bound_certified'] = raw.get('lb_certified', False)
        diagnostic['native_rounding_improvements'] = raw.get('forward_rounding_improvements', 0)
        diagnostic['native_partial_bound_certificates'] = raw.get('forward_partial_bound_certificates', 0)
        diagnostic['native_approximate_cg_returns'] = raw.get('forward_approximate_cg_returns', 0)
        if raw.get('abort_reason'):
            return dict(ok=False, reason='native_internal_error: ' + str(raw['abort_reason']), diagnostic=diagnostic)
        if not raw.get('feasible'):
            return dict(ok=False, reason='native_no_feasible_policy', diagnostic=diagnostic)
        if raw.get('timed_out') and not policy['accept_timeout']:
            return dict(ok=False, reason='native_timeout_rejected', diagnostic=diagnostic)
        alpha, dispatch = raw['alpha'], raw['y']
        if len(alpha) != len(available) or len(dispatch) != len(available) or any(len(row) != ctx.n for row in alpha):
            raise InvalidForwardPolicy('native assignment shape mismatch')
        for pos, i in enumerate(available):
            empty[f'u[{i}]'] = dispatch[pos]
            for j in range(ctx.n):
                empty[f'alpha[{i},{j}]'] = alpha[pos][j]
        # Certification checks every returned alpha/u bit before these rounded
        # totals can be accepted.  Native s=1 on inactive nodes is not LRP e=1.
        for j in range(ctx.n):
            empty[f'e[{j}]'] = int(ctx.active[j]) - sum(float(row[j]) for row in alpha)
        return _score(prob_data, node, ctx, x_prev, empty, cuts, successors, diagnostic,
                      reference_lower_bound=reference)
    except (InvalidForwardPolicy, RuntimeError, ValueError, TypeError, KeyError, OverflowError) as exc:
        return dict(ok=False, reason=f'native_policy_rejected: {type(exc).__name__}: {exc}',
                    diagnostic=dict(diagnostic, native_executed=True))


__all__ = ['solve_s2_forward_with_bpc', 'NativeBPCUnavailable']
