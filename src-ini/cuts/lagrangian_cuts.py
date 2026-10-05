"""LRP Lagrangian cuts using the original norm/level/bundle algorithm."""
import math
import os
import time
from copy import copy, deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Optional
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from cuts.cut_manager import CutManager
from core.exact_solver_log import log as exact_log
from core.backend_telemetry import backend_call, backend_scope, record_backend_event
from core.solve_deadline import SolveDeadlineReached, bounded_solve_time
from core.s2_levelset_budget import S2LevelSetBudget
from core.solver_settings import (configured_gurobi_threads, phase2_oracle_mip_gap,
                                 configured_backward_s3_backend,
                                 configured_backward_s2_backend,
                                 configured_s3_native_cap, configured_s3_native_time_limit,
                                 configured_s3_native_options)
from models.subproblem_builder import SubproblemBuilder
from models.stage_builder import StageModelBuilder, _instance, _node_context, _state_keys
from models.stage_model_core import evaluate_model, tour_from_evaluation, InvalidSolverPrimal
from solvers.lrp_native_oracle import LRPNativeRouteOracle, NativeUnavailable
from solvers.forward_policy_certification import certify_stage3_forward_policy, InvalidForwardPolicy
from solvers.free_route_lp_separation import separate_free_route_lp, matrix_audit_reserve
def remaining_seconds(deadline):
    """Remaining wall time; ``None`` preserves an unbudgeted call."""
    if deadline is None:
        return math.inf
    value = float(deadline)
    if not math.isfinite(value):
        raise ValueError("solve deadline must be finite or None")
    return max(0.0, value - time.monotonic())


def clipped_time_limit(time_limit, deadline):
    """Never grant a new solve more than the node's remaining budget."""
    if deadline is None:
        return time_limit
    remaining = remaining_seconds(deadline)
    if time_limit is None:
        return remaining
    limit = float(time_limit)
    if math.isnan(limit) or limit < 0.0:
        raise ValueError("solve time limit must be nonnegative")
    return min(limit, remaining)


def prepare_model_solve(model, deadline, *, time_limit=None):
    """Clip a model's next optimize call, or return False without solving.

    With no deadline this does not modify model parameters. Gurobi stops
    cooperatively: checking the deadline does not kill a solver or a worker.
    """
    if deadline is None:
        return True
    if time_limit is None:
        try:
            time_limit = float(model.Params.TimeLimit)
        except (AttributeError, TypeError, ValueError):
            time_limit = math.inf
    limit = clipped_time_limit(time_limit, deadline)
    if limit <= 0.0:
        return False
    model.setParam("TimeLimit", limit)
    return True


def _evaluate_fixed_target(model, *, stage_no, counters, **options):
    """Retry a rejected S2 target once, sharing its original allowance.

    A rejected primal produces no Evaluation, hence no target certificate.
    Never read its raw objective or bounds. Other errors and S3 retain their
    original behavior; a normal first call receives exactly its old options.
    """
    def evaluate(solve_options):
        with backend_call('gurobi', 'level_target_fixed_mip', model=model):
            return evaluate_model(model, **solve_options)

    started = time.monotonic()
    try:
        return evaluate(options), None
    except InvalidSolverPrimal as rejected:
        if stage_no != 2:
            raise
        record = dict(original_rejection_type=type(rejected).__name__,
            original_rejection_reason=str(rejected), retry_count=0,
            retry_reset_performed=False)
        counters['s2_target_primal_rejections'] = counters.get('s2_target_primal_rejections', 0) + 1

    allowance = float(options['time_limit'])
    stop = started + allowance
    if options.get('deadline') is not None:
        stop = min(stop, options['deadline'])
    try:
        remaining = bounded_solve_time(allowance, stop)
        model.reset()
        model.Params.Presolve = 1
        record['retry_reset_performed'] = True
        remaining = bounded_solve_time(remaining, stop)
        record.update(retry_requested_seconds=remaining, retry_deadline=stop)
        result = evaluate(dict(options, time_limit=remaining, deadline=stop))
    except SolveDeadlineReached:
        record['retry_status'] = 'deadline_before_retry_optimize'
        return None, record
    except InvalidSolverPrimal as rejected:
        counters['s2_target_primal_rejections'] += 1
        record.update(retry_status='invalid_solver_primal',
                      retry_rejection_reason=str(rejected))
        result = None
    else:
        record['retry_status'] = result.report.get('status')
    record['retry_count'] = 1
    for key in ('target_solves', 'gurobi_target_solves', 's2_target_primal_retries'):
        counters[key] = counters.get(key, 0) + 1
    return result, record


def _evaluate_lagrangian_oracle(model, *, stage_no, counters, **options):
    """Recover a rejected S2 oracle within the same query's allowance.

    Invalid primals have no accepted Evaluation. Do not use their raw bounds
    or states in a cut or bundle support, including when retry time runs out.
    """
    def rejected_solve():
        for key in ('oracle_solves', 'gurobi_oracle_solves', 's2_oracle_primal_rejections'):
            counters[key] = counters.get(key, 0) + 1

    started = time.monotonic()
    try:
        return evaluate_model(model, **options), None
    except InvalidSolverPrimal as rejected:
        if stage_no != 2:
            raise
        rejected_solve()
        record = dict(original_rejection_type=type(rejected).__name__,
            original_rejection_reason=str(rejected), retry_count=0,
            retry_reset_performed=False)

    allowance = float(options['time_limit'])
    stop = started + allowance
    if options.get('deadline') is not None:
        stop = min(stop, options['deadline'])
    try:
        remaining = bounded_solve_time(allowance, stop)
        model.reset()
        model.Params.Presolve = 1
        record['retry_reset_performed'] = True
        remaining = bounded_solve_time(remaining, stop)
        record.update(retry_requested_seconds=remaining, retry_deadline=stop)
        result = evaluate_model(model, **dict(options, time_limit=remaining, deadline=stop))
    except SolveDeadlineReached:
        record['retry_status'] = 'deadline_before_retry_optimize'
        return None, record
    except InvalidSolverPrimal as rejected:
        rejected_solve()
        record.update(retry_status='invalid_solver_primal',
                      retry_rejection_reason=str(rejected))
        result = None
    else:
        record['retry_status'] = result.report.get('status')
    record['retry_count'] = 1
    counters['s2_oracle_primal_retries'] = counters.get('s2_oracle_primal_retries', 0) + 1
    # Successful returned solves are counted by the original caller below.
    return result, record


_CERT_CHECKPOINT_MAX_ROUNDS = 4
_CERT_CHECKPOINT_GAP_ABS_TOL = 1e-6
_CERT_CHECKPOINT_GAP_REL_TOL = 0.0
_CERT_CHECKPOINT_PI_ABS_TOL = 1e-8
_CERT_CHECKPOINT_PI_REL_TOL = 1e-8
def _levelset_trace_enabled():
    # Resolve per call: persistent workers receive a new environment per job.
    # Default off: Phase 2 should look like Phase 1 (outer LB/UB lines only).
    # Set LRP_LEVELSET_TRACE=1 / VRP_LEVELSET_TRACE=1 for per-node LS/checkpoint detail.
    raw = os.environ.get('LRP_LEVELSET_TRACE', os.environ.get('VRP_LEVELSET_TRACE', '0'))
    return raw not in ('0', 'false', 'False', '')


def _levelset_detail_log(direction, stage, backend, message, *, indent=6):
    if _levelset_trace_enabled():
        exact_log(direction, stage, backend, message, indent=indent)


def _configured_native_probe_seconds():
    """Per-free-oracle native allowance, before the existing Gurobi fallback."""
    from core.solver_settings import configured_native_probe_seconds
    return configured_native_probe_seconds()


def _set_levelset_tolerances(model, level_tol):
    """Keep LP/QP feasibility error below the requested Level Set accuracy.

    Gurobi's default 1e-6 primal tolerance can admit an unchanged projection
    whose level violation exceeds the usual 1e-7 Delta stopping tolerance.
    The 1e-9 floor is Gurobi's minimum primal/dual feasibility tolerance;
    clipping here never changes Delta or the separate oracle certificate.
    """
    model.Params.Threads = configured_gurobi_threads()
    level_tol = float(level_tol)
    if not math.isfinite(level_tol) or level_tol <= 0.:
        raise ValueError('Level Set tolerance must be finite and positive')
    solver_tol = max(1e-9, min(1e-6, .01 * level_tol))
    for name in ('FeasibilityTol', 'OptimalityTol', 'BarConvTol', 'BarQCPConvTol'):
        # Preserve a stricter caller/environment setting, including barrier
        # convergence for the squared-L2 master and quadratic level row.
        model.setParam(name, min(float(getattr(model.Params, name)), solver_tol))


def _optimize_gurobi(model, operation, optimize=None):
    """Count an actual optimize, separately from native route queries."""
    with backend_call("gurobi", operation, model=model):
        return model.optimize() if optimize is None else optimize(model)


def _incumbent_cut_slack_at_trial(L_value, pi_value, x_prev, inner_value, pi_keys):
    """``L - (V_inner + pi.x)``; None without a feasible incumbent value."""
    if inner_value is None or not np.isfinite(inner_value):
        return None
    cut_at_trial = float(inner_value) + sum(
        float(pi_value[k]) * float(x_prev[k]) for k in pi_keys
    )
    return float(L_value) - cut_at_trial


def _is_tight_exit_slack(slack, tolerance):
    """Return whether a finite signed residual is close to zero.

    The residual is signed: a large negative value means the incumbent
    support and the Level-Set value disagree just as materially as a large
    positive value.  Never interpret ``slack <= tolerance`` as tightness.
    """
    return bool(
        tolerance is not None
        and slack is not None
        and np.isfinite(slack)
        and abs(float(slack)) <= float(tolerance)
    )


def _tight_exit_tolerance(L_value):
    """Original fixed anchor tolerance; two nonpositive settings disable it.

    Resolve per call so persistent workers respect the job's legacy VRP
    switches and LRP aliases. Outer-gap-scaled widening is not used here.
    """
    def setting(suffix):
        value = float(os.environ.get('LRP_LEVELSET_TIGHT_EXIT_' + suffix,
            os.environ.get('VRP_LEVELSET_TIGHT_EXIT_' + suffix, '1e-6')))
        if not math.isfinite(value):
            raise ValueError('Level Set tight-exit tolerances must be finite')
        return value
    absolute, relative = setting('ABS'), setting('REL')
    if absolute <= 0. and relative <= 0.:
        return None
    return max(absolute, relative * max(1., abs(float(L_value))))


def _scheduled_gap_setting(suffix, default):
    value = float(os.environ.get('LRP_LEVELSET_TIGHT_EXIT_' + suffix,
        os.environ.get('VRP_LEVELSET_TIGHT_EXIT_' + suffix, default)))
    if not math.isfinite(value):
        raise ValueError('Scheduled gap effort settings must be finite')
    return value


def outer_gap_tight_exit_abs(outer_gap_abs):
    """Final's S2 effort allowance; no oracle exactness or norm proof changes."""
    fraction = _scheduled_gap_setting('GAP_FRAC', '.002')
    if fraction <= 0. or outer_gap_abs is None:
        return None
    gap = float(outer_gap_abs)
    if not math.isfinite(gap) or gap <= 0.:
        return None
    value = fraction * gap
    return value if math.isfinite(value) else None


def _scheduled_gap_effort_tolerance(original_target, extra_abs):
    """Keep the original fixed rule unchanged; cap only a supplied allowance."""
    fixed = _tight_exit_tolerance(original_target)
    if fixed is None or extra_abs is None:
        return None
    extra = float(extra_abs)
    if not math.isfinite(extra) or extra <= 0.:
        return None
    cap = _scheduled_gap_setting('GAP_REL_CAP', '.01')
    capped = min(extra, cap * max(1., abs(float(original_target))))
    # A nonpositive cap or an allowance inside the fixed band adds no rule.
    # The separate original fixed-tolerance exits remain responsible there.
    return max(fixed, capped) if capped > fixed else None


def _certified_scheduled_gap_pair(stage, original_target, pi, trial, evaluation,
                                  shifts, keys, extra_abs):
    """Return this physical certified pair for an S2 effort exit only.

    Exact signed slack is measured against the ORIGINAL fixed-state lower
    target, never a bisected residual level. An incumbent upper endpoint is
    irrelevant. This does not establish fixed-band target or norm feasibility.
    """
    if stage != 2:
        return None
    tolerance = _scheduled_gap_effort_tolerance(original_target, extra_abs)
    if tolerance is None or not evaluation.has_outer:
        return None
    physical_pi, physical_eval = _lift_centered_evaluation(pi, evaluation, shifts)
    slack = _certified_anchor_tight_slack(original_target, physical_pi, trial,
        physical_eval.outer_lb, keys, tolerance)
    if slack is None:
        return None
    return physical_pi, physical_eval, slack, tolerance


def _certified_anchor_tight_slack(L_value, pi, trial, outer_lb, keys, tolerance):
    """Return exact signed slack only for a certified near-anchor endpoint.

    This is a search stopping rule, not a minimum-norm certificate. The
    caller supplies the oracle's certified lower channel; an incumbent upper
    support cannot establish this condition. Both signs must be close to zero.
    """
    if outer_lb is None or tolerance is None or tolerance < 0.:
        return None
    values = [L_value, outer_lb, tolerance]
    values.extend(value for key in keys for value in (pi[key], trial[key]))
    if not all(math.isfinite(float(value)) for value in values):
        return None
    slack = Fraction.from_float(float(L_value)) - Fraction.from_float(float(outer_lb))
    for key in keys:
        slack -= Fraction.from_float(float(pi[key])) * Fraction.from_float(float(trial[key]))
    return slack if abs(slack) <= Fraction.from_float(float(tolerance)) else None


def _certified_level_target_attained(L_value, pi, trial, outer_lb, keys, tolerance):
    """Prove feasibility of a lower-bound level using the outer certificate.

    The fixed-state target can be strictly below the physical optimum. A
    certified cut above that target is therefore valid, even if its signed
    residual is not close to zero. Incumbent upper supports cannot prove this.
    Exact binary64 arithmetic avoids cancellation in the anchor evaluation.
    This is target feasibility, not minimum-norm or physical optimality.
    """
    if outer_lb is None or not math.isfinite(float(outer_lb)):
        return False
    if not math.isfinite(float(L_value)) or not math.isfinite(float(tolerance)) or tolerance < 0:
        return False
    anchor = Fraction.from_float(float(outer_lb))
    for key in keys:
        if not math.isfinite(float(pi[key])) or not math.isfinite(float(trial[key])):
            return False
        anchor += Fraction.from_float(float(pi[key])) * Fraction.from_float(float(trial[key]))
    return anchor >= Fraction.from_float(float(L_value))-Fraction.from_float(float(tolerance))


def _select_levelset_result(pi, evaluation, records, trial, keys, *, checkpoint_proven,
                            effective_target=None, norm_option=1, selection_diagnostic=None):
    """Keep proofs paired; prefer the best certified norm-feasible incumbent.

    An oracle lower certificate at the same multiplier proves target
    feasibility. With no norm proof, select the smallest residual norm among
    strictly target-feasible recorded pairs; this does not prove optimality.
    If no such pair exists (or no target was supplied by a legacy caller),
    retain the strongest certified anchor fallback. A successful checkpoint
    always keeps its own multiplier and evaluation.
    """
    if norm_option not in (1, 2):
        raise ValueError('The residual norm option must be 1 or 2')
    if effective_target is not None and not math.isfinite(float(effective_target)):
        raise ValueError('The effective residual target must be finite')

    def norm_at(point):
        values = [Fraction.from_float(float(point[key])) for key in keys]
        return sum((abs(value) if norm_option == 1 else value*value
                    for value in values), Fraction())

    def report(reason, feasible_count):
        if selection_diagnostic is not None:
            selection_diagnostic.update(reason=reason,
                effective_residual_target=effective_target, norm_option=norm_option,
                strict_target_feasible_records=feasible_count,
                selected_residual_norm=float(norm_at(pi)))

    if checkpoint_proven:
        if not evaluation.has_outer:
            raise ValueError('A proved norm checkpoint requires its oracle lower certificate')
        report('checkpoint_pair_preserved', None)
        return pi, evaluation
    eligible = [(p, ev) for p, ev in records if ev.has_outer]
    feasible = []
    if effective_target is not None:
        target_exact = Fraction.from_float(float(effective_target))
        trial_exact = {key: Fraction.from_float(float(trial[key])) for key in keys}
        for point, candidate in eligible:
            anchor = Fraction.from_float(float(candidate.outer_lb)) + sum(
                (Fraction.from_float(float(point[key]))*trial_exact[key] for key in keys), Fraction())
            if anchor >= target_exact:
                feasible.append((norm_at(point), anchor, point, candidate))
    if feasible:
        _, _, pi, evaluation = min(feasible, key=lambda item: (item[0], -item[1]))
        report('minimum_norm_strict_target_feasible', len(feasible))
    elif eligible:
        pi, evaluation = max(eligible,
            key=lambda pair: pair[1].outer_lb+math.fsum(pair[0][key]*trial[key] for key in keys))
        report('maximum_certified_anchor', 0)
    else:
        report('no_outer_certificate', 0)
    return pi, evaluation


@dataclass(frozen=True)
class _OracleEval:
    """Information certified by one oracle evaluation at one fixed ``pi``.

    ``inner_value`` and ``inner_xcp`` are an inseparable feasible-incumbent
    pair.  They alone may support the Level-Set bundle.  ``outer_lb`` is an
    independent lower-bound certificate and may support an outer Lagrangian
    cut even when no incumbent was found.
    """

    inner_value: Optional[float] = None
    inner_xcp: Optional[Mapping[str, float]] = None
    outer_lb: Optional[float] = None
    needs_outer_fallback: bool = False
    exact: bool = False
    status: str = "unknown"
    source: str = "unknown"

    @property
    def has_inner(self):
        return self.inner_value is not None and self.inner_xcp is not None

    @property
    def has_outer(self):
        return self.outer_lb is not None and np.isfinite(self.outer_lb)


def build_s2_cut_diagnostic(*, node, L, pi, x_trial, V_inner, V_outer,
                            actual_iterations, stop_reason, backend,
                            checkpoint_requested=False,
                            checkpoint_proven=False,
                            selected_is_checkpoint=False,
                            tight_exit=False,
                            budget_enabled=False,
                            budget_exhausted=False,
                            cut_refine_attempted=False,
                            cut_refine_gap_before=None,
                            cut_refine_gap_after=None):
    """Return a JSON/pickle-safe account of one selected S2->S1 endpoint."""
    selected_pi = {
        str(key): float(value)
        for key, value in sorted(dict(pi or {}).items())
    }
    trial = dict(x_trial or {})
    pi_dot_trial = sum(
        value * float(trial.get(key, 0.0))
        for key, value in selected_pi.items()
    )
    level = _finite_scalar_or_none(L)
    inner = _finite_scalar_or_none(V_inner)
    outer = _finite_scalar_or_none(V_outer)

    def _difference(left, right):
        return None if left is None or right is None else float(left - right)

    return {
        "node": int(node) if isinstance(node, (int, np.integer)) else str(node),
        "L": level,
        "selected_pi": selected_pi,
        "V_inner": inner,
        "V_outer": outer,
        "pi_dot_trial": pi_dot_trial,
        "cut_at_trial": None if outer is None else float(outer + pi_dot_trial),
        "incumbent_at_trial": None if inner is None else float(inner + pi_dot_trial),
        "oracle_gap": _difference(inner, outer),
        "dual_gap": (
            None if level is None or inner is None
            else float(level - (inner + pi_dot_trial))
        ),
        "emitted_gap": (
            None if level is None or outer is None
            else float(level - (outer + pi_dot_trial))
        ),
        "actual_iterations": int(actual_iterations),
        "stop_reason": str(stop_reason),
        "checkpoint_requested": bool(checkpoint_requested),
        "checkpoint_proven": bool(checkpoint_proven),
        "selected_is_checkpoint": bool(selected_is_checkpoint),
        "tight_exit": bool(tight_exit),
        "budget_enabled": bool(budget_enabled),
        "budget_exhausted": bool(budget_exhausted),
        "cut_refine_attempted": bool(cut_refine_attempted),
        "cut_refine_gap_before": _finite_scalar_or_none(
            cut_refine_gap_before
        ),
        "cut_refine_gap_after": _finite_scalar_or_none(
            cut_refine_gap_after
        ),
        "backend": str(backend),
    }


def _finite_scalar_or_none(value):
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(scalar) or abs(scalar) >= 0.5 * float(GRB.INFINITY):
        return None
    return scalar


def _canonical_finite_float_hex(value):
    """Lossless binary64 key, with mathematically identical zeros unified."""
    scalar = _finite_scalar_or_none(value)
    if scalar is None:
        raise ValueError("value must be a finite binary64 scalar")
    if scalar == 0.0:
        scalar = 0.0
    return scalar.hex()


def _make_oracle_eval(*, inner_value=None, inner_xcp=None, outer_lb=None,
                      needs_outer_fallback=False, exact=False,
                      status="unknown", source="unknown",
                      exact_abs_tol=_CERT_CHECKPOINT_GAP_ABS_TOL):
    """Normalize backend data without inventing a missing incumbent/xcp.

    ``exact_abs_tol`` is the absolute inner/outer gap still counted as exact
    (the fixed certificate band, or the scheduled Stage-2 tolerance).
    """
    inner = _finite_scalar_or_none(inner_value)
    xcp = None
    if inner is not None and inner_xcp is not None:
        try:
            candidate = {str(k): float(v) for k, v in dict(inner_xcp).items()}
        except (TypeError, ValueError, OverflowError):
            candidate = None
        if candidate is not None and all(np.isfinite(v) for v in candidate.values()):
            xcp = candidate
    if xcp is None:
        inner = None

    outer = _finite_scalar_or_none(outer_lb)
    if inner is not None and outer is not None:
        if outer > inner:
            # Drop outer if outer > inner
            outer = None

    exact_tol = float(exact_abs_tol) + _CERT_CHECKPOINT_GAP_REL_TOL * max(
        1.0,
        abs(inner) if inner is not None else 0.0,
        abs(outer) if outer is not None else 0.0,
    )
    exact_proven = bool(
        exact and inner is not None and outer is not None
        and inner - outer <= exact_tol
    )

    return _OracleEval(
        inner_value=inner,
        inner_xcp=xcp,
        outer_lb=outer,
        needs_outer_fallback=bool(needs_outer_fallback and outer is None),
        exact=exact_proven,
        status=str(status),
        source=str(source),
    )


def _complete_xcp_or_none(xcp, expected_keys):
    """Return exactly the expected finite xcp payload, or reject it."""
    try:
        payload = dict(xcp)
        result = {key: float(payload[key]) for key in expected_keys}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return result if all(np.isfinite(v) for v in result.values()) else None


def _complete_binary_xcp_or_none(xcp, expected_keys):
    """Return a complete exact-binary payload, or fail closed.

    Stage-2 BPC's current pybind ABI returns integer ``z`` values.  A
    near-binary float therefore signals an old/corrupt boundary payload, not a
    mathematical incumbent.  Its independently certified relaxation LB can
    still be retained, but the point must never enter the Level-Set bundle.
    """
    result = _complete_xcp_or_none(xcp, expected_keys)
    if result is None or any(value not in (0.0, 1.0) for value in result.values()):
        return None
    return result


def _conservative_incumbent_endpoint(rebuilt_endpoint, raw_obj_val):
    """Enclose a certified policy cost without trusting raw ``ObjVal``.

    ``rebuilt_endpoint`` is already a directed-up feasible-policy value.  A
    solver can evaluate its auxiliary objective a few ulps above or below that
    independently reconstructed number.  Taking the maximum cannot invalidate
    the feasible upper bound, avoids pairing a numerical lower bound with a
    smaller endpoint, and never lets a downward raw objective weaken the
    independent certificate.
    """
    rebuilt = _finite_scalar_or_none(rebuilt_endpoint)
    raw = _finite_scalar_or_none(raw_obj_val)
    if rebuilt is None or raw is None:
        raise ValueError("incumbent endpoints must be finite binary64 values")
    return max(rebuilt, raw)


def _bundle_support_from_incumbent(inner_value, inner_xcp, pi, expected_keys):
    """Build a mathematically valid upper support in binary64.

    For an incumbent ``x`` at multiplier ``pi`` the concave Lagrangian value
    has the affine upper support

    ``Q(pi') <= V(pi, x) - x * (pi' - pi)``.

    ``inner_value`` is required to be an upper enclosure of the represented
    incumbent objective.  Computing ``V + x*pi`` with ordinary binary64 can
    round *down* after cancellation, which would move the alleged upper
    support below the represented incumbent even at the generating point.
    Recover the exact rationals represented by every binary64 input and round
    the intercept once, toward ``+inf``.  The discrete subgradient is required
    to be a complete exact 0/1 vector; a relaxed or incomplete point cannot
    support the integer oracle.
    """
    keys = tuple(expected_keys)
    complete_xcp = _complete_binary_xcp_or_none(inner_xcp, keys)
    if complete_xcp is None:
        raise ValueError("bundle support requires a complete exact-binary xcp")
    try:
        value = float(inner_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bundle support inner value must be finite") from exc
    if not math.isfinite(value):
        raise ValueError("bundle support inner value must be finite")

    if set(pi) != set(keys):
        raise ValueError("bundle support requires a complete multiplier vector")
    intercept_exact = Fraction.from_float(value)
    coefficients = {}
    for key in keys:
        try:
            multiplier = float(pi[key])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"bundle support multiplier {key!r} must be finite"
            ) from exc
        if not math.isfinite(multiplier):
            raise ValueError(
                f"bundle support multiplier {key!r} must be finite"
            )
        x_value = complete_xcp[key]
        coefficients[key] = -x_value
        intercept_exact += (
            Fraction.from_float(x_value) * Fraction.from_float(multiplier)
        )

    try:
        intercept = float(intercept_exact)
    except OverflowError as exc:
        raise ValueError("bundle support intercept exceeds binary64") from exc
    if not math.isfinite(intercept):
        raise ValueError("bundle support intercept exceeds binary64")
    if Fraction.from_float(intercept) < intercept_exact:
        intercept = math.nextafter(intercept, math.inf)
    if not math.isfinite(intercept):
        raise ValueError("bundle support intercept exceeds binary64")
    return coefficients, intercept


class _FixedTargetIntervalError(ValueError):
    """Contradictory target certificates must not enter ordinary fallback."""


def _audited_fixed_trial_witness(prob_data, node, trial, decisions, *, source):
    """Store a real fixed-state route, never a target lower bound or LP point."""
    ctx = _node_context(_instance(prob_data), node, stage=3)
    keys = _state_keys(ctx, int(node.info))
    state = _complete_binary_xcp_or_none(trial, keys)
    if state is None:
        raise ValueError('Fixed-trial support needs its complete binary parent')
    if not isinstance(decisions, Mapping):
        raise ValueError('Fixed-trial route witness must contain explicit decisions')
    try:
        route, upper = certify_stage3_forward_policy(prob_data, node, state, decisions)
    except InvalidForwardPolicy as exc:
        raise ValueError('Fixed-trial route failed its physical audit') from exc
    return dict(context=ctx.route_key(int(node.info)), trial=dict(state),
                route=route, physical_upper=upper, source=str(source))


def _reaudit_fixed_trial_witness(prob_data, node, trial, witness):
    """Rebuild a cached policy's cost under the current physical context."""
    ctx = _node_context(_instance(prob_data), node, stage=3)
    if (not isinstance(witness, Mapping) or
            witness.get('context') != ctx.route_key(int(node.info)) or
            witness.get('trial') != dict(trial)):
        raise ValueError('Fixed-trial witness has a different physical scope')
    # Ignore any cached scalar objective: only its physical route is evidence.
    return _audited_fixed_trial_witness(prob_data, node, trial, witness['route'],
                                       source=witness.get('source', 'cached_fixed_route'))


def _fixed_trial_bundle_support(witness, shifts, keys):
    """Affine upper support in the current residual multiplier coordinates."""
    trial = witness['trial']
    if set(shifts) != set(keys) or set(trial) != set(keys):
        raise ValueError('Fixed-trial support has incomplete coordinates')
    residual = Fraction.from_float(float(witness['physical_upper'])) - sum(
        (Fraction.from_float(float(shifts[key])) * int(trial[key]) for key in keys), Fraction())
    upper = _round_exact_endpoint(residual, upward=True)
    return _bundle_support_from_incumbent(upper, trial, dict.fromkeys(keys, 0.), keys)


def _canonical_bundle_slope(coefficients):
    """Lossless slope identity; missing and signed-zero entries agree."""
    canonical = []
    for raw_key, raw_value in dict(coefficients).items():
        key = str(raw_key)
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"bundle coefficient {key!r} must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"bundle coefficient {key!r} must be finite")
        if value != 0.0:
            canonical.append((key, value.hex()))
    return tuple(sorted(canonical))


def _store_bundle_support(cut_dict, support):
    """Retain every distinct support and exact same-slope dominance only.

    Bundle supports upper-bound a concave oracle and the bundle uses their
    pointwise minimum.  Therefore, for an exactly identical slope the smaller
    intercept globally dominates; supports with different slopes are never
    dropped by a numerical-violation threshold.  Return the archive index
    whose row must be appended to the live bundle model, or ``None`` when the
    new support is exactly dominated.
    """
    coefficients, raw_intercept = support
    intercept = float(raw_intercept)
    if not math.isfinite(intercept):
        raise ValueError("bundle support intercept must be finite")
    stored = (dict(coefficients), intercept)
    slope = _canonical_bundle_slope(coefficients)
    for index, (old_coefficients, raw_old_intercept) in enumerate(cut_dict):
        if _canonical_bundle_slope(old_coefficients) != slope:
            continue
        old_intercept = float(raw_old_intercept)
        if not math.isfinite(old_intercept):
            raise ValueError("stored bundle support intercept must be finite")
        if intercept < old_intercept:
            cut_dict[index] = stored
            return index
        return None
    cut_dict.append(stored)
    return len(cut_dict) - 1


def _canonical_exact_pi_key(pi, expected_keys):
    """Lossless key for a complete multiplier vector.

    ``float.hex`` preserves every binary64 value.  Positive and negative zero
    are mathematically identical multipliers, so both are normalized to +0.
    """
    expected = tuple(expected_keys)
    if set(pi) != set(expected):
        missing = sorted(set(expected) - set(pi))
        extra = sorted(set(pi) - set(expected))
        raise ValueError(f"incomplete pi payload: missing={missing}, extra={extra}")
    key = []
    for name in expected:
        try:
            value_hex = _canonical_finite_float_hex(pi[name])
        except ValueError as exc:
            raise ValueError(f"pi[{name}] must be finite") from exc
        key.append((name, value_hex))
    return tuple(key)


def _same_complete_pi(left, right, expected_keys):
    if left is None or right is None:
        return False
    try:
        return (
            _canonical_exact_pi_key(left, expected_keys)
            == _canonical_exact_pi_key(right, expected_keys)
        )
    except (TypeError, ValueError):
        return False


def _clone_oracle_eval(evaluation):
    return _OracleEval(
        inner_value=evaluation.inner_value,
        inner_xcp=(dict(evaluation.inner_xcp)
                   if evaluation.inner_xcp is not None else None),
        outer_lb=evaluation.outer_lb,
        needs_outer_fallback=evaluation.needs_outer_fallback,
        exact=evaluation.exact,
        status=evaluation.status,
        source=evaluation.source,
    )


class _ExactOracleMemo:
    """Per-call memo for complete exact oracle evaluations only."""

    def __init__(self, pi_keys, *, enabled=False):
        self.pi_keys = tuple(pi_keys)
        self.enabled = bool(enabled)
        self._cache = {}
        self.requests = 0
        self.hits = 0
        self.misses = 0
        self.native_solves = 0
        self.stores = 0

    def _cacheable(self, evaluation):
        if not (
            evaluation.has_inner
            and evaluation.has_outer
            and evaluation.exact
            # Solver OPTIMAL may only certify its requested MIP gap. The
            # same-pi memo must meet the strict checkpoint band regardless
            # of any broader tolerance used by the ordinary S2 search.
            and 0. <= evaluation.inner_value - evaluation.outer_lb <= _CERT_CHECKPOINT_GAP_ABS_TOL
            and _complete_binary_xcp_or_none(
                evaluation.inner_xcp, self.pi_keys
            ) is not None
        ):
            return False
        # Cache only exact native answers
        evidence = f"{evaluation.source} {evaluation.status}".lower()
        forbidden = ("+", "fallback", "timeout", "time_limit", "interrupted", "failed")
        return not any(token in evidence for token in forbidden)

    def evaluate(self, pi, native_solve):
        self.requests += 1
        if not self.enabled:
            self.native_solves += 1
            return native_solve()

        key = _canonical_exact_pi_key(pi, self.pi_keys)
        cached = self._cache.get(key)
        if cached is not None:
            record_backend_event("oracle", "cache", "exact_pi_hit")
            self.hits += 1
            return _clone_oracle_eval(cached)
        self.misses += 1

        self.native_solves += 1
        evaluation = native_solve()
        if self._cacheable(evaluation):
            self._cache[key] = _clone_oracle_eval(evaluation)
            self.stores += 1
        return evaluation

    def telemetry(self):
        return {
            "enabled": int(self.enabled),
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "native_solves": self.native_solves,
            "stores": self.stores,
            "entries": len(self._cache),
        }

    def format_telemetry(self):
        stats = self.telemetry()
        return "oracle_memo " + " ".join(
            f"{key}={value}" for key, value in stats.items()
        )


def _update_lrp_lagrangian_objective(model, pi):
    """Update the complete physical objective without changing its free domain.

    Gurobi invalidates the previous objective's tree/bound and retains its
    feasible solution. Copy the specification and objective vector so earlier
    Evaluation objects still describe their own multipliers and certificates.
    evaluate_model audits the new canonical/live coefficients before solving.
    """
    spec = model._lrp_spec
    if spec.direction != 'backward' or spec.multipliers is None:
        raise ValueError('Only a free-state backward oracle can update multipliers')
    copies = model._lrp_parent_copy
    if set(pi) != set(copies) or len(spec.multipliers) != len(copies):
        raise ValueError('The complete current physical multiplier is required')
    values = tuple(float(pi[key]) for key in copies)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Multipliers must be finite')
    columns = {name: k for k, name in enumerate(spec.linear.names)}
    indices = [columns[copies[key]] for key in copies]
    if any(spec.linear.cost[k] != -old for k, old in zip(indices, spec.multipliers)):
        raise ValueError('Canonical parent costs do not match previous multipliers')
    if all(a.hex() == float(b).hex() for a, b in zip(values, spec.multipliers)):
        return False
    linear = copy(spec.linear)
    linear.cost = list(spec.linear.cost)
    next_spec = copy(spec)
    next_spec.linear, next_spec.multipliers = linear, values
    variables = [model.getVarByName(copies[key]) for key in copies]
    for column, value in zip(indices, values):
        linear.cost[column] = -value
    # No reset or Start write: an unchanged pi must keep the entire tree;
    # changed pi uses Gurobi's previous-solution start under the new objective.
    model.setAttr('Obj', variables, [-value for value in values])
    model.update()
    model._lrp_spec = next_spec
    model._lrp_native.specification = next_spec
    return True


class _LastOracleModel:
    """Own one same-domain MIP during one Level Set call.

    Identical multipliers preserve the search tree. With objective reuse
    enabled, different multipliers preserve only the model/feasible solution;
    their objective and certificate must be recomputed before use.
    """

    def __init__(self, *, reuse_objectives=False):
        self.reuse_objectives = bool(reuse_objectives)
        self.key = None
        self.model_key = None
        self.model = None
        self.last_acquire_kind = None
        self.statistics = dict(cold_builds=0, warm_resumes=0, objective_updates=0,
                               disposals=0, reuse_objectives=self.reuse_objectives, history=[])

    def select(self, key):
        if key != self.key:
            same_domain = (self.reuse_objectives and self.key is not None
                           and key[0] == self.key[0])
            if not same_domain:
                self.close()
            self.key = key

    def acquire(self, build, update_objective=None):
        if self.key is None:
            raise RuntimeError('Select the complete oracle identity before building')
        if self.model is not None:
            if self.model_key != self.key:
                try:
                    if update_objective is None:
                        raise RuntimeError('Changed multipliers require a canonical objective update')
                    update_objective(self.model)
                except Exception:
                    self.close()
                    raise
                self.model_key = self.key
                self.statistics['objective_updates'] += 1
                self.last_acquire_kind = 'changed_pi'
            else:
                self.last_acquire_kind = 'same_pi'
            return self.model, True
        self.model = build()
        self.model_key = self.key
        self.last_acquire_kind = 'cold_build'
        self.statistics['cold_builds'] += 1
        return self.model, False

    def close(self):
        model, self.model = self.model, None
        self.key = None
        self.model_key = None
        if model is not None:
            self.statistics['disposals'] += 1
            model.dispose()


def _oracle_archive_key(stage, node, cut_lag):
    """Exact coefficients of the relevant downstream/own route-cut archive."""
    children = node.successor if stage == 2 else (node.index,)
    return tuple((int(child), tuple(
        (tuple(sorted((str(key), float(value).hex()) for key, value in pi.items())),
         float(intercept).hex())
        for pi, intercept in cut_lag.get(3, {}).get(child, ()))) for child in children)


def _oracle_model_scope(stage, ctx, facility, node, cut_lag):
    """Physical integer oracle identity, independent of redundant DFJ rows.

    Added/evicted physical connectivity rows do not change integer Q(pi).
    A retained MIP keeps all of its canonical rows even when the bounded pool
    evicts older rows. Actual S2 route envelopes and physical contexts remain
    part of the identity and cannot share stale models or exact memo answers.
    """
    return (stage, ctx.key, facility,
            'parent' if stage == 3 else 'availability_box',
            _oracle_archive_key(stage, node, cut_lag))


def _merge_same_pi_evals(primary, secondary):
    """Merge evaluations at exactly the same ``pi``.

    Keep the cheapest certified incumbent and its own xcp together.  Certified
    lower bounds are scalar certificates, so the strongest compatible one can
    be used independently of which backend supplied the incumbent.  This
    matters for a fast relaxation followed by an exact fallback: retaining the
    relaxation's weaker incumbent would leave an artificial oracle gap even
    after the fallback proved the same-pi optimum.
    """
    if primary is None:
        return secondary
    if secondary is None:
        return primary

    inner_candidates = [ev for ev in (primary, secondary) if ev.has_inner]
    inner_eval = (
        min(inner_candidates, key=lambda ev: float(ev.inner_value))
        if inner_candidates else None
    )
    inner = inner_eval.inner_value if inner_eval is not None else None
    xcp = inner_eval.inner_xcp if inner_eval is not None else None

    outer_candidates = [
        ev.outer_lb for ev in (primary, secondary) if ev.has_outer
    ]
    if inner is not None:
        outer_candidates = [
            lb for lb in outer_candidates if lb <= inner
        ]
    outer = max(outer_candidates) if outer_candidates else None

    sources = "+".join(dict.fromkeys((primary.source, secondary.source)))
    return _make_oracle_eval(
        inner_value=inner,
        inner_xcp=xcp,
        outer_lb=outer,
        needs_outer_fallback=(
            primary.needs_outer_fallback or secondary.needs_outer_fallback
        ) and outer is None,
        exact=primary.exact or secondary.exact,
        status="+".join(dict.fromkeys((primary.status, secondary.status))),
        source=sources,
    )


def _refinement_identity(scope, residual_pi, physical_pi, shifts):
    """Keep evidence in its exact physical domain AND residual coordinates."""
    if set(residual_pi) != set(physical_pi) or set(residual_pi) != set(shifts):
        raise ValueError('Refinement requires complete matching multiplier coordinates')
    def point(values):
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError('Refinement coordinates must be finite')
        return tuple((key, float(values[key]).hex()) for key in sorted(values))
    return scope, point(residual_pi), point(physical_pi), point(shifts)


class _AmbiguousPiRefinement:
    """Optional same-point budget for an unresolved target-membership test.

    The helper owns no solver or certificate cache. All extra oracle calls
    still use _ConsecutiveOracleRefinement and its exact physical identity.
    Tolerance-feasibility only stops extra work; it never proves the strict
    Level Set target or minimum norm.
    """

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.depth = 0
        self.checkpoint_active = False
        self.attempts = {}
        self.pending_supports = []
        self.trace = []

    @classmethod
    def for_stage(cls, stage, same_pi_enabled):
        return cls(stage == 3 and same_pi_enabled and
                   os.environ.get('LRP_AMBIGUOUS_PI_REFINEMENT', '0') == '1')

    def begin(self, key, evaluation, pi, trial, keys, target, tolerance,
              *, remaining, base_allowance, live_model_matches):
        if (not self.enabled or self.depth or self.checkpoint_active or
                evaluation.exact or not (evaluation.has_inner and evaluation.has_outer) or
                not live_model_matches or remaining <= float(base_allowance) or
                self.attempts.get(key, 0) >= 2):
            return None
        if set(pi) != set(keys) or set(trial) != set(keys):
            raise ValueError('Ambiguity classification requires complete residual coordinates')
        level = Fraction.from_float(float(target))
        dot = sum((Fraction.from_float(float(pi[k])) *
                   Fraction.from_float(float(trial[k])) for k in keys), Fraction())
        h_lower = level - dot - Fraction.from_float(evaluation.inner_value)
        h_upper = level - dot - Fraction.from_float(evaluation.outer_lb)
        tol = Fraction.from_float(float(tolerance))
        if not h_lower <= tol < h_upper:
            return None
        self.attempts[key] = self.attempts.get(key, 0) + 1
        self.pending_supports.append(_bundle_support_from_incumbent(
            evaluation.inner_value, evaluation.inner_xcp, pi, keys))
        return dict(pi=dict(pi), effective_residual_target=float(target),
                    h_lower_before=float(h_lower), h_upper_before=float(h_upper),
                    identity_attempt=self.attempts[key], remaining_before=remaining)

    def finish(self, record, evaluation, *, actual_mip_calls, remaining):
        record.update(lower_after=evaluation.outer_lb, upper_after=evaluation.inner_value,
                      exact_after=evaluation.exact, actual_mip_calls=actual_mip_calls,
                      remaining_after=remaining)
        self.trace.append(record)
        return actual_mip_calls > 0

    def flush_pending(self, cut_dict, changed):
        """Return every added/strengthened index for BOTH continuous models."""
        changed = list(changed)
        for support in self.pending_supports:
            index = _store_bundle_support(cut_dict, support)
            if index is not None and index not in changed:
                changed.append(index)
        self.pending_supports.clear()
        return changed

    def telemetry(self):
        return dict(enabled=self.enabled, extra_attempts=len(self.trace),
                    unique_identities=len(self.attempts),
                    actual_mip_calls=sum(row['actual_mip_calls'] for row in self.trace),
                    terminal_pending_supports=len(self.pending_supports),
                    trace=deepcopy(self.trace))


class _ConsecutiveOracleRefinement:
    """Retain audited bounds/supports only at the last exact oracle identity.

    This is distinct from the exact-answer memo: an open interval is never
    made exact by caching. Consecutive inexact visits can spend 2/4/8 times
    the base allowance continuing an existing MIP, within the node deadline.
    The switch is off by default; callers restrict it to the S3 route oracle.
    """

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.key = None
        self.evidence = None
        self.repeats = 0
        self.statistics = dict(enabled=self.enabled, refinement_calls=0,
                               native_skipped=0, lp_skipped=0, audit_deadline_skips=0, history=[])

    def begin(self, key, base_allowance, remaining, *, live_model_matches):
        if not self.enabled:
            return False, min(float(base_allowance), max(0., remaining))
        if key != self.key:
            self.key, self.evidence, self.repeats = key, None, 0
        previous = self.evidence
        refine = bool(live_model_matches and previous is not None and not previous.exact
                      and (previous.has_inner or previous.has_outer))
        self.repeats = self.repeats + 1 if refine else 0
        factor = 2 ** min(self.repeats, 3) if refine else 1
        allowance = min(float(base_allowance) * factor, max(0., remaining))
        if refine:
            self.statistics['refinement_calls'] += 1
        self.statistics['history'].append(dict(
            refinement=refine, consecutive_refinements=self.repeats,
            base_allowance=float(base_allowance), multiplier=factor,
            effective_allowance=allowance, remaining_at_entry=remaining,
            reused_evidence=refine,
            previous_lower=previous.outer_lb if refine else None,
            previous_upper=previous.inner_value if refine else None))
        return refine, allowance

    def finish(self, key, evaluation):
        if not self.enabled:
            return evaluation
        if key != self.key:
            raise ValueError('Cannot combine refinement evidence from another oracle identity')
        if evaluation.has_inner or evaluation.has_outer:
            result = _merge_same_pi_evals(self.evidence, evaluation)
        else:
            # A deadline with no new solve must preserve the previous paired
            # certificate, without assigning it a newly solved status.
            result = self.evidence if self.evidence is not None else evaluation
        self.evidence = deepcopy(result)
        return deepcopy(result)


def _tightest_certified_eval(pi_history, inner_history, outer_history, xcp_history,
                              current_pi, current_inner, current_outer, current_xcp,
                              x_prev, pi_keys):
    """Select the valid Lagrangian cut with largest value at ``x_prev``.

    Every finite ``outer`` supplied here is already a certified LB on Q(pi).
    Taking the maximum of their values at the trial point strengthens the reported
    LB without changing validity.  The selected intercept always stays paired with
    the same pi (and incumbent metadata) that produced it.
    """
    candidates = []
    for idx, (pi, inner, outer) in enumerate(
            zip(pi_history, inner_history, outer_history)):
        if outer is None or not np.isfinite(outer):
            continue
        xcp = xcp_history[idx] if idx < len(xcp_history) else None
        score = float(outer) + sum(float(pi[k]) * float(x_prev[k]) for k in pi_keys)
        candidates.append((
            score, dict(pi), inner, float(outer),
            dict(xcp) if xcp is not None else None,
        ))

    if current_outer is not None and np.isfinite(current_outer):
        score = float(current_outer) + sum(
            float(current_pi[k]) * float(x_prev[k]) for k in pi_keys
        )
        candidates.append((
            score, dict(current_pi), current_inner, float(current_outer),
            dict(current_xcp) if current_xcp is not None else None,
        ))

    return max(candidates, key=lambda rec: rec[0]) if candidates else None


def _certify_lrp_oracle_incumbent(result, model, pi, keys, node, cut_lag):
    """Rebuild a feasible LRP oracle policy and round its objective upward."""
    spec, ctx = result.problem, result.problem.context
    xcp = {key: float(round(model.getVarByName(local).X))
           for key, local in model._lrp_parent_copy.items()}
    if set(xcp) != set(keys):
        raise ValueError('Oracle parent-copy registry is incomplete')
    exact = Fraction()
    if spec.layer == 'assignment':
        from cuts.lrp_static_bounds import basic_route_cuts
        from models.stage_builder import _route_pools, _as_cut
        route_pools, _ = _route_pools(ctx, node, cut_lag)
        z, alpha, u, e = (result.values(group) for group in ('z','alpha','u','e'))
        for j in range(ctx.n):
            if sum(alpha[i,j] for i in range(ctx.m)) + e[j,] != ctx.active[j]:
                raise ValueError('Invalid LRP oracle customer service')
            exact += Fraction.from_float(float(ctx.outsourcing[j])) * e[j,]
        for i in range(ctx.m):
            assignment, dispatch = ctx.check_route_state(i,
                [alpha[i,j] for j in range(ctx.n)], u[i,])
            if dispatch > z[i,]:
                raise ValueError('Closed facility dispatched in LRP oracle')
            envelopes = [Fraction()]
            # Permanent RouteCut rows belong to the S2 epigraph even when
            # the learned archive is empty. Certify the rounded policy in
            # that complete envelope, independently of solver tolerances.
            cuts = basic_route_cuts(ctx, i) + [
                _as_cut(raw, ctx, i) for raw in route_pools[i]]
            for cut in cuts:
                envelopes.append(Fraction.from_float(float(cut.intercept)) + sum(
                    (Fraction.from_float(float(coefficient)) * value
                     for coefficient, value in zip(cut.coefficients, (*assignment, dispatch))),
                    Fraction()))
            exact += max(envelopes)
    else:
        assignment = result.values('a_copy')
        dispatch = result.values('u_copy')[()]
        ctx.check_route_state(spec.facility,
            [assignment[j,] for j in range(ctx.n)], dispatch)
        tour = tour_from_evaluation(result, spec.facility)
        for tail, head in tour['arcs']:
            exact += Fraction.from_float(float(ctx.route_cost[spec.facility,tail,head]))
    exact -= sum((Fraction.from_float(float(pi[key]))*int(xcp[key]) for key in keys), Fraction())
    value = float(exact)
    if Fraction.from_float(value) < exact:
        value = math.nextafter(value, math.inf)
    return max(value, float(result.report['objective'])), xcp


def _round_exact_endpoint(value, *, upward):
    result = float(value)
    represented = Fraction.from_float(result)
    if (represented < value) if upward else (represented > value):
        result = math.nextafter(result, math.inf if upward else -math.inf)
    if not math.isfinite(result):
        raise ValueError('Nonfinite centered Lagrangian endpoint')
    return result


def _centered_multiplier(pi, shifts):
    physical, error = {}, {}
    for key, multiplier in pi.items():
        exact = Fraction.from_float(float(multiplier)) + Fraction.from_float(shifts[key])
        physical[key] = float(exact)
        error[key] = Fraction.from_float(physical[key]) - exact
    return physical, error


def _centered_evaluation(evaluation, errors):
    # q_res(pi) = min[physical objective at round(pi+ell) + error*x].
    # Global LB pays every negative rounding error over the binary box;
    # incumbent UB uses only its own exact binary state.
    inner, outer = None, None
    if evaluation.has_inner:
        correction = sum((value * int(evaluation.inner_xcp[key])
                          for key,value in errors.items()), Fraction())
        inner = _round_exact_endpoint(Fraction.from_float(evaluation.inner_value)+correction, upward=True)
    if evaluation.has_outer:
        correction = sum((min(Fraction(), value) for value in errors.values()), Fraction())
        outer = _round_exact_endpoint(Fraction.from_float(evaluation.outer_lb)+correction, upward=False)
    return _make_oracle_eval(inner_value=inner, inner_xcp=evaluation.inner_xcp,
        outer_lb=outer, exact=evaluation.exact, status=evaluation.status, source=evaluation.source)


def _lift_centered_evaluation(pi, evaluation, shifts):
    physical, error = _centered_multiplier(pi, shifts)
    inner, outer = None, None
    if evaluation.has_outer:
        payment = sum((max(Fraction(), value) for value in error.values()), Fraction())
        outer = _round_exact_endpoint(Fraction.from_float(evaluation.outer_lb)-payment, upward=False)
    if evaluation.has_inner:
        correction = sum((value * int(evaluation.inner_xcp[key]) for key,value in error.items()), Fraction())
        inner = _round_exact_endpoint(Fraction.from_float(evaluation.inner_value)-correction, upward=True)
    return physical, _make_oracle_eval(inner_value=inner, inner_xcp=evaluation.inner_xcp,
        outer_lb=outer, exact=evaluation.exact, status=evaluation.status, source=evaluation.source)


def _certified_original_target_pair(original_target, pi, trial, evaluation, shifts, keys):
    """Retain an emitted physical cut that attains the original certified level.

    This is a value-search stopping certificate, never a norm proof. Test the
    already rounded physical pair, including the centering payment, against
    the original target; a subsequently bisected residual target is irrelevant.
    A lower-only oracle is sufficient and does not create an inner support.
    """
    tolerance = _tight_exit_tolerance(original_target)
    if tolerance is None or not evaluation.has_outer:
        return None
    physical_pi, physical_evaluation = _lift_centered_evaluation(pi, evaluation, shifts)
    if not _certified_level_target_attained(original_target, physical_pi, trial,
            physical_evaluation.outer_lb, keys, tolerance):
        return None
    slack = Fraction.from_float(float(original_target)) - Fraction.from_float(physical_evaluation.outer_lb)
    slack -= sum((Fraction.from_float(physical_pi[key])*Fraction.from_float(float(trial[key]))
                  for key in keys), Fraction())
    return physical_pi, physical_evaluation, slack, tolerance


def _certified_existing_route_target(prob_data, node, node_ind, trial,
                                     cut_lag, original_target, target_certificate):
    """A current-archive value certificate; no oracle or norm proof.

    Archive tuples inherit the same validity trust as the existing S2 model.
    They are never relabelled from another physical route or another scope.
    The extra route-upper closure gate is stricter than the ordinary search
    exit and uses independently re-audited physical arcs, not a cached scalar.
    """
    # Optional legacy-search comparison: retain fixed-target certificates,
    # but start at the bundle minimum-norm LP and query the physical oracle.
    if os.environ.get('LRP_S3_EXISTING_ROUTE_TARGET_EXIT', '1') in ('0', 'false', 'False', ''):
        return None
    if _tight_exit_tolerance(original_target) is None:
        return None
    if not isinstance(target_certificate, Mapping):
        return None
    witness = target_certificate.get('fixed_trial_witness')
    if witness is None:
        return None
    try:
        audited = _reaudit_fixed_trial_witness(prob_data, node, trial, witness)
    except (ValueError, KeyError, TypeError):
        return None
    if not math.isfinite(float(original_target)):
        return None
    target = Fraction.from_float(float(original_target))
    upper = Fraction.from_float(float(audited['physical_upper']))
    if target < 0 or target > upper:
        return None
    from models.stage_builder import _as_cut
    from cuts.benders_cuts import clean_pi
    ctx = _node_context(_instance(prob_data), node, stage=3)
    facility = int(node.info)
    keys = _state_keys(ctx, facility)
    if set(trial) != set(keys) or any(trial[k] not in (0., 1.) for k in keys):
        return None
    rows = cut_lag.get(3, {}).get(node_ind, ())
    # New inline seeds are appended last. Any qualifying row proves the same
    # closed value-search condition; there is no need to maximize its anchor.
    upper_float = float(audited['physical_upper'])
    threshold = _round_exact_endpoint(upper-Fraction.from_float(1e-6), upward=False)
    for index in range(len(rows)-1, -1, -1):
        raw = rows[index]
        cut = _as_cut(raw, ctx, facility)
        # The estimate is only a fast rejection trigger, never a certificate.
        # Directed additions enclose the exact sum, including cancellation.
        # Tiny coefficients can change on cleaning: keep their strict path.
        if not any(0. < abs(value) <= 1e-6 for value in cut.coefficients):
            terms = [float(cut.intercept)] + [float(value) for key,value in
                zip(keys,cut.coefficients) if trial[key] and value]
            try:
                estimate = math.fsum(terms)
            except (OverflowError, ValueError):
                estimate = math.nan
            if (math.isfinite(estimate) and
                    (estimate < original_target or estimate > upper_float or estimate < threshold)):
                low = high = 0.
                for value in terms:
                    low = math.nextafter(low+value, -math.inf)
                    high = math.nextafter(high+value, math.inf)
                if (math.isfinite(low) and math.isfinite(high) and
                        (high < original_target or low > upper_float or high < threshold)):
                    continue
        pi, intercept, _, _ = clean_pi(dict(zip(keys, cut.coefficients)), cut.intercept)
        pi = {k: float(pi.get(k, 0.)) for k in keys}
        anchor = Fraction.from_float(intercept) + sum(
            (Fraction.from_float(pi[k]) * int(trial[k]) for k in keys), Fraction())
        # Reject even a numerically inverted interval. A merely loose target
        # cannot authorize an exit unless the physical route also closes it.
        if anchor < target or not 0 <= upper-anchor <= Fraction.from_float(1e-6):
            continue
        evidence = _make_oracle_eval(outer_lb=intercept, exact=False,
            status='certified_archived_route_lower', source='existing_route_archive')
        pair = _certified_original_target_pair(original_target, pi, trial,
            evidence, dict.fromkeys(keys, 0.), keys)
        if pair is not None:
            return dict(pair=pair, archive_index=index, anchor=anchor,
                physical_upper=audited['physical_upper'], context=ctx.route_key(facility))
    return None

class LagrangianCutManager(CutManager):
    def _native_route_query(self, node, oracle, pi, **options):
        """Preserve the exact native result; optionally retain its paid-for primal."""
        result = oracle.solve(pi, **options)
        self._collect_native_route_witness(node, result)
        return result

    def _collect_native_route_witness(self, node, result):
        """Optional primal side channel; rejection never changes cut/target evidence."""
        sink = getattr(self, '_route_witness_sink', None)
        if sink is None:
            return
        try:
            sink.capture_native(node, result)
        except (ValueError, TypeError, KeyError):
            # An invalid optional route must neither enter the policy pool nor
            # trigger a different mathematical oracle/fallback path.
            self.route_witness_rejections = getattr(self, 'route_witness_rejections', 0) + 1

    def _initialize_cut_storage(self):
        cuts = {}
        for stage, nodes in self.scen_tree.items():
            cuts[stage] = {}
            for node_idx in range(len(nodes)):
                cuts[stage][node_idx] = []
        return cuts


    def add_cut(self, stage, node_idx, cut_data):
        if stage not in self.cuts:
            self.cuts[stage] = {}
        if node_idx not in self.cuts[stage]:
            self.cuts[stage][node_idx] = []
        self.cuts[stage][node_idx].append(cut_data)


    def get_cuts(self, stage, node_idx):
        return self.cuts.get(stage, {}).get(node_idx, [])


    @staticmethod
    def _build_lb_prob(x_prev, L_value, norm_option, cut_Dict, prob_lb=-1e7, prob_ub=1e7,
                       *, level_tol=1e-7, env=None):
        """
        构建下界问题 lb_prob:
            min  ||π||_norm
            s.t. Σ coeff_j · π + intercept_j ≥ θ,  ∀j ∈ bundle
                 π · x_prev + θ ≥ L_value
        """
        lb_prob = gp.Model("lb_prob", env=env)
        lb_prob.setParam('OutputFlag', 0)
        _set_levelset_tolerances(lb_prob, level_tol)

        x_ind_list = list(x_prev.keys())
        pi_var = lb_prob.addVars(x_ind_list, lb=prob_lb, ub=prob_ub, name="pi_var_x")
        theta = lb_prob.addVar(lb=prob_lb, name="theta")

        if norm_option == 2:
            # L2 norm
            lb_prob.setObjective(
                gp.quicksum(pi_var[i] * pi_var[i] for i in x_ind_list), GRB.MINIMIZE)
        else:
            # L1 norm
            pi_abs = lb_prob.addVars(x_ind_list, lb=0.0,
                                     ub=np.maximum(np.abs(prob_ub), np.abs(prob_lb)),
                                     name="pi_abs_x")
            lb_prob.setObjective(gp.quicksum(pi_abs[i] for i in x_ind_list), GRB.MINIMIZE)
            lb_prob.addConstrs((pi_var[i] <= pi_abs[i] for i in x_ind_list), name="pi_abs_pos_x")
            lb_prob.addConstrs((-pi_var[i] <= pi_abs[i] for i in x_ind_list), name="pi_abs_neg_x")

        # bundle 切割
        for j in range(len(cut_Dict)):
            lb_prob.addConstr(
                gp.quicksum(cut_Dict[j][0].get(i, 0.0) * pi_var[i] for i in x_ind_list)
                + cut_Dict[j][1] >= theta
            )

        # D(π) ≥ L_value 约束
        lb_prob.addConstr(
            gp.quicksum(pi_var[i] * x_prev[i] for i in x_ind_list) + theta >= L_value,
            name="cons"
        )
        lb_prob.update()
        return lb_prob


    @staticmethod
    def _update_lb_prob(lb_prob, cut_Dict, update_range):
        """向 lb_prob 追加新的 bundle 切割。"""
        for j in update_range:
            x_ind_list = list(cut_Dict[j][0].keys())
            lb_prob.addConstr(
                gp.quicksum(cut_Dict[j][0][i] * lb_prob.getVarByName(f"pi_var_x[{i}]")
                             for i in x_ind_list)
                + cut_Dict[j][1] >= lb_prob.getVarByName("theta")
            )
        lb_prob.update()


    @staticmethod
    def _build_next_pi_prob(level, alpha, x_value, L_value, cut_Dict, norm_option,
                             prob_lb=-1e7, prob_ub=1e7, *, level_tol=1e-7, env=None):
        """
        构建 next_pi 问题:
            min  ||π - π_bar||₁    (目标: 离 stability center 最近)
            s.t. bundle 切割
                 α·||π||_norm + (1-α)·(L - π·x - θ) ≤ level
        """
        prob = gp.Model("next_pi_prob", env=env)
        prob.setParam('OutputFlag', 0)
        _set_levelset_tolerances(prob, level_tol)
        x_ind_list = list(x_value.keys())

        pi_var = prob.addVars(x_ind_list, lb=prob_lb, ub=prob_ub, name="pi_var_x")
        theta = prob.addVar(lb=prob_lb, name="theta")
        pi_obj_abs = prob.addVars(x_ind_list, lb=0.0,
                                   ub=np.maximum(np.abs(prob_ub), np.abs(prob_lb)),
                                   name="pi_obj_abs_x")

        # bundle 切割
        for j in range(len(cut_Dict)):
            prob.addConstr(
                gp.quicksum(cut_Dict[j][0].get(i, 0.0) * pi_var[i] for i in x_ind_list)
                + cut_Dict[j][1] >= theta
            )

        # level 约束: α·||π|| + (1-α)·gap ≤ level
        if norm_option == 2:
            prob.addConstr(
                alpha * gp.quicksum(pi_var[i] * pi_var[i] for i in x_ind_list) +
                (1 - alpha) * (L_value - gp.quicksum(pi_var[i] * x_value[i]
                                                      for i in x_ind_list) - theta)
                <= level,
                name="level_cons"
            )
        else:
            pi_abs = prob.addVars(x_ind_list, lb=0.0,
                                   ub=np.maximum(np.abs(prob_ub), np.abs(prob_lb)),
                                   name="pi_abs_x")
            prob.addConstr(
                alpha * gp.quicksum(pi_abs[i] for i in x_ind_list) +
                (1 - alpha) * (L_value - gp.quicksum(pi_var[i] * x_value[i]
                                                      for i in x_ind_list) - theta)
                <= level,
                name="level_cons"
            )
            prob.addConstrs((pi_var[i] <= pi_abs[i] for i in x_ind_list), name="pi_abs_pos_x")
            prob.addConstrs((-pi_var[i] <= pi_abs[i] for i in x_ind_list), name="pi_abs_neg_x")

        # 目标: min ||π - π_bar||₁ (π_bar 通过 RHS 更新)
        prob.addConstrs((pi_obj_abs[i] - pi_var[i] >= 0 for i in x_ind_list), name="pi_obj_pos_x")
        prob.addConstrs((pi_obj_abs[i] + pi_var[i] >= 0 for i in x_ind_list), name="pi_obj_neg_x")
        prob.setObjective(gp.quicksum(pi_obj_abs[i] for i in x_ind_list), GRB.MINIMIZE)

        prob.update()
        return prob


    @staticmethod
    def _update_next_pi_prob(prob, cut_Dict, update_range, alpha, x_value, L_value,
                              pi_bar, level, norm_option):
        """更新 next_pi_prob: 新 bundle 切割 + 新 level 约束 + 新 π_bar。"""
        x_ind_list = list(x_value.keys())

        # 移除旧 level 约束，添加新的
        level_cons = prob.getConstrByName("level_cons")
        if level_cons is not None:
            prob.remove(level_cons)
            prob.update()
        for constraint in prob.getQConstrs():
            if constraint.QCName == "level_cons":
                prob.remove(constraint)
        prob.update()

        if norm_option == 2:
            prob.addConstr(
                alpha * gp.quicksum(
                    prob.getVarByName(f"pi_var_x[{i}]") ** 2 for i in x_ind_list) +
                (1 - alpha) * (L_value - gp.quicksum(
                    prob.getVarByName(f"pi_var_x[{i}]") * x_value[i] for i in x_ind_list)
                    - prob.getVarByName("theta"))
                <= level,
                name="level_cons"
            )
        else:
            prob.addConstr(
                alpha * gp.quicksum(
                    prob.getVarByName(f"pi_abs_x[{i}]") for i in x_ind_list) +
                (1 - alpha) * (L_value - gp.quicksum(
                    prob.getVarByName(f"pi_var_x[{i}]") * x_value[i] for i in x_ind_list)
                    - prob.getVarByName("theta"))
                <= level,
                name="level_cons"
            )

        # 追加新 bundle 切割
        for j in update_range:
            prob.addConstr(
                gp.quicksum(cut_Dict[j][0].get(i, 0.0) * prob.getVarByName(f"pi_var_x[{i}]")
                             for i in x_ind_list)
                + cut_Dict[j][1] >= prob.getVarByName("theta")
            )

        # 更新 π_bar（通过调整绝对值约束的 RHS）
        for i in x_ind_list:
            prob.getConstrByName(f"pi_obj_pos_x[{i}]").RHS = -pi_bar[i]
            prob.getConstrByName(f"pi_obj_neg_x[{i}]").RHS = pi_bar[i]
        prob.update()
        return prob


    @staticmethod
    def _obtain_alpha_bounds(pi_list, L_value, x_value, v_underbar, V_list, norm_option):
        """
        代数方法计算 α 的上下界和收敛指标 Delta。

        对每个已知 (π^k, V^k), 定义:
            γ^k = ||π^k||_norm - v_underbar - gap^k
            η^k = gap^k
        其中 gap^k = L_value - π^k · x_prev - V^k

        α 的可行域: [α_min, α_max] 使得 α·γ^k + η^k ≥ 0  ∀k
        Delta = max_α min_k (α·γ^k + η^k)  (lower envelope 的最大值)
        """
        alpha_underbar = []
        alpha_bar = []
        gamma_list = {}
        eta_list = {}
        x_ind_list = list(x_value.keys())

        for k in range(len(pi_list)):
            gap_k = L_value - sum(pi_list[k][i] * x_value[i] for i in x_ind_list) - V_list[k]

            if norm_option == 2:
                norm_k = sum(pi_list[k][i] ** 2 for i in x_ind_list)
            else:
                norm_k = sum(np.abs(pi_list[k][i]) for i in x_ind_list)

            gamma_list[k] = norm_k - v_underbar - gap_k
            eta_list[k] = gap_k
            if abs(eta_list[k]) <= 1e-7:
                eta_list[k] = 0.0

            if gamma_list[k] >= 0.0:
                alpha_bar.append(1.0)
                if eta_list[k] >= 0.0:
                    alpha_underbar.append(0.0)
                else:
                    ratio = -eta_list[k] / gamma_list[k]
                    if ratio <= 1:
                        alpha_underbar.append(ratio)
                    elif ratio >= 0:
                        alpha_underbar.append(1.0)
                    else:
                        alpha_underbar.append(0.0)
            else:
                alpha_underbar.append(0.0)
                if eta_list[k] >= 0.0:
                    ratio = -eta_list[k] / gamma_list[k]
                    if ratio <= 1 + 1e-5:
                        alpha_bar.append(ratio)
                    else:
                        alpha_bar.append(1.0)
                else:
                    alpha_bar.append(1.0)

        alpha_min_val = round(np.max(alpha_underbar), 7) if alpha_underbar else 0.0
        alpha_max_val = round(np.min(alpha_bar), 7) if alpha_bar else 1.0

        gamma_array = np.array([gamma_list[k] for k in range(len(pi_list))])
        eta_array = np.array([eta_list[k] for k in range(len(pi_list))])
        _, Delta = LagrangianCutManager._maximize_lower_envelope(gamma_array, eta_array)

        return alpha_max_val, alpha_min_val, Delta


    @staticmethod
    def _maximize_lower_envelope(gamma, eta):
        """
        求分段线性函数下包络的最大值:  max_{x∈[0,1]} min_k (γ_k·x + η_k)
        """
        n = len(gamma)
        if n == 1:
            if gamma[0] > 0:
                return 1.0, gamma[0] + eta[0]
            else:
                return 0.0, eta[0]

        xs = [0.0, 1.0]
        for j in range(n):
            for k in range(j + 1, n):
                if gamma[j] != gamma[k]:
                    x_int = (eta[k] - eta[j]) / (gamma[j] - gamma[k])
                    if 0.0 <= x_int <= 1.0:
                        xs.append(x_int)
        xs = sorted(set(xs))

        best_x, best_val = 0.0, float('-inf')
        for x in xs:
            v = np.min(gamma * x + eta)
            if v > best_val:
                best_val = v
                best_x = x
        return best_x, best_val


    @staticmethod
    def _lb_optimizer_pi(lb_prob, pi_keys):
        return {
            key: float(lb_prob.getVarByName(f"pi_var_x[{key}]").X)
            for key in pi_keys
        }


    @staticmethod
    def _pi_moved(pi_before, pi_after, pi_keys):
        for key in pi_keys:
            before = float(pi_before[key])
            after = float(pi_after[key])
            tolerance = (
                _CERT_CHECKPOINT_PI_ABS_TOL
                + _CERT_CHECKPOINT_PI_REL_TOL * max(1.0, abs(before), abs(after))
            )
            if abs(after - before) > tolerance:
                return True
        return False


    def _run_certificate_checkpoint(
            self, *, stage, node_tag, backend, lb_prob, cut_Dict, pi_keys,
            x_prev, L_value, level_tol, solve_at_pi, deadline=None,
            exact_abs_tol=_CERT_CHECKPOINT_GAP_ABS_TOL):
        """Certify the current ``lb_prob`` optimizer with a same-pi oracle.

        A small Level-Set ``Delta`` only starts this procedure.  Each feasible
        incumbent contributes its own affine bundle support.  If that support
        moves the lower-bound optimizer, the new optimizer is evaluated in the
        next round.  Convergence requires a stable optimizer, a finite feasible
        incumbent, a certified LB at that exact pi, and a strict oracle gap.

        ``exact_abs_tol`` widens both the oracle-gap and the level-error band
        (Stage 2 under the scheduled absolute tolerance: the trial value L
        itself is only that exact).

        Returns ``(pi, evaluation, converged, effective_L, records)``.  Every
        record is safe for outer-cut selection even when certification fails.
        """
        effective_L = float(L_value)
        exact_abs_tol = max(_CERT_CHECKPOINT_GAP_ABS_TOL, float(exact_abs_tol))
        records = []
        last_pi = self._lb_optimizer_pi(lb_prob, pi_keys)
        last_eval = _make_oracle_eval(source=backend, status="not_evaluated")

        for checkpoint_round in range(1, _CERT_CHECKPOINT_MAX_ROUNDS + 1):
            if remaining_seconds(deadline) <= 0.0:
                return last_pi, last_eval, False, effective_L, records
            checkpoint_pi = self._lb_optimizer_pi(lb_prob, pi_keys)
            evaluation = solve_at_pi(checkpoint_pi)
            records.append((dict(checkpoint_pi), evaluation))
            last_pi, last_eval = checkpoint_pi, evaluation

            if not evaluation.has_inner:
                _bound_text = (f"{evaluation.outer_lb:.6f}"
                               if evaluation.has_outer else "none")
                _levelset_detail_log(
                    "backward", stage, backend,
                    f"{node_tag} | checkpoint#{checkpoint_round} bound-only/failed "
                    f"outer={_bound_text}; conv=N",
                )
                return last_pi, last_eval, False, effective_L, records

            xcp = dict(evaluation.inner_xcp)
            support_value = float(evaluation.inner_value)
            if cut_Dict:
                theta_at_pi = min(
                    sum(float(coeffs.get(key, 0.0)) * checkpoint_pi[key]
                        for key in pi_keys) + float(intercept)
                    for coeffs, intercept in cut_Dict
                )
            else:
                theta_at_pi = float("inf")

            support = _bundle_support_from_incumbent(
                support_value,
                xcp,
                checkpoint_pi,
                pi_keys,
            )
            support_index = _store_bundle_support(cut_Dict, support)
            support_added = support_index is not None
            if support_added:
                self._update_lb_prob(lb_prob, cut_Dict, [support_index])

            if not prepare_model_solve(lb_prob, deadline):
                return last_pi, last_eval, False, effective_L, records
            _optimize_gurobi(lb_prob, "level_lp")
            if lb_prob.Status != GRB.OPTIMAL:
                if remaining_seconds(deadline) <= 0.0:
                    return last_pi, last_eval, False, effective_L, records
                effective_L = self._fix_L_value_bisect(
                    lb_prob, x_prev, pi_keys, effective_L,
                    **({"deadline": deadline} if deadline is not None else {}),
                )
            if lb_prob.Status != GRB.OPTIMAL:
                _levelset_detail_log(
                    "backward", stage, backend,
                    f"{node_tag} | checkpoint#{checkpoint_round} lb_prob "
                    "未恢复可行；conv=N",
                )
                return last_pi, last_eval, False, effective_L, records

            optimizer_after = self._lb_optimizer_pi(lb_prob, pi_keys)
            moved = self._pi_moved(checkpoint_pi, optimizer_after, pi_keys)

            oracle_gap = float("inf")
            dual_lb = float("-inf")
            level_error = float("inf")
            strict_gap_tol = float("nan")
            oracle_proven = False
            if evaluation.has_outer:
                oracle_gap = max(
                    0.0, float(evaluation.inner_value) - float(evaluation.outer_lb)
                )
                strict_gap_tol = (
                    exact_abs_tol
                    + _CERT_CHECKPOINT_GAP_REL_TOL * max(
                        1.0,
                        abs(float(evaluation.inner_value)),
                        abs(float(evaluation.outer_lb)),
                    )
                )
                oracle_proven = bool(
                    evaluation.exact or oracle_gap <= strict_gap_tol
                )
                dual_lb = float(evaluation.outer_lb) + sum(
                    checkpoint_pi[key] * float(x_prev[key]) for key in pi_keys
                )
                level_error = max(0.0, effective_L - dual_lb)

            allowed_level_error = max(
                exact_abs_tol,
                float(level_tol) if np.isfinite(level_tol) and level_tol >= 0 else 0.0,
            )
            proof_error = max(oracle_gap, level_error)
            certified = bool(
                not moved
                and evaluation.has_outer
                and oracle_proven
                and level_error <= allowed_level_error
            )
            _levelset_detail_log(
                "backward", stage, backend,
                f"{node_tag} | checkpoint#{checkpoint_round} "
                f"support={int(support_added)} moved={int(moved)} "
                f"exact={int(evaluation.exact)} status={evaluation.status} "
                f"oracle_gap={oracle_gap:.3e} level_error={level_error:.3e} "
                f"proof_error={proof_error:.3e} conv={'Y' if certified else 'N'}",
            )
            if moved:
                continue
            return last_pi, last_eval, certified, effective_L, records

        _levelset_detail_log(
            "backward", stage, backend,
            f"{node_tag} | checkpoint 达到 {_CERT_CHECKPOINT_MAX_ROUNDS} 轮且 "
            "optimizer 仍移动；conv=N",
        )
        return last_pi, last_eval, False, effective_L, records


    @staticmethod
    def _levelset_trace(stage, tag, counter, lb_prob, pi_keys, Delta, alpha, bisected):
        """逐轮打印 lb_prob 的 π/theta、Delta、α 及 bisect 是否触发 (VRP_LEVELSET_TRACE 控制)."""
        if not _levelset_trace_enabled():
            return
        try:
            theta = float(lb_prob.getVarByName("theta").X)
        except Exception:
            theta = float('nan')
        try:
            pi = {k: float(lb_prob.getVarByName(f"pi_var_x[{k}]").X) for k in pi_keys}
        except Exception:
            pi = {}
        pi_norm = sum(abs(v) for v in pi.values())
        pi_str = ", ".join(f"{k}={v:.4g}" for k, v in pi.items() if abs(v) > 1e-9) or "all~0"
        print(
            f"    [LS-trace s{stage}] {tag} it={counter} "
            f"theta={theta:.4f} Delta={Delta:.3e} alpha={alpha:.4f} "
            f"bisect={int(bool(bisected))} ||pi||1={pi_norm:.4f} pi[{pi_str}]",
            flush=True,
        )


    @staticmethod
    def _fix_L_value_bisect(lb_prob, x_prev, x_ind_list, L_value, sub_lb=-1e7,
                            *, deadline=None):
        """当 lb_prob 不可行时，二分搜索修正 L_value 使其可行。"""
        L_value_in = float(L_value)

        def _set_cons_and_optimize(L_rhs):
            if remaining_seconds(deadline) <= 0.0:
                return False
            cons = lb_prob.getConstrByName("cons")
            if cons is not None:
                lb_prob.remove(cons)
                lb_prob.update()
            lb_prob.addConstr(
                gp.quicksum(lb_prob.getVarByName(f"pi_var_x[{i}]") * x_prev[i]
                            for i in x_ind_list)
                + lb_prob.getVarByName("theta") >= L_rhs,
                name="cons"
            )
            lb_prob.update()
            if not prepare_model_solve(lb_prob, deadline):
                return False
            _optimize_gurobi(lb_prob, "level_bisection")
            return lb_prob.Status == GRB.OPTIMAL

        L_lb, L_ub = min(sub_lb, L_value), L_value
        found_feasible = _set_cons_and_optimize(L_lb)
        if not found_feasible:
            # 极端情况下 sub_lb 也不可行，直接把模型保留在当前状态并返回最保守值
            return L_lb

        for _ in range(50):
            if remaining_seconds(deadline) <= 0.0:
                return L_lb
            L_test = (L_lb + L_ub) / 2
            if _set_cons_and_optimize(L_test):
                L_lb = L_test
                if abs(L_ub - L_lb) < 1e-2:
                    break
            else:
                L_ub = L_test

        # restore lb_prob at L_lb
        _set_cons_and_optimize(L_lb)
        try:
            _drop = L_value_in - L_lb
            if _drop > 1e-6:
                print(f"    [bisect] L_value: {L_value_in:.4f} → {L_lb:.4f} "
                      f"(drop={_drop:.4f})", flush=True)
        except Exception:
            pass
        return L_lb


    @staticmethod
    def _fixed_target_budget(stage, limit, remaining):
        """Reserve the finite S2 search budget for multiplier oracle calls.

        An unfinished fixed solve still supplies its certified lower endpoint;
        it does not close the target interval or prove the norm checkpoint.
        """
        if stage == 2 and math.isfinite(remaining):
            return min(limit, 5., max(0., remaining) / 3.)
        return limit

    def solve_lagrangian_dual(self, probData, stage_no, node, cut_lag, cut_Dict,
                              lambda_level, mu_level, norm_option, tol, iter_limit, alpha_level,
                              node_ind=None, x_prev=None, adaptive_alpha=False,
                              sub_time_limit=60.0, tight_exit_abs=None, s2_oracle_state=None,
                              deadline=None, s2_abs_tol=None, s2_rel_cap=None):
        """Original norm/level/bundle search, with the LRP free-state oracle.

        Incumbent objectives only supply affine upper supports of the concave
        dual. The separately certified oracle lower bound supplies the outer
        cut intercept. Identical multipliers continue the same MIP search tree;
        changed multipliers update the objective and retain feasible starts.
        Changed physical domains or cut archives always build a new model.
        S2 may stop at a certified near-anchor cut without a norm proof.
        Either stage may stop when its emitted physical cut attains the
        original certified fixed-state target, also without a norm proof;
        ``tight_exit_abs`` supplies a separate S2 scheduled-gap effort rule
        at the original physical target. Fixed target and norm proofs retain
        their original strict tolerances.
        """
        if stage_no not in (2, 3) or norm_option not in (1, 2):
            raise ValueError('Stage must be 2/3 and norm_option must be 1/2')
        if not 0 < lambda_level < 1 or not 0 < mu_level < 1:
            raise ValueError('Level parameters must lie in (0,1)')
        if iter_limit < 1 or not math.isfinite(float(alpha_level)):
            raise ValueError('A positive iteration limit and finite trial value are required')
        # Legacy node allowance is soft (the final MIP may use its minimum
        # solve allowance); the caller's absolute outer deadline stays hard.
        hard_deadline = deadline
        resume_incomplete_s3_checkpoint = (stage_no == 3 and
            os.environ.get('LRP_S3_RESUME_INCOMPLETE_CHECKPOINT', '0') == '1')
        node_budget = S2LevelSetBudget.from_environment(deadline) if stage_no == 2 else None
        if node_budget is not None:
            deadline = node_budget.search_deadline
        oracle_mip_gap = phase2_oracle_mip_gap()
        native_probe_seconds = (configured_s3_native_cap()[0] if stage_no == 3
                                else _configured_native_probe_seconds())
        refinement = _ConsecutiveOracleRefinement(
            stage_no == 3 and os.environ.get('LRP_SAME_PI_REFINEMENT', '0') == '1')
        instance = _instance(probData)
        ctx = _node_context(instance, node, stage=stage_no)
        from core.stage2_tolerance import effective_abs_gap
        s2_mip_abs_gap = 1e-8
        if stage_no == 2:
            reference = float(alpha_level)
            s2_mip_abs_gap = max(s2_mip_abs_gap, effective_abs_gap(s2_abs_tol, reference, s2_rel_cap) or 0.)

        facility = None if stage_no == 2 else int(node.info)
        keys = _state_keys(ctx, facility)
        trial = {key: float(x_prev[key]) for key in keys}
        if any(value not in (0., 1.) for value in trial.values()):
            raise ValueError('The Level Set trial state must be exactly binary')
        shifts = dict.fromkeys(keys, 0.)
        centered = stage_no == 3 and os.environ.get('LRP_RESIDUAL_CENTERING', '1') not in ('0','false','False')
        if centered:
            active_vertices = [0] + [j+1 for j in range(ctx.n) if ctx.active[j]]
            for j in range(ctx.n):
                if ctx.active[j]:
                    shifts[f'alpha[{facility},{j}]'] = min(
                        float(ctx.route_cost[facility,v,j+1]) for v in active_vertices if v != j+1)
        removed_at_trial = sum((Fraction.from_float(shifts[key])*int(trial[key]) for key in keys), Fraction())
        bundle_reset = False
        if stage_no == 3:
            registry = getattr(self, '_bundle_coordinates', None)
            if registry is None:
                self._bundle_coordinates = registry = {}
            coordinate_id = (ctx.route_key(facility), tuple(shifts.items()))
            if registry.get(id(cut_Dict)) != coordinate_id:
                bundle_reset = bool(cut_Dict)
                cut_Dict.clear()
                registry[id(cut_Dict)] = coordinate_id
        # An S2 bundle upper support uses its current route epigraph archive.
        # After route cuts increase, the old policy objective is no longer a
        # feasible upper support. Never reuse these stale surrogate supports.
        if stage_no == 2:
            cut_Dict.clear()
        elif any(set(coefficients) - set(keys) for coefficients, _ in cut_Dict):
            raise ValueError('Stored route bundle belongs to another LRP state domain')
        counters = getattr(self, 'solve_counts', None)
        if counters is None:
            self.solve_counts = counters = dict(levelset_calls=0, s2_levelset_calls=0,
                s3_levelset_calls=0, oracle_solves=0, level_iterations=0,
                level_lp_solves=0, level_projection_solves=0, checkpoint_calls=0)
        counters['levelset_calls'] += 1
        counters[f's{stage_no}_levelset_calls'] += 1
        self.last_cut_diagnostic = None
        if stage_no == 2:
            self.last_s2_cut_diagnostic = None
        builder = SubproblemBuilder(instance,
            lazy_threshold=getattr(self, "stage2_lazy_threshold", None),
            env=getattr(self, "_model_env", None))
        native_oracle = None
        native_free_oracle = None
        native_probe_time_limits = []
        native_residual_enabled = centered and os.environ.get('LRP_NATIVE_RESIDUAL', '0') == '1'
        route_backend = configured_backward_s3_backend(2)
        assignment_backend = configured_backward_s2_backend(2) if stage_no == 2 else None
        if stage_no == 3 and route_backend == 'native':
            cache = getattr(self, '_native_route_cache', None)
            if cache is None:
                self._native_route_cache = cache = {}
            scope = ctx.route_key(facility)
            if scope not in cache:
                cache[scope] = LRPNativeRouteOracle(instance, node)
            native_oracle = cache[scope]
            native_free_oracle = native_oracle
            if native_residual_enabled:
                residual_cache = getattr(self, '_native_residual_route_cache', None)
                if residual_cache is None:
                    self._native_residual_route_cache = residual_cache = {}
                residual_scope = (scope, tuple(shifts.items()))
                if residual_scope not in residual_cache:
                    residual_cache[residual_scope] = native_oracle.residual_view(shifts)
                native_free_oracle = residual_cache[residual_scope]
        oracle_count_before = counters['oracle_solves']
        memo = _ExactOracleMemo(keys, enabled=True)
        oracle_model = _LastOracleModel(reuse_objectives=
            os.environ.get('LRP_ORACLE_OBJECTIVE_REUSE', '1') not in ('0', 'false', 'False'))
        memo_scope = None
        last_s2_query = None
        records, pi_history, values, outer_values, states = [], [], [], [], []
        model_pair = []
        pi = {key: 0. for key in keys}
        requested_target = float(alpha_level)
        L = 0.0
        target_diagnostic = None
        trial_support_enabled = stage_no == 3 and os.environ.get('LRP_FIXED_TRIAL_SUPPORT', '0') == '1'
        fixed_trial_witness = None
        trial_support_diagnostic = dict(enabled=trial_support_enabled, inserted=False, rejections=[])
        has_warm_start = bool(cut_Dict)
        alpha, delta_initial = 0.5, None
        converged, checkpoint_requested = False, False
        tight_exit, tight_exit_slack, tight_exit_tolerance = False, None, None
        target_exit_pair = None
        scheduled_gap_pair = None
        stop_reason = 'iteration_limit'
        last = _make_oracle_eval(status='not_evaluated', source='gurobi_lrp')
        actual_iterations, projection_count, checkpoint_resumptions = 0, 0, 0
        oracle_precision = dict(search_mip_gap=oracle_mip_gap,
            search_mip_abs_gap=s2_mip_abs_gap,
            checkpoint_mip_gap=0. if stage_no == 2 else oracle_mip_gap,
            checkpoint_mip_abs_gap=min(s2_mip_abs_gap, 1e-8),
            strict_checkpoint_requests=0, strict_checkpoint_solves=0,
            repeated_pi_strict_requests=0, repeated_pi_strict_solves=0)
        tag = f'node={node_ind} context={ctx.key[:10]}'
        ambiguity = _AmbiguousPiRefinement.for_stage(stage_no, refinement.enabled)

        def evaluate(pi_value, *, checkpoint=False):
            nonlocal memo_scope, last_s2_query
            # The ordinary S2 search may deliberately use a loose absolute
            # gap. A norm certificate still requires the unchanged strict
            # oracle interval, so tighten this same live model only for the
            # checkpoint query. Call-local settings cannot leak into a later
            # search query, even after an exception or an expired deadline.
            strict_checkpoint = stage_no == 2 and checkpoint
            query_mip_gap = 0. if strict_checkpoint else oracle_mip_gap
            query_mip_abs_gap = min(s2_mip_abs_gap, 1e-8) if strict_checkpoint else s2_mip_abs_gap
            if strict_checkpoint:
                oracle_precision['strict_checkpoint_requests'] += 1
            # The native residual view scores its own incumbent in the exact
            # original-cost residual objective. Gurobi retains the physical
            # costs, so translate its rounded multiplier evidence first.
            # Merge only after both channels refer to this same residual pi.
            physical_pi, errors = _centered_multiplier(pi_value, shifts)
            scope = _oracle_model_scope(stage_no, ctx, facility, node, cut_lag)
            if memo_scope is not None and scope != memo_scope:
                # Exact answers also belong to their actual route envelope.
                # Changed S2 cuts or physical domains cannot reuse models or
                # memo values. Globally valid DFJ rows preserve integer Q(pi).
                memo._cache.clear()
            memo_scope = scope
            physical_key = tuple((key, float(physical_pi[key]).hex()) for key in keys)
            oracle_model.select((scope, physical_key))
            refine_key = _refinement_identity(scope, pi_value, physical_pi, shifts)
            # Preserve strict same-pi fallback for optional native/auto queries.
            # Match Final compact ordinary scheduled precision at repeated pi.
            # Our separate certificate checkpoint remains strict.
            # Changed pi also uses ordinary precision in every backend.
            strict_search_refinement = bool(stage_no == 2 and not strict_checkpoint
                and assignment_backend != 'gurobi'
                and last_s2_query is not None and last_s2_query[0] == refine_key
                and last_s2_query[1].has_inner and last_s2_query[1].has_outer
                and last_s2_query[1].inner_value - last_s2_query[1].outer_lb
                    > _CERT_CHECKPOINT_GAP_ABS_TOL)
            if strict_search_refinement:
                query_mip_gap, query_mip_abs_gap = 0., min(s2_mip_abs_gap, 1e-8)
                oracle_precision['repeated_pi_strict_requests'] += 1
            allowance = node_budget.solve_allowance(sub_time_limit) if node_budget is not None else sub_time_limit
            # Bound ordinary queries so one uncapped Gurobi solve cannot
            # consume the whole S2 search. Explicit caller/legacy budgets,
            # native sharing, and strict checkpoints keep their allowance.
            if (stage_no == 2 and not strict_checkpoint
                    and assignment_backend == 'gurobi'
                    and float(sub_time_limit) == math.inf
                    and not any(name in os.environ for name in (
                        'LRP_PHASE2_S2_LEVELSET_MIN_SOLVE_TIME',
                        'VRP_PHASE2_S2_LEVELSET_MIN_SOLVE_TIME'))):
                allowance = min(allowance, 2.)
            if allowance <= 0 or remaining_seconds(deadline) <= 0:
                if stage_no == 2 and last_s2_query is not None and last_s2_query[0] == refine_key:
                    return last_s2_query[1]
                return _make_oracle_eval(status='time_limit', source='gurobi_lrp')
            refine, effective_allowance = refinement.begin(
                refine_key, allowance, remaining_seconds(hard_deadline),
                live_model_matches=(oracle_model.model is not None
                    and oracle_model.model_key == (scope, physical_key)))
            def native_solve():
                left = remaining_seconds(hard_deadline)
                if left <= 0:
                    return _make_oracle_eval(status='time_limit', source='gurobi_lrp')
                native_evidence = last_s2_query[1] if strict_search_refinement else None
                raw = None
                oracle_deadline = time.monotonic() + min(effective_allowance, left)
                if stage_no == 2 and assignment_backend in ('auto', 'bpc') and not strict_search_refinement:
                    from solvers.lrp_backward_s2_bpc import solve_s2_backward_with_bpc
                    configured_cap = float(os.environ.get('LRP_S2_BP_TIME_LIMIT_S',
                                           os.environ.get('VRP_S2_BP_TIME_LIMIT_S', '2')))
                    if not math.isfinite(configured_cap) or configured_cap <= 0:
                        raise ValueError('S2_BP_TIME_LIMIT_S must be positive and finite')
                    # This oracle is the assignment/cut epigraph MILP, not a
                    # route TSP. Native root pricing can remain unclosed even
                    # when the compact same-pi model is cheap. Bound the probe
                    # and reserve the rest of this SAME query allowance for
                    # that model; a timed-out pricing pass proves no new LB.
                    native_limit = min(configured_cap, effective_allowance / 3., left / 3.)
                    raw = solve_s2_backward_with_bpc(instance, node, cut_lag,
                        physical_pi, time_limit_s=native_limit,
                        deadline=oracle_deadline if math.isfinite(oracle_deadline) else None,
                        root_bound_only=False, phase=2,
                        backward_gap_abs=query_mip_abs_gap,
                        backward_gap_rel=query_mip_gap)
                    counters['native_s2_oracle_calls'] = counters.get('native_s2_oracle_calls', 0) + 1
                    if raw.get('diagnostic', {}).get('native_executed'):
                        counters['native_cpp_solves'] = counters.get('native_cpp_solves', 0) + 1
                        counters['oracle_solves'] += 1
                    native_evidence = _make_oracle_eval(
                        inner_value=raw.get('inner_value') if raw.get('incumbent_policy_certified') is True else None,
                        inner_xcp=raw.get('xcp') if raw.get('incumbent_policy_certified') is True else None,
                        outer_lb=raw.get('outer_lb') if raw.get('lb_certified') is True else None,
                        exact=raw.get('exact', False),
                        status='native_s2_interval' if raw.get('ok') else raw.get('reason', 'unavailable'),
                        source='lrp_native_s2_bpc')
                    # A repaired root LB is valid but can remain strictly
                    # below the integer target even after native tree search
                    # has finished. Repeating that open interval cannot close
                    # Level Set. Apply the ordinary query's stopping precision
                    # before accepting it without the same-pi fallback.
                    native_gap_closed = (native_evidence.has_inner and native_evidence.has_outer
                        and native_evidence.inner_value - native_evidence.outer_lb <= max(
                            query_mip_abs_gap,
                            query_mip_gap * abs(native_evidence.inner_value)))
                    if (native_evidence.has_inner and native_evidence.has_outer
                            and ((not strict_checkpoint and native_gap_closed)
                                 or (strict_checkpoint and native_evidence.exact))):
                        return native_evidence
                    counters['native_s2_gurobi_fallbacks'] = counters.get('native_s2_gurobi_fallbacks', 0) + 1
                if refine and native_free_oracle is not None:
                    refinement.statistics['native_skipped'] += 1
                    counters['same_pi_native_skipped'] = counters.get('same_pi_native_skipped', 0) + 1
                if native_free_oracle is not None and not refine:
                    # Parse configuration outside the backend-failure handler:
                    # a malformed setting is not a request for Gurobi fallback.
                    native_limit = configured_s3_native_time_limit(
                        min(float(sub_time_limit), effective_allowance),
                        checkpoint=checkpoint, remaining_seconds=left)
                    if native_limit is not None and native_limit <= 0.:
                        return _make_oracle_eval(status='time_limit', source='lrp_native_pctsp')
                    native_options = configured_s3_native_options()
                    try:
                        native_pi = pi_value if native_residual_enabled else physical_pi
                        # Final distinguishes ordinary ESP probes (1s) from
                        # strict checkpoint finalization (10s). Zero disables
                        # that extra cap, never the shared query/node deadline.
                        native_probe_time_limits.append(native_limit)
                        raw = self._native_route_query(node, native_free_oracle, native_pi, time_limit=native_limit,
                            deadline=oracle_deadline if math.isfinite(oracle_deadline) else None,
                            **native_options)
                        counters['native_oracle_calls'] = counters.get('native_oracle_calls', 0) + 1
                        if raw.get('native_executed'):
                            counters['native_cpp_solves'] = counters.get('native_cpp_solves', 0) + 1
                            counters['oracle_solves'] += 1
                        else:
                            counters['analytic_oracle_results'] = counters.get('analytic_oracle_results', 0) + 1
                        native_evidence = _make_oracle_eval(
                            inner_value=raw.get('inner_value'), inner_xcp=raw.get('inner_xcp'),
                            outer_lb=raw.get('outer_lb'), exact=raw.get('exact', False),
                            status=raw.get('status', 'INVALID'), source=raw.get('source', 'lrp_native_pctsp'))
                        if not native_residual_enabled:
                            native_evidence = _centered_evaluation(native_evidence, errors)
                        # Original ordinary ESP/PCTSP search accepts a
                        # certified interval: its incumbent supplies the
                        # bundle support, its LB the merit/cut endpoint.
                        # Only a checkpoint needs to close that interval;
                        # forcing a route MIP after every inexact probe can
                        # spend the remaining search budget on the same pi.
                        # Explicit refinement policies retain their requested
                        # live-MIP path; their same-pi continuation requires it.
                        if (native_evidence.has_inner and native_evidence.has_outer
                                and (native_evidence.exact or (not checkpoint
                                    and not refinement.enabled and not ambiguity.enabled))):
                            return native_evidence
                    except SolveDeadlineReached:
                        return _make_oracle_eval(status='time_limit', source='lrp_native_pctsp')
                    except (NativeUnavailable, ValueError) as exc:
                        record_backend_event('gurobi', 'fallback', f'native_oracle:{type(exc).__name__}', stage=3)
                    counters['native_gurobi_fallbacks'] = counters.get('native_gurobi_fallbacks', 0) + 1
                if min(oracle_deadline-time.monotonic(), remaining_seconds(hard_deadline)) <= 0:
                    return native_evidence or _make_oracle_eval(status='time_limit', source='gurobi_lrp')
                model, retained = oracle_model.acquire(
                    lambda: builder.build_subproblem(stage_no, node, cut_lag, physical_pi),
                    lambda current: _update_lrp_lagrangian_objective(current, physical_pi))
                warm_resume = oracle_model.last_acquire_kind == 'same_pi'
                objective_updated = oracle_model.last_acquire_kind == 'changed_pi'
                if not retained:
                    counters['gurobi_oracle_cold_builds'] = counters.get('gurobi_oracle_cold_builds', 0) + 1
                elif objective_updated:
                    counters['gurobi_oracle_objective_updates'] = counters.get('gurobi_oracle_objective_updates', 0) + 1
                    record_backend_event('gurobi', 'cache', 'changed_physical_pi_objective_update', stage=stage_no)
                try:
                    limit = min(max(0., oracle_deadline-time.monotonic()), remaining_seconds(hard_deadline))
                    if limit <= 0:
                        return native_evidence or _make_oracle_eval(status='time_limit', source='gurobi_lrp')
                    lp_summary = None
                    combined_evidence = native_evidence
                    use_free_dfj = (stage_no == 3 and np.count_nonzero(ctx.active) >= 8
                        and os.environ.get('LRP_FREE_ROUTE_DFJ', '1') not in ('0', 'false', 'False'))
                    if use_free_dfj and refine:
                        refinement.statistics['lp_skipped'] += 1
                        counters['same_pi_lp_skipped'] = counters.get('same_pi_lp_skipped', 0) + 1
                    if use_free_dfj and not refine:
                        separated = separate_free_route_lp(model, time_limit=.6*limit,
                            deadline=min(oracle_deadline, hard_deadline if hard_deadline is not None else math.inf))
                        count = separated['lp_solves']
                        counters['oracle_solves'] += count
                        counters['gurobi_free_route_lp_solves'] = counters.get('gurobi_free_route_lp_solves', 0) + count
                        counters['free_route_dfj_rows'] = counters.get('free_route_dfj_rows', 0) + len(separated['rows'])
                        lp_summary = {key: value for key, value in separated.items() if key not in ('rows', 'history', 'multipliers')}
                        lp_summary['rows_added'] = len(separated['rows'])
                        # A free parent-domain route can always be empty. This
                        # is a real integer upper support even if the native
                        # backend is disabled or the MIP has no remaining time.
                        # Fractional LP states never enter this evidence.
                        ctx.check_route_state(facility, [0]*ctx.n, 0)
                        lp_evidence = _make_oracle_eval(inner_value=0.,
                            inner_xcp=dict.fromkeys(keys, 0.),
                            outer_lb=separated['certified_lower_bound'], exact=False,
                            status='certified_free_lp' if separated['certified_lower_bound'] is not None else 'empty_route',
                            source='gurobi_free_route_lp')
                        lp_evidence = _centered_evaluation(lp_evidence, errors)
                        combined_evidence = _merge_same_pi_evals(combined_evidence, lp_evidence)
                    if use_free_dfj or refine:
                        # evaluate_model includes full matrix and final primal
                        # audits outside its optimizer TimeLimit. Reserve them
                        # using the actual accumulated physical row matrix.
                        model.update()
                        reserve = matrix_audit_reserve(model.NumNZs)
                        limit = min(oracle_deadline-time.monotonic(), remaining_seconds(hard_deadline)) - reserve
                        if limit <= .01:
                            oracle_model.statistics['history'].append(dict(
                                model_sequence=oracle_model.statistics['cold_builds'],
                                warm_resume=False, objective_updated=objective_updated,
                                objective_sequence=oracle_model.statistics['objective_updates'],
                                time_limit=0., status='MIP_skipped_audit_budget',
                                lower_bound=combined_evidence.outer_lb if combined_evidence else None,
                                incumbent_value=combined_evidence.inner_value if combined_evidence else None,
                                free_route_lp=lp_summary))
                            return combined_evidence or _make_oracle_eval(status='time_limit', source='gurobi_lrp')
                    # New constraints invalidate the previous search tree even
                    # when pi is unchanged, while retaining a feasible start.
                    warm_resume = warm_resume and not (lp_summary and lp_summary['rows_added'])
                    # Preserve Gurobi's automatic previous-solution start after
                    # objective changes, and the full tree at identical pi.
                    if (stage_no == 2 and raw is not None
                            and raw.get('incumbent_policy_certified') is True
                            and isinstance(raw.get('decisions'), Mapping)):
                        # The native physical assignment can be cheaper than
                        # the incumbent retained at a previous multiplier.
                        # Pass every audited decision, including the complete
                        # epigraph, on cold AND changed/same-pi models. An A
                        # key is a parent name, not an extra local column.
                        for variable in model.getVars():
                            if variable.VarName in raw['decisions']:
                                variable.Start = raw['decisions'][variable.VarName]
                        counters['native_s2_primal_starts'] = counters.get('native_s2_primal_starts', 0) + 1
                    if not retained and raw is not None and raw.get('incumbent_policy_certified') and raw.get('x') is not None:
                        for parent_key, local in model._lrp_parent_copy.items():
                            model.getVarByName(local).Start = raw['inner_xcp'][parent_key]
                        for arc, variable in model._lrp_variables.get('r', {}).items():
                            variable.Start = raw['x'].get('r['+','.join(map(str,arc))+']', 0.)
                        model.getVarByName('stage_cost').Start = raw['route_cost']
                    result, oracle_retry = _evaluate_lagrangian_oracle(model,
                        stage_no=stage_no, counters=counters,
                        time_limit=limit, mip_gap=query_mip_gap,
                        mip_abs_gap=query_mip_abs_gap, threads=configured_gurobi_threads(),
                        deadline=min(oracle_deadline, hard_deadline if hard_deadline is not None else math.inf),
                        optimize_context=lambda: backend_call('gurobi', 'lagrangian_mip', model=model))
                    if result is None:
                        oracle_model.statistics['history'].append(dict(
                            model_sequence=oracle_model.statistics['cold_builds'],
                            warm_resume=False, objective_updated=objective_updated,
                            objective_sequence=oracle_model.statistics['objective_updates'],
                            time_limit=limit, status='invalid_solver_primal',
                            primal_recovery=oracle_retry,
                            lower_bound=combined_evidence.outer_lb if combined_evidence else None,
                            incumbent_value=combined_evidence.inner_value if combined_evidence else None))
                        oracle_model.close()
                        return combined_evidence or _make_oracle_eval(
                            status='invalid_solver_primal', source='gurobi_lrp')
                    if oracle_retry is not None and oracle_retry['retry_reset_performed']:
                        warm_resume = False
                    # Canonical auditing may exhaust the deadline before
                    # optimize. Count only an actual returned optimizer call.
                    counters['oracle_solves'] += 1
                    counters['gurobi_oracle_solves'] = counters.get('gurobi_oracle_solves', 0) + 1
                    if strict_checkpoint:
                        oracle_precision['strict_checkpoint_solves'] += 1
                    if strict_search_refinement:
                        oracle_precision['repeated_pi_strict_solves'] += 1
                    if warm_resume:
                        oracle_model.statistics['warm_resumes'] += 1
                        counters['gurobi_oracle_warm_resumes'] = counters.get('gurobi_oracle_warm_resumes', 0) + 1
                        record_backend_event('gurobi', 'cache', 'same_physical_pi_resume', stage=stage_no)
                    inner, xcp = None, None
                    if result.x is not None:
                        try:
                            inner, xcp = _certify_lrp_oracle_incumbent(result, model,
                                                                      physical_pi, keys, node, cut_lag)
                        except ValueError:
                            # A tolerance-feasible MIP incumbent can fail the
                            # exact physical capacity audit. It cannot enter
                            # the bundle; its independent lower bound remains.
                            counters['rejected_incumbents'] = counters.get('rejected_incumbents', 0) + 1
                    gurobi_evidence = _make_oracle_eval(inner_value=inner, inner_xcp=xcp,
                        outer_lb=result.certified_lower_bound,
                        exact=result.optimal, status=result.report['status'],
                        source='gurobi_lrp')
                    oracle_model.statistics['history'].append(dict(
                        model_sequence=oracle_model.statistics['cold_builds'],
                        warm_resume=warm_resume, objective_updated=objective_updated,
                        objective_sequence=oracle_model.statistics['objective_updates'], time_limit=limit,
                        same_pi_refinement=refine, effective_oracle_allowance=effective_allowance,
                        strict_checkpoint=bool(strict_checkpoint),
                        repeated_pi_strict=bool(strict_search_refinement),
                        requested_mip_gap=query_mip_gap, requested_mip_abs_gap=query_mip_abs_gap,
                        node_count=float(model.NodeCount), status=result.report['status'],
                        lower_bound=result.certified_lower_bound, incumbent_value=inner,
                        objective_value=result.report['objective'],
                        solver_runtime=result.report['solver_runtime'], free_route_lp=lp_summary,
                        **({'primal_recovery': oracle_retry} if oracle_retry is not None else {})))
                    gurobi_evidence = _centered_evaluation(gurobi_evidence, errors)
                    return _merge_same_pi_evals(combined_evidence, gurobi_evidence)
                except SolveDeadlineReached:
                    # No MIP optimize took place. Keep the same-coordinate
                    # audited LP/native pair from this query; refinement.finish
                    # also retains an earlier pair when this query has none.
                    refinement.statistics['audit_deadline_skips'] += 1
                    return combined_evidence or _make_oracle_eval(
                        status='not_solved_deadline', source='gurobi_lrp')
                except Exception:
                    oracle_model.close()
                    raise
            answer = refinement.finish(refine_key, memo.evaluate(pi_value, native_solve))
            if stage_no == 2:
                last_s2_query = (refine_key, answer)
            if not ambiguity.enabled:
                return answer
            while True:
                attempt = ambiguity.begin(refine_key, answer, pi_value, trial, keys, L, tol,
                    remaining=remaining_seconds(deadline), base_allowance=sub_time_limit,
                    live_model_matches=(oracle_model.model is not None and
                        oracle_model.model_key == (scope, physical_key)))
                if attempt is None:
                    break
                before = counters.get('gurobi_oracle_solves', 0)
                ambiguity.depth += 1
                try:
                    answer = evaluate(pi_value, checkpoint=checkpoint)
                finally:
                    ambiguity.depth -= 1
                if not ambiguity.finish(attempt, answer,
                        actual_mip_calls=counters.get('gurobi_oracle_solves', 0)-before,
                        remaining=remaining_seconds(deadline)):
                    break
            return answer

        def retain_fixed_trial_route(decisions, source):
            nonlocal fixed_trial_witness
            if stage_no != 3:
                return
            try:
                witness = _audited_fixed_trial_witness(probData, node, trial, decisions, source=source)
            except ValueError as exc:
                trial_support_diagnostic['rejections'].append(dict(source=source, reason=str(exc)))
                return
            if fixed_trial_witness is None or witness['physical_upper'] < fixed_trial_witness['physical_upper']:
                fixed_trial_witness = witness
            # Preserve paid-for feasible work even if the remaining deadline
            # prevents the target fallback or Level Set solve. The optional
            # bundle hook below remains independently off.
            record = deepcopy(target_cache.get(target_key,
                dict(used_lower_bound=0., cache_closed=False)))
            if record['used_lower_bound'] > fixed_trial_witness['physical_upper']:
                raise _FixedTargetIntervalError('Fixed-target lower bound exceeds the audited route upper bound')
            record['fixed_trial_witness'] = deepcopy(fixed_trial_witness)
            target_cache[target_key] = record

        def solve_level(model):
            if not prepare_model_solve(model, deadline):
                return False
            counters['level_lp_solves'] += 1
            _optimize_gurobi(model, 'level_lp')
            return model.Status == GRB.OPTIMAL

        try:
            # The level feasibility problem asks D(pi)+pi*x_trial >= L.
            # A time-limited forward incumbent is an UPPER bound and may be
            # strictly larger than the attainable dual optimum. Rebuild the
            # fixed-state problem under the CURRENT downstream cut archive;
            # only its certified LOWER bound is a guaranteed attainable target
            # at the binary parent state. No oracle incumbent supplies L.
            if remaining_seconds(deadline) <= 0:
                return pi, {}, None, cut_Dict, False
            target_cache = getattr(self, '_fixed_target_cache', None)
            if target_cache is None:
                self._fixed_target_cache = target_cache = {}
            downstream = tuple((int(child), tuple((tuple(sorted((str(k), float(v).hex()) for k,v in pi_cut.items())),
                                                    float(beta).hex())
                for pi_cut,beta in cut_lag.get(3, {}).get(child, ()))) for child in node.successor) if stage_no == 2 else ()
            target_key = (stage_no, ctx.key, facility, tuple(trial.items()), downstream)
            cached_target = target_cache.get(target_key)
            if stage_no == 3 and cached_target is not None and cached_target.get('fixed_trial_witness') is not None:
                try:
                    fixed_trial_witness = _reaudit_fixed_trial_witness(
                        probData, node, trial, cached_target['fixed_trial_witness'])
                    trial_support_diagnostic['cached_route_reaudited'] = True
                except (ValueError, KeyError) as exc:
                    trial_support_diagnostic['rejections'].append(dict(source='cached_fixed_route', reason=str(exc)))
            if cached_target is not None and cached_target.get('cache_closed'):
                target_diagnostic = deepcopy(cached_target)
                target_diagnostic['cache_hit'] = True
                L = float(target_diagnostic['used_lower_bound'])
                counters['target_cache_hits'] = counters.get('target_cache_hits', 0) + 1
            if target_diagnostic is None and native_oracle is not None:
                try:
                    native_target = native_oracle.solve_fixed(
                        [trial[f'alpha[{facility},{j}]'] for j in range(ctx.n)], trial[f'u[{facility}]'],
                        time_limit=min(1., float(sub_time_limit), remaining_seconds(deadline)),
                        deadline=deadline)
                    retain_fixed_trial_route(native_target['x'], 'native_fixed_route')
                    target_diagnostic = dict(
                        backend=native_target['diagnostic'].get('backend'),
                        native_executed=native_target['diagnostic'].get('native_executed'),
                        integer_optimality_proven=native_target['diagnostic'].get('integer_optimality_proven', False),
                        wall_seconds=native_target['diagnostic'].get('wall_seconds'),
                        context=native_target['diagnostic'].get('context'),
                        objective=native_target['objective'],
                        certified_lower_bound=native_target['lower_bound'],
                        closed_numerical_optimality_certificate=native_target['exact'])
                    L = max(0., float(native_target['lower_bound']))
                    counters['target_solves'] = counters.get('target_solves', 0) + 1
                    counters['native_target_calls'] = counters.get('native_target_calls', 0) + 1
                    if native_target['diagnostic'].get('native_executed'):
                        counters['native_target_cpp_solves'] = counters.get('native_target_cpp_solves', 0) + 1
                    if not (native_target['exact'] or native_target['diagnostic'].get('integer_optimality_proven', False)):
                        # An open force-visit PCTSP bound can transform to zero.
                        # It is valid but should not weaken the LRP level target.
                        target_diagnostic = None
                except _FixedTargetIntervalError:
                    raise
                except (NativeUnavailable, ValueError) as exc:
                    record_backend_event('gurobi', 'fallback', f'native_target:{type(exc).__name__}', stage=3)
            if target_diagnostic is None:
                fixed = StageModelBuilder(instance,
                    lazy_threshold=getattr(self, "stage2_lazy_threshold", None),
                    env=getattr(self, "_model_env", None)).build_stage_problem(
                    stage_no, node, cut_lag, x_prev)
                try:
                    target_remaining = remaining_seconds(deadline)
                    limit = self._fixed_target_budget(stage_no,
                        min(float(sub_time_limit), target_remaining), target_remaining)
                    if limit <= 0:
                        return pi, {}, None, cut_Dict, False
                    counters['target_solves'] = counters.get('target_solves', 0) + 1
                    counters['gurobi_target_solves'] = counters.get('gurobi_target_solves', 0) + 1
                    target_result, target_retry = _evaluate_fixed_target(fixed,
                        stage_no=stage_no, counters=counters, time_limit=limit,
                        mip_gap=oracle_mip_gap, mip_abs_gap=s2_mip_abs_gap,
                        threads=configured_gurobi_threads(), deadline=deadline)
                    if target_result is None:
                        # The caller can still commit its independently verified
                        # SBC seed. This failed target supplies no LevelSet row,
                        # cached endpoint, or convergence/norm certificate.
                        diagnostic = dict(stage=stage_no, node=node_ind, context=ctx.key,
                            method='fixed_target_primal_recovery',
                            stop_reason='fixed_target_primal_rejected',
                            fixed_target_primal_retry=target_retry,
                            target_accepted=False, selected_cut_valid=False,
                            requested_target=requested_target, certified_target=None,
                            actual_iterations=0, oracle_solves=0,
                            level_target_feasibility_certified=False,
                            norm_checkpoint_proven=False, checkpoint_proven=False,
                            target_attained_exit=False, converged=False)
                        self.last_cut_diagnostic = diagnostic
                        self.last_s2_cut_diagnostic = diagnostic
                        return pi, {}, None, cut_Dict, False
                    target_diagnostic = target_result.summary()
                    if target_retry is not None:
                        target_diagnostic['fixed_target_primal_retry'] = target_retry
                    if stage_no == 3 and target_result.x is not None:
                        retain_fixed_trial_route(dict(zip(target_result.problem.linear.names,
                                                          map(float, target_result.x))),
                                                 'gurobi_fixed_route')
                    if target_result.certified_lower_bound is not None:
                        L = max(0., float(target_result.certified_lower_bound))
                finally:
                    fixed.dispose()
            # Nonnegative physical costs certify a zero target when the
            # current fixed solve has no stronger certified lower bound.
            if cached_target is not None:
                L = max(L, float(cached_target['used_lower_bound']))
            target_diagnostic['used_lower_bound'] = L
            target_diagnostic['cache_closed'] = bool(
                target_diagnostic.get('closed_numerical_optimality_certificate') or
                target_diagnostic.get('integer_optimality_proven'))
            if stage_no == 3 and fixed_trial_witness is not None:
                if L > fixed_trial_witness['physical_upper']:
                    raise _FixedTargetIntervalError('Fixed-target lower bound exceeds the audited route upper bound')
                target_diagnostic['fixed_trial_witness'] = deepcopy(fixed_trial_witness)
            target_cache[target_key] = deepcopy(target_diagnostic)
            original_target = max(L, _round_exact_endpoint(removed_at_trial, upward=False))
            L = max(0., _round_exact_endpoint(Fraction.from_float(original_target)-removed_at_trial, upward=False))
            archived_target = (_certified_existing_route_target(
                probData, node, node_ind, trial, cut_lag, original_target, target_diagnostic)
                if stage_no == 3 else None)
            if archived_target is not None:
                physical_pi, evidence, slack, exit_tolerance = archived_target['pair']
                diagnostic = build_s2_cut_diagnostic(node=node_ind, L=original_target,
                    pi=physical_pi, x_trial=trial, V_inner=None, V_outer=evidence.outer_lb,
                    actual_iterations=0, stop_reason='certified_target_attained_exit',
                    backend='gurobi_lrp', checkpoint_requested=False, checkpoint_proven=False,
                    selected_is_checkpoint=False, tight_exit=False)
                diagnostic.update(stage=3, norm_option=norm_option,
                    configured_route_backend=route_backend, configured_assignment_backend=None,
                    selected_oracle_source=evidence.source, projection_solves=0,
                    checkpoint_resumptions=0, oracle_solves=0, oracle_evaluations=0,
                    oracle_memo=memo.telemetry(), context=ctx.key,
                    oracle_precision=oracle_precision, oracle_model_reuse=oracle_model.statistics,
                    same_pi_refinement=refinement.statistics, selected_cut_valid=True,
                    requested_target=requested_target, certified_target=original_target,
                    residual_centering=centered, residual_target=L,
                    native_residual_view=native_residual_enabled and native_free_oracle is not None,
                    native_probe_seconds=native_probe_seconds, native_probe_time_limits=[],
                    incoming_shifts=dict(shifts), bundle_reset_for_coordinate_change=bundle_reset,
                    target_certificate=deepcopy(target_diagnostic),
                    merit_endpoint='certified_archived_route_lower',
                    level_target_feasibility_certified=True, norm_checkpoint_proven=False,
                    target_attained_exit=True, target_exit_original_target=original_target,
                    target_exit_signed_slack=float(slack), target_exit_tolerance=exit_tolerance,
                    target_exit_evidence='existing_route_archive',
                    existing_route_archive_index=archived_target['archive_index'],
                    existing_route_anchor=float(archived_target['anchor']),
                    existing_route_physical_upper=archived_target['physical_upper'],
                    scheduled_gap_effort_exit=False,
                    result_selection={'reason':'certified_existing_route_target_pair_preserved'},
                    initial_pi_source='certified_current_archive_value',
                    fixed_trial_support=dict(trial_support_diagnostic))
                self.last_cut_diagnostic = diagnostic
                counters['existing_route_target_exits'] = counters.get('existing_route_target_exits', 0) + 1
                return dict(physical_pi), {}, evidence.outer_lb, cut_Dict, False

            if trial_support_enabled and fixed_trial_witness is not None:
                support = _fixed_trial_bundle_support(fixed_trial_witness, shifts, keys)
                index = _store_bundle_support(cut_Dict, support)
                trial_support_diagnostic.update(inserted=index is not None,
                    physical_upper=fixed_trial_witness['physical_upper'], residual_upper=support[1],
                    source=fixed_trial_witness['source'], full_parent_state=dict(trial))
            lb_prob = self._build_lb_prob(trial, L, norm_option, cut_Dict, level_tol=tol, env=getattr(self, "_model_env", None))
            model_pair.append(lb_prob)
            if not solve_level(lb_prob):
                if remaining_seconds(deadline) <= 0:
                    stop_reason = 'deadline'
                    return pi, {}, None, cut_Dict, False
                L = self._fix_L_value_bisect(lb_prob, trial, keys, L, deadline=deadline)
            if lb_prob.Status != GRB.OPTIMAL:
                return pi, {}, None, cut_Dict, False
            pi = self._lb_optimizer_pi(lb_prob, keys)
            initial_pi_source = 'bundle_minimum_norm'
            # Keep the original Level Set minimum-norm bundle starting point.
            # A tight SBC slope certifies an anchor value, not minimum norm.
            # Existing certified outer cuts stay in cut_lag independently.
            next_prob = self._build_next_pi_prob(1e7, alpha, trial, L, cut_Dict, norm_option,
                                                level_tol=tol, env=getattr(self, "_model_env", None))
            model_pair.append(next_prob)
            last = evaluate(pi)
            for iteration in range(int(iter_limit)):
                records.append((dict(pi), last))
                scheduled_candidate = _certified_scheduled_gap_pair(stage_no,
                    original_target, pi, trial, last, shifts, keys, tight_exit_abs)
                if not last.has_inner:
                    target_exit_pair = _certified_original_target_pair(
                        original_target, pi, trial, last, shifts, keys)
                    if target_exit_pair is not None:
                        converged = False
                        stop_reason = 'certified_target_attained_exit'
                        break
                    if scheduled_candidate is not None:
                        scheduled_gap_pair = scheduled_candidate
                        converged = False
                        stop_reason = 'scheduled_gap_effort_exit'
                        break
                    stop_reason = 'oracle_no_incumbent'
                    break
                actual_iterations += 1
                counters['level_iterations'] += 1
                pi_history.append(dict(pi)); values.append(last.inner_value)
                outer_values.append(last.outer_lb); states.append(dict(last.inner_xcp))
                support = _bundle_support_from_incumbent(last.inner_value, last.inner_xcp, pi, keys)
                index = _store_bundle_support(cut_Dict, support)
                changed = [] if index is None else [index]
                if ambiguity.enabled:
                    changed = ambiguity.flush_pending(cut_Dict, changed)
                self._update_lb_prob(lb_prob, cut_Dict, changed)
                candidate_target_pair = _certified_original_target_pair(
                    original_target, pi, trial, last, shifts, keys)
                if candidate_target_pair is not None:
                    # Preserve S2's existing near-anchor classification and
                    # trace path. A distinct attained-target exit already has
                    # its valid support; it needs no further level LP solve.
                    near_anchor = (stage_no == 2 and _certified_anchor_tight_slack(
                        L, pi, trial, last.outer_lb, keys, _tight_exit_tolerance(L)) is not None)
                    if not near_anchor:
                        target_exit_pair = candidate_target_pair
                        converged = False
                        stop_reason = 'certified_target_attained_exit'
                        break
                if candidate_target_pair is None and scheduled_candidate is not None:
                    scheduled_gap_pair = scheduled_candidate
                    converged = False
                    stop_reason = 'scheduled_gap_effort_exit'
                    break
                if not solve_level(lb_prob):
                    if remaining_seconds(deadline) <= 0:
                        stop_reason = 'deadline'; break
                    L = self._fix_L_value_bisect(lb_prob, trial, keys, L, deadline=deadline)
                if lb_prob.Status != GRB.OPTIMAL:
                    stop_reason = 'level_not_optimal'; break
                # Restore Final's incumbent endpoint for S3 search geometry
                # and effort scheduling. For an inexact oracle this merit is
                # optimistic: its Delta/level may request an early checkpoint
                # or fail projection, but cannot certify convergence or a cut.
                # S2 retains the conservative same-pi lower endpoint. Targets,
                # emitted cuts and strict checkpoints still use certified LBs.
                search_endpoints = values if stage_no == 3 else outer_values
                merit_points = [(old_pi, bound) for old_pi, bound in zip(pi_history, search_endpoints)
                                if bound is not None and math.isfinite(bound)]
                if not merit_points:
                    stop_reason = 'no_certified_merit_endpoint'; break
                merit_pi = [point[0] for point in merit_points]
                merit_lower = [point[1] for point in merit_points]
                alpha_max, alpha_min, Delta = self._obtain_alpha_bounds(
                    merit_pi, L, trial, lb_prob.ObjVal, merit_lower, norm_option)
                if adaptive_alpha and alpha_max > alpha_min + 1e-10:
                    if delta_initial is None:
                        delta_initial = max(Delta, 1e-10)
                    if stage_no == 3 and iteration == 0:
                        # Final's first adaptive S3 step distinguishes an
                        # inherited physical bundle from a cold start.
                        init_ratio = 0.35 if has_warm_start else 0.2
                        alpha = alpha_min + init_ratio * (alpha_max-alpha_min)
                    else:
                        progress = max(0., min(1., 1. - Delta / delta_initial))
                        alpha = alpha_min + (0.2 + 0.15 * progress) * (alpha_max-alpha_min)
                elif iteration == 0 or (alpha_max > alpha_min + 1e-10 and
                        not mu_level / 2 <= (alpha-alpha_min)/(alpha_max-alpha_min) <= 1-mu_level/2):
                    alpha = (alpha_max+alpha_min)/2
                self._levelset_trace(stage_no, tag, iteration, lb_prob, keys, Delta, alpha, False)
                if stage_no == 2:
                    tight_exit_tolerance = _tight_exit_tolerance(L)
                    tight_exit_slack = _certified_anchor_tight_slack(
                        L, pi, trial, last.outer_lb, keys, tight_exit_tolerance)
                    if tight_exit_slack is not None:
                        # The emitted certified cut already reaches this
                        # trial target. Stop the secondary norm search without
                        # claiming its optimum or global SDDP convergence.
                        tight_exit = True
                        stop_reason = 'certified_anchor_tight_exit'
                        break
                target_exit_pair = _certified_original_target_pair(
                    original_target, pi, trial, last, shifts, keys)
                if target_exit_pair is not None:
                    # A valid cut can exceed a lower-bound target. Meeting
                    # that level stops this search, not the norm certificate
                    # or the outer SDDP gap computation.
                    converged = False
                    stop_reason = 'certified_target_attained_exit'
                    break
                if Delta < tol:
                    checkpoint_requested = True
                    counters['checkpoint_calls'] += 1
                    previous_pair = (dict(pi), last)
                    checkpoint_was_active = ambiguity.checkpoint_active
                    ambiguity.checkpoint_active = True
                    try:
                        pi, last, converged, L, checkpoint_records = self._run_certificate_checkpoint(
                            stage=stage_no, node_tag=tag, backend='gurobi_lrp', lb_prob=lb_prob,
                            cut_Dict=cut_Dict, pi_keys=keys, x_prev=trial, L_value=L,
                            level_tol=min(float(tol), _CERT_CHECKPOINT_GAP_ABS_TOL),
                            solve_at_pi=lambda point: evaluate(point, checkpoint=True), deadline=deadline)
                    finally:
                        ambiguity.checkpoint_active = checkpoint_was_active
                    records.extend(checkpoint_records)
                    if converged:
                        stop_reason = 'checkpoint_proven'
                        break
                    if stage_no == 3 and not resume_incomplete_s3_checkpoint:
                        # Final's ordinary route search pays for one bounded
                        # checkpoint, then returns its best certified cut.
                        # An unfinished norm proof is not a reason to spend
                        # the whole route budget before visiting new trials.
                        converged = False
                        stop_reason = ('deadline' if remaining_seconds(deadline) <= 0
                                       else 'iteration_limit' if iteration + 1 >= int(iter_limit)
                                       else 'scheduled_checkpoint_incomplete')
                        break
                    if stage_no == 3:
                        # Opt-in S3 continuation: keep the original target proof
                        # and incumbent/lower channels separate, but spend
                        # remaining node budget if its target is still unmet.
                        for point, evaluation in checkpoint_records:
                            target_exit_pair = _certified_original_target_pair(
                                original_target, point, trial, evaluation, shifts, keys)
                            if target_exit_pair is not None:
                                break
                        if target_exit_pair is not None:
                            converged = False
                            stop_reason = 'certified_target_attained_exit'
                            break
                        if not any(evaluation.has_inner for _, evaluation in checkpoint_records):
                            converged = False
                            stop_reason = 'scheduled_checkpoint_no_incumbent'
                            break
                        # The unchanged deadline/iteration guards and full
                        # bundle/paired-record resumption below apply to S3.
                    if remaining_seconds(deadline) <= 0:
                        stop_reason = 'deadline'
                        break
                    if iteration + 1 >= int(iter_limit):
                        stop_reason = 'iteration_limit'
                        break
                    # A bounded checkpoint batch is a certification attempt,
                    # not a stopping certificate. Continue the same bundle
                    # problem while the node still has iteration/time budget.
                    # Keep every oracle value paired with its own multiplier:
                    # upper incumbents add supports; only lower certificates
                    # enter the merit history. A final bound-only checkpoint
                    # remains available for cut selection, but cannot become
                    # the next iteration's incumbent support.
                    incumbent_records = [(point, evaluation) for point, evaluation
                                         in checkpoint_records if evaluation.has_inner]
                    pi, last = incumbent_records[-1] if incumbent_records else previous_pair
                    for point, evaluation in incumbent_records[:-1]:
                        pi_history.append(dict(point)); values.append(evaluation.inner_value)
                        outer_values.append(evaluation.outer_lb)
                        states.append(dict(evaluation.inner_xcp))
                    # Checkpoint separation updated lb_prob, possibly replacing
                    # a same-slope support or decreasing the effective target.
                    # Rebuilding from the full bundle synchronizes all such
                    # changes, including the ordinary iteration's pending row.
                    replacement = self._build_next_pi_prob(
                        1e7, alpha, trial, L, cut_Dict, norm_option, level_tol=tol, env=getattr(self, "_model_env", None))
                    previous_projection = next_prob
                    next_prob = replacement
                    model_pair[-1] = next_prob
                    previous_projection.dispose()
                    checkpoint_resumptions += 1
                    counters['checkpoint_resumptions'] = counters.get('checkpoint_resumptions', 0) + 1
                    continue
                if remaining_seconds(deadline) <= 0:
                    stop_reason = 'deadline'; break
                norms = [sum((abs(v) if norm_option == 1 else v*v) for v in old.values())
                         for old in merit_pi]
                gaps = [L - math.fsum(old[k]*trial[k] for k in keys) - val
                        for old, val in zip(merit_pi, merit_lower)]
                upper = min(alpha*n + (1-alpha)*g for n,g in zip(norms,gaps))
                level = lambda_level*upper + (1-lambda_level)*alpha*lb_prob.ObjVal
                self._update_next_pi_prob(next_prob, cut_Dict, changed, alpha, trial,
                                          L, pi, level, norm_option)
                if not prepare_model_solve(next_prob, deadline):
                    stop_reason = 'deadline'; break
                counters['level_projection_solves'] += 1
                projection_count += 1
                _optimize_gurobi(next_prob, 'level_projection')
                if next_prob.Status != GRB.OPTIMAL:
                    stop_reason = 'projection_not_optimal'; break
                pi = self._lb_optimizer_pi(next_prob, keys)
                last = evaluate(pi)
            records.append((dict(pi), last))
            norm_checkpoint_proven = bool(checkpoint_requested and converged)
            selection_diagnostic = {}
            if tight_exit or target_exit_pair is not None or scheduled_gap_pair is not None:
                # Preserve the exact pi/lower-certificate pair that triggered
                # the stop. A different previously feasible pair need not be
                # near this anchor, and this is not a checkpoint proof.
                selection_diagnostic.update(reason=('scheduled_gap_pair_preserved'
                    if scheduled_gap_pair is not None else 'certified_target_pair_preserved'
                    if target_exit_pair is not None else 'certified_anchor_pair_preserved'),
                    effective_residual_target=L, norm_option=norm_option,
                    strict_target_feasible_records=None,
                    selected_residual_norm=float(sum((abs(Fraction.from_float(float(pi[key])))
                        if norm_option == 1 else Fraction.from_float(float(pi[key]))**2
                        for key in keys), Fraction())))
            else:
                pi, last = _select_levelset_result(pi, last, records, trial, keys,
                    checkpoint_proven=norm_checkpoint_proven, effective_target=L,
                    norm_option=norm_option, selection_diagnostic=selection_diagnostic)
            residual_pi, residual_target = dict(pi), L
            if target_exit_pair is not None:
                pi, last = target_exit_pair[:2]
            elif scheduled_gap_pair is not None:
                pi, last = scheduled_gap_pair[:2]
            else:
                pi, last = _lift_centered_evaluation(pi, last, shifts)
            intercept = last.outer_lb if last.has_outer else None
            subgradient = ({key: -last.inner_xcp[key] for key in keys} if last.has_inner else {})
            diagnostic = build_s2_cut_diagnostic(node=node_ind, L=original_target, pi=pi,
                x_trial=trial, V_inner=last.inner_value, V_outer=intercept,
                actual_iterations=actual_iterations, stop_reason=stop_reason,
                backend='gurobi_lrp', checkpoint_requested=checkpoint_requested,
                checkpoint_proven=converged, selected_is_checkpoint=norm_checkpoint_proven,
                tight_exit=tight_exit)
            diagnostic.update(stage=stage_no, norm_option=norm_option,
                configured_route_backend=route_backend if stage_no == 3 else None,
                configured_assignment_backend=assignment_backend,
                selected_oracle_source=last.source,
                levelset_solver_tolerances={
                    'requested_delta_tolerance': float(tol),
                    'master': {name: float(getattr(lb_prob.Params, name)) for name in
                        ('FeasibilityTol', 'OptimalityTol', 'BarConvTol', 'BarQCPConvTol')},
                    'projection': {name: float(getattr(next_prob.Params, name)) for name in
                        ('FeasibilityTol', 'OptimalityTol', 'BarConvTol', 'BarQCPConvTol')}},
                projection_solves=projection_count,
                checkpoint_resumptions=checkpoint_resumptions,
                oracle_solves=counters['oracle_solves']-oracle_count_before,
                oracle_evaluations=memo.native_solves,
                oracle_memo=memo.telemetry(), context=ctx.key,
                oracle_precision=oracle_precision,
                oracle_model_reuse=oracle_model.statistics,
                same_pi_refinement=refinement.statistics,
                selected_cut_valid=intercept is not None,
                requested_target=requested_target, certified_target=original_target,
                residual_centering=centered, residual_target=residual_target,
                native_residual_view=native_residual_enabled and native_free_oracle is not None,
                native_probe_seconds=native_probe_seconds,
                native_probe_time_limits=native_probe_time_limits,
                residual_pi=residual_pi, incoming_shifts=shifts,
                bundle_reset_for_coordinate_change=bundle_reset,
                target_certificate=target_diagnostic,
                merit_endpoint=('heuristic_same_multiplier_incumbent_upper' if stage_no == 3 else 'same_multiplier_certified_lower_bound'),
                level_target_feasibility_certified=_certified_level_target_attained(
                    original_target, pi, trial, intercept, keys, _CERT_CHECKPOINT_GAP_ABS_TOL),
                norm_checkpoint_proven=norm_checkpoint_proven,
                anchor_exit_tolerance=tight_exit_tolerance if tight_exit else None,
                anchor_exit_signed_slack=float(tight_exit_slack) if tight_exit else None,
                anchor_exit_effective_target=residual_target if tight_exit else None,
                target_attained_exit=target_exit_pair is not None,
                target_exit_original_target=original_target if target_exit_pair is not None else None,
                target_exit_signed_slack=float(target_exit_pair[2]) if target_exit_pair is not None else None,
                target_exit_tolerance=target_exit_pair[3] if target_exit_pair is not None else None,
                scheduled_gap_effort_exit=scheduled_gap_pair is not None,
                scheduled_gap_original_target=original_target if scheduled_gap_pair is not None else None,
                scheduled_gap_signed_slack=float(scheduled_gap_pair[2]) if scheduled_gap_pair is not None else None,
                scheduled_gap_tolerance=scheduled_gap_pair[3] if scheduled_gap_pair is not None else None,
                result_selection=selection_diagnostic,
                initial_pi_source=initial_pi_source)
            if stage_no == 3:
                diagnostic['fixed_trial_support'] = trial_support_diagnostic
            if ambiguity.enabled:
                # A terminal oracle query after the final geometric iteration
                # may leave pending witnesses. No unsolved model update or
                # norm proof is inferred from those unflushed terminal rows.
                diagnostic['ambiguous_pi_refinement'] = ambiguity.telemetry()
            self.last_cut_diagnostic = diagnostic
            if stage_no == 2:
                self.last_s2_cut_diagnostic = diagnostic
            return dict(pi), subgradient, intercept, cut_Dict, converged
        except SolveDeadlineReached:
            # A pre-solve budget check can pass and the matrix audit/model
            # preparation can consume its remaining time. This is a normal
            # node/epoch stop, including in a worker, not a failed outer run.
            # No endpoint from the interrupted solve is usable. The caller
            # retains its existing cuts/SBC seed; caches and bundle supports
            # already accepted before this call stopped remain available.
            counters['deadline_stops'] = counters.get('deadline_stops', 0) + 1
            diagnostic = dict(stage=stage_no, node=node_ind, context=ctx.key,
                method='levelset', stop_reason='deadline', deadline=deadline,
                hard_deadline=hard_deadline, requested_target=requested_target,
                target_accepted=False, selected_cut_valid=False,
                actual_iterations=actual_iterations,
                oracle_solves=counters['oracle_solves']-oracle_count_before,
                level_target_feasibility_certified=False,
                norm_checkpoint_proven=False, checkpoint_proven=False,
                target_attained_exit=False, converged=False)
            self.last_cut_diagnostic = diagnostic
            if stage_no == 2:
                self.last_s2_cut_diagnostic = diagnostic
            return dict(pi), {}, None, cut_Dict, False
        finally:
            if stage_no == 3 and self.last_cut_diagnostic is not None:
                self.last_cut_diagnostic['s3_resume_incomplete_checkpoint'] = bool(
                    resume_incomplete_s3_checkpoint)
            oracle_model.close()
            for model in model_pair:
                model.dispose()
