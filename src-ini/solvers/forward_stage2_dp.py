"""Adapter: certified-gap subset-DP forward Stage-2.

Used by both forward passes (serial and pool workers) and by the Phase-2
backward S2 refresh.  ``try_dp_forward_stage2`` returns ``None`` when the DP is
not applicable or disabled (``VRP_S2_FORWARD_SOLVER=gurobi``); callers then
run the Gurobi model exactly as before.  Every DP result passes policy
feasibility certification and complete-archive rescoring.  The explicit solve
purpose makes Phase-2 subset-DP results require a rigorous interval no wider
than ``1e-6``.  Phase 1 accepts either a certified feasible probe incumbent or
a feasible subset-DP warm-start policy without claiming optimality.  A
full-Gurobi fallback retains its caller's existing MIPGap and time-limit
contract, so the DP acceptance rule is not an all-backend proof.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict

import core.customized_subprob  # noqa: F401  (sys.path for s2forward)
from core.backend_telemetry import backend_call, record_backend_event
from core.exact_solver_log import log as exact_log, log_fallback as exact_log_fallback
from core.solver_bounds import certified_gurobi_minimization_lower_bound
from models.learned_cut_dispatch import (
    collect_s2_model_stats,
    merge_s2_model_stats,
)
from s2forward.purpose import (
    PHASE1_FEASIBLE_TRIAL,
    PHASE2_FORWARD,
    PHASE2_REFRESH,
    ForwardStage2Purpose,
    normalize_forward_stage2_purpose,
    forward_stage2_backend_scope,
)
from s2forward.outer_reuse import (
    ForwardStage2OuterReuseCache,
    format_outer_reuse_stats,
    stage2_outer_reuse_key,
)
from s2forward.verified_lp_bound import stabilize_fixed_fleet_bound, tiny_bound_conflict

_LOG_BACKEND = "subset_dp"


@dataclass
class ForwardStage2SolveResult:
    """Backend-neutral, independently certified Stage-2 forward result."""

    x_dict: Dict[str, float]
    cost_star_value: float
    stage_cost_value: float
    theta_by_succ: Dict[Any, float]
    model_stats: Dict[str, Any]
    exact_optimal: bool
    backend: str
    decision_reason: str
    objective_lower_bound: float = float("-inf")
    status: Any = None


def _from_dp_result(result, *, backend: str, reason: str, extra_stats=None,
                    kernel: str = "dp"):
    stats = dict(result.model_stats)
    if extra_stats:
        stats = merge_s2_model_stats([extra_stats, stats])
    stats["backend_dp"] = int(kernel == "dp")
    stats["backend_bp"] = int(kernel == "bp")
    stats["backend_gurobi"] = int(bool(extra_stats))
    stats["backend_probe"] = int(bool(extra_stats))
    return ForwardStage2SolveResult(
        x_dict=dict(result.x_dict),
        cost_star_value=float(result.cost_star_value),
        stage_cost_value=float(result.stage_cost_value),
        theta_by_succ=dict(result.theta_by_succ),
        model_stats=stats,
        exact_optimal=bool(result.optimality_certified),
        backend=backend,
        decision_reason=reason,
        objective_lower_bound=float(result.objective_lower_bound),
        status="certified_interval",
    )


def _extract_gurobi_result(
    model,
    prob_data,
    node,
    cut_lag,
    x_prev,
    *,
    node_idx,
    optimize_model,
    solve_seconds,
    exact_optimal,
    backend,
    reason,
    direction,
    had_probe=False,
):
    """Certify and rescore a Gurobi incumbent; never trust raw ``ObjVal``."""
    from gurobipy import GRB
    from solvers.forward_incumbent import (
        certify_stage2_incumbent_with_retighten,
        stage2_all_outsource_if_interrupted_without_incumbent,
    )
    from solvers.forward_stage2_policy import (
        all_outsource_forward_solution,
        score_forward_stage2_policy_from_archive,
    )

    certification_started = time.perf_counter()
    fallback = stage2_all_outsource_if_interrupted_without_incumbent(
        model,
        prob_data,
        node,
        cut_lag,
        node_idx=node_idx,
    )
    if fallback is not None:
        exact_log_fallback(
            direction,
            2,
            "gurobi",
            "time/resource limit without incumbent -> certified all-outsource policy",
            indent=4,
        )
        x_dict = dict(fallback["x_star_dict"])
        cost_star = float(fallback["cost_star_value"])
        stage_cost = float(fallback["stage_cost_value"])
        theta_by_succ = dict(fallback["theta_by_succ"])
        exact_optimal = False
    else:
        x_dict, _ = certify_stage2_incumbent_with_retighten(
            model,
            prob_data,
            node,
            x_prev,
            node_idx=node_idx,
            optimize=optimize_model,
        )
        score = score_forward_stage2_policy_from_archive(
            prob_data, node, x_dict, cut_lag
        )
        cost_star = float(score["cost_star_value"])
        stage_cost = float(score["stage_cost_value"])
        theta_by_succ = dict(score["theta_by_succ"])
        # Exact-probe status must describe the *final* model solve.  Capacity
        # certification may add valid cover rows, reset, and re-optimize; a
        # time-limited retry still gives a valid UB but cannot seed the exact
        # outer-reuse cache.
        exact_optimal = bool(
            exact_optimal and int(model.Status) == int(GRB.OPTIMAL)
            and not getattr(model, "_forward_certification_fallback", False)
        )
    # Read the bound after any certification re-solve, not from the rejected
    # pre-retighten model state.
    lower_bound = certified_gurobi_minimization_lower_bound(model)
    prior_bound = getattr(model, "_forward_certification_lower_bound", None)
    candidates = [float(value) for value in (lower_bound, prior_bound)
                  if value is not None and math.isfinite(float(value))
                  and float(value) <= cost_star]
    # Retightening may expose a numerical contradiction in an earlier bound.
    # Keep only actual solver bounds consistent with the exact policy rescore;
    # never substitute that feasible upper cost for a rejected certificate.
    lower_bound = max(candidates) if candidates else None
    raw_bound = raw_incumbent = None
    if lower_bound is None:
        try:
            raw_bound = float(model.ObjBound)
            raw_incumbent = float(model.ObjVal)
        except Exception:
            pass
    recover_roundoff = lower_bound is None and (
        tiny_bound_conflict(raw_bound, cost_star)
        or tiny_bound_conflict(raw_bound, raw_incumbent)
    )
    if lower_bound is not None or recover_roundoff:
        # A forward assignment is physically certified, not necessarily in
        # the canonical rank domain. All-outsource is canonical for every
        # fixed fleet, so only its rescored cost supplies the LP's safe box.
        all_outsource = all_outsource_forward_solution(prob_data, node, cut_lag)
        verification = stabilize_fixed_fleet_bound(
            model, lower_bound,
            policy_upper_bound=all_outsource["cost_star_value"],
            label=f"Stage2 fixed forward node={node_idx}",
            recover_missing_bound=recover_roundoff,
        )
        lower_bound = verification["bound"]
        if lower_bound is not None and lower_bound > cost_star:
            lower_bound = None
        if recover_roundoff:
            record_backend_event(
                "gurobi", "bound_recovery",
                "verified_lp" if lower_bound is not None else "no_verified_bound",
                node=node_idx, raw_bound=raw_bound, raw_incumbent=raw_incumbent,
                rescored_policy_upper=cost_star, lower_bound=lower_bound,
                verification_status=verification["status"],
                verification_seconds=verification["seconds"],
            )
    solve_seconds = float(solve_seconds) + (
        time.perf_counter() - certification_started
    )
    model_stats = collect_s2_model_stats(model, solve_seconds=solve_seconds)
    model_stats["backend_dp"] = 0
    model_stats["backend_gurobi"] = 1
    model_stats["backend_probe"] = int(
        bool(had_probe) or backend == "gurobi_probe"
    )
    # GRB.OPTIMAL is a numerical solver status.  Cross-outer reuse makes a
    # stronger claim, so require its directed lower endpoint to equal the
    # policy's exact rational score over the represented binary64 data.
    if exact_optimal:
        from s2forward.mip_start import exact_forward_policy_score

        try:
            policy_exact = exact_forward_policy_score(
                prob_data, node, cut_lag, x_dict
            )
            exact_optimal = bool(
                lower_bound is not None
                and Fraction.from_float(float(lower_bound)) == policy_exact
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            exact_optimal = False
    record_backend_event(
        "gurobi", "accepted", "certified_forward_policy", node=node_idx,
        selected_backend=backend, status=int(model.Status),
        exact_cache_proven=bool(exact_optimal),
        all_outsource_fallback=bool(
            fallback is not None or getattr(model, "_forward_certification_fallback", False)
        ),
    )
    return ForwardStage2SolveResult(
        x_dict=dict(x_dict),
        cost_star_value=cost_star,
        stage_cost_value=stage_cost,
        theta_by_succ=theta_by_succ,
        model_stats=model_stats,
        exact_optimal=bool(exact_optimal),
        backend=backend,
        decision_reason=reason,
        objective_lower_bound=(
            float(lower_bound) if lower_bound is not None else float("-inf")
        ),
        status=int(model.Status),
    )


def try_dp_forward_stage2(
    prob_data,
    node,
    cut_lag,
    x_prev,
    *,
    node_idx=None,
    concurrent_slots=None,
    purpose: ForwardStage2Purpose | str = PHASE2_FORWARD,
):
    from s2forward.forward_stage2_dp import solve_forward_stage2_by_dp

    purpose = normalize_forward_stage2_purpose(purpose)
    direction = purpose.log_direction

    def _log(message):
        exact_log(direction, 2, _LOG_BACKEND, message)

    started = time.time()
    try:
        result = solve_forward_stage2_by_dp(
            prob_data,
            node,
            cut_lag,
            x_prev,
            node_idx=node_idx,
            log=_log,
            concurrent_slots=concurrent_slots,
            purpose=purpose,
        )
    except Exception as exc:
        record_backend_event("subset_dp", "error", type(exc).__name__,
                             node=node_idx, detail=str(exc))
        raise
    if result is None:
        return None
    _log(
        f"node={node_idx} n_active={result.n_active} fleet={result.counts} "
        f"cost_star={result.cost_star_value:,.6f} "
        f"cert_gap={result.certified_gap:.3e} "
        f"purpose={purpose.value} "
        f"tight_required={int(purpose.requires_tight_dp_gap)} "
        f"in {time.time() - started:.2f}s"
    )
    return result


def try_bp_forward_stage2(
    prob_data,
    node,
    cut_lag,
    x_prev,
    *,
    node_idx=None,
    purpose: ForwardStage2Purpose | str = PHASE2_FORWARD,
    full_time_limit=None,
):
    """Run fixed-fleet branch-and-price; never raises for kernel problems.

    Returns a ``BPForwardOutcome``; ``result`` is ``None`` when the caller
    must continue with Gurobi (``warm_start`` then carries any certified
    incumbent).  The kernel budget is the smaller of the caller's Stage-2
    time limit and ``VRP_S2_BP_TIME_LIMIT``.
    """
    from s2forward.bp_solver import BPForwardOutcome, bp_time_limit, solve_forward_stage2_by_bp

    purpose = normalize_forward_stage2_purpose(purpose)
    direction = purpose.log_direction

    def _log(message):
        exact_log(direction, 2, "branch_price", message)

    limit = bp_time_limit()
    try:
        caller = float(full_time_limit) if full_time_limit is not None else math.inf
    except (TypeError, ValueError, OverflowError):
        caller = math.inf
    if math.isfinite(caller) and caller > 0.0:
        limit = min(limit, caller)
    try:
        return solve_forward_stage2_by_bp(
            prob_data,
            node,
            cut_lag,
            x_prev,
            node_idx=node_idx,
            log=_log,
            purpose=purpose,
            time_limit=limit,
        )
    except Exception as exc:  # defensive: a kernel defect must not kill the pass
        exact_log_fallback(
            direction, 2, "branch_price",
            f"node={node_idx} branch-and-price raised {type(exc).__name__}: {exc}; Gurobi",
        )
        return BPForwardOutcome(None, None, f"exception {type(exc).__name__}: {exc}")


@forward_stage2_backend_scope
def solve_forward_stage2_dispatched(
    prob_data,
    node,
    cut_lag,
    x_prev,
    *,
    node_idx,
    purpose: ForwardStage2Purpose | str = PHASE2_FORWARD,
    build_gurobi_model: Callable[[], Any],
    configure_full_gurobi: Callable[[Any], None],
    optimize_gurobi: Callable[[Any], None],
    full_time_limit=None,
    concurrent_slots=None,
) -> ForwardStage2SolveResult:
    """Run the certificate-safe DP/Gurobi crossover policy for one fixed fleet.

    In Phase 1, any incumbent found by the short Gurobi probe is independently
    feasibility-certified and archive-rescored, then accepted as a warm-start
    trial point; neither its MIP gap nor exact-cache certificate is required.
    Phase 2 accepts a probe directly only with the strict exact-cache proof;
    otherwise the fixed-fleet DP must provide a certified interval no wider
    than 1e-6.  If DP is outside its memory/domain gate (or its Phase-2
    interval is wider than 1e-6), the same Gurobi model continues under the
    caller's normal MIPGap/time budget; that fallback is not relabelled as a
    tight DP solve.
    Thus the dispatcher changes runtime only; every returned policy still
    passes the shared feasibility check and complete-archive rescore.  Only a
    strictly closed interval may seed cross-outer exact reuse.
    """
    from gurobipy import GRB
    from s2forward.dispatch_policy import build_forward_dispatch_plan

    purpose = normalize_forward_stage2_purpose(purpose)
    direction = purpose.log_direction
    plan = build_forward_dispatch_plan(
        prob_data,
        node,
        cut_lag,
        x_prev,
        phase=purpose.dispatch_phase,
        concurrent_slots=concurrent_slots,
    )
    if plan.features.phase != purpose.dispatch_phase:
        raise RuntimeError(
            "forward Stage-2 dispatcher returned a plan for "
            f"{plan.features.phase!r}, expected {purpose.dispatch_phase!r}"
        )
    exact_log(
        direction,
        2,
        "dispatch",
        f"node={node_idx} purpose={purpose.value} "
        f"phase={purpose.dispatch_phase} backend={plan.backend} "
        f"active={plan.features.active_customers} "
        f"purchased={plan.features.purchased_vehicles} "
        f"cuts_by_rank={plan.features.private_cut_pool_sizes} "
        f"slots={plan.features.concurrent_slots} "
        f"memory_safe={plan.features.memory_safe} "
        f"memory_max={plan.features.memory_max_customers} "
        f"probe={float(plan.probe_seconds):.3f}s reason={plan.reason}",
    )
    record_backend_event(
        "dispatch", "selection", plan.reason, node=node_idx,
        selected_backend=plan.backend, purpose=purpose.value,
        active_customers=plan.features.active_customers,
        memory_safe=plan.features.memory_safe,
        memory_max_customers=plan.features.memory_max_customers,
        probe_seconds=float(plan.probe_seconds),
    )
    if plan.backend not in ("dp", "probe_then_dp"):
        record_backend_event("subset_dp", "skip", plan.reason, node=node_idx)
    if plan.backend != "bp_then_gurobi":
        record_backend_event("branch_price", "skip", plan.reason, node=node_idx)

    if plan.backend == "dp":
        dp_result = try_dp_forward_stage2(
            prob_data,
            node,
            cut_lag,
            x_prev,
            node_idx=node_idx,
            concurrent_slots=concurrent_slots,
            purpose=purpose,
        )
        if dp_result is not None:
            return _from_dp_result(
                dp_result, backend="subset_dp", reason=plan.reason
            )
        # ``auto`` can reach this only if a structural DP assumption rejected
        # the node after the cheap policy screen.  Preserve the old Gurobi
        # fallback.  Forced ``dp`` already raises in the lower layer.
        record_backend_event(
            "subset_dp", "fallback", "dp_not_accepted", node=node_idx,
            fallback_backend="gurobi",
        )

    bp_warm_start = None
    if plan.backend == "bp_then_gurobi":
        bp_outcome = try_bp_forward_stage2(
            prob_data,
            node,
            cut_lag,
            x_prev,
            node_idx=node_idx,
            purpose=purpose,
            full_time_limit=full_time_limit,
        )
        if bp_outcome.result is not None:
            return _from_dp_result(
                bp_outcome.result,
                backend="branch_price",
                reason=plan.reason,
                kernel="bp",
            )
        bp_warm_start = bp_outcome.warm_start
        record_backend_event(
            "branch_price", "fallback", bp_outcome.reason, node=node_idx,
            fallback_backend="gurobi", warm_start=bool(bp_warm_start),
        )
        exact_log(
            direction,
            2,
            "dispatch",
            f"node={node_idx} branch-and-price not accepted "
            f"({bp_outcome.reason}); full Gurobi"
            f"{' from its incumbent' if bp_warm_start else ''}",
        )

    model = build_gurobi_model()
    optimize_seconds = 0.0
    try:
        if bp_warm_start:
            from s2forward.mip_start import apply_forward_binary_start

            apply_forward_binary_start(model, prob_data, node, x_prev, bp_warm_start)
        if plan.backend == "feasible_probe":
            if purpose is not PHASE1_FEASIBLE_TRIAL:
                raise RuntimeError(
                    "feasible_probe is valid only for PHASE1_FEASIBLE_TRIAL"
                )
            configure_full_gurobi(model)
            probe_limit = float(plan.probe_seconds)
            if full_time_limit is not None:
                try:
                    limit = float(full_time_limit)
                except (TypeError, ValueError, OverflowError):
                    limit = math.inf
                if math.isfinite(limit) and limit > 0.0:
                    probe_limit = min(probe_limit, limit)
            model.setParam("MIPGap", 0.0)
            model.setParam("MIPGapAbs", 0.0)
            model.setParam("TimeLimit", max(probe_limit, 1e-6))
            started = time.perf_counter()
            with backend_call("gurobi", "optimize", model=model,
                              attempt_kind="feasible_probe", node=node_idx):
                optimize_gurobi(model)
            optimize_seconds += time.perf_counter() - started
            result = _extract_gurobi_result(
                model,
                prob_data,
                node,
                cut_lag,
                x_prev,
                node_idx=node_idx,
                optimize_model=optimize_gurobi,
                solve_seconds=optimize_seconds,
                exact_optimal=False,
                backend="gurobi_probe_feasible",
                reason=plan.reason,
                direction=direction,
                had_probe=True,
            )
            exact_log(
                direction,
                2,
                "dispatch",
                f"node={node_idx} probe_status={int(model.Status)} Phase1 "
                "accepts certified feasible policy (incumbent or "
                "all-outsource fallback)",
            )
            return result

        if plan.backend == "probe_then_dp":
            configure_full_gurobi(model)
            # Save the caller's real solve policy before temporarily forcing
            # an exact probe.  In particular, Phase 2 relies on Gurobi's
            # ordinary MIPGap; without this restore a failed probe would make
            # the subsequent full solve silently use MIPGap=0 forever.
            full_mip_gap = float(model.Params.MIPGap)
            full_mip_gap_abs = float(model.Params.MIPGapAbs)
            full_time_limit_param = float(model.Params.TimeLimit)
            probe_limit = float(plan.probe_seconds)
            if full_time_limit is not None:
                try:
                    limit = float(full_time_limit)
                except (TypeError, ValueError, OverflowError):
                    limit = math.inf
                if math.isfinite(limit) and limit > 0.0:
                    probe_limit = min(probe_limit, limit)
            model.setParam("MIPGap", 0.0)
            model.setParam("MIPGapAbs", 0.0)
            model.setParam("TimeLimit", max(probe_limit, 1e-6))
            started = time.perf_counter()
            with backend_call("gurobi", "optimize", model=model,
                              attempt_kind="exact_probe", node=node_idx):
                optimize_gurobi(model)
            optimize_seconds += time.perf_counter() - started
            if int(model.Status) == int(GRB.OPTIMAL):
                probe_backend = (
                    "gurobi_probe_feasible"
                    if purpose is PHASE1_FEASIBLE_TRIAL
                    else "gurobi_probe"
                )
                certified_probe = _extract_gurobi_result(
                    model,
                    prob_data,
                    node,
                    cut_lag,
                    x_prev,
                    node_idx=node_idx,
                    optimize_model=optimize_gurobi,
                    solve_seconds=optimize_seconds,
                    exact_optimal=True,
                    backend=probe_backend,
                    reason=plan.reason,
                    direction=direction,
                    had_probe=True,
                )
                if (
                    certified_probe.exact_optimal
                    or purpose is PHASE1_FEASIBLE_TRIAL
                ):
                    if not certified_probe.exact_optimal:
                        exact_log(
                            direction,
                            2,
                            "dispatch",
                            f"node={node_idx} Phase1 accepts certified "
                            "feasible probe incumbent; exact-cache proof "
                            "not required",
                        )
                    return certified_probe
                # The original solve proved its numerical model, but exact
                # incumbent certification had to re-optimize and the final
                # solve was interrupted.  Keep only telemetry/bound and run
                # the certified-gap DP as for every other failed probe.
                probe_bound = certified_probe.objective_lower_bound
                probe_stats = certified_probe.model_stats
            else:
                probe_bound = certified_gurobi_minimization_lower_bound(model)
                probe_stats = collect_s2_model_stats(
                    model, solve_seconds=optimize_seconds
                )
                if (
                    purpose is PHASE1_FEASIBLE_TRIAL
                    and int(getattr(model, "SolCount", 0)) > 0
                ):
                    feasible_probe = _extract_gurobi_result(
                        model,
                        prob_data,
                        node,
                        cut_lag,
                        x_prev,
                        node_idx=node_idx,
                        optimize_model=optimize_gurobi,
                        solve_seconds=optimize_seconds,
                        exact_optimal=False,
                        backend="gurobi_probe_feasible",
                        reason=(
                            f"{plan.reason}; Phase1 feasible probe incumbent"
                        ),
                        direction=direction,
                        had_probe=True,
                    )
                    exact_log(
                        direction,
                        2,
                        "dispatch",
                        f"node={node_idx} probe_status={int(model.Status)} "
                        "Phase1 accepts certified feasible incumbent",
                    )
                    return feasible_probe
            exact_log(
                direction,
                2,
                "dispatch",
                f"node={node_idx} probe_status={int(model.Status)} "
                f"bound={probe_bound} -> subset DP purpose={purpose.value}",
            )
            record_backend_event(
                "gurobi", "fallback", "probe_not_accepted", node=node_idx,
                fallback_backend="subset_dp", probe_status=int(model.Status),
            )
            dp_result = try_dp_forward_stage2(
                prob_data,
                node,
                cut_lag,
                x_prev,
                node_idx=node_idx,
                concurrent_slots=concurrent_slots,
                purpose=purpose,
            )
            if dp_result is not None:
                return _from_dp_result(
                    dp_result,
                    backend="gurobi_probe_then_subset_dp",
                    reason=plan.reason,
                    extra_stats=probe_stats,
                )
            # Structural rejection discovered after the screen: continue the
            # already-built model under its normal budget.
            record_backend_event(
                "subset_dp", "fallback", "dp_not_accepted_after_probe",
                node=node_idx, fallback_backend="gurobi",
            )
            model.setParam("MIPGap", full_mip_gap)
            model.setParam("MIPGapAbs", full_mip_gap_abs)
            model.setParam("TimeLimit", full_time_limit_param)

        configure_full_gurobi(model)
        started = time.perf_counter()
        with backend_call("gurobi", "optimize", model=model,
                          attempt_kind="full", node=node_idx):
            optimize_gurobi(model)
        optimize_seconds += time.perf_counter() - started
        return _extract_gurobi_result(
            model,
            prob_data,
            node,
            cut_lag,
            x_prev,
            node_idx=node_idx,
            optimize_model=optimize_gurobi,
            solve_seconds=optimize_seconds,
            exact_optimal=False,
            backend="gurobi",
            reason=plan.reason,
            direction=direction,
            had_probe=(plan.backend == "probe_then_dp"),
        )
    finally:
        try:
            model.dispose()
        except Exception:
            pass


__all__ = [
    "ForwardStage2OuterReuseCache",
    "ForwardStage2Purpose",
    "ForwardStage2SolveResult",
    "PHASE1_FEASIBLE_TRIAL",
    "PHASE2_FORWARD",
    "PHASE2_REFRESH",
    "format_outer_reuse_stats",
    "solve_forward_stage2_dispatched",
    "stage2_outer_reuse_key",
    "try_dp_forward_stage2",
]
