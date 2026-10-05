"""Feasible Stage-2 targets obtained without optimizing another MIP."""
from __future__ import annotations

import math

from solvers.forward_policy_certification import certify_stage2_forward_policy
from solvers.forward_stage2_policy import score_forward_stage2_policy_from_archive


def weighted_stage2_interval(scen_tree, scores):
    """Diagnostic width for the current fixed-fleet surrogate, not a global gap.

    Every period/scenario contributes its original weight, including dedup
    copies. Rescoring alone has no finite lower endpoint: report unknown
    instead of turning a good score at one assignment into S2 closure.
    """
    widths = []
    for index in scen_tree[1][0].successor:
        score = scores.get(index, {})
        try:
            lower = float(score["objective_lower_bound"])
            upper = float(score["cost_star_value"])
            weight = float(scen_tree[2][index].multi_coeff)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if (not all(math.isfinite(value) for value in (lower, upper, weight))
                or weight < 0.0 or lower > upper):
            return None
        widths.append(weight * (upper - lower))
    total = math.fsum(widths)
    return total if math.isfinite(total) else None


def stage2_search_diagnostic(scen_tree, scores, abs_tol=None):
    """Separate fixed-fleet search accuracy from one policy's route residual.

    The scheduled per-node epsilon is aggregated with the original scenario
    weights. Without a schedule only the 1e-6 numerical band is allowed. This
    local surrogate interval is not the outer problem's optimality gap.
    """
    tolerance = 1e-6 if abs_tol is None else float(abs_tol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Stage-2 absolute tolerance must be finite and nonnegative")
    weights = [float(scen_tree[2][index].multi_coeff)
               for index in scen_tree[1][0].successor]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("Stage-2 weights must be finite and nonnegative")
    limit = math.fsum(weights) * tolerance
    interval = weighted_stage2_interval(scen_tree, scores)
    sufficient = interval is not None and interval <= limit
    return {
        "s2_interval": interval,
        "s2_gap_tol": limit,
        "s2_search_sufficient": sufficient,
        "s2_search_status": "unknown" if interval is None else
                            ("within_tolerance" if sufficient else "open"),
    }


def retry_fleet_table_state(state, counts):
    """Ignore only this fleet's stall hint for an explicitly budgeted retry.

    Copy-on-write preserves shared period tables, all certified lower bounds,
    stored policies and other fleets' scheduling hints. Closed intervals still
    use the normal cache hit; clearing a hint never asks for exact optimality.
    """
    if state is None or counts is None:
        return state
    record = state.get("records", {}).get(counts)
    if record is None or record.get("stalled_at") is None:
        return state
    return {**state, "records": {
        **state["records"], counts: {**record, "stalled_at": None},
    }}


def rescore_stage2_policy(prob_data, node, x_prev, policy, cut_lag):
    """Certify this trial fleet/assignment and score the complete new archive.

    The returned objective is a feasible surrogate upper bound, not a lower
    bound or an optimality certificate.  Re-scoring does not create another
    refresh-handoff opportunity.
    """
    certified, _ = certify_stage2_forward_policy(prob_data, node, x_prev, policy)
    score = score_forward_stage2_policy_from_archive(
        prob_data, node, certified, cut_lag
    )
    return certified, {
        "stage_cost": score["stage_cost_value"],
        "theta_by_succ": dict(score["theta_by_succ"]),
        "cost_star_value": score["cost_star_value"],
        "objective_lower_bound": float("-inf"),
        "exact_optimal": False,
        "backend": "rescore",
        "status": None,
        "fresh_solve": False,
    }
