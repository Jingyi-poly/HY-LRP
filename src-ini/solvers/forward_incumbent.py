"""Feasible-incumbent boundary shared by Phase-1/2 forward solvers."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from gurobipy import GRB
from core.backend_telemetry import backend_call, record_backend_event

import core.customized_subprob  # noqa: F401
from core.solver_bounds import certified_gurobi_minimization_lower_bound
from solver_budget import (
    forward_retry_deadline,
    prepare_model_solve,
    remaining_seconds,
)
from solvers.forward_stage2_policy import all_outsource_forward_solution


_INTERRUPTED_WITH_SAFE_FALLBACK = frozenset(
    status
    for status in (
        getattr(GRB, "TIME_LIMIT", None),
        getattr(GRB, "NODE_LIMIT", None),
        getattr(GRB, "ITERATION_LIMIT", None),
        getattr(GRB, "WORK_LIMIT", None),
        getattr(GRB, "MEM_LIMIT", None),
        getattr(GRB, "INTERRUPTED", None),
    )
    if status is not None
)


def require_gurobi_forward_incumbent(model, *, stage: int, node_idx: Any) -> None:
    """Fail before any ``.X``/``ObjVal`` read when no incumbent exists."""
    try:
        sol_count = int(model.SolCount)
        status = int(model.Status)
    except Exception as exc:
        raise RuntimeError(
            f"forward stage {stage} node {node_idx}: unavailable Gurobi status"
        ) from exc
    if sol_count <= 0:
        raise RuntimeError(
            f"forward stage {stage} node {node_idx}: no feasible incumbent "
            f"(status={status}, SolCount={sol_count})"
        )


def stage2_all_outsource_if_interrupted_without_incumbent(
    model,
    prob_data,
    node,
    cut_lag,
    *,
    node_idx: Any,
):
    """Return a certified fallback only for an interrupted no-solution solve.

    ``None`` means the model has an incumbent and callers may safely read it.
    Infeasible/unbounded/numeric statuses are not masked: those indicate a
    formulation or data failure even though all-outsource should be feasible.
    """
    try:
        sol_count = int(model.SolCount)
        status = int(model.Status)
    except Exception as exc:
        raise RuntimeError(
            f"forward stage 2 node {node_idx}: unavailable Gurobi status"
        ) from exc
    if sol_count > 0:
        return None
    if status not in _INTERRUPTED_WITH_SAFE_FALLBACK:
        require_gurobi_forward_incumbent(
            model, stage=2, node_idx=node_idx
        )
    result = all_outsource_forward_solution(prob_data, node, cut_lag)
    if not result.get("ok", False) or not result.get("ub_certified", False):
        raise RuntimeError(
            f"forward stage 2 node {node_idx}: all-outsource fallback "
            "was not certified feasible"
        )
    record_backend_event(
        "gurobi", "fallback", "interrupted_without_incumbent",
        node=node_idx, stage=2, status=status,
        fallback_backend="all_outsource_policy",
    )
    return result


def extract_stage2_incumbent(model, prob_data) -> dict:
    """Raw ``alpha``/``y`` values of the model's current incumbent."""
    x_dict = {}
    for j in prob_data.J:
        for v in prob_data.V:
            var = model.getVarByName(f"alpha[{j},{v}]")
            x_dict[f"alpha[{j},{v}]"] = var.X if var is not None else 0.0
    for v in prob_data.V:
        var = model.getVarByName(f"y[{v}]")
        x_dict[f"y[{v}]"] = var.X if var is not None else 0.0
    return x_dict


# Incumbent may violate original capacity under IntFeasTol/presolve; retry
# uses tight tols + Presolve=0 + exact cover cuts.
_TIGHT_INT_FEAS_TOL = 1e-9
_TIGHT_FEAS_TOL = 1e-9
_RETRY_PRESOLVE = 0
# Cap retries; each round adds cover cuts for overloaded sets.
_MAX_CERTIFY_RETRIES = 25


def exact_overloaded_assignments(prob_data, node, x_dict):
    """``{vehicle: [customers]}`` whose rounded load exceeds capacity exactly."""
    overloaded = {}
    for v in prob_data.V:
        assigned = [
            j for j in prob_data.J
            if float(x_dict.get(f"alpha[{j},{v}]", 0.0)) >= 0.5
        ]
        if not assigned:
            continue
        load = sum((Fraction(float(node.volume[j])) for j in assigned), Fraction(0))
        if load > Fraction(float(prob_data.Qv[v])):
            overloaded[v] = assigned
    return overloaded


def add_exact_capacity_cover_cuts(model, overloaded) -> int:
    """Forbid each exactly-infeasible set: ``sum_{j in S} alpha[j,v] <= |S|-1``.

    Tolerance-free and valid for the original model (S really does not fit
    in vehicle v), so it never excludes a truly feasible point and leaves the
    model's optimum, and hence any bound derived from it, unchanged.
    """
    import gurobipy as gp

    added = 0
    for v, customers in overloaded.items():
        variables = [model.getVarByName(f"alpha[{j},{v}]") for j in customers]
        if any(var is None for var in variables):
            continue
        model.addConstr(
            gp.quicksum(variables) <= len(variables) - 1,
            name=f"exact_capacity_cover[{v},{added}]",
        )
        added += 1
    model.update()
    return added


def certify_stage2_incumbent_with_retighten(
    model, prob_data, node, x_prev, *, node_idx: Any, optimize, deadline=None
):
    """Certify incumbent; cover-cut retries share the first solve's budget."""
    from solvers.forward_policy_certification import (
        InvalidForwardPolicy,
        certify_stage2_forward_policy,
    )

    retry_deadline = forward_retry_deadline(model, deadline)
    x_dict = extract_stage2_incumbent(model, prob_data)
    try:
        return certify_stage2_forward_policy(prob_data, node, x_prev, x_dict)
    except InvalidForwardPolicy as first_exc:
        first_reason = str(first_exc)
        record_backend_event(
            "gurobi", "rejected", "incumbent_failed_exact_certification",
            node=node_idx, stage=2, detail=first_reason,
        )

    def preserve_bound():
        bound = certified_gurobi_minimization_lower_bound(model)
        previous = getattr(model, "_forward_certification_lower_bound", None)
        if bound is not None and (previous is None or bound > previous):
            model._forward_certification_lower_bound = bound

    def budget_fallback():
        decisions = {f"y[{v}]": 0.0 for v in prob_data.V}
        decisions.update({
            f"alpha[{j},{v}]": 0.0 for j in prob_data.J for v in prob_data.V
        })
        certified = certify_stage2_forward_policy(
            prob_data, node, x_prev, decisions
        )
        model._forward_certification_fallback = True
        record_backend_event(
            "gurobi", "fallback", "certification_budget_exhausted",
            node=node_idx, stage=2, fallback_backend="all_outsource_policy",
        )
        print(
            f"    [forward stage2 gurobi] node={node_idx}: certification "
            "budget exhausted -> certified all-outsource policy", flush=True,
        )
        return certified

    preserve_bound()
    if remaining_seconds(retry_deadline) <= 0.0:
        return budget_fallback()
    constr_vio = None
    try:
        constr_vio = float(model.ConstrVio)
    except Exception:
        pass
    model.setParam("IntFeasTol", _TIGHT_INT_FEAS_TOL)
    model.setParam("FeasibilityTol", _TIGHT_FEAS_TOL)
    model.setParam("Presolve", _RETRY_PRESOLVE)
    total_cuts = 0
    reason = first_reason
    for attempt in range(1, _MAX_CERTIFY_RETRIES + 1):
        if remaining_seconds(retry_deadline) <= 0.0:
            return budget_fallback()
        overloaded = exact_overloaded_assignments(prob_data, node, x_dict)
        if overloaded:
            total_cuts += add_exact_capacity_cover_cuts(model, overloaded)
        elif attempt > 1:
            # Non-capacity failure: stop retrying
            break
        if not prepare_model_solve(model, retry_deadline):
            return budget_fallback()
        # Discard rejected incumbent (no MIP start reuse)
        model.reset()
        with backend_call("gurobi", "optimize", model=model,
                          attempt_kind="certification_retry", node=node_idx,
                          stage=2, retry=attempt, reason=reason):
            optimize(model)
        preserve_bound()
        if (int(model.SolCount) <= 0
                and int(model.Status) in _INTERRUPTED_WITH_SAFE_FALLBACK
                and remaining_seconds(retry_deadline) <= 0.0):
            return budget_fallback()
        require_gurobi_forward_incumbent(model, stage=2, node_idx=node_idx)
        x_dict = extract_stage2_incumbent(model, prob_data)
        try:
            certified = certify_stage2_forward_policy(prob_data, node, x_prev, x_dict)
        except InvalidForwardPolicy as exc:
            reason = str(exc)
            continue
        print(
            f"    [forward stage2 gurobi] node={node_idx}: incumbent failed exact "
            f"certification ({first_reason}; gurobi ConstrVio={constr_vio}); "
            f"re-solved with IntFeasTol={_TIGHT_INT_FEAS_TOL:g} "
            f"FeasibilityTol={_TIGHT_FEAS_TOL:g} Presolve={_RETRY_PRESOLVE} "
            f"+ {total_cuts} exact capacity cover cut(s) in {attempt} round(s) -> certified"
        )
        return certified
    if remaining_seconds(retry_deadline) <= 0.0:
        return budget_fallback()
    raise InvalidForwardPolicy(
        f"{reason} (after tight-tolerance/no-presolve re-solve with "
        f"{total_cuts} exact cover cuts; first: {first_reason}; "
        f"gurobi ConstrVio={constr_vio})"
    )


__all__ = [
    "add_exact_capacity_cover_cuts",
    "certify_stage2_incumbent_with_retighten",
    "exact_overloaded_assignments",
    "extract_stage2_incumbent",
    "require_gurobi_forward_incumbent",
    "stage2_all_outsource_if_interrupted_without_incumbent",
]
