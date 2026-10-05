"""Solve one fixed-fleet piece ``C(n)`` with Gurobi and certify the outcome.

The model is the ordinary forward Stage-2 assignment model
(``StageModelBuilder.build_stage_problem(2, ...)``) with the fleet fixed to the
leading run ``n``; no Lagrangian term.  Two certified artefacts come out:

* a lower bound on ``C(n)``: Gurobi's ``ObjBound`` filtered through
  ``core.solver_bounds.certified_gurobi_minimization_lower_bound`` (valid even
  after a time limit, never an incumbent value);
* optionally a feasible policy: the incumbent's ``alpha``/``y`` rounded and
  certified exactly by ``cuts.exact_subroutines.certify_s2_lagrangian_policy``
  (capacity, fulfilment, activation, assignment order and the canonical
  purchase prefix).  An incumbent that is over capacity by a solver tolerance
  is cut off with an exact cover cut and the model re-solved, as in the forward
  pass.

The fleet oracle uses this backend above the exact subset-DP size limit and
for any piece that a cheaper certified bound cannot exclude.
"""

from __future__ import annotations

import time
from typing import Dict, Mapping, Optional

from gurobipy import GRB

from core.solver_bounds import certified_gurobi_minimization_lower_bound
from core.backend_telemetry import backend_call, record_backend_event
from core.stage2_tolerance import apply_abs_gap
from cuts import exact_subroutines as exact_sub
from models.learned_cut_dispatch import optimize_with_learned_cut_lifecycle
from models.stage_builder import StageModelBuilder, lazy_cut_callback
from solvers.forward_incumbent import (
    add_exact_capacity_cover_cuts,
    exact_overloaded_assignments,
)

from s2forward.fleet_pieces import Counts, FleetLayout
from s2forward.verified_lp_bound import stabilize_fixed_fleet_bound, tiny_bound_conflict
from .piece_table import PolicyRecord

_MAX_COVER_CUT_ROUNDS = 3


def fleet_x_prev(layout: FleetLayout, counts: Counts, period: int) -> Dict[str, float]:
    prefix = set(layout.prefix_vehicles(counts))
    return {f"z[{v},{period}]": (1.0 if v in prefix else 0.0) for v in layout.vehicles}


def _optimize(model):
    with backend_call("gurobi", "fleet_piece_mip", model=model):
        if getattr(model, "_lazy_cuts", ()):
            optimize_with_learned_cut_lifecycle(model, lazy_cut_callback)
        else:
            optimize_with_learned_cut_lifecycle(model)


def _raw_incumbent(model, prob_data) -> Dict[str, float]:
    out = {}
    for v in prob_data.V:
        var = model.getVarByName(f"y[{v}]")
        out[f"y[{v}]"] = float(var.X) if var is not None else 0.0
        for j in prob_data.J:
            var = model.getVarByName(f"alpha[{j},{v}]")
            out[f"alpha[{j},{v}]"] = float(var.X) if var is not None else 0.0
    return out


def certify_policy_dict(prob_data, node, cuts_payload, x_dict: Mapping[str, float],
                        layout: FleetLayout, *, binary_tolerance: float,
                        source: str, return_certificate: bool = False):
    """Exactly certify ``alpha``/``y`` values; return the record and its cost.

    The cost is the directed-up binary64 value of outsourcing + sum of the
    Stage-3 cut maxima under ``cuts_payload`` (``pi = 0``, ``z = y``).  Raises
    ``exact_sub.InvalidS2LagrangianPolicy`` if the point is infeasible.
    """
    vehicles = list(prob_data.V)
    customers = list(prob_data.J)
    raw_y = [x_dict.get(f"y[{v}]", 0.0) for v in vehicles]
    raw_alpha = [[x_dict.get(f"alpha[{j},{v}]", 0.0) for j in customers] for v in vehicles]
    certified = exact_sub.certify_s2_lagrangian_policy(
        prob_data, node, None, {},
        {"z": raw_y, "y": raw_y, "alpha": raw_alpha},
        binary_tolerance=binary_tolerance,
        cuts_payload=cuts_payload,
        require_assignment_order=True,
        require_purchase_order=True,
    )
    used = {v: certified["y"][pos] for pos, v in enumerate(vehicles)}
    if not layout.is_leading_run(used):
        raise exact_sub.InvalidS2LagrangianPolicy("used_vehicles_not_leading_run")
    record = PolicyRecord(
        counts=layout.counts_from_vehicle_flags(used),
        alpha={
            f"alpha[{j},{v}]": 1
            for vpos, v in enumerate(vehicles)
            for jpos, j in enumerate(customers)
            if certified["alpha"][vpos][jpos]
        },
        y={f"y[{v}]": 1 for pos, v in enumerate(vehicles) if certified["y"][pos]},
        source=source,
    )
    result = (record, float(certified["V"]))
    return (*result, certified) if return_certificate else result


def solve_piece(prob_data, node, cut_lag, layout: FleetLayout, counts: Counts, *,
                cuts_payload, stage_builder_options: Optional[Mapping] = None,
                mip_gap: float = 1e-7, time_limit: Optional[float] = None,
                mip_start: Optional[PolicyRecord] = None,
                binary_tolerance: float = 1e-5,
                bound_stop: Optional[float] = None,
                abs_gap: Optional[float] = None) -> dict:
    """Solve ``C(counts)``; returns certified ``lb``, optional policy, status.

    ``bound_stop`` (Gurobi ``BestBdStop``) ends the search as soon as the
    dual bound proves ``C(counts) >= bound_stop``: enough to show the piece
    cannot attain the minimum at the current multiplier, without paying for
    its exact value.  ``abs_gap`` (Gurobi ``MIPGapAbs``) is the scheduled
    absolute Stage-2 tolerance; ``lb`` stays the certified dual bound.
    """
    period = int(node.info[1])
    started = time.time()
    builder = StageModelBuilder(prob_data, **dict(stage_builder_options or {}))
    model = builder.build_stage_problem(2, node, cut_lag, fleet_x_prev(layout, counts, period))
    model.setParam("MIPGap", float(mip_gap))
    apply_abs_gap(model, abs_gap)
    if time_limit is not None and time_limit > 0:
        model.setParam("TimeLimit", float(time_limit))
    if bound_stop is not None:
        model.setParam("BestBdStop", float(bound_stop))
    if mip_start is not None:
        for name, value in mip_start.as_x_dict().items():
            var = model.getVarByName(name)
            if var is not None:
                var.Start = value
        for v in prob_data.V:
            if f"y[{v}]" not in mip_start.y:
                var = model.getVarByName(f"y[{v}]")
                if var is not None:
                    var.Start = 0.0
        model.update()

    _optimize(model)
    policy = None
    policy_cost = None
    policy_certificate = None
    failure = None
    for _round in range(_MAX_COVER_CUT_ROUNDS + 1):
        if int(getattr(model, "SolCount", 0)) <= 0:
            break
        x_dict = _raw_incumbent(model, prob_data)
        try:
            policy, policy_cost, policy_certificate = certify_policy_dict(
                prob_data, node, cuts_payload, x_dict, layout,
                binary_tolerance=binary_tolerance, source=f"solve{counts}",
                return_certificate=True,
            )
            failure = None
            break
        except exact_sub.InvalidS2LagrangianPolicy as exc:
            failure = str(exc)
            overloaded = exact_overloaded_assignments(prob_data, node, x_dict)
            if not overloaded or _round == _MAX_COVER_CUT_ROUNDS:
                break
            add_exact_capacity_cover_cuts(model, overloaded)
            model.reset()
            record_backend_event("gurobi", "retry", "capacity_certification")
            _optimize(model)

    lb = certified_gurobi_minimization_lower_bound(model)
    raw_bound = raw_incumbent = None
    if lb is None:
        try:
            raw_bound, raw_incumbent = float(model.ObjBound), float(model.ObjVal)
        except Exception:
            pass
    recover_roundoff = tiny_bound_conflict(lb, policy_cost) or (
        lb is None and (tiny_bound_conflict(raw_bound, policy_cost)
                       or tiny_bound_conflict(raw_bound, raw_incumbent))
    )
    verification = stabilize_fixed_fleet_bound(
        model, lb, policy_upper_bound=policy_cost,
        label=f"Stage2 fixed piece {counts}",
        recover_missing_bound=recover_roundoff,
    )
    lb = verification["bound"]
    if recover_roundoff:
        record_backend_event(
            "gurobi", "bound_recovery", "verified_lp" if lb is not None else "no_verified_bound",
            stage=2, path="backward", node=getattr(node, "index", None),
            counts=counts, raw_bound=verification["raw_bound"] if raw_bound is None else raw_bound,
            rescored_policy_upper=policy_cost, lower_bound=lb,
            verification_status=verification["status"],
        )
    status = int(getattr(model, "Status", -1))
    return {
        "counts": counts,
        "lb": lb,
        "bound_source": verification["source"],
        "policy": policy,
        "policy_cost": policy_cost,
        "policy_certificate": policy_certificate,
        "status": status,
        "optimal": status == GRB.OPTIMAL and lb is not None and (
            not recover_roundoff or (policy_cost is not None and float(lb) == float(policy_cost))),
        "seconds": time.time() - started,
        "certification_failure": failure,
        "node_count": float(getattr(model, "NodeCount", 0.0)),
    }
