"""Forward Stage-2 assignment by exhaustive subset DP.

The forward Stage-2 node has the fleet ``z`` fixed by Stage 1, so it is a
fixed-fleet piece ``C(n(z))``.  The output goes through the *same* feasibility
certification and exact archive rescoring as a Gurobi incumbent
(``certify_stage2_forward_policy`` + ``score_forward_stage2_policy``), so the
returned ``cost_star_value`` is a directed-up feasible bound and no raw DP
objective is trusted.  The DP also returns a rigorously directed lower bound;
only equality of those exact endpoints is labelled strict optimality.

Production backend order is selected by ``s2forward.dispatch_policy`` using
the frozen customer/cut-density crossover table.  This module is the lower
DP execution layer: ``auto`` attempts DP when the dispatcher asks for it,
``dp`` raises on an unsupported request (tests/parity tools), and ``gurobi``
returns ``None``.  Callers retain Gurobi as the structural/memory fallback.
"""
from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, Optional

from cuts import exact_subroutines as exact_sub
from solvers.forward_policy_certification import (
    InvalidForwardPolicy,
    certify_stage2_forward_policy,
)
from solvers.forward_stage2_policy import score_forward_stage2_policy

from .fleet_pieces import FleetLayout
from .mip_start import exact_forward_policy_score
from core.backend_telemetry import record_backend_event

from .purpose import (
    PHASE2_FORWARD,
    ForwardStage2Purpose,
    normalize_forward_stage2_purpose,
)
from .subset_dp import (
    SubsetDPNotApplicable,
    SubsetDPPieceSolver,
    default_max_customers,
)

MODE_ENV = "VRP_S2_FORWARD_SOLVER"
MAX_CUSTOMERS_ENV = "VRP_S2_FORWARD_DP_MAX_CUSTOMERS"
_MODES = ("auto", "dp", "gurobi")
# The DP optimum and the exact upward rescoring of its assignment are two
# evaluations of the same objective; a disagreement beyond round-off means the
# DP tables and the model disagree, in which case the DP result is not used.
_SELF_CHECK_REL_TOL = 1e-9
_SELF_CHECK_ABS_TOL = 1e-7
# User-selected numerical zero band for paths that require a tight Stage-2
# surrogate solve (Phase 2 forward and backward refresh).  Phase 1 forward
# needs only a certified feasible policy/UB and therefore does not use this
# gate; its result still cannot claim optimality or enter exact outer reuse.
_FORWARD_CERTIFIED_GAP_ABS_TOL = 1e-6


class ForwardStage2DPMismatch(RuntimeError):
    """DP optimum and exact rescoring of the DP assignment disagree."""


def forward_solver_mode() -> str:
    mode = os.environ.get(MODE_ENV, "auto").strip().lower() or "auto"
    if mode not in _MODES:
        raise ValueError(f"{MODE_ENV}={mode!r}; expected one of {_MODES}")
    return mode


def dp_max_customers(concurrent_slots: Optional[int] = None) -> int:
    raw = os.environ.get(MAX_CUSTOMERS_ENV, "").strip()
    return int(raw) if raw else default_max_customers(concurrent_slots)


@dataclass
class ForwardStage2DPResult:
    x_dict: Dict[str, float]
    cost_star_value: float
    stage_cost_value: float
    theta_by_succ: Dict[Any, float]
    counts: tuple
    n_active: int
    dp_value: float
    seconds: float
    model_stats: Dict[str, Any] = field(default_factory=dict)
    # Certified lower endpoint of the exact fixed-fleet surrogate solve.  The
    # forward score above is a directed-up feasible-policy value; callers must
    # not confuse it with this lower certificate.
    objective_lower_bound: float = float("-inf")
    lb_certified: bool = False
    optimality_certified: bool = False
    certified_gap: float = math.inf
    certificate_source: str = ""


def dense_stage2_x_dict(prob_data, sparse: Dict[str, Any]) -> Dict[str, float]:
    """Full ``alpha[j,v]``/``y[v]`` dictionary (zeros filled) from a DP result."""
    x_dict: Dict[str, float] = {}
    for v in prob_data.V:
        for j in prob_data.J:
            x_dict[f"alpha[{j},{v}]"] = 0.0
        x_dict[f"y[{v}]"] = 0.0
    for name, value in sparse.items():
        if name not in x_dict:
            raise InvalidForwardPolicy(f"DP produced unknown decision {name}")
        x_dict[name] = float(value)
    return x_dict


def _model_stats(payload_size: int, dp: SubsetDPPieceSolver, seconds: float) -> dict:
    """Same schema as ``models.learned_cut_dispatch.collect_s2_model_stats``."""
    return {
        "models": 1,
        "backend_gurobi": 0,
        "backend_dp": 1,
        "backend_probe": 0,
        "cuts": int(payload_size),
        "explicit": int(payload_size),
        "lazy": 0,
        "callback_calls": 0,
        "callback_checks": 0,
        "callback_added": 0,
        "callback_add_calls": 0,
        "build_seconds": float(dp.stats.get("theta_seconds", 0.0)),
        "solve_seconds": max(0.0, float(seconds)),
        "epochs": 1,
    }


def solve_forward_stage2_by_dp(
    prob_data,
    node,
    cut_lag,
    x_prev,
    *,
    node_idx: Any = None,
    log=None,
    max_customers: Optional[int] = None,
    concurrent_slots: Optional[int] = None,
    purpose: ForwardStage2Purpose | str = PHASE2_FORWARD,
) -> Optional[ForwardStage2DPResult]:
    """Solve fixed-fleet forward Stage 2 by exhaustive floating-point DP.

    Returns ``None`` when the DP must not / cannot be used (caller falls back
    to Gurobi).  ``cut_lag`` is ``{3: cuts_of_successors}`` exactly as the
    Gurobi stage builder receives it.

    The default purpose is the strict ``PHASE2_FORWARD`` contract.  Phase 1
    must explicitly request ``PHASE1_FEASIBLE_TRIAL``: a feasible policy with
    an independently rescored objective is a valid UB and useful trial point
    even when the certified DP interval is wider than ``1e-6``.  Purpose never
    changes ``optimality_certified``; only exact endpoint equality may set that
    flag and make a result eligible for exact outer reuse.
    """
    purpose = normalize_forward_stage2_purpose(purpose)
    mode = forward_solver_mode()
    if mode == "gurobi":
        record_backend_event("subset_dp", "skip", "forced_gurobi", node=node_idx)
        return None
    started = time.time()
    layout = FleetLayout(prob_data)
    period = int(node.info[1])
    flags = {v: x_prev.get(f"z[{v},{period}]", 0.0) for v in layout.vehicles}
    # assignment_order makes any fleet equivalent to its leading run, so the
    # Gurobi model with this z and the DP with these counts share an optimum.
    counts = layout.counts_from_vehicle_flags(flags)

    succ_to_pos = {succ: pos for pos, succ in enumerate(node.successor)}
    payload = exact_sub.build_s2_bp_cuts(prob_data, node, cut_lag, succ_to_pos)
    try:
        dp = SubsetDPPieceSolver(
            prob_data, node, payload, layout,
            max_customers=(
                dp_max_customers(concurrent_slots)
                if max_customers is None
                else int(max_customers)
            ),
        )
    except SubsetDPNotApplicable as exc:
        record_backend_event("subset_dp", "skip", "not_applicable",
                             node=node_idx, detail=str(exc))
        if mode == "dp":
            raise
        if log is not None:
            log(f"node={node_idx}: subset DP not applicable ({exc}); Gurobi")
        return None

    record_backend_event("subset_dp", "attempt", "fixed_fleet", node=node_idx,
                         purpose=purpose.value, active_customers=dp.n)
    result = dp.solve(counts)
    x_dict = dense_stage2_x_dict(prob_data, result["x_dict"])
    certified, _stage_cost = certify_stage2_forward_policy(prob_data, node, x_prev, x_dict)
    score = score_forward_stage2_policy(prob_data, node, certified, payload)
    cost_star = float(score["cost_star_value"])
    dp_value = float(result["value"])
    objective_lower_bound = float(result["lb"])
    objective_exact = exact_forward_policy_score(
        prob_data, node, cut_lag, certified
    )
    lower_exact = (
        Fraction.from_float(objective_lower_bound)
        if math.isfinite(objective_lower_bound)
        else None
    )
    lb_certified = bool(
        result.get("optimal", False)
        and lower_exact is not None
        and lower_exact <= objective_exact
    )
    # Exhaustive binary64 DP plus its directed error margin gives a rigorous
    # interval [LB, exact feasible score].  It is useful in Forward even when
    # the endpoints differ, but only a genuinely closed interval is an exact
    # certificate that may skip a later solve.  Never promote LB to the
    # incumbent merely because the gap is small.
    optimality_certified = bool(
        lb_certified and lower_exact == objective_exact
    )
    if abs(cost_star - dp_value) > _SELF_CHECK_ABS_TOL + _SELF_CHECK_REL_TOL * abs(cost_star):
        record_backend_event("subset_dp", "rejected", "objective_rescore_mismatch",
                             node=node_idx, dp_value=dp_value, policy_cost=cost_star)
        message = (
            f"node={node_idx}: DP value {dp_value!r} != exact rescoring "
            f"{cost_star!r} of the DP assignment"
        )
        if mode == "dp":
            raise ForwardStage2DPMismatch(message)
        if log is not None:
            log(message + "; Gurobi")
        return None
    certified_gap = (
        max(0.0, cost_star - objective_lower_bound)
        if lb_certified
        else math.inf
    )
    if (
        purpose.requires_tight_dp_gap
        and certified_gap > _FORWARD_CERTIFIED_GAP_ABS_TOL
    ):
        record_backend_event("subset_dp", "rejected", "certified_interval_too_wide",
                             node=node_idx, certified_gap=certified_gap)
        message = (
            f"node={node_idx}: DP certified interval is too wide "
            f"(UB-LB={certified_gap:.3e} > "
            f"{_FORWARD_CERTIFIED_GAP_ABS_TOL:.1e})"
        )
        if mode == "dp":
            raise ForwardStage2DPMismatch(message)
        if log is not None:
            log(message + "; Gurobi")
        return None
    seconds = time.time() - started
    record_backend_event("subset_dp", "accepted",
                         "exact_interval" if optimality_certified else "certified_policy",
                         node=node_idx, certified_gap=certified_gap,
                         proven=optimality_certified)
    return ForwardStage2DPResult(
        x_dict=dict(certified),
        cost_star_value=cost_star,
        stage_cost_value=float(score["stage_cost_value"]),
        theta_by_succ=dict(score["theta_by_succ"]),
        counts=tuple(counts),
        n_active=int(dp.n),
        dp_value=dp_value,
        seconds=seconds,
        model_stats=_model_stats(len(payload), dp, seconds),
        objective_lower_bound=(
            objective_lower_bound if lb_certified else float("-inf")
        ),
        lb_certified=lb_certified,
        optimality_certified=optimality_certified,
        certified_gap=certified_gap,
        certificate_source=str(result.get("status", "subset_dp")),
    )
