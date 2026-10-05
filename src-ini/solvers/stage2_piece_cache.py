"""The fleet piece table as the single Phase-2 Stage-2 cache.

Forward Stage 2, the backward Stage-2 refresh and the S2->S1 fleet oracle all
evaluate the same fixed-fleet value ``C(n)`` for one node, one canonical
purchased fleet ``n`` and one Stage-3 cut archive.  The per-node
``FleetPieceTable`` (``customized-subprob/s2backward/piece_table.py``) already
stores, for every fleet, a certified lower bound that stays valid while the
archive grows and the feasible policies found so far.  This module lets the
two forward-style call sites read and write that table too:

* lookup: the cheapest stored policy feasible for the trial fleet is re-scored
  against the *current* archive (exact, upward-rounded) and compared with the
  table's monotone lower bound.  When the interval is within the scheduled
  absolute tolerance the policy is returned without any MIP;
* otherwise that policy warm-starts the ordinary dispatcher solve, whose
  Gurobi stop is the scheduled absolute gap, and the certified lower bound and
  the new policy are written back so the S2->S1 oracle never re-solves the
  trial piece.

Everything returned keeps the dispatcher's certificates: policies are
feasibility-certified and archive re-scored, bounds are solver-certified
lower bounds.  ``exact_optimal`` is never claimed for a table hit.

Stall memory.  Within one round the refresh, the S2->S1 oracle and the next
forward pass all see the same archive for a node.  A fixed-fleet MIP whose
time-limited solve (at least ``STALL_MIN_SOLVE_SECONDS`` long) improved the
certified bound by no more than the scheduled ``eps`` has stalled at its
root relaxation; the table remembers that under the archive fingerprint and
every later caller under the same fingerprint answers from the certified
interval ``[lb, ub]`` (best archived policy, table lower bound) instead of
restarting the same branch-and-bound.  The UB stays a real policy cost and
every cut keeps the certified lower endpoint; only wall clock is saved.  On
C50 (tol 1e-6, eps ~ 70) the unclosable t=0/t=1 pieces otherwise cost the
refresh 300s, the Level Set 600s and the forward its full 1800s TimeLimit
per node and per iteration, for the same interval each time.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import core.customized_subprob  # noqa: F401  (sys.path for s2forward/s2backward)
from core.exact_solver_log import log as exact_log
from core.backend_telemetry import record_backend_event
from core.stage2_tolerance import apply_abs_gap, effective_abs_gap
from solvers.forward_period_dedup import stage2_archive_fingerprint
from solvers.forward_stage2_dp import (
    ForwardStage2SolveResult,
    solve_forward_stage2_dispatched,
)
from s2forward.purpose import (
    forward_stage2_backend_scope,
    PHASE2_FORWARD,
    ForwardStage2Purpose,
    normalize_forward_stage2_purpose,
)

TABLE_BACKEND = "piece_table"
_LOG_BACKEND = "piece_table"
# A time-limited solve shorter than this (e.g. a refresh that only had a few
# seconds of shared budget left) says nothing about the MIP and never marks
# the piece stalled.
STALL_MIN_SOLVE_SECONDS = float(
    os.environ.get("VRP_PHASE2_S2_STALL_MIN_SECONDS", "30")
)
STATUS_STALLED = "piece_table_stalled"


class Stage2PieceCache:
    """Per-node ``FleetPieceTable`` states shared by forward, refresh and oracle."""

    def __init__(self):
        # second_ind -> FleetPieceTable.to_state() (plain dict, pickle-safe)
        self.tables: Dict[Any, dict] = {}
        self.stats = {
            "hits": 0, "stalled": 0, "warm_starts": 0, "solves": 0, "stores": 0,
        }

    def state_for(self, node_ind) -> Optional[dict]:
        return self.tables.get(node_ind)

    def store(self, node_ind, table_state, copies=()) -> None:
        """Keep ``table_state`` for ``node_ind`` and its exact-copy nodes."""
        if table_state is None:
            return
        self.tables[node_ind] = table_state
        for copy_ind in copies:
            self.tables[copy_ind] = table_state
        self.stats["stores"] += 1

    def record(self, outcome: "FixedFleetSolveOutcome") -> None:
        if outcome.from_table:
            if outcome.result.status == STATUS_STALLED:
                self.stats["stalled"] += 1
            else:
                self.stats["hits"] += 1
        else:
            self.stats["solves"] += 1
            if outcome.warm_started:
                self.stats["warm_starts"] += 1

    def format_stats(self) -> str:
        s = self.stats
        return (
            f"tables={len(self.tables)} hits={s['hits']} stalled={s['stalled']} "
            f"solves={s['solves']} warm_starts={s['warm_starts']}"
        )


@dataclass
class FixedFleetSolveOutcome:
    result: ForwardStage2SolveResult
    table_state: Optional[dict]
    counts: Optional[tuple]
    from_table: bool
    warm_started: bool
    abs_gap: Optional[float]
    table_lb: Optional[float]
    table_ub: Optional[float]


def _full_policy_dict(prob_data, policy) -> Dict[str, float]:
    out = {f"y[{v}]": 0.0 for v in prob_data.V}
    for j in prob_data.J:
        for v in prob_data.V:
            out[f"alpha[{j},{v}]"] = 0.0
    out.update(policy.as_x_dict())
    return out


def trial_fleet_counts(prob_data, node, x_prev):
    """``(layout, counts)`` of the purchased fleet at ``node``'s period.

    ``counts`` is None when the fleet is not a canonical leading run; both
    are None when the problem carries no fleet-type structure at all.
    """
    from s2backward import FleetLayout

    try:
        layout = FleetLayout(prob_data)
        period = int(node.info[1])
        flags = {v: x_prev.get(f"z[{v},{period}]", 0.0) for v in layout.vehicles}
        if not layout.is_leading_run(flags):
            return layout, None
        return layout, layout.counts_from_vehicle_flags(flags)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None, None


def seed_policy_hint(prob_data, node, cuts_payload, table, counts, hint_policy, *,
                     source: str, log) -> bool:
    """Archive a caller-supplied feasible policy (e.g. the last forward trial).

    Gives a cold table a reference value for the relative cap and a warm
    start.  Returns False (and logs) when the hint is not a certified policy
    inside the trial fleet.
    """
    from cuts import exact_subroutines as exact_sub
    from s2backward.piece_solver import certify_policy_dict

    if hint_policy is None:
        return False
    try:
        record, _cost = certify_policy_dict(
            prob_data, node, cuts_payload, dict(hint_policy), table.layout,
            binary_tolerance=1e-5, source=source,
        )
    except exact_sub.InvalidS2LagrangianPolicy as exc:
        log(f"policy hint rejected ({exc})")
        return False
    if not table.layout.dominates(counts, record.counts):
        log("policy hint uses vehicles outside the trial fleet; ignored")
        return False
    table.assert_policy_upper_bound(
        record.counts, _cost, source=f"{source}: certified policy hint",
    )
    table.add_policy(record)
    return True


def build_cuts_payload(prob_data, node, cut_lag):
    from cuts import exact_subroutines as exact_sub

    return exact_sub.build_s2_bp_cuts(
        prob_data, node, cut_lag,
        {successor: position for position, successor in enumerate(node.successor)},
    )


def best_cached_policy(prob_data, node, cut_lag, table, counts, cuts_payload=None):
    """Cheapest stored policy feasible for ``counts`` under the current archive.

    Returns ``(x_dict, score, cuts_payload)``; ``x_dict``/``score`` are None
    when the table holds no usable policy for the fleet.
    """
    from solvers.forward_stage2_policy import score_forward_stage2_policy

    if cuts_payload is None:
        cuts_payload = build_cuts_payload(prob_data, node, cut_lag)
    best_x = None
    best_score = None
    for policy in table.policies_for(counts):
        x_dict = _full_policy_dict(prob_data, policy)
        try:
            score = score_forward_stage2_policy(prob_data, node, x_dict, cuts_payload)
        except ValueError:
            continue
        table.assert_policy_upper_bound(
            policy.counts, score["cost_star_value"],
            source=f"cached policy rescore ({policy.source})",
        )
        if best_score is None or score["cost_star_value"] < best_score["cost_star_value"]:
            best_x, best_score = x_dict, score
    return best_x, best_score, cuts_payload


def _store_result_in_table(prob_data, node, table, counts, cuts_payload, result, *,
                           source: str, log, known_policy_upper_bound=None) -> None:
    from cuts import exact_subroutines as exact_sub
    from s2backward.piece_solver import certify_policy_dict

    lower = result.objective_lower_bound
    if not math.isfinite(float(result.cost_star_value)):
        raise ValueError("Stage-2 solve must return a finite certified policy cost")
    # A physically feasible assignment need not satisfy the canonical rank
    # constraints defining C(n). Only a policy certified in that same domain
    # can refute the table's lower bound.
    upper = math.inf
    if known_policy_upper_bound is not None:
        known_upper = float(known_policy_upper_bound)
        if not math.isfinite(known_upper):
            raise ValueError("cached Stage-2 policy cost must be finite")
        upper = min(upper, known_upper)
        table.assert_policy_upper_bound(
            counts, upper, source=f"{source}: cached canonical policy",
        )
    try:
        record, _cost = certify_policy_dict(
            prob_data, node, cuts_payload, result.x_dict, table.layout,
            binary_tolerance=1e-5, source=source,
        )
    except exact_sub.InvalidS2LagrangianPolicy as exc:
        log(f"policy not archived in the piece table ({exc})")
        record = None
    if record is not None and not table.layout.dominates(counts, record.counts):
        log("policy uses vehicles outside the trial fleet; not archived")
        record = None
    if record is not None:
        table.assert_policy_upper_bound(
            record.counts, _cost, source=f"{source}: certified piece policy",
        )
        upper = min(upper, float(_cost))
        table.add_policy(record)
    if lower is not None and math.isfinite(float(lower)):
        lower = float(lower)
        if lower > upper:
            # A new contradictory bound has not entered the shared table or
            # supported a cut. Reject it, retaining the feasible policy only.
            log(f"new piece bound rejected: fleet={counts} lb={lower!r} "
                f"certified_policy_ub={upper!r} source={source!r}")
        else:
            table.update_lb(counts, lower, status=str(result.backend))


def _is_time_limit_status(status) -> bool:
    try:
        from gurobipy import GRB

        return int(status) == int(GRB.TIME_LIMIT)
    except (TypeError, ValueError, ImportError):
        return False


def _note_stall(table, counts, fingerprint, result, eps, *, lb_before,
                solve_seconds, log) -> bool:
    """Mark the piece stalled when a long time-limited solve left its bound.

    Conditions (all required): a scheduled ``eps``; the solve ended on the
    Gurobi time limit after at least ``STALL_MIN_SOLVE_SECONDS``; the table
    held a finite bound before and the solve improved it by at most ``eps``;
    the interval is still wider than ``eps`` (a closed one is a plain hit).
    """
    from s2backward.piece_table import should_mark_stalled

    lb_after = float(table.lower_bound(counts))
    ub_after = float(result.cost_star_value)
    if not should_mark_stalled(
        time_limited=_is_time_limit_status(result.status),
        solve_seconds=solve_seconds,
        min_solve_seconds=STALL_MIN_SOLVE_SECONDS,
        tolerance=eps,
        lb_before=lb_before,
        lb_after=lb_after,
        ub_after=ub_after,
        fingerprint=fingerprint,
    ):
        return False
    if table.mark_stalled(counts, fingerprint):
        log(
            f"piece stalled: fleet={counts} time limit after {solve_seconds:.1f}s "
            f"moved lb {float(lb_before):,.6f} -> {lb_after:,.6f} (<= eps={eps:.3e}) "
            f"with ub={ub_after:,.6f}; later solves under this archive answer "
            "from the certified interval"
        )
    return True


@forward_stage2_backend_scope
def solve_fixed_fleet_stage2(
    prob_data,
    node,
    cut_lag,
    x_prev,
    *,
    node_idx,
    purpose: ForwardStage2Purpose | str = PHASE2_FORWARD,
    build_gurobi_model,
    configure_full_gurobi,
    optimize_gurobi,
    full_time_limit=None,
    concurrent_slots=None,
    abs_tol=None,
    rel_cap=None,
    table_state=None,
    hint_policy=None,
) -> FixedFleetSolveOutcome:
    """Fixed-fleet Stage-2 solve through the shared piece table.

    ``abs_tol`` is the scheduled per-node absolute tolerance (None = the
    caller's plain MIPGap contract, table used for warm starts only).
    ``table_state`` is the node's ``FleetPieceTable.to_state()`` or None.
    ``hint_policy`` (an ``x_dict`` feasible for this fleet, e.g. the previous
    forward trial) is archived first so a cold table still has a reference
    for the relative cap and a warm start.
    """
    from s2backward import FleetPieceTable

    purpose = normalize_forward_stage2_purpose(purpose)
    direction = purpose.log_direction

    def _log(message):
        exact_log(direction, 2, _LOG_BACKEND, f"node={node_idx} {message}")

    started = time.perf_counter()
    layout, counts = trial_fleet_counts(prob_data, node, x_prev)
    table = None
    warm_x = None
    warm_score = None
    cuts_payload = None
    table_lb = None
    fingerprint = None
    if counts is not None:
        table = FleetPieceTable.from_state(layout, table_state)
        fingerprint = stage2_archive_fingerprint(node, cut_lag)
        table.archive_fingerprint = fingerprint
        table_lb = float(table.lower_bound(counts))
        cuts_payload = build_cuts_payload(prob_data, node, cut_lag)
        if hint_policy is not None:
            seed_policy_hint(
                prob_data, node, cuts_payload, table, counts, hint_policy,
                source=f"{purpose.value}_hint", log=_log,
            )
        warm_x, warm_score, cuts_payload = best_cached_policy(
            prob_data, node, cut_lag, table, counts, cuts_payload
        )
        if warm_score is not None:
            table.assert_policy_upper_bound(
                counts, warm_score["cost_star_value"],
                source=f"{purpose.value}: node={node_idx} table lookup",
            )
    elif layout is not None:
        _log("purchased fleet is not a canonical leading run; piece table skipped")

    eps = None
    if abs_tol is not None and math.isfinite(float(abs_tol)) and float(abs_tol) > 0.0:
        eps = float(abs_tol)

    def _answer_from_table(status, reason):
        ub = float(warm_score["cost_star_value"])
        record_backend_event(
            TABLE_BACKEND, "cache", status, node=node_idx,
            detail=reason, lower_bound=table_lb, upper_bound=ub,
            requested_abs_gap=eps,
        )
        table.assert_policy_upper_bound(
            counts, ub, source=f"{purpose.value}: node={node_idx} table answer",
        )
        result = ForwardStage2SolveResult(
            x_dict=dict(warm_x),
            cost_star_value=ub,
            stage_cost_value=float(warm_score["stage_cost_value"]),
            theta_by_succ=dict(warm_score["theta_by_succ"]),
            model_stats={},
            exact_optimal=False,
            backend=TABLE_BACKEND,
            decision_reason=reason,
            objective_lower_bound=table_lb,
            status=status,
        )
        return FixedFleetSolveOutcome(
            result=result, table_state=table.to_state(), counts=counts,
            from_table=True, warm_started=False, abs_gap=eps,
            table_lb=table_lb, table_ub=ub,
        )

    if eps is not None and warm_score is not None and table_lb is not None:
        gap = warm_score["cost_star_value"] - table_lb
        if 0.0 <= gap <= eps:
            _log(
                f"table hit: fleet={counts} ub={warm_score['cost_star_value']:,.6f} "
                f"lb={table_lb:,.6f} gap={gap:.3e} <= eps={eps:.3e}; no MIP "
                f"({time.perf_counter() - started:.3f}s incl. rescoring "
                f"{sum(1 for _ in table.policies_for(counts))} policies)"
            )
            return _answer_from_table(
                "piece_table_hit", f"piece table gap {gap:.3e} <= eps {eps:.3e}"
            )
        if table.is_stalled(counts, fingerprint):
            _log(
                f"table stalled under this archive: fleet={counts} "
                f"ub={warm_score['cost_star_value']:,.6f} lb={table_lb:,.6f} "
                f"gap={gap:.3e} > eps={eps:.3e}; a time-limited solve of this "
                "model already left the bound in place, answering from the "
                f"certified interval; no MIP ({time.perf_counter() - started:.3f}s)"
            )
            return _answer_from_table(
                STATUS_STALLED,
                f"piece MIP stalled under this archive; interval gap {gap:.3e}",
            )

    reference = None if warm_score is None else warm_score["cost_star_value"]
    abs_gap = effective_abs_gap(eps, reference, rel_cap)
    warm_started = False

    def _configure(model):
        nonlocal warm_started
        configure_full_gurobi(model)
        apply_abs_gap(model, abs_gap)
        if warm_x is not None:
            from s2forward.mip_start import apply_forward_binary_start

            try:
                apply_forward_binary_start(model, prob_data, node, x_prev, warm_x)
                warm_started = True
            except Exception as exc:  # a rejected start must not fail the solve
                _log(f"warm start rejected ({type(exc).__name__}: {exc})")

    if counts is not None:
        _log(
            f"table miss: fleet={counts} lb={table_lb:,.6f} "
            f"ub={'n/a' if reference is None else f'{reference:,.6f}'} "
            f"-> MIP with MIPGapAbs={'off' if abs_gap is None else f'{abs_gap:.4g}'}"
            f"{' warm start' if warm_x is not None else ''}"
        )
    solve_started = time.perf_counter()
    result = solve_forward_stage2_dispatched(
        prob_data,
        node,
        cut_lag,
        x_prev,
        node_idx=node_idx,
        purpose=purpose,
        build_gurobi_model=build_gurobi_model,
        configure_full_gurobi=_configure,
        optimize_gurobi=optimize_gurobi,
        full_time_limit=full_time_limit,
        concurrent_slots=concurrent_slots,
    )
    solve_seconds = time.perf_counter() - solve_started
    new_state = table_state
    if table is not None:
        _store_result_in_table(
            prob_data, node, table, counts, cuts_payload, result,
            source=purpose.value, log=_log,
            known_policy_upper_bound=reference,
        )
        _note_stall(
            table, counts, fingerprint, result, eps,
            lb_before=table_lb, solve_seconds=solve_seconds, log=_log,
        )
        new_state = table.to_state()
    return FixedFleetSolveOutcome(
        result=result, table_state=new_state, counts=counts,
        from_table=False, warm_started=warm_started, abs_gap=abs_gap,
        table_lb=table_lb, table_ub=reference,
    )


__all__ = [
    "FixedFleetSolveOutcome",
    "STALL_MIN_SOLVE_SECONDS",
    "STATUS_STALLED",
    "Stage2PieceCache",
    "TABLE_BACKEND",
    "best_cached_policy",
    "build_cuts_payload",
    "seed_policy_hint",
    "solve_fixed_fleet_stage2",
    "trial_fleet_counts",
]
