"""Exact-safe Phase-1 Stage-2 strengthening dispatcher.

The standard fixed-RHS LP cut is built by the caller before this module is
entered.  ``auto`` then spends only short, independent budgets on optional
strengthening oracles and returns the strongest *certified* lower bound:

1. a zero-MIPGap Gurobi probe;
2. the purchase-prefix BPC root relaxation;
3. the exact free-fleet subset DP, but only when both the aggregate memory
   gate and a cold-runtime prediction say it is cheap.

Failure or timeout can therefore weaken a cut only back to the already valid
LP intercept; it can never promote an incumbent or an uncertified backend
number into a backward bound.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional


PHASE1_S2_ORACLE_ENV = "VRP_PHASE1_S2_ORACLE"
PHASE1_S2_ORACLE_CHECK_ENV = "VRP_PHASE1_S2_ORACLE_CHECK"
PHASE1_S2_AUTO_GRB_PROBE_ENV = "VRP_PHASE1_S2_AUTO_GRB_PROBE_S"
PHASE1_S2_AUTO_BPC_LIMIT_ENV = "VRP_PHASE1_S2_AUTO_BPC_TIME_LIMIT_S"
PHASE1_S2_AUTO_BPC_MAX_ACTIVE_ENV = "VRP_PHASE1_S2_AUTO_BPC_MAX_ACTIVE"
PHASE1_S2_AUTO_DP_MAX_ACTIVE_ENV = "VRP_PHASE1_S2_AUTO_DP_MAX_ACTIVE"
PHASE1_S2_AUTO_DP_PREDICTED_LIMIT_ENV = (
    "VRP_PHASE1_S2_AUTO_DP_MAX_PREDICTED_S"
)

PHASE1_S2_ORACLE_MODES = ("auto", "gurobi", "fleet_enum")
DEFAULT_GRB_PROBE_SECONDS = 0.1
DEFAULT_BPC_SECONDS = 5.0
DEFAULT_BPC_MAX_ACTIVE = 49
DEFAULT_DP_MAX_ACTIVE = 18
DEFAULT_DP_MAX_PREDICTED_SECONDS = 1.0


def phase1_s2_oracle_mode() -> str:
    """Configured Phase-1 Stage-2 oracle mode (read at call time)."""
    raw = os.environ.get(PHASE1_S2_ORACLE_ENV, "auto").strip().lower() or "auto"
    if raw not in PHASE1_S2_ORACLE_MODES:
        raise ValueError(
            f"{PHASE1_S2_ORACLE_ENV}={raw!r}; expected one of "
            f"{PHASE1_S2_ORACLE_MODES}"
        )
    return raw


def active_customer_count(prob_data, node) -> int:
    active = getattr(node, "active", None)
    if active is None:
        return len(prob_data.J)
    return sum(
        1 for customer in prob_data.J if float(active[customer]) > 0.5
    )


def phase1_s2_auto_bpc_gate_reason(prob_data, node) -> Optional[str]:
    """Return why optional BPC is skipped, or ``None`` when measured useful.

    C50 six-worker A/B found that root BPC dominated the Phase-1 wall clock
    while its rare, tiny local gains did not improve the handoff bounds.  The
    C25/C50 trace audit also found every positive marginal BPC gain at period
    zero; noninitial-period calls added none over the LP/probe certificate.
    standard LP cut and Gurobi-probe ObjBound remain certified when this
    optional strengthening is skipped.  An explicit environment override is
    retained for the active-customer crossover; the period gate deliberately
    follows the frozen measured policy.
    """
    maximum = _nonnegative_env_int(
        PHASE1_S2_AUTO_BPC_MAX_ACTIVE_ENV, DEFAULT_BPC_MAX_ACTIVE
    )
    if maximum <= 0 or active_customer_count(prob_data, node) > maximum:
        return "active_customer_size_gate"
    info = getattr(node, "info", None)
    if not isinstance(info, (tuple, list)) or len(info) < 2:
        raise ValueError("Phase-1 Stage-2 node.info must contain (omega, period)")
    if int(info[1]) != 0:
        return "noninitial_period_gate"
    return None


def _nonnegative_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        value = float(default if raw is None or raw.strip() == "" else raw)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _nonnegative_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        value = int(default if raw is None or raw.strip() == "" else raw)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _finite_certified_lb(payload: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not payload or not bool(payload.get("lb_certified", False)):
        return None
    try:
        value = float(payload.get("lb"))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _remaining_seconds(started: float, total: Optional[float], clock) -> float:
    if total is None or math.isinf(total):
        return math.inf
    return max(0.0, total - max(0.0, float(clock()) - started))


def _safe_call(callback, *args) -> dict:
    try:
        result = callback(*args)
    except Exception as exc:  # Optional strengthening must safely fall back.
        return {
            "ok": False,
            "lb_certified": False,
            "reason": f"{type(exc).__name__}:{exc}",
        }
    if not isinstance(result, Mapping):
        return {
            "ok": False,
            "lb_certified": False,
            "reason": "backend_returned_non_mapping",
        }
    return dict(result)


@dataclass
class Phase1S2AutoResult:
    bound: float
    source: str
    exact: bool
    candidates: dict[str, Optional[float]]
    probe: dict = field(default_factory=dict)
    bpc: dict = field(default_factory=dict)
    direct_dp: dict = field(default_factory=dict)
    seconds: float = 0.0

    def format_tag(self) -> str:
        def value(name: str) -> str:
            candidate = self.candidates.get(name)
            return "n/a" if candidate is None else f"{candidate:.2f}"

        def status(payload: Mapping[str, Any], candidate_name: str) -> str:
            for key in ("status_name", "status", "termination_reason", "reason"):
                if payload.get(key) not in (None, ""):
                    return str(payload[key])
            return (
                "certified"
                if self.candidates.get(candidate_name) is not None
                else "skip"
            )

        probe_status = status(self.probe, "gurobi_probe")
        bpc_status = status(self.bpc, "bpc")
        dp_status = status(self.direct_dp, "direct_dp")
        return (
            " | [auto-composite "
            f"best={self.source}:{self.bound:.2f} exact={int(self.exact)} "
            f"probe={value('gurobi_probe')}({probe_status}) "
            f"bpc={value('bpc')}({bpc_status}) "
            f"dp={value('direct_dp')}({dp_status}) "
            f"t={self.seconds:.2f}s]"
        )


def run_phase1_s2_auto(
    v_lp: float,
    total_time_limit: Optional[float],
    *,
    gurobi_probe: Callable[[float], Mapping[str, Any]],
    bpc_root: Callable[[float], Mapping[str, Any]],
    direct_dp: Callable[[int, float, float], Mapping[str, Any]],
    bpc_enabled: bool,
    bpc_skip_reason: Optional[str] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Phase1S2AutoResult:
    """Run the auto composite and return the strongest certified endpoint.

    Callback payloads use ``lb`` plus ``lb_certified``.  A Gurobi probe may
    additionally set ``exact=True``; that is the only early-return condition.
    In particular, ``SolCount`` is deliberately irrelevant to accepting its
    certified ``ObjBound``.
    """
    baseline = float(v_lp)
    if not math.isfinite(baseline):
        raise ValueError("Phase-1 Stage-2 LP intercept must be finite")
    if total_time_limit is None:
        total = None
    else:
        total = float(total_time_limit)
        if math.isnan(total) or total < 0.0:
            raise ValueError(
                "Phase-1 Stage-2 total time limit must be nonnegative"
            )
        if math.isinf(total):
            total = None

    started = float(clock())
    candidates: dict[str, Optional[float]] = {
        "v_lp": baseline,
        "gurobi_probe": None,
        "bpc": None,
        "direct_dp": None,
    }

    probe_result: dict = {"reason": "probe_budget_exhausted"}
    configured_probe = _nonnegative_env_float(
        PHASE1_S2_AUTO_GRB_PROBE_ENV, DEFAULT_GRB_PROBE_SECONDS
    )
    probe_limit = min(configured_probe, _remaining_seconds(started, total, clock))
    if probe_limit > 0.0:
        probe_result = _safe_call(gurobi_probe, probe_limit)
        candidates["gurobi_probe"] = _finite_certified_lb(probe_result)
        if (
            bool(probe_result.get("exact", False))
            and candidates["gurobi_probe"] is not None
        ):
            bound = max(baseline, float(candidates["gurobi_probe"]))
            source = (
                "gurobi_probe" if float(candidates["gurobi_probe"]) >= baseline
                else "v_lp"
            )
            return Phase1S2AutoResult(
                bound=bound,
                source=source,
                exact=(source == "gurobi_probe"),
                candidates=candidates,
                probe=probe_result,
                bpc={"reason": "skipped_after_exact_probe"},
                direct_dp={"reason": "skipped_after_exact_probe"},
                seconds=max(0.0, float(clock()) - started),
            )

    bpc_result: dict = {
        "reason": (
            bpc_skip_reason or "phase1_s2_bpc_disabled"
            if not bpc_enabled
            else "bpc_budget_exhausted"
        )
    }
    remaining = _remaining_seconds(started, total, clock)
    configured_bpc = _nonnegative_env_float(
        PHASE1_S2_AUTO_BPC_LIMIT_ENV, DEFAULT_BPC_SECONDS
    )
    bpc_limit = min(configured_bpc, remaining)
    if bpc_enabled and bpc_limit > 0.0:
        bpc_result = _safe_call(bpc_root, bpc_limit)
        # ``ok`` and incumbent availability are intentionally irrelevant;
        # the native root endpoint is usable only with its explicit certificate.
        candidates["bpc"] = _finite_certified_lb(bpc_result)

    dp_result: dict = {"reason": "direct_dp_budget_exhausted"}
    remaining = _remaining_seconds(started, total, clock)
    dp_max_active = _nonnegative_env_int(
        PHASE1_S2_AUTO_DP_MAX_ACTIVE_ENV, DEFAULT_DP_MAX_ACTIVE
    )
    dp_prediction_limit = _nonnegative_env_float(
        PHASE1_S2_AUTO_DP_PREDICTED_LIMIT_ENV,
        DEFAULT_DP_MAX_PREDICTED_SECONDS,
    )
    if remaining > 0.0 and dp_max_active > 0 and dp_prediction_limit > 0.0:
        dp_result = _safe_call(
            direct_dp,
            dp_max_active,
            min(dp_prediction_limit, remaining),
            remaining,
        )
        candidates["direct_dp"] = _finite_certified_lb(dp_result)

    source = "v_lp"
    bound = baseline
    for name in ("gurobi_probe", "bpc", "direct_dp"):
        candidate = candidates[name]
        if candidate is not None and candidate > bound:
            source = name
            bound = float(candidate)
    selected_payload = {
        "gurobi_probe": probe_result,
        "bpc": bpc_result,
        "direct_dp": dp_result,
    }.get(source, {})
    return Phase1S2AutoResult(
        bound=bound,
        source=source,
        exact=bool(selected_payload.get("exact", False)),
        candidates=candidates,
        probe=probe_result,
        bpc=bpc_result,
        direct_dp=dp_result,
        seconds=max(0.0, float(clock()) - started),
    )


def try_direct_dp_bound(
    prob_data,
    node,
    cut_lag,
    pi_value,
    *,
    max_active: int,
    max_predicted_seconds: float,
    remaining_seconds: float,
) -> dict:
    """Cold-build and run DirectDP only when memory and time predictors agree."""
    from cuts import exact_subroutines as exact_sub
    from s2forward.fleet_pieces import FleetLayout
    from s2forward.subset_dp import (
        SubsetDPNotApplicable,
        SubsetDPPieceSolver,
        default_max_customers,
    )
    from .direct_dp import DirectDPOracle

    active = active_customer_count(prob_data, node)
    memory_ceiling = int(default_max_customers())
    effective_ceiling = min(int(max_active), memory_ceiling)
    if active > effective_ceiling:
        return {
            "ok": False,
            "lb_certified": False,
            "reason": "active_or_memory_gate",
            "active_n": active,
            "max_active": int(max_active),
            "memory_ceiling": memory_ceiling,
        }

    prepared_at = time.perf_counter()
    try:
        layout = FleetLayout(prob_data)
        successors = list(node.successor)
        payload = exact_sub.build_s2_bp_cuts(
            prob_data,
            node,
            cut_lag,
            {successor: position for position, successor in enumerate(successors)},
        )
        piece_solver = SubsetDPPieceSolver(
            prob_data,
            node,
            payload,
            layout,
            max_customers=effective_ceiling,
        )
        full_counts = tuple(len(group) for group in layout.groups)
        solve_prediction = piece_solver.predicted_seconds(full_counts)
    except (AttributeError, KeyError, TypeError, ValueError, SubsetDPNotApplicable) as exc:
        return {
            "ok": False,
            "lb_certified": False,
            "reason": f"direct_dp_not_applicable:{type(exc).__name__}:{exc}",
            "active_n": active,
            "memory_ceiling": memory_ceiling,
        }

    prediction_seconds = time.perf_counter() - prepared_at
    if solve_prediction is None:
        return {
            "ok": False,
            "lb_certified": False,
            "reason": "direct_dp_prediction_unavailable",
            "active_n": active,
            "prediction_seconds": prediction_seconds,
        }
    # The calibrated min-plus estimate is the dominant solve component.  A
    # 50% reserve covers theta-table construction and cold-cache variability;
    # the actually measured preparation/prediction work is added separately.
    cold_prediction = prediction_seconds + 1.5 * float(solve_prediction)
    allowed = min(float(max_predicted_seconds), float(remaining_seconds))
    if not math.isfinite(cold_prediction) or cold_prediction > allowed:
        return {
            "ok": False,
            "lb_certified": False,
            "reason": "direct_dp_prediction_too_expensive",
            "active_n": active,
            "predicted_seconds": cold_prediction,
            "allowed_seconds": allowed,
        }

    try:
        oracle = DirectDPOracle(
            prob_data,
            node,
            cut_lag,
            max_customers=effective_ceiling,
            piece_solver=piece_solver,
        )
        result = dict(oracle.evaluate(pi_value))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError,
            SubsetDPNotApplicable) as exc:
        return {
            "ok": False,
            "lb_certified": False,
            "reason": f"direct_dp_failed:{type(exc).__name__}:{exc}",
            "active_n": active,
            "predicted_seconds": cold_prediction,
        }
    result.setdefault("lb", result.get("outer_lb"))
    result["predicted_seconds"] = cold_prediction
    result["memory_ceiling"] = memory_ceiling
    return result


__all__ = [
    "DEFAULT_BPC_SECONDS",
    "DEFAULT_BPC_MAX_ACTIVE",
    "DEFAULT_DP_MAX_ACTIVE",
    "DEFAULT_DP_MAX_PREDICTED_SECONDS",
    "DEFAULT_GRB_PROBE_SECONDS",
    "PHASE1_S2_AUTO_BPC_LIMIT_ENV",
    "PHASE1_S2_AUTO_BPC_MAX_ACTIVE_ENV",
    "PHASE1_S2_AUTO_DP_MAX_ACTIVE_ENV",
    "PHASE1_S2_AUTO_DP_PREDICTED_LIMIT_ENV",
    "PHASE1_S2_AUTO_GRB_PROBE_ENV",
    "PHASE1_S2_ORACLE_CHECK_ENV",
    "PHASE1_S2_ORACLE_ENV",
    "PHASE1_S2_ORACLE_MODES",
    "Phase1S2AutoResult",
    "active_customer_count",
    "phase1_s2_oracle_mode",
    "phase1_s2_auto_bpc_gate_reason",
    "run_phase1_s2_auto",
    "try_direct_dp_bound",
]
