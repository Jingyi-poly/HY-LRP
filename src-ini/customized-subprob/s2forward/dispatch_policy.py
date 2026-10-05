"""Pure policy for choosing a certified forward Stage-2 backend.

The policy observes only cheap, immutable features of one fixed-fleet
Stage-2 call.  It does not build a model, run a solver, or certify a bound.
The execution layer remains responsible for closing or bounding the DP
interval, checking Gurobi status, validating incumbents, and rescoring the
complete cut archive.

``auto`` is deliberately a frozen lookup table backed by the crossover
matrix in ``benchmarks/forward_crossover_report.md``.  Unmeasured customer
counts are not interpolated: when DP fits the aggregate memory gate they use
a short zero-gap Gurobi probe, followed by DP only when the current solve
purpose does not accept the probe.  Phase 1 may use any independently
feasibility-certified and archive-rescored incumbent as a trial policy;
Phase 2 may skip DP only when the directed bound equals the exact rational
policy score.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


MODE_ENV = "VRP_S2_FORWARD_SOLVER"
PROBE_SECONDS_ENV = "VRP_S2_FORWARD_AUTO_GRB_PROBE_S"
DEFAULT_PROBE_SECONDS = 0.2
LARGE_DP_PROBE_SECONDS = 1.5
PHASE1_LARGE_PROBE_SECONDS = 0.1
_MODES = ("auto", "dp", "gurobi", "bp")
# ``bp_then_gurobi``: fixed-fleet branch-and-price (C++ kernel) first; the
# execution layer falls back to full Gurobi, warm-started from the
# branch-and-price incumbent, when the certified interval is too wide.
_BACKENDS = ("dp", "gurobi", "feasible_probe", "probe_then_dp", "bp_then_gurobi")


@dataclass(frozen=True)
class ForwardDispatchFeatures:
    """Cheap features used by :class:`ForwardDispatchPlan`.

    Cut counts include learned cuts only, and only the private successor pools
    of vehicles bought at the current period.  Static RouteCuts are present in
    both backends and therefore are intentionally excluded.
    """

    mode: str
    phase: str
    active_customers: int
    purchased_vehicles: Tuple[Any, ...]
    private_cut_pool_sizes: Tuple[int, ...]
    max_private_cuts: int
    total_private_cuts: int
    concurrent_slots: int
    memory_gate_evaluated: bool
    memory_safe: Optional[bool]
    memory_max_customers: Optional[int]
    memory_budget_mb: Optional[float]
    memory_budget_source: Optional[str]
    estimated_aggregate_peak_mb: Optional[float]


@dataclass(frozen=True)
class ForwardDispatchPlan:
    """Backend order selected for one fixed-fleet forward Stage-2 solve."""

    backend: str
    probe_seconds: float
    reason: str
    features: ForwardDispatchFeatures

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError(f"unknown forward Stage-2 backend {self.backend!r}")
        if self.backend in {"feasible_probe", "probe_then_dp"}:
            if not math.isfinite(self.probe_seconds) or self.probe_seconds <= 0.0:
                raise ValueError(
                    f"{self.backend} requires a finite positive probe"
                )
        elif self.probe_seconds != 0.0:
            raise ValueError("non-probe plans must have probe_seconds=0")


def _configured_mode() -> str:
    raw = os.environ.get(MODE_ENV, "auto")
    mode = str(raw).strip().lower() or "auto"
    if mode not in _MODES:
        raise ValueError(f"{MODE_ENV}={mode!r}; expected one of {_MODES}")
    return mode


def _normalise_phase(phase: object) -> str:
    raw = str(phase).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "1": "phase1",
        "p1": "phase1",
        "phase1": "phase1",
        "2": "phase2",
        "p2": "phase2",
        "phase2": "phase2",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(
            f"phase={phase!r}; expected phase1/P1/1 or phase2/P2/2"
        ) from exc


def _probe_seconds(default: float) -> float:
    raw = os.environ.get(PROBE_SECONDS_ENV)
    try:
        value = float(default if raw is None or raw.strip() == "" else raw)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{PROBE_SECONDS_ENV} must be numeric") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{PROBE_SECONDS_ENV} must be finite and positive")
    return value


def branch_price_available() -> bool:
    """True when the branch-and-price kernel is built and not disabled."""
    from .bp_solver import kernel_available

    return kernel_available()


BP_AUTO_ENV = "VRP_S2_BP_AUTO"


def branch_price_auto_enabled() -> bool:
    """True when ``auto`` mode may route memory-rejected nodes to branch-and-price.

    Opt-in: on C50 the kernel currently leaves a ~0.3% interval after its
    default budget and the node still goes to Gurobi, so ``auto`` keeps the
    plain Gurobi path unless the operator asks for the detour explicitly
    (``VRP_S2_FORWARD_SOLVER=bp`` always uses it).
    """
    raw = os.environ.get(BP_AUTO_ENV, "").strip().lower()
    return raw in ("1", "on", "true", "yes") and branch_price_available()


def _dp_memory_policy(concurrent_slots: Optional[int]) -> Mapping[str, object]:
    # Lazy import keeps this policy module solver-free and lets forced Gurobi
    # bypass every DP/kernel/memory decision.
    from .subset_dp import dp_memory_policy

    return dp_memory_policy(concurrent_slots)


def _dp_concurrent_slots(concurrent_slots: Optional[int]) -> tuple[int, str]:
    from .subset_dp import dp_concurrent_slots

    return dp_concurrent_slots(concurrent_slots)


def _stage3_cut_pools(cut_lag) -> Mapping:
    if cut_lag is None:
        return {}
    if isinstance(cut_lag, Mapping):
        pools = cut_lag.get(3, {})
    else:
        try:
            pools = cut_lag[3]
        except (IndexError, KeyError, TypeError):
            pools = {}
    if pools is None:
        return {}
    if not isinstance(pools, Mapping):
        raise ValueError("cut_lag[3] must be a mapping of successor cut pools")
    return pools


def _cheap_features(prob_data, node, cut_lag, x_prev) -> tuple:
    customers = tuple(prob_data.J)
    vehicles = tuple(prob_data.V)
    successors = tuple(node.successor)
    if len(successors) != len(vehicles):
        raise ValueError(
            "Stage-2 node must have exactly one private Stage-3 successor "
            f"per vehicle ({len(successors)} != {len(vehicles)})"
        )
    try:
        period = int(node.info[1])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Stage-2 node.info must contain (scenario, period)") from exc

    active = sum(
        1 for customer in customers if float(node.active[customer]) > 0.5
    )
    purchased_positions = tuple(
        position
        for position, vehicle in enumerate(vehicles)
        if float(x_prev.get(f"z[{vehicle},{period}]", 0.0)) >= 0.5
    )
    purchased = tuple(vehicles[position] for position in purchased_positions)
    pools = _stage3_cut_pools(cut_lag)
    sizes = tuple(
        len(pools.get(successors[position], ()))
        for position in purchased_positions
    )
    return active, purchased, sizes


def _validated_memory_policy(
    concurrent_slots: Optional[int],
) -> tuple[Mapping[str, object], int, int]:
    policy = _dp_memory_policy(concurrent_slots)
    try:
        slots = int(policy["concurrent_slots"])
        maximum = int(policy["max_customers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("DP memory policy returned an invalid schema") from exc
    if slots <= 0 or maximum < 0:
        raise ValueError("DP memory policy returned invalid limits")
    if concurrent_slots is not None and slots != int(concurrent_slots):
        raise ValueError("DP memory policy ignored the actual concurrent slots")
    return policy, slots, maximum


def _optional_float(policy: Mapping[str, object], key: str) -> Optional[float]:
    raw = policy.get(key)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"DP memory policy field {key!r} must be numeric") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"DP memory policy field {key!r} must be finite/nonnegative")
    return value


def _make_features(
    *,
    mode: str,
    phase: str,
    active: int,
    purchased: tuple,
    sizes: tuple[int, ...],
    slots: int,
    memory_policy: Optional[Mapping[str, object]],
    memory_max: Optional[int],
) -> ForwardDispatchFeatures:
    memory_safe = None if memory_max is None else active <= memory_max
    return ForwardDispatchFeatures(
        mode=mode,
        phase=phase,
        active_customers=active,
        purchased_vehicles=purchased,
        private_cut_pool_sizes=sizes,
        max_private_cuts=max(sizes, default=0),
        total_private_cuts=sum(sizes),
        concurrent_slots=slots,
        memory_gate_evaluated=memory_policy is not None,
        memory_safe=memory_safe,
        memory_max_customers=memory_max,
        memory_budget_mb=(
            _optional_float(memory_policy, "memory_budget_mb")
            if memory_policy is not None
            else None
        ),
        memory_budget_source=(
            str(memory_policy.get("budget_source", "unknown"))
            if memory_policy is not None
            else None
        ),
        estimated_aggregate_peak_mb=(
            _optional_float(memory_policy, "estimated_aggregate_peak_mb")
            if memory_policy is not None
            else None
        ),
    )


def _plan(backend: str, reason: str, features: ForwardDispatchFeatures, *,
          probe_default: float = DEFAULT_PROBE_SECONDS) -> ForwardDispatchPlan:
    return ForwardDispatchPlan(
        backend=backend,
        probe_seconds=(
            _probe_seconds(probe_default)
            if backend in {"feasible_probe", "probe_then_dp"}
            else 0.0
        ),
        reason=reason,
        features=features,
    )


def build_forward_dispatch_plan(
    prob_data,
    node,
    cut_lag,
    x_prev,
    *,
    phase,
    concurrent_slots=None,
) -> ForwardDispatchPlan:
    """Return the solver order for one fixed-fleet forward Stage-2 call.

    ``mode=dp`` and ``mode=gurobi`` are explicit user choices.  The selector
    never overrides them: in particular, an explicitly requested DP remains
    the execution layer's responsibility to run or reject with a useful error.

    ``auto`` first enforces the aggregate DP memory gate, then applies only
    measured crossover buckets.  The gray and unmeasured buckets mean
    "brief zero-gap Gurobi attempt; apply the purpose-specific acceptance
    contract; otherwise run the certified-gap DP, or full Gurobi if DP is
    rejected".  The execution layer owns that contract: Phase 1 accepts a
    certified/rescored feasible incumbent, while Phase 2 requires strict
    exact closure before a probe can replace DP.
    """
    mode = _configured_mode()
    phase_name = _normalise_phase(phase)
    probe_contract = (
        "phase1_feasible_probe_else_dp"
        if phase_name == "phase1"
        else "phase2_strict_probe_else_dp"
    )
    active, purchased, sizes = _cheap_features(
        prob_data, node, cut_lag, x_prev
    )

    if mode != "auto":
        slots, _source = _dp_concurrent_slots(concurrent_slots)
        features = _make_features(
            mode=mode,
            phase=phase_name,
            active=active,
            purchased=purchased,
            sizes=sizes,
            slots=slots,
            memory_policy=None,
            memory_max=None,
        )
        explicit = {"dp": "dp", "gurobi": "gurobi", "bp": "bp_then_gurobi"}[mode]
        return _plan(
            explicit,
            f"explicit_{mode}_execution_layer_controls_applicability",
            features,
        )

    memory, slots, memory_max = _validated_memory_policy(concurrent_slots)
    features = _make_features(
        mode=mode,
        phase=phase_name,
        active=active,
        purchased=purchased,
        sizes=sizes,
        slots=slots,
        memory_policy=memory,
        memory_max=memory_max,
    )
    if not features.memory_safe:
        if phase_name == "phase1":
            return _plan(
                "feasible_probe",
                f"auto_phase1_memory_reject_n{active}_max{memory_max}_"
                f"slots{slots}_feasible_probe",
                features,
                probe_default=PHASE1_LARGE_PROBE_SECONDS,
            )
        # Beyond the DP memory gate the Phase-2 node is a large assignment
        # problem; branch-and-price may precede the compact Gurobi MIP there.
        if branch_price_auto_enabled():
            return _plan(
                "bp_then_gurobi",
                f"auto_memory_reject_n{active}_max{memory_max}_slots{slots}_branch_price",
                features,
            )
        return _plan(
            "gurobi",
            f"auto_memory_reject_n{active}_max{memory_max}_slots{slots}",
            features,
        )

    cuts = features.max_private_cuts
    if active <= 10:
        return _plan("dp", "auto_measured_c10_or_less_dp", features)
    if active == 12:
        return _plan("dp", "auto_measured_c12_dp", features)
    if active in (15, 16):
        if cuts >= 16:
            return _plan("dp", f"auto_measured_c{active}_k{cuts}_dp", features)
        return _plan(
            "gurobi", f"auto_measured_c{active}_k{cuts}_gurobi", features
        )
    if active in (18, 19):
        if cuts <= 16:
            return _plan(
                "gurobi", f"auto_measured_c{active}_k{cuts}_gurobi", features
            )
        if cuts >= 32:
            return _plan(
                "dp", f"auto_measured_c{active}_k{cuts}_dp", features
            )
        return _plan(
            "feasible_probe" if phase_name == "phase1" else "probe_then_dp",
            f"auto_measured_c{active}_k{cuts}_gray_{probe_contract}",
            features,
        )
    if active == 20:
        if cuts <= 16:
            return _plan(
                "gurobi", f"auto_measured_c20_k{cuts}_gurobi", features
            )
        if cuts >= 20:
            return _plan("dp", f"auto_measured_c20_k{cuts}_dp", features)
        return _plan(
            "feasible_probe" if phase_name == "phase1" else "probe_then_dp",
            f"auto_measured_c20_k{cuts}_gray_{probe_contract}",
            features,
        )
    if active in (21, 22):
        return _plan(
            "feasible_probe" if phase_name == "phase1" else "probe_then_dp",
            f"auto_c21_c22_{probe_contract}",
            features,
            probe_default=(
                PHASE1_LARGE_PROBE_SECONDS
                if phase_name == "phase1"
                else LARGE_DP_PROBE_SECONDS
            ),
        )

    # C11/C13/C14/C17 are inside the DP domain but have no crossover matrix.
    # Do not silently interpolate their nearest measured neighbour.
    return _plan(
        "feasible_probe" if phase_name == "phase1" else "probe_then_dp",
        f"auto_unmeasured_c{active}_benchmark_required_{probe_contract}",
        features,
    )


__all__ = [
    "DEFAULT_PROBE_SECONDS",
    "LARGE_DP_PROBE_SECONDS",
    "PHASE1_LARGE_PROBE_SECONDS",
    "MODE_ENV",
    "PROBE_SECONDS_ENV",
    "ForwardDispatchFeatures",
    "ForwardDispatchPlan",
    "build_forward_dispatch_plan",
]
