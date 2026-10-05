"""Explicit acceptance contracts for forward-style Stage-2 solves."""

from __future__ import annotations

from enum import Enum
from functools import wraps

from core.backend_telemetry import backend_scope


class ForwardStage2Purpose(str, Enum):
    """Why a fixed-fleet Stage-2 policy is being requested.

    Phase 1 needs a certified feasible trial policy.  Phase 2 keeps the
    project's numerical-zero requirement for subset-DP results.  Exact
    cross-outer reuse is a separate, stricter certificate and is never implied
    by any purpose here.
    """

    PHASE1_FEASIBLE_TRIAL = "phase1_feasible_trial"
    PHASE2_FORWARD = "phase2_forward"
    PHASE2_REFRESH = "phase2_refresh"

    @property
    def dispatch_phase(self) -> str:
        if self is ForwardStage2Purpose.PHASE1_FEASIBLE_TRIAL:
            return "phase1"
        return "phase2"

    @property
    def requires_tight_dp_gap(self) -> bool:
        return self is not ForwardStage2Purpose.PHASE1_FEASIBLE_TRIAL

    @property
    def log_direction(self) -> str:
        if self is ForwardStage2Purpose.PHASE2_REFRESH:
            return "backward-refresh"
        return "forward"


PHASE1_FEASIBLE_TRIAL = ForwardStage2Purpose.PHASE1_FEASIBLE_TRIAL
PHASE2_FORWARD = ForwardStage2Purpose.PHASE2_FORWARD
PHASE2_REFRESH = ForwardStage2Purpose.PHASE2_REFRESH


def normalize_forward_stage2_purpose(value) -> ForwardStage2Purpose:
    """Return a validated purpose; generic ``phase1``/``phase2`` is rejected."""
    if isinstance(value, ForwardStage2Purpose):
        return value
    try:
        return ForwardStage2Purpose(str(value).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in ForwardStage2Purpose)
        raise ValueError(
            f"forward Stage-2 purpose={value!r}; expected one of {allowed}"
        ) from exc


def forward_stage2_backend_scope(function):
    """Attribute shared fixed-fleet calls, including backward refreshes."""
    @wraps(function)
    def scoped(*args, **kwargs):
        purpose = normalize_forward_stage2_purpose(
            kwargs.get("purpose", PHASE2_FORWARD)
        )
        with backend_scope(
            phase=1 if purpose is PHASE1_FEASIBLE_TRIAL else 2,
            path=purpose.log_direction, stage=2
        ):
            return function(*args, **kwargs)
    return scoped


__all__ = [
    "ForwardStage2Purpose",
    "PHASE1_FEASIBLE_TRIAL",
    "PHASE2_FORWARD",
    "PHASE2_REFRESH",
    "normalize_forward_stage2_purpose",
]
