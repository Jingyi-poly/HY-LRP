"""Fixed-fleet Stage-2 assignment (the *forward* Stage-2 subproblem).

Given the Stage-1 fleet ``z`` (a leading run of every vehicle type, see
``fleet_pieces``), Stage 2 chooses ``alpha``/``y`` minimising outsourcing
plus the Stage-3 value-function cut maxima.  ``subset_dp`` exhaustively
evaluates customer subsets in binary64 (C++ kernel in ``cpp/``, numpy
fallback) for nodes with up to ~20 active customers; ``forward_stage2_dp``
adds a directed lower endpoint, feasible-policy certification, and complete
archive rescoring.  ``purpose`` keeps the Phase-1 feasible-trial and Phase-2
DP-gap acceptance contracts explicit.

``s2backward`` (Lagrangian oracle by fleet enumeration) reuses the same
solver for its fixed-fleet pieces ``C(n)``.
"""

from .fleet_pieces import Counts, FleetLayout
from .purpose import (
    PHASE1_FEASIBLE_TRIAL,
    PHASE2_FORWARD,
    PHASE2_REFRESH,
    ForwardStage2Purpose,
)
from .subset_dp import (
    DEFAULT_MAX_CUSTOMERS,
    SubsetDPNotApplicable,
    SubsetDPPieceSolver,
    dp_concurrent_slots,
    dp_memory_budget_mb,
    dp_memory_policy,
    default_max_customers,
    kernel_available,
    physical_memory_mb,
    subset_dp_solver_or_none,
)

__all__ = [
    "Counts",
    "FleetLayout",
    "ForwardStage2Purpose",
    "PHASE1_FEASIBLE_TRIAL",
    "PHASE2_FORWARD",
    "PHASE2_REFRESH",
    "DEFAULT_MAX_CUSTOMERS",
    "SubsetDPNotApplicable",
    "SubsetDPPieceSolver",
    "dp_concurrent_slots",
    "dp_memory_budget_mb",
    "dp_memory_policy",
    "default_max_customers",
    "kernel_available",
    "physical_memory_mb",
    "subset_dp_solver_or_none",
]
