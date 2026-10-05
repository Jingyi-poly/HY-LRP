"""Forward and backward solvers for SDDP.

The public classes are loaded on first access.  Keeping package import itself
lightweight prevents a cycle when cut modules import small solver helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


__all__ = ["ForwardSolver", "BackwardSolverSBC", "BackwardSolverLagrangian"]

if TYPE_CHECKING:
    from .backward_solver_lag import BackwardSolverLagrangian
    from .backward_solver_sbc_phase1 import BackwardSolverSBC
    from .forward_solver import ForwardSolver


def __getattr__(name: str):
    if name == "ForwardSolver":
        from .forward_solver import ForwardSolver

        value = ForwardSolver
    elif name == "BackwardSolverSBC":
        from .backward_solver_sbc_phase1 import BackwardSolverSBC

        value = BackwardSolverSBC
    elif name == "BackwardSolverLagrangian":
        from .backward_solver_lag import BackwardSolverLagrangian

        value = BackwardSolverLagrangian
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
