"""Gap-scaled absolute tolerance shared by every Phase-2 Stage-2 MIP.

Forward Stage 2, the backward Stage-2 refresh and the fixed-fleet pieces of
the S2->S1 fleet oracle all compute the same fixed-fleet value ``C(n)`` (one
node, one purchased fleet, one Stage-3 cut archive).  Each of those MIPs may
stop as soon as its *absolute* gap is within

    eps_n     = max(eps_floor, kappa * (UB - LB)) / W,
    eps_floor = floor_share * phase2_tol * |UB|,
    W         = sum of the Stage-2 node weights (``multi_coeff``),

where ``UB``/``LB`` are the current outer Phase-2 bounds.  ``W`` is the
weight the forward UB and the Stage-1 objective attach to the Stage-2 value
functions (node count only when every weight is one).

This is a requested stopping tolerance, not a guarantee about a returned
interval. Time limits and cached stalled pieces may return wider intervals;
rescoring an assignment alone supplies no new lower bound. Callers must
check the actual certified endpoints. Only when every relevant fixed-fleet
interval has width at most ``eps_n`` is their weighted sum bounded by
``W * eps_n``. This does not bound the error of the Stage-3 cut approximation
or imply that the outer algorithm will close its gap.

Cut validity comes from using certified lower endpoints; feasible physical
policies provide upper bounds. Only the resulting outer LB/UB interval can
certify outer convergence. The positive tolerance floor can impede further
progress and is not a convergence proof.

On C50 the fixed-fleet piece is a
multiple-subset-sum (outsourcing cost ~ volume): its compact MIP bound
stalls at the root about 230-440 below the optimum.  With ``kappa = 0.1``
and ``phase2_tol = 1e-6`` the schedule gave ``eps ~ 70`` at a 1.1% gap, below
that plateau, and every t=0/t=1 node ran to its time limit in the refresh
(300s), the Level Set (2 x 300s) and the forward pass (1800s) each
iteration; at ``eps ~ 350`` the same pieces close in 0.1-160s and the
Level Set and forward pass are then answered from the table.

A relative cap ``rel_cap * |reference|`` (``reference`` = the best cached
policy value for the same fleet, when one exists) only ever *tightens* a MIP
stop so that small-objective nodes are not solved with a slack that is a
large fraction of their own value when a cheap exact solve is available.
Acceptance/exactness tests (piece "exact", table hits) use the uncapped
``eps_n`` so that a piece closed by a tighter MIP is trivially accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


DEFAULT_KAPPA = 0.5
DEFAULT_FLOOR_SHARE = 0.5
DEFAULT_REL_CAP = 0.02


def _finite(value) -> Optional[float]:
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return scalar if math.isfinite(scalar) else None


def effective_abs_gap(abs_tol, reference=None, rel_cap=None) -> Optional[float]:
    """MIP stopping gap: ``abs_tol`` capped at ``rel_cap * max(1, |reference|)``.

    ``None`` when no absolute tolerance is scheduled.  The cap is skipped
    without a finite reference or a positive ``rel_cap``.
    """
    eps = _finite(abs_tol)
    if eps is None or eps <= 0.0:
        return None
    ref = _finite(reference)
    cap = _finite(rel_cap)
    if ref is None or cap is None or cap <= 0.0:
        return eps
    return min(eps, cap * max(1.0, abs(ref)))


def apply_abs_gap(model, abs_gap) -> bool:
    """Set Gurobi ``MIPGapAbs`` when a positive absolute gap is scheduled."""
    eps = _finite(abs_gap)
    if eps is None or eps <= 0.0:
        return False
    model.setParam("MIPGapAbs", float(eps))
    return True


def stage2_weight_mass(nodes) -> float:
    """``sum(multi_coeff)`` over the Stage-2 nodes (node count for unit weights).

    The forward UB adds ``multi_coeff * cost`` per node and the Stage-1
    objective weighs each S2->S1 value function by the same coefficient, so
    a per-node slack ``eps`` moves either bound by at most ``mass * eps``.
    """
    mass = 0.0
    count = 0
    for node in nodes:
        count += 1
        weight = _finite(getattr(node, "multi_coeff", 1.0))
        mass += abs(weight) if weight is not None else 1.0
    return mass if mass > 0.0 else float(count)


@dataclass(frozen=True)
class Stage2ToleranceSchedule:
    """Per-node absolute Stage-2 tolerance derived from the outer gap.

    ``weight_mass`` is the total Stage-2 node weight (``stage2_weight_mass``);
    it defaults to ``n_nodes`` (unit weights).
    """

    phase2_tol: float
    n_nodes: int
    kappa: float = DEFAULT_KAPPA
    floor_share: float = DEFAULT_FLOOR_SHARE
    rel_cap: float = DEFAULT_REL_CAP
    weight_mass: Optional[float] = None

    def __post_init__(self):
        if int(self.n_nodes) < 1:
            raise ValueError("Stage-2 tolerance schedule needs at least one node")
        if not (float(self.phase2_tol) >= 0.0):
            raise ValueError("phase2_tol must be non-negative")
        if not (float(self.kappa) >= 0.0):
            raise ValueError("kappa must be non-negative")
        if not (0.0 <= float(self.floor_share) < 1.0):
            # The floor must leave part of the target for the true gap.
            raise ValueError("floor_share must lie in [0, 1)")
        if not (float(self.rel_cap) >= 0.0):
            raise ValueError("rel_cap must be non-negative")
        if self.weight_mass is not None and not (
            _finite(self.weight_mass) is not None and float(self.weight_mass) > 0.0
        ):
            raise ValueError("weight_mass must be a positive finite number")

    @property
    def divisor(self) -> float:
        """Total Stage-2 weight that a per-node ``eps`` is multiplied by."""
        if self.weight_mass is not None:
            return float(self.weight_mass)
        return float(self.n_nodes)

    @classmethod
    def from_config(
        cls, config, n_nodes: int, weight_mass: Optional[float] = None
    ) -> Optional["Stage2ToleranceSchedule"]:
        """Build from ``AlgorithmConfig``; ``None`` when the rule is disabled."""
        kappa = float(config.get("phase2_s2_abs_tol_kappa", DEFAULT_KAPPA))
        floor_share = float(
            config.get("phase2_s2_abs_tol_floor_share", DEFAULT_FLOOR_SHARE)
        )
        if kappa <= 0.0 and floor_share <= 0.0:
            return None
        return cls(
            phase2_tol=float(config.get("phase2_tol", 1e-2)),
            n_nodes=int(n_nodes),
            kappa=kappa,
            floor_share=floor_share,
            rel_cap=float(config.get("phase2_s2_abs_tol_rel_cap", DEFAULT_REL_CAP)),
            weight_mass=weight_mass,
        )

    def total_slack(self, ub, lb) -> Optional[float]:
        """``max(eps_floor, kappa*(UB-LB))``; ``None`` without a finite UB."""
        ub_value = _finite(ub)
        if ub_value is None:
            return None
        floor = self.floor_share * self.phase2_tol * abs(ub_value)
        lb_value = _finite(lb)
        scaled = 0.0
        if lb_value is not None and ub_value > lb_value:
            scaled = self.kappa * (ub_value - lb_value)
        total = max(floor, scaled)
        return total if total > 0.0 else None

    def node_tolerance(self, ub, lb) -> Optional[float]:
        """``eps_n`` for the current bounds; ``None`` when unavailable."""
        total = self.total_slack(ub, lb)
        if total is None:
            return None
        return total / self.divisor

    def mip_gap(self, ub, lb, reference=None) -> Optional[float]:
        """Capped MIP stop for one node (see ``effective_abs_gap``)."""
        return effective_abs_gap(self.node_tolerance(ub, lb), reference, self.rel_cap)

    def describe(self, ub, lb) -> str:
        eps = self.node_tolerance(ub, lb)
        if eps is None:
            return "S2-eps=n/a (no finite UB yet)"
        total = self.total_slack(ub, lb)
        ub_value = _finite(ub)
        floor = self.floor_share * self.phase2_tol * abs(ub_value)
        source = "floor" if total <= floor * (1.0 + 1e-12) else "kappa*gap"
        return (
            f"S2-eps={eps:,.3f}/node ({source}; total={total:,.1f}, "
            f"N={self.n_nodes}, weight_mass={self.divisor:g}, kappa={self.kappa:g}, "
            f"floor_share={self.floor_share:g}, rel_cap={self.rel_cap:g})"
        )


__all__ = [
    "DEFAULT_FLOOR_SHARE",
    "DEFAULT_KAPPA",
    "DEFAULT_REL_CAP",
    "Stage2ToleranceSchedule",
    "apply_abs_gap",
    "stage2_weight_mass",
    "effective_abs_gap",
]
