"""Same-instance global bounds from separate physical search strategies.

This owner never combines cut archives or scenario blocks. Its lower-bound
entry point is restricted to reports from the free-investment Stage-1 refresh;
the caller must pass that actual report, not a fixed-fleet or restricted-master
bound with a renamed source. Report validation preserves the solver certificate
contract but cannot independently prove an arbitrary caller-supplied number.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral

from core.solver_bounds import minimization_bounds_inverted, minimization_gap_percent
from solvers.forward_period_dedup import forward_semantic_fingerprint


_MASTER_SOURCE = "refresh_stage1_bound"
_USABLE_MASTER_STATUSES = frozenset((2, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17))


class PortfolioBounds:
    """Keep max global LB and min independently verified whole-policy UB.

    A complete candidate may use a different investment trajectory. It replaces
    the whole incumbent only if its original physical objective is smaller.
    Search parameters may differ; the actual mathematical instance may not.
    """

    def __init__(self, prob_data, tree):
        self._prob_data, self._tree = prob_data, tree
        self._semantic_fingerprint = forward_semantic_fingerprint(prob_data, tree)
        self._lb = self._ub = None
        self._policy = None
        self._lb_origin = self._ub_origin = None
        self._master_offers = self._policy_offers = 0

    @property
    def semantic_fingerprint(self):
        return self._semantic_fingerprint

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    @property
    def policy(self):
        return deepcopy(self._policy)

    def _check_identity(self, semantic_fingerprint):
        if semantic_fingerprint != self._semantic_fingerprint:
            raise ValueError("portfolio candidate belongs to a different mathematical instance")
        if forward_semantic_fingerprint(self._prob_data, self._tree) != self._semantic_fingerprint:
            raise ValueError("portfolio mathematical instance changed after initialization")

    @staticmethod
    def _origin(origin):
        if not isinstance(origin, str) or not origin.strip():
            raise ValueError("portfolio origin must be a nonempty strategy label")
        return origin

    @staticmethod
    def _finite_bound(value, label):
        if isinstance(value, bool):
            raise ValueError(f"{label} must be a finite numeric bound")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} must be a finite numeric bound") from exc
        if not math.isfinite(value) or abs(value) >= 5e99:
            raise ValueError(f"{label} is nonfinite or a solver infinity sentinel")
        return value

    def offer_master(self, report, *, origin, semantic_fingerprint, source):
        """Accept the actual ``refresh_stage1_bound`` result, never an RMP LB.

        The free-z Stage-1 builder and its solver-bound extractor establish the
        global scope. This method only verifies that their report is usable and
        consistent with the independently checked incumbent. A certification
        flag by itself is not a proof. No arbitrary scalar-LB setter is exposed.
        """
        self._check_identity(semantic_fingerprint)
        origin = self._origin(origin)
        if source != _MASTER_SOURCE:
            raise ValueError("portfolio LB requires a free-investment Stage-1 refresh report")
        if not isinstance(report, Mapping):
            raise ValueError("portfolio master report must be a mapping")
        lower = report.get("lb")
        if report.get("lb_certified") is not True:
            if lower is not None:
                raise ValueError("uncertified master report contains a claimed lower bound")
            return False
        lower = self._finite_bound(lower, "master lower bound")
        status = report.get("status")
        if (isinstance(status, bool) or not isinstance(status, Integral)
                or status not in _USABLE_MASTER_STATUSES):
            raise ValueError("portfolio master status cannot support a certified lower bound")
        if self._ub is not None and minimization_bounds_inverted(lower, self._ub):
            raise ValueError("portfolio global lower bound exceeds a certified complete-policy UB")
        improved = self._lb is None or lower > self._lb
        if improved:
            self._lb, self._lb_origin = lower, origin
        self._master_offers += 1
        return improved

    def offer_policy(self, policy, *, origin, semantic_fingerprint):
        """Recheck every physical decision and recompute the complete cost."""
        self._check_identity(semantic_fingerprint)
        origin = self._origin(origin)
        from .routeopt.restricted_master import certify_complete_policy

        normalized, upper = certify_complete_policy(
            self._prob_data, self._tree, deepcopy(policy),
        )
        upper = self._finite_bound(upper, "complete-policy upper bound")
        if self._lb is not None and minimization_bounds_inverted(self._lb, upper):
            raise ValueError("portfolio certified complete-policy UB contradicts its global LB")
        improved = self._ub is None or upper < self._ub
        if improved:
            self._policy = deepcopy(normalized)
            self._ub, self._ub_origin = upper, origin
        self._policy_offers += 1
        return improved

    def report(self):
        """JSON-safe scalar report; no policy, cuts or fixed-fleet bounds."""
        self._check_identity(self._semantic_fingerprint)
        gap = (minimization_gap_percent(self._lb, self._ub)
               if self._lb is not None and self._ub is not None else math.inf)
        return dict(
            semantic_model_fingerprint=self._semantic_fingerprint,
            lb=self._lb, ub=self._ub,
            gap_percent=gap if math.isfinite(gap) else None,
            lb_origin=self._lb_origin, ub_origin=self._ub_origin,
            master_source=_MASTER_SOURCE if self._lb is not None else None,
            master_offers=self._master_offers, policy_offers=self._policy_offers,
        )


__all__ = ["PortfolioBounds"]
