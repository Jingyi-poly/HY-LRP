"""Necessary Stage-1 budget domain for a free-fleet Stage-2 oracle.

With zero initial fleet, ``z[v,t] = sum(a[v,tau], tau <= t)`` implies
``sum(cost[v] * z[v,t]) <= sum(B[tau], tau <= t)``.  This is only a
necessary condition: the helper does not claim that every retained fleet
can be scheduled within the individual period budgets.

All comparisons use exact rational representations of the binary64 inputs.
The optional per-period slack and the final upward-rounded RHS only enlarge
the domain.  Nonstandard or incomplete budget data disable the restriction.
No solver is imported, built, or configured by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral


def _nonnegative_binary64(value):
    if isinstance(value, (bool, str, bytes)):
        raise ValueError("expected a nonnegative numeric scalar")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError("expected a finite nonnegative scalar")
    return number, Fraction.from_float(number)


def _round_up(value: Fraction) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("cumulative budget is not finite binary64")
    if Fraction.from_float(number) < value:
        number = math.nextafter(number, math.inf)
    if not math.isfinite(number):
        raise ValueError("cumulative budget upper endpoint overflowed")
    return number


@dataclass(frozen=True)
class BudgetFleetDomain:
    applicable: bool
    reason: str
    allowed_counts: tuple
    fingerprint: tuple
    cumulative_budget_upper: float | None
    purchase_costs: tuple
    groups: tuple
    all_counts: tuple

    @property
    def restricted(self) -> bool:
        return len(self.allowed_counts) < len(self.all_counts)

    def contains(self, counts) -> bool:
        # Do not silently convert a fractional count to an integer prefix.
        values = tuple(counts)
        if any(not isinstance(value, Integral) or isinstance(value, bool)
               for value in values):
            return False
        return values in self.allowed_counts

    def best_extension_counts(self, required, prefix_credits):
        """Best affordable prefix containing ``required``, or ``None``.

        Budget couples the vehicle types, so independent per-type maximisers
        are no longer valid.  Credits should be the exact Fraction tables
        returned by ``FleetLayout.prefix_credit_tables``.
        """
        required = tuple(required)
        if len(required) != len(self.groups) or any(
            not isinstance(value, Integral) or isinstance(value, bool)
            or value < 0 or value > len(group)
            for value, group in zip(required, self.groups)
        ):
            raise ValueError("invalid required canonical fleet")
        if len(prefix_credits) != len(self.groups) or any(
            len(credits) != len(group) + 1
            for credits, group in zip(prefix_credits, self.groups)
        ):
            raise ValueError("prefix credit tables do not match vehicle groups")
        best, best_credit = None, None
        for counts in self.allowed_counts:
            if any(count < minimum for count, minimum in zip(counts, required)):
                continue
            credit = sum((credits[count] for credits, count
                          in zip(prefix_credits, counts)), Fraction(0))
            if best is None or credit > best_credit:
                best, best_credit = counts, credit
        return best

    def add_gurobi_constraint(self, model, z_by_vehicle,
                             name="stage1_cumulative_budget"):
        """Add the same necessary row to a free-z model, when restrictive.

        A solver incumbent still needs the caller's ordinary binary/physical
        certification and ``contains`` check; solver feasibility tolerance is
        not an exact-membership certificate at a budget boundary.
        """
        if not self.restricted:
            return None
        expression = sum(cost * z_by_vehicle[vehicle]
                         for vehicle, cost in self.purchase_costs)
        return model.addConstr(expression <= self.cumulative_budget_upper,
                               name=name)


def build_budget_fleet_domain(prob_data, period, layout,
                              *, per_period_slack=1e-6) -> BudgetFleetDomain:
    """Build a conservative canonical fleet mask for standard Stage-1 data.

    Caller contract: use only with the repository's zero-initial-fleet,
    time-independent nonnegative purchase costs and cumulative z transition.
    Missing data, an invalid period, negative/nonfinite numbers, or non-finite
    cumulative arithmetic fail open.  Invalid ``layout`` is a caller error.

    The fingerprint encodes the *binary domain*, not period or budget RHS.
    Thus later periods with the same allowed fleets can still be deduplicated.
    Existing physical fixed-fleet tables must remain unfiltered; apply this
    mask only to the outer minimum and the feasible inner extensions.
    """
    groups = tuple(tuple(group) for group in layout.groups)
    all_counts = tuple(tuple(counts) for counts in layout.all_counts())

    def result(applicable, reason, allowed, upper=None, costs=()):
        return BudgetFleetDomain(
            applicable, reason, allowed,
            ("stage1_budget_fleet_v1", groups, allowed),
            upper, costs, groups, all_counts,
        )

    try:
        periods = prob_data.T
        if (not isinstance(period, Integral) or isinstance(period, bool)
                or not isinstance(periods, Integral) or isinstance(periods, bool)
                or period < 0 or period >= periods):
            return result(False, "invalid_period", all_counts)
        budget_data = prob_data.B_t0
        if len(budget_data) != periods:
            return result(False, "nonstandard_budget_horizon", all_counts)
        costs, exact_costs = [], {}
        for vehicle in layout.vehicles:
            number, exact = _nonnegative_binary64(prob_data.cost_purchase[vehicle])
            costs.append((vehicle, number))
            exact_costs[vehicle] = exact
        budgets = [_nonnegative_binary64(budget_data[index])[1]
                   for index in range(period + 1)]
        slack = _nonnegative_binary64(per_period_slack)[1]
        upper = _round_up(sum(budgets, Fraction(0)) + (period + 1) * slack)
        exact_upper = Fraction.from_float(upper)
        allowed = tuple(counts for counts in all_counts if sum(
            (exact_costs[vehicle]
             for count, group in zip(counts, groups) for vehicle in group[:count]),
            Fraction(0),
        ) <= exact_upper)
        return result(True, "cumulative_budget", allowed, upper, tuple(costs))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return result(False, "nonstandard_budget_data", all_counts)


__all__ = ["BudgetFleetDomain", "build_budget_fleet_domain"]
