"""Directed-rounding accumulator for feasible forward-policy costs."""

from __future__ import annotations

import math
import sys
from fractions import Fraction


def _finite_binary64(value, *, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite binary64 value, got {value!r}")
    return converted


def _finite_nonnegative_binary64(value, *, name: str) -> float:
    converted = _finite_binary64(value, name=name)
    if converted < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
    return converted


def round_fraction_up(value: Fraction) -> float:
    """Return the least binary64 value known to be at least ``value``."""
    try:
        rounded = float(value)
    except OverflowError:
        return math.inf if value > 0 else -sys.float_info.max  # directed overflow

    if not math.isfinite(rounded):
        return rounded if value > 0 else -sys.float_info.max
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    return rounded


class ForwardUBAccumulator(list):
    """Lossless exact sum of binary64 inputs with a float upper enclosure.

    The one-element list interface preserves the existing internal forward
    solver contract.  Its exact state is the rational value represented by the
    supplied binary64 first-stage cost plus every binary64
    ``multi_coeff * stage_cost`` product.  Index 0 is refreshed with directed
    rounding toward ``+inf`` after every addition.
    """

    def __init__(self, initial_cost):
        initial = _finite_nonnegative_binary64(
            initial_cost, name="forward initial cost"
        )
        self._exact_total = Fraction.from_float(initial)
        super().__init__([round_fraction_up(self._exact_total)])

    def add_weighted(self, multi_coeff, stage_cost) -> float:
        weight = _finite_nonnegative_binary64(
            multi_coeff, name="forward multi_coeff"
        )
        cost = _finite_nonnegative_binary64(
            stage_cost, name="forward stage_cost"
        )
        self._exact_total += Fraction.from_float(weight) * Fraction.from_float(cost)
        upper = round_fraction_up(self._exact_total)
        list.__setitem__(self, 0, upper)
        return upper

    def upper_bound(self) -> float:
        return self[0]


def add_weighted_forward_cost(total_cost, multi_coeff, stage_cost) -> float:
    """Add one weighted stage cost without permitting downward rounding."""
    if isinstance(total_cost, ForwardUBAccumulator):
        return total_cost.add_weighted(multi_coeff, stage_cost)

    # Legacy one-element list total_cost API
    if len(total_cost) != 1:
        raise ValueError("forward total_cost must contain exactly one value")
    current = _finite_nonnegative_binary64(
        total_cost[0], name="forward accumulated cost"
    )
    weight = _finite_nonnegative_binary64(
        multi_coeff, name="forward multi_coeff"
    )
    cost = _finite_nonnegative_binary64(
        stage_cost, name="forward stage_cost"
    )
    exact = (
        Fraction.from_float(current)
        + Fraction.from_float(weight) * Fraction.from_float(cost)
    )
    upper = round_fraction_up(exact)
    total_cost[0] = upper
    return upper


def forward_ub_value(total_cost) -> float:
    """Read the directed-upward float enclosure from an accumulator/list."""
    if isinstance(total_cost, ForwardUBAccumulator):
        return total_cost.upper_bound()
    if len(total_cost) != 1:
        raise ValueError("forward total_cost must contain exactly one value")
    return _finite_nonnegative_binary64(
        total_cost[0], name="forward accumulated cost"
    )
