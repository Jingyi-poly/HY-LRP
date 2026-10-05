"""Exact integer arithmetic for a fleet-piece objective at one multiplier.

Every value entering a Stage-2 Level-Set oracle is a finite binary64 number,
so it is an integer divided by a power of two.  Rescaling all fixed-fleet
lower bounds and multipliers to their common power-of-two denominator turns

    C_lb(n) - sum_{v in prefix(n)} pi_v

into plain Python integer arithmetic.  Comparisons remain exact while a
large fleet grid no longer constructs thousands of ``Fraction`` objects on
every refinement round.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Mapping

from s2forward.fleet_pieces import Counts, FleetLayout, fraction_from_float


class Binary64PieceValues:
    """Exact scaled values and credits for one ``(lower_bounds, pi)`` pair."""

    def __init__(
        self,
        layout: FleetLayout,
        lower_bounds: Mapping[Counts, float],
        pi: Mapping[int, Fraction],
    ):
        lb_fractions = {
            counts: fraction_from_float(value)
            for counts, value in lower_bounds.items()
        }
        pi_fractions = {vehicle: Fraction(value) for vehicle, value in pi.items()}
        values = tuple(lb_fractions.values()) + tuple(pi_fractions.values())
        if not values:
            raise ValueError("fleet-piece objective cannot be empty")

        scale_bits = 0
        for value in values:
            denominator = value.denominator
            if denominator & (denominator - 1):
                raise ValueError("fleet-piece inputs must be binary64 rationals")
            scale_bits = max(scale_bits, denominator.bit_length() - 1)
        self.denominator = 1 << scale_bits

        def scaled(value: Fraction) -> int:
            multiplier, remainder = divmod(self.denominator, value.denominator)
            if remainder:
                raise ValueError("fleet-piece inputs do not share a dyadic lattice")
            return value.numerator * multiplier

        pi_scaled = {
            vehicle: scaled(value) for vehicle, value in pi_fractions.items()
        }
        group_prefixes = []
        for group in layout.groups:
            running = 0
            prefixes = [running]
            for vehicle in group:
                running += pi_scaled[vehicle]
                prefixes.append(running)
            group_prefixes.append(tuple(prefixes))

        self.credits: Dict[Counts, int] = {}
        self.values: Dict[Counts, int] = {}
        for counts, lower_bound in lb_fractions.items():
            credit = sum(
                prefixes[count]
                for prefixes, count in zip(group_prefixes, counts)
            )
            self.credits[counts] = credit
            self.values[counts] = scaled(lower_bound) - credit

    def as_fraction(self, scaled_value: int) -> Fraction:
        return Fraction(int(scaled_value), self.denominator)

    def credit_fraction(self, counts: Counts) -> Fraction:
        return self.as_fraction(self.credits[counts])

    def tolerance_units(self, tolerance: Fraction) -> int:
        """Largest lattice delta not exceeding a nonnegative tolerance."""
        tolerance = Fraction(tolerance)
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        return (tolerance.numerator * self.denominator) // tolerance.denominator


__all__ = ["Binary64PieceValues"]
