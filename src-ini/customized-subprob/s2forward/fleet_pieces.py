"""Fleet pieces of the Stage-2 Lagrangian dual function.

A *piece* is a count vector ``n = (n_k)_k`` with ``0 <= n_k <= |V_k|``: the
purchased fleet made of the leading ``n_k`` vehicles of every type ``k`` in
``prob_data.V_k[k]`` order.  It is exactly one feasible point of the shared
``purchase_order`` domain, not an equivalence class of arbitrary vehicle
subsets.  ``C(n)`` is the optimal Stage-2 value with that purchased prefix.

The Lagrangian oracle therefore ranges only over canonical purchase prefixes:

    D(pi) = min_n [ C(n) - credit(n, pi) ],
    credit(n, pi) = sum_{v in prefix(n)} pi_v.

For a stored policy using prefix ``r``, its best compatible purchase decision
is the prefix ``n >= r`` with largest prefix sum of ``pi``.  Thus both the
outer piece value and the inner feasible support use the same exact domain.

All arithmetic here is exact (``Fraction``); callers round once, in the safe
direction, when they need a binary64 value.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Counts = Tuple[int, ...]


def fraction_from_float(value) -> Fraction:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"non-finite value {value!r}")
    return Fraction(scalar)


def float_down(value: Fraction) -> float:
    """Largest binary64 that is <= ``value``."""
    approx = float(value)
    if not math.isfinite(approx):
        raise OverflowError("value not representable")
    if Fraction(approx) > value:
        approx = math.nextafter(approx, -math.inf)
    return approx


def float_up(value: Fraction) -> float:
    """Smallest binary64 that is >= ``value``."""
    approx = float(value)
    if not math.isfinite(approx):
        raise OverflowError("value not representable")
    if Fraction(approx) < value:
        approx = math.nextafter(approx, math.inf)
    return approx


class FleetLayout:
    """Vehicle types in canonical assignment and purchase order."""

    def __init__(self, prob_data):
        self.types: List[int] = list(prob_data.K)
        self.groups: List[List[int]] = [list(prob_data.V_k[k]) for k in self.types]
        self.vehicles: List[int] = list(prob_data.V)
        flat = [v for group in self.groups for v in group]
        if sorted(flat) != sorted(self.vehicles) or len(set(flat)) != len(flat):
            raise ValueError("prob_data.V_k must partition prob_data.V")
        self._position: Dict[int, Tuple[int, int]] = {
            v: (g, i) for g, group in enumerate(self.groups) for i, v in enumerate(group)
        }
        self.full: Counts = tuple(len(group) for group in self.groups)

    # ------------------------------------------------------------------ pieces
    def all_counts(self) -> Iterable[Counts]:
        return itertools.product(*(range(len(group) + 1) for group in self.groups))

    @property
    def n_pieces(self) -> int:
        out = 1
        for group in self.groups:
            out *= len(group) + 1
        return out

    def check_counts(self, counts: Sequence[int]) -> Counts:
        counts = tuple(int(c) for c in counts)
        if len(counts) != len(self.groups) or any(
            c < 0 or c > len(group) for c, group in zip(counts, self.groups)
        ):
            raise ValueError(f"invalid fleet counts {counts!r} for {self.full!r}")
        return counts

    def prefix_vehicles(self, counts: Sequence[int]) -> List[int]:
        counts = self.check_counts(counts)
        return [v for c, group in zip(counts, self.groups) for v in group[:c]]

    def counts_from_vehicle_flags(self, flags: Mapping[int, object]) -> Counts:
        """Leading-run counts of the vehicle set ``{v: flags[v] truthy}``."""
        out = []
        for group in self.groups:
            run = 0
            for v in group:
                if self._is_on(flags.get(v, 0)):
                    run += 1
                else:
                    break
            out.append(run)
        return tuple(out)

    def counts_from_z(self, z_by_key: Mapping[str, object], period: int) -> Counts:
        return self.counts_from_vehicle_flags(
            {v: z_by_key.get(f"z[{v},{period}]", 0.0) for v in self.vehicles}
        )

    def is_leading_run(self, flags: Mapping[int, object]) -> bool:
        """True iff the flagged vehicles form a leading run in every type."""
        for group in self.groups:
            seen_off = False
            for v in group:
                on = self._is_on(flags.get(v, 0))
                if on and seen_off:
                    return False
                if not on:
                    seen_off = True
        return True

    @staticmethod
    def _is_on(value) -> bool:
        return float(value) >= 0.5

    @staticmethod
    def dominates(larger: Sequence[int], smaller: Sequence[int]) -> bool:
        """``larger >= smaller`` componentwise (a superset fleet)."""
        return all(a >= b for a, b in zip(larger, smaller))

    # ------------------------------------------------------------- Lagrangian
    def pi_by_vehicle(self, pi_value: Mapping[str, object], period: int) -> Dict[int, Fraction]:
        return {
            v: fraction_from_float(pi_value.get(f"z[{v},{period}]", 0.0))
            for v in self.vehicles
        }

    def prefix_credit_tables(
        self, pi: Mapping[int, Fraction]
    ) -> Tuple[Tuple[Fraction, ...], ...]:
        """Exact credit of every within-type purchase prefix.

        A Level-Set query evaluates many fleet pieces at one fixed ``pi``.
        Building these short tables once avoids summing the same vehicle
        multipliers again for every piece and every refinement round.
        """
        tables = []
        for group in self.groups:
            running = Fraction(0)
            credits = [running]
            for vehicle in group:
                running += pi[vehicle]
                credits.append(running)
            tables.append(tuple(credits))
        return tuple(tables)

    def lagrangian_credit(
        self,
        counts: Sequence[int],
        pi: Mapping[int, Fraction],
        prefix_credits: Optional[Sequence[Sequence[Fraction]]] = None,
    ) -> Fraction:
        """Exact ``sum(pi[v] for v in purchased prefix(counts))``."""
        counts = self.check_counts(counts)
        if prefix_credits is not None:
            if len(prefix_credits) != len(self.groups):
                raise ValueError("prefix credit tables do not match vehicle types")
            return sum(
                (credits[count] for count, credits in zip(counts, prefix_credits)),
                Fraction(0),
            )
        return sum(
            (pi[v] for c, group in zip(counts, self.groups) for v in group[:c]),
            Fraction(0),
        )

    def piece_value(self, lower_bound, counts: Sequence[int], pi: Mapping[int, Fraction]) -> Fraction:
        return fraction_from_float(lower_bound) - self.lagrangian_credit(counts, pi)

    def best_extension_counts(
        self,
        required: Sequence[int],
        prefix_credits: Sequence[Sequence[Fraction]],
    ) -> Counts:
        """Best purchased prefix containing the required leading runs."""
        required = self.check_counts(required)
        if len(prefix_credits) != len(self.groups):
            raise ValueError("prefix credit tables do not match vehicle types")
        chosen = []
        for minimum, credits in zip(required, prefix_credits):
            best_count = minimum
            best_credit = credits[minimum]
            for count in range(minimum + 1, len(credits)):
                if credits[count] > best_credit:
                    best_count = count
                    best_credit = credits[count]
            chosen.append(best_count)
        return tuple(chosen)

    def flags_for_counts(self, counts: Sequence[int]) -> Dict[int, int]:
        prefix = set(self.prefix_vehicles(counts))
        return {vehicle: int(vehicle in prefix) for vehicle in self.vehicles}

    def best_z_for_policy(self, used: Mapping[int, object], pi: Mapping[int, Fraction]) -> Dict[int, int]:
        """Best purchased prefix containing every vehicle used by ``policy``."""
        if not self.is_leading_run(used):
            raise ValueError("used vehicles must form a canonical leading run")
        required = self.counts_from_vehicle_flags(used)
        chosen = self.best_extension_counts(required, self.prefix_credit_tables(pi))
        return self.flags_for_counts(chosen)

    def brute_force_dual(self, cost_of_counts: Mapping[Counts, object], pi: Mapping[int, Fraction]) -> Fraction:
        """Reference minimum over all purchase-order-feasible binary vectors."""
        best = None
        for bits in itertools.product((0, 1), repeat=len(self.vehicles)):
            flags = dict(zip(self.vehicles, bits))
            if not self.is_leading_run(flags):
                continue
            counts = self.counts_from_vehicle_flags(flags)
            value = fraction_from_float(cost_of_counts[counts]) - sum(
                (pi[v] for v, on in flags.items() if on), Fraction(0)
            )
            if best is None or value < best:
                best = value
        return best
