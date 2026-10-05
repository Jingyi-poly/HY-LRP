"""Exact capacity-aware lower bounds for physical elementary-route pricing.

Every visited customer has one incoming arc. After subtracting its minimum
incoming cost, the positive customer prizes form a fractional knapsack. Its
exact optimum bounds the prize of every capacity-feasible elementary route.
Repeated-customer NG walks are deliberately not part of this certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class CapacityPriceBound:
    demands: tuple[Fraction, ...]
    capacity: Fraction
    incoming: tuple[Fraction, ...]

    @classmethod
    def from_group(cls, demands, group):
        demands = tuple(Fraction(value) for value in demands)
        capacity = Fraction(group["capacity"])
        costs = group["costs"]
        size = len(demands) + 1
        if capacity < 0 or any(value < 0 for value in demands):
            raise ValueError("physical resources must be nonnegative")
        if len(costs) != size or any(len(row) != size for row in costs):
            raise ValueError("route costs must include the depot and every customer")
        incoming = tuple(min(Fraction(costs[i][j]) for i in range(size) if i != j)
                         for j in range(1, size))
        if any(value < 0 for row in costs for value in row):
            raise ValueError("physical route costs must be nonnegative")
        return cls(demands, capacity, incoming)

    def lower_bound(self, u):
        """Bound ``min_route(cost - sum(u))`` using original exact resources.

        Arithmetic and profit/demand comparisons use exact rational values;
        callers round only the final certificate toward minus infinity.
        Customers too large for this vehicle cannot occur in a physical route.
        Zero-demand positive prizes must be included even at zero capacity.
        """
        if len(u) != len(self.demands):
            raise ValueError("customer duals must cover the physical domain")
        free_profit = Fraction()
        items = []
        for price, demand, incoming in zip(u, self.demands, self.incoming):
            profit = Fraction(price) - incoming
            if demand > self.capacity or profit <= 0:
                continue
            if demand == 0:
                free_profit += profit
            else:
                items.append((profit / demand, demand, profit))
        remaining, upper_profit = self.capacity, free_profit
        for ratio, demand, profit in sorted(items, reverse=True):
            if remaining <= 0:
                break
            taken = min(demand, remaining)
            upper_profit += ratio * taken
            remaining -= taken
        return -upper_profit


__all__ = ["CapacityPriceBound"]
