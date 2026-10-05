"""Bind ordinary pricing certificates to their complete physical route domain.

Identity includes active customer IDs, original binary64 demands and outsourcing
charges, and every vehicle group's capacity and full active-route cost matrix.
Unavailable vehicles remain in the identity: their coefficients may matter away
from the certificate's anchor. Period, node ID, probability/operating weight and
anchor availability are deliberately excluded and checked separately by users
of the certificate.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import math
from numbers import Integral, Real


def _identifier(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("physical domain IDs must be integers")
    return int(value)


def _represented(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("physical data must be real binary64 values")
    number = float(value)
    if not math.isfinite(number) or number < 0 or Fraction(value) != Fraction(number):
        raise ValueError("physical data must be finite nonnegative binary64 values")
    # Signed zero does not change the represented route problem.
    return (0.0 if number == 0.0 else number).hex()


def physical_fingerprint_from_groups(jobs, demands, outsourcing, groups):
    """Hash `_physical_groups` output, without its anchor availability counts."""
    jobs = tuple(_identifier(j) for j in jobs)
    demands, outsourcing = tuple(demands), tuple(outsourcing)
    if len(set(jobs)) != len(jobs) or len(demands) != len(jobs) or len(outsourcing) != len(jobs):
        raise ValueError("physical customer domain is incomplete or duplicated")
    domain_groups, seen = [], set()
    for group in groups:
        vehicles = tuple(_identifier(v) for v in group["vehicles"])
        if not vehicles or len(set(vehicles)) != len(vehicles) or seen.intersection(vehicles):
            raise ValueError("physical vehicle groups must partition the vehicle IDs")
        seen.update(vehicles)
        matrix = tuple(tuple(_represented(value) for value in row) for row in group["costs"])
        size = len(jobs) + 1
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError("physical cost matrix does not cover the active route domain")
        domain_groups.append((vehicles, _represented(group["capacity"]), matrix))
    identity = (
        "physical_route_domain_v1", jobs,
        tuple(_represented(value) for value in demands),
        tuple(_represented(value) for value in outsourcing), tuple(domain_groups),
    )
    return hashlib.sha256(repr(identity).encode("ascii")).hexdigest()


def physical_fingerprint(pd, node):
    """Rebuild the original physical domain with all vehicle types retained."""
    from .root_lp import _physical_groups

    data = _physical_groups(pd, node, {v: 0 for v in pd.V})
    return physical_fingerprint_from_groups(*data)


__all__ = ["physical_fingerprint", "physical_fingerprint_from_groups"]
