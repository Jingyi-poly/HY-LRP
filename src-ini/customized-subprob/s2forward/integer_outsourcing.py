"""Integer outsourcing floors from an exact aggregate-capacity relaxation.

For anchor fleet F, maximize the number of served packages subject to its
total capacity. This ignores individual vehicle packing, so the remaining
package charge is a lower bound on outsourcing. The conditional model row
keeps this bound valid when linking RHS values change during backward LPs.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math


_MAX_STATES = 4096
_MAX_WORK = 200000


def _represented(value):
    if isinstance(value, (str, bytes)):
        raise ValueError("expected numeric model data")
    number = float(value)
    if not math.isfinite(number) or number < 0 or Fraction(value) != Fraction(number):
        raise ValueError("expected nonnegative binary64 model data")
    return Fraction(number)


@lru_cache(maxsize=16)
def _minimum_loads(items):
    # Binary64 demands are dyadic: use a common exact integer unit instead
    # of repeatedly constructing Fractions in the knapsack recurrence.
    ratios = [demand.as_integer_ratio() for _, demand in items]
    denominator = max((d for _, d in ratios), default=1)
    total = sum(count for count, _ in items)
    loads = [None] * (total + 1)
    loads[0] = 0
    processed = 0
    for (count, _), (numerator, divisor) in zip(items, ratios):
        weight = numerator * (denominator // divisor)
        for previous in range(processed, -1, -1):
            if loads[previous] is not None:
                target, value = previous + count, loads[previous] + weight
                if loads[target] is None or value < loads[target]:
                    loads[target] = value
        processed += count
    return tuple(loads), denominator


@dataclass(frozen=True)
class IntegerOutsourcingCertificate:
    """Exact package data and a proved aggregate-capacity outsourcing floor."""

    jobs: tuple
    counts: tuple
    demands: tuple
    charges: tuple
    unit: Fraction
    flags: tuple
    capacities: tuple
    lower_packages: int


def integer_outsourcing_certificate(prob_data, node, flags):
    """Return the exact profile and integer floor, or None if unsupported.

    Original integer package counts and an exactly common per-package charge
    are required. Missing metadata, fractional fleet bits, large profiles or
    unsupported charges leave the ordinary model unchanged. Active customers
    with zero packages remain aligned with route-pricing customer indices.
    """
    try:
        if (set(flags) != set(prob_data.V)
                or len(set(prob_data.V)) != len(prob_data.V)
                or len(set(prob_data.J)) != len(prob_data.J)):
            return None
        bits = {v: _represented(flags[v]) for v in prob_data.V}
        if any(bit not in (0, 1) for bit in bits.values()):
            return None
        capacities = tuple((v, _represented(prob_data.Qv[v])) for v in prob_data.V)
        capacity = sum((value * bits[v] for v, value in capacities), Fraction())
        items, unit = [], None
        jobs, counts, demands, charges = [], [], [], []
        for j in prob_data.J:
            active = _represented(node.active[j])
            charge = _represented(node.c_out[j])
            demand = _represented(node.volume[j])
            count = _represented(node.n[j])
            if active not in (0, 1) or count.denominator != 1:
                return None
            if not active or not count:
                if charge:
                    return None
                if not active:
                    continue
            else:
                ratio = charge / count
                if unit is None:
                    unit = ratio
                elif unit != ratio:
                    return None
                items.append((int(count), float(demand)))
            jobs.append(j)
            counts.append(int(count))
            demands.append(demand)
            charges.append(charge)
        unit = Fraction() if unit is None else unit
        total = sum(counts)
        if total + 1 > _MAX_STATES or len(items) * (total + 1) > _MAX_WORK:
            return None
        loads, denominator = _minimum_loads(tuple(sorted(items)))
        served = max(p for p, load in enumerate(loads)
                     if load is not None
                     and load * capacity.denominator <= capacity.numerator * denominator)
        return IntegerOutsourcingCertificate(
            tuple(jobs), tuple(counts), tuple(demands), tuple(charges), unit,
            tuple((v, int(bits[v])) for v in prob_data.V), capacities, total - served,
        )
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None


def integer_outsourcing_floor(prob_data, node, flags):
    """Return a downward-rounded cost floor from the shared exact profile."""
    certificate = integer_outsourcing_certificate(prob_data, node, flags)
    if certificate is None:
        return None
    try:
        exact = certificate.unit * certificate.lower_packages
        lower = float(exact)
        if not math.isfinite(lower):
            return None
        if Fraction(lower) > exact:
            lower = math.nextafter(lower, -math.inf)
        return lower
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None


def add_integer_outsourcing_floor(model, prob_data, node, x_prev, z, stage_cost):
    """Add an all-binary-fleet-valid row, not a trial-only constant.

    At the anchor this is stage_cost >= L. For any purchased fleet outside
    the anchor, its RHS is nonpositive. Thus backward LP linking duals keep
    the necessary z coefficients when they produce cuts for other fleets.
    """
    period = int(node.info[1])
    flags = {v: x_prev.get(f"z[{v},{period}]") for v in prob_data.V}
    lower = integer_outsourcing_floor(prob_data, node, flags)
    if lower is None or lower <= 0.0:
        return None
    outside = [v for v in prob_data.V if flags[v] == 0]
    return model.addConstr(
        stage_cost + lower * sum(z[v] for v in outside) >= lower,
        name="integer_outsourcing_floor",
    )


__all__ = [
    "IntegerOutsourcingCertificate", "integer_outsourcing_certificate",
    "integer_outsourcing_floor", "add_integer_outsourcing_floor",
]
