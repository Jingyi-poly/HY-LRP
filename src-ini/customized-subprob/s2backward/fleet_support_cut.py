"""Affine supports of certified fixed-fleet lower bounds, without piece solves.

Every canonical binary fleet is a vertex, so its tabulated lower bound admits
an affine support over any specified subset of the canonical fleet domain.
A tiny LP chooses small slopes; exact binary64-rational enumeration certifies
the returned row independently of LP feasibility/optimality tolerances.

``allowed_counts`` is a validity domain, not a heuristic sample.  A caller
restricting it must enforce that same domain in the receiving master.  No
claim of optimal recourse, a closed oracle, or convergence follows from a
support touching an inexact tabulated lower bound.
"""

from __future__ import annotations

import math
import time
from fractions import Fraction
from typing import Mapping

from s2forward.fleet_pieces import float_down, float_up


def _minimum_l1_slopes(rows, trial, target, seconds):
    import gurobipy as gp
    from gurobipy import GRB

    if seconds <= 0.0:
        return None
    model = gp.Model("fleet_lower_support")
    try:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = seconds
        model.Params.FeasibilityTol = 1e-9
        slope = model.addVars(len(trial), lb=-GRB.INFINITY)
        magnitude = model.addVars(len(trial), lb=0.0)
        for index in range(len(trial)):
            model.addConstr(slope[index] <= magnitude[index])
            model.addConstr(-slope[index] <= magnitude[index])
        for flags, bound in rows:
            model.addConstr(
                gp.quicksum(
                    (flag - anchor) * slope[index]
                    for index, (flag, anchor) in enumerate(zip(flags, trial))
                    if flag != anchor
                ) <= float(bound - target)
            )
        model.setObjective(magnitude.sum(), GRB.MINIMIZE)
        model.optimize()
        if model.SolCount <= 0:
            return None
        result = tuple(float(slope[index].X) for index in range(len(trial)))
        return result if all(map(math.isfinite, result)) else None
    finally:
        model.dispose()


def _certify(slopes, rows, trial, target):
    rational = tuple(Fraction(value) for value in slopes)
    intercept = float_down(min(
        bound - sum((p * z for p, z in zip(rational, flags)), Fraction(0))
        for flags, bound in rows
    ))
    intercept_q = Fraction(intercept)
    # Check the actual returned binary64 coefficients, not the LP solution.
    if any(
        intercept_q + sum((p * z for p, z in zip(rational, flags)), Fraction(0)) > bound
        for flags, bound in rows
    ):
        return None
    value = intercept_q + sum(
        (p * z for p, z in zip(rational, trial)), Fraction(0)
    )
    return intercept, value, target - value


def build_fleet_support_cut(
    table,
    trial_z: Mapping[str, float],
    period: int,
    *,
    allowed_counts=None,
    time_limit: float = 0.2,
    max_trial_loss: float = 1e-6,
):
    """Return a domain-certified ``eta >= intercept + pi_value * z`` or None.

    Requires exact 0/1 trial flags (no rounding), a canonical purchase prefix,
    and finite certified table bounds at every allowed fleet.  Missing trial
    keys, fractional/noncanonical trials, malformed domains and invalid
    bounds fail closed.  A supplied domain must include the trial.

    The LP only chooses the row's shape.  It never solves a fixed-fleet MIP.
    If unavailable or numerically unsuitable, a finite Hamming support has
    the same validity guarantee.  ``time_limit=0`` selects that algebraic
    construction directly.  Fraction certification is never time-limited.
    """
    started = time.monotonic()
    try:
        if int(period) != period or period < 0:
            return None
        period = int(period)
        seconds = float(time_limit)
        tolerance = float(max_trial_loss)
        if not all(map(math.isfinite, (seconds, tolerance))) or min(seconds, tolerance) < 0:
            return None
        layout = table.layout
        keys = tuple(f"z[{vehicle},{period}]" for vehicle in layout.vehicles)
        trial = tuple(float(trial_z[key]) for key in keys)
        if any(value not in (0.0, 1.0) for value in trial):
            return None
        flags = dict(zip(layout.vehicles, trial))
        if not layout.is_leading_run(flags):
            return None
        trial_counts = layout.counts_from_vehicle_flags(flags)
        expected = tuple(layout.all_counts())
        if set(table.records) != set(expected):
            return None
        if any(not math.isfinite(float(record.lb)) for record in table.records.values()):
            return None
        if allowed_counts is None:
            domain = expected
        else:
            checked = set()
            for counts in allowed_counts:
                counts = tuple(counts)
                validated = layout.check_counts(counts)
                if counts != validated:
                    return None
                checked.add(validated)
            domain = tuple(counts for counts in expected if counts in checked)
        if not domain or trial_counts not in domain:
            return None
        bounds = table.closed_lower_bounds()
        rows = tuple(
            (
                tuple(layout.flags_for_counts(counts)[v] for v in layout.vehicles),
                Fraction(float(bounds[counts])),
            )
            for counts in domain
        )
        target = Fraction(float(bounds[trial_counts]))
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        return None

    try:
        slopes = _minimum_l1_slopes(rows, trial, target, seconds)
    except (ImportError, RuntimeError, ValueError, OverflowError):
        slopes = None
    # Gurobi errors do not compromise the algebraic fallback.
    except Exception as exc:
        if exc.__class__.__module__.split(".")[0] != "gurobipy":
            raise
        slopes = None
    try:
        certificate = _certify(slopes, rows, trial, target) if slopes is not None else None
        method = "minimum_l1"
        if certificate is None or certificate[2] > Fraction(tolerance):
            # h(z) = LB(trial) - M * Hamming(z, trial).  Distinct binary
            # vertices have distance >= 1, so finite M always exists.
            penalty = max((
                (target - bound) / sum(left != right for left, right in zip(flags, trial))
                for flags, bound in rows if flags != trial
            ), default=Fraction(0))
            magnitude = float_up(max(Fraction(0), penalty))
            slopes = tuple(magnitude if flag else -magnitude for flag in trial)
            certificate = _certify(slopes, rows, trial, target)
            method = "hamming"
        if certificate is None or not 0 <= certificate[2] <= Fraction(tolerance):
            return None
        intercept, value, loss = certificate
        return {
            "pi_value": dict(zip(keys, slopes)),
            "intercept": intercept,
            "trial_lb": float(target),
            "trial_value": float_down(value),
            "rounding_loss": float_up(loss),
            "certified": True,
            "domain": domain,
            "trial_counts": trial_counts,
            "method": method,
            "n_pieces": len(domain),
            "seconds": time.monotonic() - started,
        }
    except (TypeError, ValueError, OverflowError):
        return None


__all__ = ["build_fleet_support_cut"]
