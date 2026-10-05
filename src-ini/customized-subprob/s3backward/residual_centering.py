"""Conservative cost and cut translations for residual Stage-3 oracles.

Subtract each active customer's minimum incoming cost before solving its
residual routing problem. Adding that cost back to the returned alpha slope
gives a valid cut for the original problem. All conversions preserve the
required inequality over the represented binary64 inputs.

This module owns no solver, bundle archive, installation hook, or configuration.
The caller must keep oracle memo and bundle supports in residual coordinates.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from fractions import Fraction
import math

import numpy as np

__all__ = [
    "round_fraction",
    "prepare_residual",
    "residual_target",
    "lift_residual_cut",
    "original_trial_closed",
]

_CACHE = OrderedDict()
_CACHE_LIMIT = 64


def _fraction(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("residual centering requires finite binary64 inputs") from exc
    if not math.isfinite(number):
        raise ValueError("residual centering requires finite binary64 inputs")
    return Fraction.from_float(number)


def round_fraction(value, *, upward):
    """Return the smallest upward / largest downward binary64 enclosure."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("residual conversion exceeds binary64") from exc
    if not math.isfinite(result):
        raise ValueError("residual conversion exceeds binary64")
    represented = Fraction.from_float(result)
    if upward and represented < value:
        result = math.nextafter(result, math.inf)
    elif not upward and represented > value:
        result = math.nextafter(result, -math.inf)
    if not math.isfinite(result):
        raise ValueError("residual conversion exceeds binary64")
    return result


def prepare_residual(prob_data, node):
    """Copy only this vehicle's matrix, using exact active incoming minima.

    Every changed arc is rounded downward after subtracting its head's minimum
    incoming cost. The cached matrix is immutable; the problem and cost-map
    containers returned on every call are new shallow copies of current data.
    """
    vehicle = node.info
    matrix = np.asarray(prob_data.c_routing[vehicle], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("residual routing costs must be a square matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("residual routing costs must be finite and nonnegative")
    customers = tuple(prob_data.J)
    nodes = tuple(prob_data.N)
    active = tuple(j for j in customers if node.active[j] == 1)
    key = (vehicle, customers, nodes, active, matrix.shape, matrix.tobytes())
    hit = _CACHE.get(key)
    if hit is None:
        residual = matrix.copy()
        ell = {}
        for j in active:
            incoming = [float(matrix[i, j]) for i in nodes if i != j]
            if not incoming:
                raise ValueError("active customer has no incoming arc")
            ell[j] = min(incoming)
            exact_ell = _fraction(ell[j])
            for i in nodes:
                if i == j:
                    continue
                exact = _fraction(matrix[i, j]) - exact_ell
                residual[i, j] = round_fraction(exact, upward=False)
        residual.flags.writeable = False
        hit = (residual, ell)
        _CACHE[key] = hit
        if len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(key)
    residual, ell = hit
    clone = copy.copy(prob_data)
    clone.c_routing = dict(prob_data.c_routing)
    clone.c_routing[vehicle] = residual
    return clone, dict(ell)


def residual_target(alpha_level, ell, vehicle, x_prev):
    """Translate a feasible original route UB upward into residual costs."""
    removed = Fraction(0)
    for customer, value in ell.items():
        shift = _fraction(value)
        if shift < 0:
            raise ValueError("residual incoming shifts must be nonnegative")
        bit = float((x_prev or {}).get(f"alpha[{customer},{vehicle}]", 0.0))
        if bit not in (0.0, 1.0):
            raise ValueError("residual trial assignment must be exactly binary")
        removed += shift * int(bit)
    target = _fraction(alpha_level) - removed
    if target < 0:
        raise ValueError("forward route upper bound is below its incoming-arc lower bound")
    return round_fraction(target, upward=True)


def lift_residual_cut(pi, intercept, ell, vehicle):
    """Add incoming costs to slopes and pay positive rounding error in b.

    For every binary assignment, the returned represented affine expression
    is at most the exact translated source cut. Neither input mapping changes.
    A missing certified intercept remains missing.
    """
    lifted = {name: float(_fraction(value)) for name, value in pi.items()}
    error = Fraction(0)
    for customer, value in ell.items():
        shift = _fraction(value)
        if shift < 0:
            raise ValueError("residual incoming shifts must be nonnegative")
        name = f"alpha[{customer},{vehicle}]"
        exact = _fraction(lifted.get(name, 0.0)) + shift
        try:
            nearest = float(exact)
        except OverflowError as exc:
            raise ValueError("lifted Stage-3 slope exceeds binary64") from exc
        if not math.isfinite(nearest):
            raise ValueError("lifted Stage-3 slope exceeds binary64")
        lifted[name] = nearest
        error += max(Fraction(0), _fraction(nearest) - exact)
    if intercept is None:
        return lifted, None
    bound = round_fraction(_fraction(intercept) - error, upward=False)
    return lifted, bound


def original_trial_closed(pi, intercept, x_prev, alpha_level, *, tolerance=1e-6):
    """Check translated original-cut tightness, never residual LS status alone."""
    exact_tolerance = _fraction(tolerance)
    if exact_tolerance < 0:
        raise ValueError("cut tightness tolerance must be nonnegative")
    if intercept is None:
        return False
    value = _fraction(intercept)
    for name, coefficient in pi.items():
        bit = float((x_prev or {}).get(name, 0.0))
        if bit not in (0.0, 1.0):
            return False
        value += _fraction(coefficient) * int(bit)
    gap = _fraction(alpha_level) - value
    return 0 <= gap <= exact_tolerance
