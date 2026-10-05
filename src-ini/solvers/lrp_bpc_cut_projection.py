"""Exactly equivalent forward-only projection of local LRP route envelopes.

The caller validates each cut against its node/facility with ``_as_cut`` first.
This module does not mutate or replace that original archive.  The full archive
must still be used to audit and reprice every returned native policy.

For a fixed first-stage policy, closed-facility alpha/u and inactive-customer
alpha are zero.  After that substitution, equal slopes need only their largest
intercept.  If both the intercept and every coefficient of A are >= those of B,
A implies B on nonnegative states.  The native theta >= 0 is another such row.
These comparisons are exact order/equality tests on the stored binary64 inputs:
no tolerance, subtraction, coefficient rounding, or weakening is used.  Thus
the entire theta epigraph, including its continuous nonnegative relaxation, is
unchanged after the fixed-zero substitution; this is not a cut approximation.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral

import numpy as np


def project_route_cuts(cuts_by_facility, available, active):
    """Return ``(native_payload, statistics)`` for already validated local cuts.

    ``cuts_by_facility`` maps every physical facility ``0..m-1`` to AffineCuts
    in alpha[0:n],u order.  ``available`` gives the native dense facility order;
    successor numbers remain physical facility IDs, including closed ones.
    """
    if not isinstance(cuts_by_facility, Mapping):
        raise ValueError('cuts_by_facility must be a physical-facility mapping')
    m = len(cuts_by_facility)
    if set(cuts_by_facility) != set(range(m)):
        raise ValueError('cuts_by_facility must contain exactly facilities 0..m-1')
    active = tuple(active)
    if any(value not in (0, 1) for value in active):
        raise ValueError('active must contain exact binary values')
    n = len(active)
    available = tuple(available)
    if (any(not isinstance(i, Integral) or isinstance(i, bool) or not 0 <= i < m
            for i in available) or len(set(available)) != len(available)):
        raise ValueError('available must contain unique physical facility IDs')
    local = {i: pos for pos, i in enumerate(available)}
    payload, per_facility = [], []
    total_input = total_equal = total_dominated = total_zero = 0
    for i in range(m):
        original = list(cuts_by_facility[i])
        total_input += len(original)
        # A Python tuple key compares the exact stored coefficients.  Inactive
        # coordinates are fixed zero by the physical service equalities.
        slopes = {}
        for cut in original:
            if cut.level != 'route' or len(cut.coefficients) != n + 1:
                raise ValueError('Expected local route cuts in alpha[0:n],u order')
            beta = float(cut.intercept)
            coefficients = tuple(float(c) for c in cut.coefficients)
            if not math.isfinite(beta) or not all(map(math.isfinite, coefficients)):
                raise ValueError('Cut coefficients and intercept must be finite')
            slope = tuple(c if i in local and (j == n or active[j]) else 0.
                          for j, c in enumerate(coefficients))
            if slope not in slopes or beta > slopes[slope]:
                slopes[slope] = beta
        equal_removed = len(original) - len(slopes)
        total_equal += equal_removed
        # Include theta >= 0 in the dominance logic without transmitting a
        # redundant row to the native kernel.  No differences are computed.
        candidates = [(slope, beta) for slope, beta in slopes.items()
                      if not (beta <= 0. and all(c <= 0. for c in slope))]
        zero_removed = len(slopes) - len(candidates)
        total_zero += zero_removed
        coefficients = np.asarray([slope for slope, _ in candidates], dtype=float)
        betas = np.asarray([beta for _, beta in candidates], dtype=float)
        retained = []
        for index, (slope, beta) in enumerate(candidates):
            dominates = ((betas >= beta) & np.all(coefficients >= coefficients[index], axis=1))
            dominates[index] = False
            if not np.any(dominates):
                retained.append((slope, beta))
        dominated_removed = len(candidates) - len(retained)
        total_dominated += dominated_removed
        for slope, beta in retained:
            y = [0.] * len(available)
            alpha = [[0.] * n for _ in available]
            if i in local:
                pos = local[i]
                y[pos] = slope[-1]
                alpha[pos] = list(slope[:-1])
            payload.append(dict(succ=i, beta=beta, piY=y, piAlpha=alpha))
        per_facility.append(dict(facility=i, available=i in local,
            original=len(original), retained=len(retained),
            equal_slopes_removed=equal_removed, theta_zero_removed=zero_removed,
            dominated_removed=dominated_removed))
    return payload, dict(original_count=total_input, projected_count=len(payload),
        equal_slopes_removed=total_equal, theta_zero_removed=total_zero,
        dominated_removed=total_dominated, per_facility=per_facility,
        projection='fixed_zero_exact_binary64_componentwise_v1')


__all__ = ['project_route_cuts']
