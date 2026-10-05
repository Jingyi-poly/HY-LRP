"""Optional monotonicity lifting, only after a strict physical metric proof.

For a valid route cut v + pi*a + sigma*u, remove negative customer slopes and
cap sigma at min(sigma, r_min-v). If a chosen positive-slope subset is nonempty,
shortcutting to that subset proves validity. Otherwise the cheapest feasible
singleton r_min is a lower bound on every nonempty tour. Empty routes retain v.
Nonmetric instances keep their original cuts; no costs are modified.
"""
from fractions import Fraction
import math
import numpy as np


def _floor(value):
    result = float(value)
    return math.nextafter(result, -math.inf) if Fraction.from_float(result)>value else result


def route_metric_certificate(ctx, facility):
    cache = getattr(ctx, '_lrp_metric_certificate_cache', None)
    if cache is not None and facility in cache:
        return cache[facility]
    cap = Fraction.from_float(float(ctx.capacity[facility]))
    eligible = tuple(j for j in range(ctx.n) if ctx.active[j]
                     and Fraction.from_float(float(ctx.demand[j])) <= cap)
    vertices = [0]+[j+1 for j in eligible]
    cost = ctx.route_cost[facility][np.ix_(vertices, vertices)]
    passed = True
    off_diagonal = ~np.eye(len(vertices), dtype=bool)
    for k in range(len(vertices)):
        left, right = cost[:,k,None], cost[k,None,:]
        summed = left+right
        # One nextafter pays any upward addition rounding. Adding an exact
        # zero needs no rounding guard. Failure means no metric optimization.
        lower = np.where((left == 0)|(right == 0), summed,
                         np.nextafter(summed, -np.inf))
        check = off_diagonal.copy(); check[k,:]=False; check[:,k]=False
        if np.any(cost[check] > lower[check]):
            passed = False
            break
    minimum = min((Fraction.from_float(float(ctx.route_cost[facility,0,j+1]))
                   +Fraction.from_float(float(ctx.route_cost[facility,j+1,0]))
                   for j in eligible), default=Fraction())
    result = {'proved_directed_triangle': passed, 'eligible': eligible,
              'minimum_singleton_exact': minimum}
    # NodeContext owns immutable input bytes. Its cache is scoped to this
    # context object, never a facility count or a mutable instance filename.
    if cache is None:
        cache = {}
        object.__setattr__(ctx, '_lrp_metric_certificate_cache', cache)
    cache[facility] = result
    return result


def lift_metric_route_cut(ctx, facility, pi, intercept):
    """Return a proved lifted tuple, or None if unchanged/not applicable."""
    if intercept > 0 or not math.isfinite(float(intercept)):
        return None
    certificate = route_metric_certificate(ctx, facility)
    if not certificate['proved_directed_triangle']:
        return None
    customer_keys = {f'alpha[{facility},{j}]' for j in range(ctx.n)}
    if not any(k in customer_keys and value < 0 for k,value in pi.items()):
        return None
    lifted = {k:float(value) for k,value in pi.items()
              if value and not (k in customer_keys and value < 0)}
    key = f'u[{facility}]'
    sigma = min(Fraction.from_float(float(pi.get(key, 0.))),
                certificate['minimum_singleton_exact']-Fraction.from_float(float(intercept)))
    lifted[key] = _floor(sigma)
    return lifted, float(intercept)
