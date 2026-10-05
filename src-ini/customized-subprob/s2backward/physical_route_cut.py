"""True physical LRP recourse bounds, used only by the root eta master.

For 0 <= lambda_j <= outsourcing_j, each facility supplies a certified
beta_i <= min_route(cost_i - lambda * served). Empty routes imply beta_i <= 0.
Then Q(A) >= sum(lambda) + sum(beta_i * A_i). Costs and cuts are unweighted;
scenario probabilities belong to the root objective. A restricted route
master objective and a pricing incumbent are never certificates for beta.
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import math
from numbers import Real

from models.stage_builder import _instance, _node_context


def _exact(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('expected a finite real scalar')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('expected a finite real scalar')
    return Fraction.from_float(number)


def round_down(value):
    value = Fraction(value)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('coefficient exceeds binary64 range')
    return math.nextafter(result, -math.inf) if Fraction.from_float(result) > value else result


def build_physical_pricing_cut(prob_data, node, rewards, beta, *, beta_certified):
    """Build the global availability cut from caller-certified pricing bounds.

    Certification concerns this exact context, physical facility, original
    route costs and reward vector. This helper checks all coefficient domains;
    the producer is responsible for the supplied lower-bound provenance.
    """
    if beta_certified is not True:
        raise ValueError('physical pricing coefficients need certified lower bounds')
    ctx = _node_context(_instance(prob_data), node, stage=2)
    if len(rewards) != ctx.n or len(beta) != ctx.m:
        raise ValueError('wrong physical pricing dimension')
    lam = tuple(_exact(v) for v in rewards)
    bounds = tuple(_exact(v) for v in beta)
    for j, value in enumerate(lam):
        if value < 0 or value > _exact(ctx.outsourcing[j]) or (not ctx.active[j] and value != 0):
            raise ValueError('lambda must be zero if inactive and between zero and outsourcing')
    if any(value > 0 for value in bounds):
        raise ValueError('empty physical route requires beta <= 0')
    if any(float(v) < 0 or not math.isfinite(float(v)) for v in ctx.route_cost.flat):
        raise ValueError('nonnegative finite physical routing costs are required')
    intercept = round_down(sum(lam, Fraction()))
    return ({f'A[{i},{ctx.interval}]': round_down(value)
             for i, value in enumerate(bounds) if value != 0}, intercept)


def build_physical_route_cut(prob_data, node, flags, lb, *, lb_certified):
    """Compatibility helper for a certified fixed-availability physical bound.

    Extra facilities can idle, so Q(A) >= L(1-sum(A_i outside F)) for
    0 <= L <= Q(F). The pricing driver uses the stronger dual cut above;
    this function neither solves a model nor writes any surrogate bundle.
    """
    if lb_certified is not True or not isinstance(flags, Mapping):
        return None
    try:
        ctx = _node_context(_instance(prob_data), node, stage=2)
        bound = _exact(lb)
        if bound < 0:
            return None
        values = [flags.get(f'A[{i},{ctx.interval}]', flags.get(i)) for i in range(ctx.m)]
        if any(value not in (0, 1) for value in values):
            return None
        return ({f'A[{i},{ctx.interval}]': round_down(-bound)
                 for i, value in enumerate(values) if not value}, round_down(bound))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None


__all__ = ['build_physical_pricing_cut', 'build_physical_route_cut']
