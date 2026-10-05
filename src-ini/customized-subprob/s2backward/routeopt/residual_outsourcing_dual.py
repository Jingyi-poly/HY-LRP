"""Physical dual search in a smaller auxiliary outsourcing-price box.

Only a copied node has smaller charges. Its routing-price certificate is
rebuilt against the original package cover; shadow objectives stay diagnostic.
"""
from copy import copy
from collections.abc import Mapping
from fractions import Fraction
from functools import wraps
import math
import time

from s2backward.routeopt.capacity_price_bound import CapacityPriceBound
from s2backward.routeopt.package_cover import (
    export_package_certificate, initial_package_certificate, make_package_cover,
)
from s2backward.routeopt.physical_identity import physical_fingerprint_from_groups
from s2backward.routeopt.root_lp import (
    _certificate_update_reason, _directed, _dual_certificate, _physical_groups, _route_data,
)
from s2backward.routeopt.seed import build_routeopt_eta_cut


SCALE = Fraction(1, 16)


def _checked_scale(scale):
    scale = Fraction(scale)
    if not 0 < scale <= 1:
        raise ValueError("residual price scale must be in (0, 1]")
    return scale


def shadow_node(node, *, scale=SCALE):
    scale = _checked_scale(scale)
    shadow = copy(node)
    charges = {}
    items = node.c_out.items() if isinstance(node.c_out, Mapping) else enumerate(node.c_out)
    for job, value in items:
        exact = Fraction(value) * scale
        scaled = float(exact)
        if not math.isfinite(scaled) or Fraction(scaled) != exact:
            raise ValueError("shadow charges are not exactly representable binary64")
        charges[job] = scaled
    shadow.c_out = charges
    return shadow


def lift_shadow_certificate(pd, node, shadow, flags, certificate, *, package_enabled, scale=SCALE):
    """Recheck both physical domains and rebuild an original-price certificate."""
    scale = _checked_scale(scale)
    original = _physical_groups(pd, node, flags)
    modified = _physical_groups(pd, shadow, flags)
    jobs, demands, charges, groups = original
    sjobs, sdemands, scharges, sgroups = modified
    if jobs != sjobs or demands != sdemands or groups != sgroups:
        raise ValueError("shadow changed routing domain or anchor fleet")
    if any(scaled != scale * value for scaled, value in zip(scharges, charges)):
        raise ValueError("shadow charges do not match the exact scale")
    expected_shadow = physical_fingerprint_from_groups(*modified)
    if certificate.get("physical_fingerprint") != expected_shadow:
        raise ValueError("shadow certificate physical identity mismatch")
    if build_routeopt_eta_cut(pd, shadow, certificate, anchor_flags=flags) is None:
        raise ValueError("shadow certificate fails its original consumer contract")

    u = tuple(Fraction(value) for value in certificate["dual_u"])
    beta = tuple(Fraction(value) for value in certificate["dual_beta"])
    original_cover = make_package_cover(pd, node, flags) if package_enabled else None
    gamma_shadow = Fraction(certificate.get("dual_gamma", 0.))
    if original_cover is not None:
        modified_cover = make_package_cover(pd, shadow, flags)
        if modified_cover is None:
            raise ValueError("shadow package profile is unavailable")
        a, b = original_cover.profile, modified_cover.profile
        for field in ("jobs", "counts", "demands", "flags", "capacities", "lower_packages"):
            if getattr(a, field) != getattr(b, field):
                raise ValueError("shadow package profile changed: " + field)
        if b.unit != scale * a.unit or original_cover.groups != modified_cover.groups:
            raise ValueError("shadow package unit or physical groups changed")
        if any(price + gamma_shadow * count > charge
               for price, count, charge in zip(u, b.counts, b.charges)):
            raise ValueError("shadow outsourcing dual constraints violated")
        gamma = Fraction(_directed(gamma_shadow + (1 - scale) * a.unit))
        lifted = export_package_certificate(original_cover, u, beta, gamma)
    else:
        # Dropping a nonnegative shadow package multiplier is also safe.
        if any(price > charge for price, charge in zip(u, scharges)):
            raise ValueError("shadow ordinary outsourcing dual constraints violated")
        lifted = _dual_certificate(u, beta, groups)
        lifted.update(lb_certified=True, certificate_source="routeopt_pricing_dual")
        gamma = Fraction()
    lifted.update(
        groups=[dict(vehicles=g["vehicles"], available=g["available"], capacity=float(g["capacity"]))
                for g in groups],
        physical_fingerprint=physical_fingerprint_from_groups(*original),
        certificate_method="residual_outsourcing_dual", certificate_domain="physical_elementary_routes",
    )
    if build_routeopt_eta_cut(pd, node, lifted, anchor_flags=flags) is None:
        raise ValueError("rebuilt original-price certificate fails consumer contract")
    return lifted, float(gamma)


def make_residual_outsourcing_solver(base_solver, *, scale=SCALE):
    scale = _checked_scale(scale)
    @wraps(base_solver)
    def solve(pd, node, flags, **kwargs):
        started = time.monotonic()
        budget = float(kwargs.get("time_limit_s", 5.))
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("positive finite shared budget required")
        deadline = started + budget
        original = _physical_groups(pd, node, flags)
        jobs, demands, charges, groups = original
        package_enabled = bool(kwargs.get("package_cover", False))
        cover = make_package_cover(pd, node, flags) if package_enabled else None
        initial_u = charges if not any(group["available"] for group in groups) else [Fraction()] * len(jobs)
        baseline = _dual_certificate(initial_u, [CapacityPriceBound.from_group(demands, group).lower_bound(initial_u)
                                               for group in groups], groups)
        baseline.update(lb_certified=True, certificate_source="physical_capacity_dual" if any(initial_u)
                        else "nonnegative_route_bound")
        if cover is not None:
            package_floor = initial_package_certificate(cover)
            if _certificate_update_reason(package_floor, baseline, groups):
                baseline = package_floor
        shadow = shadow_node(node, scale=scale)
        consumer = kwargs.get("route_consumer")
        delivered = rejected = 0
        group_by_vehicles = {tuple(group["vehicles"]): group for group in groups}
        local = {job: index + 1 for index, job in enumerate(jobs)}

        def checked_consumer(vehicles, route, reported_cost):
            nonlocal delivered, rejected
            group = group_by_vehicles.get(tuple(vehicles))
            if group is None or any(job not in local for job in route):
                rejected += 1
                return
            checked = _route_data(tuple(local[job] for job in route), group, demands)
            if checked is None or Fraction(reported_cost) != checked[1]:
                rejected += 1
                return
            if consumer is not None:
                consumer(tuple(vehicles), tuple(route), checked[1])
                delivered += 1

        preparation_seconds = time.monotonic() - started
        validation_reserve = min(.25 * budget, max(.05, .1 * budget + 3 * preparation_seconds))
        remaining = deadline - time.monotonic()
        base_requested = max(0., remaining - validation_reserve)
        base_seconds = validation_seconds = 0.
        selected = "original_analytic"
        if base_requested <= 0:
            result = dict(baseline)
            shadow_result = None
            lifted = None
            gamma = None
        else:
            options = dict(kwargs, time_limit_s=base_requested,
                           route_consumer=checked_consumer if consumer is not None else None)
            base_started = time.monotonic()
            shadow_result = base_solver(pd, shadow, flags, **options)
            base_seconds = time.monotonic() - base_started
            validation_started = time.monotonic()
            lifted, gamma = lift_shadow_certificate(
                pd, node, shadow, flags, shadow_result, package_enabled=package_enabled, scale=scale)
            use_lifted = _certificate_update_reason(lifted, baseline, groups)
            result = dict(lifted if use_lifted else baseline)
            selected = "lifted" if use_lifted else "original_analytic"
            validation_seconds = time.monotonic() - validation_started
        result.update(
            physical_fingerprint=physical_fingerprint_from_groups(*original),
            groups=[dict(vehicles=g["vehicles"], available=g["available"], capacity=float(g["capacity"]))
                    for g in groups],
            certificate_domain="physical_elementary_routes", physical_optimality_proven=False,
            rmp_obj=None, root_lp_gap=None, rmp_minus_physical_lb=None, history=[],
            status="residual_outsourcing_certificate" if shadow_result is not None else "preparation_deadline",
            seconds=time.monotonic() - started,
            residual_outsourcing=dict(
                scale=str(scale), baseline_lb=baseline["lb"],
                lifted_lb=None if lifted is None else lifted["lb"],
                lifted_gamma=gamma, selected=selected, shadow_result=shadow_result,
                original_routes_delivered=delivered, rejected_routes=rejected,
                preparation_seconds=preparation_seconds, validation_reserve_seconds=validation_reserve,
                base_requested_seconds=base_requested, base_seconds=base_seconds,
                validation_seconds=validation_seconds,
                requested_seconds=budget, deadline_exhausted=time.monotonic() >= deadline,
            ),
        )
        for name in ("pricing_calls", "completed_pricing_calls", "pricing_seconds", "iterations",
                     "route_count", "relaxed_route_count", "improved_columns"):
            result[name] = 0 if shadow_result is None else shadow_result.get(name, 0)
        result["relaxed_columns"] = bool(kwargs.get("relaxed_columns", False))
        return result
    return solve


__all__ = ["SCALE", "shadow_node", "lift_shadow_certificate", "make_residual_outsourcing_solver"]
