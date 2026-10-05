"""Exact package-cover dual certificates for physical route/outsource costs.

For anchor fleet F, an aggregate-capacity knapsack proves at least P packages
must be outsourced. The row sum(n*s) + P*sum(z outside F) >= P is valid for
every binary fleet. Its multiplier changes eta, but never routing prices or
the single-vehicle theta certificate. RMP objectives are not certificates.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from numbers import Real

from s2forward.integer_outsourcing import (
    IntegerOutsourcingCertificate, integer_outsourcing_certificate,
)


PACKAGE_DP = "package_knapsack_nonnegative_routes"
PACKAGE_DUAL = "routeopt_package_pricing_dual"


def binary64(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("certificate needs a real binary64 value")
    number = float(value)
    if not math.isfinite(number) or Fraction(value) != Fraction(number):
        raise ValueError("certificate value is nonfinite or not binary64")
    return Fraction(number)


def down(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("certificate exceeds binary64 range")
    return math.nextafter(number, -math.inf) if Fraction(number) > value else number


@dataclass(frozen=True)
class PackageCover:
    profile: IntegerOutsourcingCertificate
    groups: tuple
    physical_fingerprint: str

    def metadata(self):
        profile = self.profile
        return dict(
            jobs=list(profile.jobs), counts=list(profile.counts), unit=float(profile.unit),
            lower_packages=profile.lower_packages, anchor_flags=dict(profile.flags),
            physical_fingerprint=self.physical_fingerprint,
        )


def make_package_cover(pd, node, flags):
    """Return None for unsupported package data, leaving the same root CG intact."""
    profile = integer_outsourcing_certificate(pd, node, flags)
    if profile is None or profile.unit <= 0:
        return None
    try:
        binary64(profile.unit)
    except ValueError:
        return None
    # Local import avoids root_lp -> package_cover -> root_lp at import time.
    from .root_lp import _physical_groups

    jobs, demands, charges, physical = _physical_groups(pd, node, dict(profile.flags))
    if (tuple(jobs) != profile.jobs or tuple(demands) != profile.demands
            or tuple(charges) != profile.charges):
        raise ValueError("package metadata does not align with physical route jobs")
    groups = tuple((tuple(g["vehicles"]), g["capacity"], g["available"]) for g in physical)
    # Period weights are external to this physical cost function. Original
    # costs, demands, charges, package counts and full vehicle domains are not.
    semantic = (
        profile.jobs, profile.counts, profile.demands, profile.charges, profile.capacities,
        tuple((tuple(g["vehicles"]), g["capacity"],
               tuple(x.hex() for row in g["costs"] for x in row)) for g in physical),
    )
    fingerprint = hashlib.sha256(repr(semantic).encode()).hexdigest()
    return PackageCover(profile, groups, fingerprint)


def project_package_duals(cover, raw_u, raw_gamma):
    """Cap u downward to satisfy u_j + gamma*n_j <= outsourcing_charge_j."""
    if len(raw_u) != len(cover.profile.jobs):
        raise ValueError("customer dual dimensions differ")
    gamma = max(Fraction(), binary64(raw_gamma))
    u = tuple(Fraction(down(min(binary64(value), charge - gamma * count)))
              for value, charge, count in
              zip(raw_u, cover.profile.charges, cover.profile.counts))
    return u, gamma


def export_package_certificate(cover, u, beta_lower, gamma, *, source=PACKAGE_DUAL):
    """Export a physical-Q affine lower bound for every binary purchased fleet.

    beta_lower[k] must already bound min_R(c_k(R)-sum(u*a)) over ALL physical
    routes, including unavailable types. An analytic fallback is permitted;
    finite restricted-column bounds and uncompleted pricing minima are not.
    """
    if source not in (PACKAGE_DP, PACKAGE_DUAL):
        raise ValueError("unknown certificate source")
    u = tuple(binary64(value) for value in u)
    gamma = binary64(gamma)
    if len(u) != len(cover.profile.jobs) or len(beta_lower) != len(cover.groups):
        raise ValueError("missing customer/physical vehicle group certificate")
    beta = tuple(Fraction(down(min(Fraction(), Fraction(value)))) for value in beta_lower)
    if gamma < 0 or any(price + gamma * count > charge for price, count, charge in
                        zip(u, cover.profile.counts, cover.profile.charges)):
        raise ValueError("outsourcing dual constraints violated")
    if source == PACKAGE_DP and (gamma != cover.profile.unit or any(u) or any(beta)):
        raise ValueError("DP-only certificate must be u=beta=0, gamma=unit")
    extra = gamma * cover.profile.lower_packages
    intercept = down(sum(u, Fraction()) + extra)
    flags = dict(cover.profile.flags)
    slope = {
        v: down(value - (extra if not flags[v] else Fraction()))
        for (vehicles, _capacity, _available), value in zip(cover.groups, beta)
        for v in vehicles
    }
    bound = down(Fraction(intercept) + sum(
        (Fraction(slope[v]) * bit for v, bit in flags.items()), Fraction()))
    return dict(
        lb=bound, intercept=intercept, eta_slope_by_vehicle=slope,
        dual_u=list(map(float, u)), dual_beta=list(map(float, beta)), dual_gamma=float(gamma),
        package_cover=cover.metadata(), lb_certified=True, certificate_source=source,
        groups=[dict(vehicles=list(vehicles), capacity=float(capacity), available=available)
                for vehicles, capacity, available in cover.groups],
    )


def initial_package_certificate(cover):
    """Nonnegative physical routes + exact DP: no pricing call is claimed."""
    return export_package_certificate(
        cover, [Fraction()] * len(cover.profile.jobs), [Fraction()] * len(cover.groups),
        cover.profile.unit, source=PACKAGE_DP,
    )


@dataclass(frozen=True)
class VerifiedPackageCertificate:
    intercept: float
    eta_slope_by_vehicle: tuple
    jobs: tuple
    u: tuple
    beta_by_vehicle: tuple
    anchor_lb: float

    def eta_cut(self, period):
        if isinstance(period, bool) or not isinstance(period, int) or period < 0:
            raise ValueError("period must be a nonnegative integer")
        return ({f"z[{v},{period}]": value for v, value in self.eta_slope_by_vehicle if value},
                self.intercept)

    def theta_slope(self, vehicle):
        """Use only original u and beta; no package/outside-fleet lift enters theta."""
        beta = dict(self.beta_by_vehicle)[vehicle]
        slope = {f"alpha[{j},{vehicle}]": value for j, value in zip(self.jobs, self.u) if value}
        if beta:
            slope[f"y[{vehicle}]"] = beta
        return slope


def validate_package_certificate(pd, node, flags, result):
    """Verify the physical mapping and directed export, not the RMP objective.

    The root oracle supplies the all-route beta contract. TIME_LIMIT is valid
    when every beta has a certified fallback. Positivity is not a mathematical
    requirement; a useful partial-anchor cut may be negative at the full fleet.
    """
    if (not isinstance(result, dict) or result.get("lb_certified") is not True
            or result.get("certificate_source") not in (PACKAGE_DP, PACKAGE_DUAL)):
        return None
    cover = make_package_cover(pd, node, flags)
    if cover is None:
        return None
    if result.get("package_cover") != cover.metadata():
        raise ValueError("physical/package/anchor certificate mismatch")
    expected_groups = [dict(vehicles=list(v), capacity=float(c), available=a)
                       for v, c, a in cover.groups]
    groups = result.get("groups")
    if not isinstance(groups, list) or len(groups) != len(expected_groups):
        raise ValueError("missing physical group")
    for row, expected in zip(groups, expected_groups):
        if (tuple(row["vehicles"]) != tuple(expected["vehicles"])
                or type(row["available"]) is not int
                or row["available"] != expected["available"]
                or binary64(row["capacity"]) != Fraction(expected["capacity"])):
            raise ValueError("physical group layout mismatch")
    u = tuple(binary64(x) for x in result["dual_u"])
    beta = tuple(binary64(x) for x in result["dual_beta"])
    gamma = binary64(result["dual_gamma"])
    if any(x > 0 for x in beta):
        raise ValueError("beta must cover empty route at zero")
    rebuilt = export_package_certificate(cover, u, beta, gamma, source=result["certificate_source"])
    for key in ("lb", "intercept"):
        if binary64(result[key]) != Fraction(rebuilt[key]):
            raise ValueError("dual export rounding mismatch")
    slopes = result["eta_slope_by_vehicle"]
    if set(slopes) != set(rebuilt["eta_slope_by_vehicle"]) or any(
            binary64(slopes[v]) != Fraction(value)
            for v, value in rebuilt["eta_slope_by_vehicle"].items()):
        raise ValueError("outside-fleet lift/rounding mismatch")
    return VerifiedPackageCertificate(
        rebuilt["intercept"], tuple(rebuilt["eta_slope_by_vehicle"].items()),
        cover.profile.jobs, tuple(map(float, u)),
        tuple((v, float(b)) for (vehicles, _c, _a), b in zip(cover.groups, beta) for v in vehicles),
        rebuilt["lb"],
    )
