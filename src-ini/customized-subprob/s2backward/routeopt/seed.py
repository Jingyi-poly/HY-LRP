"""Budgeted full/partial-fleet physical route cuts for Stage-1 eta.

The route/outsource dual yields eta >= sum(u) + sum_v beta[type(v)] z[v,t].
It is not a bound on the learned-theta Stage-2 surrogate, and must never enter
that surrogate's piece table or Level-Set bundle. Probabilities, operating
weights, and purchase costs remain outside this unweighted physical cut.
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import math
from numbers import Integral, Real
import os
from pathlib import Path
import time

from gurobipy import GRB, GurobiError

from cuts.benders_cuts import add_unique_cut
from ..physical_route_seed import _finite, _physical_groups
from .root_lp import _physical_groups as _route_groups
from .root_lp import solve_routeopt_root_lp
from .physical_identity import physical_fingerprint_from_groups


_EXECUTABLE = Path(__file__).resolve().parent / "build/routeopt_pricing"
_UNSUPPORTED = frozenset({
    "this RouteOpt adapter requires a symmetric cost matrix",
    "RouteOpt CVRP pricing requires symmetric costs",
    "RouteOpt internal integer resources could overflow",
    "integer costs exceed exact arithmetic range",
    "integer prices exceed exact arithmetic range",
    "half-integer label arithmetic could round",
    "route cost arithmetic could round",
    "pricing requires strictly positive integer resources",
})


def _binary64(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("certificate coefficient must be a real binary64 scalar")
    number = float(value)
    if not math.isfinite(number) or Fraction(value) != Fraction(number):
        raise ValueError("certificate coefficient is nonfinite or not exactly binary64")
    return number


def build_routeopt_eta_cut(pd, node, result, *, anchor_flags=None):
    """Validate the physical certificate at its original purchased fleet.

    ``lb_certified is True`` is the root-pricing helper's assertion that each
    vehicle group's beta is a lower bound on c(route)-sum(u) for ALL physical
    routes, not just retained columns. This helper checks its data mapping and
    exact binary64 arithmetic; it does not rerun pricing or substitute RMP ObjVal.
    Ordinary roots use the full fleet unless explicit anchor flags are given.
    The ordinary fingerprint binds demands, charges and every vehicle's route
    costs/capacity; old certificates without that identity are rejected.
    A partial-fleet affine row remains valid outside its anchor. Its anchor
    value need not be positive; usefulness is separate from validity.
    Package rows have a separate exact
    metadata/dual check, including their outside-fleet coefficients.
    """
    if not isinstance(result, Mapping) or result.get("lb_certified") is not True:
        return None
    try:
        bound, intercept = _binary64(result["lb"]), _binary64(result["intercept"])
        period = node.info[1]
        if (isinstance(period, bool) or not isinstance(period, Integral)
                or isinstance(pd.T, bool) or not isinstance(pd.T, Integral)
                or not 0 <= period < pd.T
                or len(set(pd.V)) != len(pd.V) or len(set(pd.J)) != len(pd.J)):
            return None
        flags = {v: 1 for v in pd.V} if anchor_flags is None else anchor_flags
        if (not isinstance(flags, Mapping) or set(flags) != set(pd.V)
                or any(_binary64(flags[v]) not in (0., 1.) for v in pd.V)):
            return None
        if result.get("certificate_source") in (
                "routeopt_package_pricing_dual", "package_knapsack_nonnegative_routes"):
            from .package_cover import validate_package_certificate
            checked = validate_package_certificate(pd, node, flags, result)
            return None if checked is None else checked.eta_cut(int(period))
        if result.get("certificate_source") not in (
                "routeopt_pricing_dual", "physical_capacity_dual",
                "nonnegative_route_bound"):
            return None
        u = [_binary64(value) for value in result["dual_u"]]
        beta = [_binary64(value) for value in result["dual_beta"]]
        groups = result["groups"]
        jobs, demands, outsourcing, physical = _route_groups(pd, node, flags)
        if result.get("physical_fingerprint") != physical_fingerprint_from_groups(
                jobs, demands, outsourcing, physical):
            return None
        if len(u) != len(jobs) or len(beta) != len(physical) or len(groups) != len(physical):
            return None
        if any(Fraction(value) > out for value, out in zip(u, outsourcing)):
            return None
        if Fraction(intercept) > sum(map(Fraction, u), Fraction()) or any(value > 0 for value in beta):
            return None
        coefficients = {}
        for returned, expected, value in zip(groups, physical, beta):
            # Compare the exact grouping reconstructed from capacity and the
            # active routing matrix, not a user-supplied vehicle type label.
            if (tuple(returned["vehicles"]) != tuple(expected["vehicles"])
                    or _binary64(returned["capacity"]) != expected["capacity"]
                    or type(returned["available"]) is not int
                    or returned["available"] != expected["available"]):
                return None
            if value:
                for vehicle in expected["vehicles"]:
                    coefficients[f"z[{vehicle},{int(period)}]"] = value
        anchor_value = Fraction(intercept) + sum((Fraction(value) * group["available"]
                                                  for value, group in zip(beta, physical)), Fraction())
        if Fraction(bound) > anchor_value:
            return None
        return coefficients, intercept
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None


def _seed_full_fleet_bounds(pd, tree, cut_lag, *, time_limit_s, per_node_limit_s=5.0,
                           initial_policy=None):
    """Run a single serial, fair-budgeted pass and modify only cut_lag[2].

    Identical physical nodes share a solve across periods, without sharing
    learned/private cuts. Each root call is limited to min(per_node_limit,
    remaining/groups_left), uses one Gurobi thread and NG8 relaxed columns.
    Native pricing has a subprocess watchdog; Python/RMP construction and
    certificate checking are cooperative and are included in reported time.
    Any overrun prevents another call; there is no fallback or automatic build.
    Pricing route priors are returned for a later backward pass, not inserted
    here. An optional complete policy can reuse physical columns for a bounded
    restricted-master improvement, within this same seed deadline.
    """
    started = time.monotonic()
    budget = _finite(time_limit_s, "time_limit_s")
    per_node = _finite(per_node_limit_s, "per_node_limit_s")
    if per_node <= 0:
        raise ValueError("per_node_limit_s must be positive")
    report = dict(attempted=False, attempts=0, solves=0, copied=0, added=0,
                  seconds=0.0, bounds=[], failures=[], nodes=0, groups=0,
                  skipped=0, deadline_exhausted=False, backend="routeopt_ng8",
                  ng_size=8, relaxed_columns=True, status="not_started",
                  route_priors=[])
    if budget <= 0:
        report["seconds"] = time.monotonic() - started
        return report
    deadline = started + budget
    if not _EXECUTABLE.is_file() or not os.access(_EXECUTABLE, os.X_OK):
        report["status"] = "backend_unavailable"
        report["failures"].append(dict(node=None, reason="RouteOpt pricing executable unavailable"))
        report["seconds"] = time.monotonic() - started
        return report
    groups, flags = _physical_groups(pd, tree, report)
    report["groups"] = len(groups)
    report["status"] = "complete"
    improver = None
    if groups and initial_policy is not None:
        from .restricted_master import RoutePolicyImprover
        improver = RoutePolicyImprover(pd, tree, initial_policy)
    for position, members in enumerate(groups):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            report["deadline_exhausted"] = True
            report["status"] = "time_limit"
            report["skipped"] += sum(map(len, groups[position:]))
            break
        node = members[0]
        # Reject known physical incompatibilities before entering root CG.
        try:
            jobs, _, _, _ = _route_groups(pd, node, flags)
            if len(jobs) >= 1002:
                raise ValueError("RouteOpt customer bitset capacity exceeded")
        except ValueError as exc:
            if str(exc) not in _UNSUPPORTED and str(exc) != "RouteOpt customer bitset capacity exceeded":
                raise
            report["failures"].append(dict(node=int(node.index), reason=str(exc)))
            report["skipped"] += len(members)
            continue
        primal_options = (
            {"route_consumer": improver.start_group(node)} if improver is not None else {}
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            report["deadline_exhausted"] = True
            report["status"] = "time_limit"
            report["skipped"] += sum(map(len, groups[position:]))
            break
        requested = min(per_node, remaining / (len(groups) - position))
        report["attempted"] = True
        report["attempts"] += 1
        call_started = time.monotonic()
        try:
            result = solve_routeopt_root_lp(
                pd, node, flags, time_limit_s=requested, threads=1,
                ng_size=8, relaxed_columns=True,
                **primal_options,
            )
        except ValueError as exc:
            if str(exc) not in _UNSUPPORTED:
                raise
            report["failures"].append(dict(node=int(node.index), reason=str(exc)))
            report["skipped"] += len(members)
            continue
        except GurobiError as exc:
            if exc.errno not in (GRB.Error.NO_LICENSE, GRB.Error.SIZE_LIMIT_EXCEEDED):
                raise
            report["failures"].append(dict(node=int(node.index), reason=str(exc)))
            report["skipped"] += sum(map(len, groups[position:]))
            report["status"] = "solver_unavailable"
            break
        except RuntimeError as exc:
            if str(exc) != "Build the adapter with routeopt/build.py first":
                raise
            report["failures"].append(dict(node=int(node.index), reason=str(exc)))
            report["skipped"] += sum(map(len, groups[position:]))
            report["status"] = "backend_unavailable"
            break
        except OSError as exc:
            # A binary removed or made non-executable after the preflight is
            # unavailable; unrelated filesystem/program errors still propagate.
            if (exc.errno not in (2, 8, 13)
                    or Path(exc.filename or "") != _EXECUTABLE):
                raise
            report["failures"].append(dict(node=int(node.index), reason=str(exc)))
            report["skipped"] += sum(map(len, groups[position:]))
            report["status"] = "backend_unavailable"
            break
        report["solves"] += 1
        cut = build_routeopt_eta_cut(pd, node, result)
        accepted, added = cut is not None, 0
        if accepted:
            # The certificate was checked once for this exact physical group;
            # only period names change. Coefficients are copied without cleaning.
            for member in members:
                period = int(member.info[1])
                coefficients = {f"z[{vehicle},{period}]": value
                                for group, value in zip(result["groups"], result["dual_beta"])
                                for vehicle in group["vehicles"] if value != 0}
                added += int(add_unique_cut(
                    cut_lag.setdefault(2, {}).setdefault(member.index, []), coefficients, cut[1]))
            report["copied"] += len(members) - 1
            report["added"] += added
            if tree.get(3):
                from .route_priors import build_route_priors
                report["route_priors"].extend(build_route_priors(pd, tree, members, result))
        else:
            report["skipped"] += len(members)
        if improver is not None:
            improver.improve_group(members, deadline)
        report["bounds"].append(dict(
            node=int(node.index), members=[int(member.index) for member in members],
            lb=result.get("lb"), ub=None, status=result.get("status"),
            accepted=accepted, added=added, requested_seconds=requested,
            seconds=time.monotonic() - call_started,
            certificate_source=result.get("certificate_source"),
            completed_pricing_calls=result.get("completed_pricing_calls", 0),
            pricing_calls=result.get("pricing_calls", 0),
            rmp_obj=result.get("rmp_obj"),
        ))
    if improver is not None:
        report["policy_candidate"] = improver.report_candidate()
        report["policy_improvement"] = dict(improver.stats)
    report["seconds"] = time.monotonic() - started
    report["deadline_exhausted"] |= report["seconds"] >= budget
    if report["deadline_exhausted"]:
        report["status"] = "time_limit"
    elif not groups:
        report["status"] = "no_supported_nodes"
    return report


def _partial_anchors(pd, tree, policy, report, eligible_nodes):
    physical, _ = _physical_groups(pd, tree, report)
    anchors = []
    for members in physical:
        if members[0].index not in eligible_nodes:
            continue
        same_fleet = {}
        for node in members:
            flags = tuple(policy[1][0][f"z[{v},{node.info[1]}]"] for v in pd.V)
            if any(flag not in (0., 1.) for flag in flags):
                raise ValueError("partial route seed needs exact binary purchase flags")
            if not all(flags):
                same_fleet.setdefault(flags, []).append(node)
        anchors.extend((tuple(members), tuple(targets), dict(zip(pd.V, flags)))
                       for flags, targets in same_fleet.items())
    return anchors


def _seed_partial_fleet_bounds(pd, tree, cut_lag, policy, *, deadline, eligible_nodes,
                               partial_node_limit_s=8.):
    from .restricted_master import RoutePolicyImprover, _certify_node
    from .route_priors import build_route_priors

    started = time.monotonic()
    per_node = _finite(partial_node_limit_s, "partial_node_limit_s")
    if per_node <= 0:
        raise ValueError("partial_node_limit_s must be positive")
    report = dict(attempts=0, solves=0, copied=0, added=0, skipped=0,
                  failures=[], bounds=[], route_priors=[], seconds=0.,
                  deadline_exhausted=False, status="complete", ng_size=4)
    anchors = _partial_anchors(pd, tree, policy, report, eligible_nodes)
    report["groups"] = len(anchors)
    if not anchors or time.monotonic() >= deadline:
        report["seconds"] = time.monotonic() - started
        report["deadline_exhausted"] = time.monotonic() >= deadline
        return report
    owner = RoutePolicyImprover(pd, tree, policy)
    for position, (members, targets, flags) in enumerate(anchors):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            report["skipped"] += sum(len(group[1]) for group in anchors[position:])
            break
        node = targets[0]
        group_deadline = time.monotonic() + remaining / (len(anchors) - position)
        consumer = owner.start_group(node)
        share = max(0., group_deadline - time.monotonic())
        primal_reserve = min(1., share / 3.)
        requested = min(per_node, max(0., group_deadline - time.monotonic() - primal_reserve))
        if requested <= 0:
            break
        report["attempts"] += 1
        call_started = time.monotonic()
        try:
            result = solve_routeopt_root_lp(
                pd, node, flags, time_limit_s=requested, threads=1,
                ng_size=4, relaxed_columns=True, package_cover=True,
                route_consumer=consumer,
            )
        except (ValueError, GurobiError, RuntimeError, OSError) as exc:
            if isinstance(exc, ValueError) and str(exc) in _UNSUPPORTED:
                report["failures"].append(dict(node=int(node.index), reason=str(exc)))
                report["skipped"] += len(members)
                continue
            solver_missing = isinstance(exc, GurobiError) and exc.errno in (
                GRB.Error.NO_LICENSE, GRB.Error.SIZE_LIMIT_EXCEEDED)
            backend_missing = (
                isinstance(exc, RuntimeError) and str(exc) == "Build the adapter with routeopt/build.py first"
            ) or (isinstance(exc, OSError) and exc.errno in (2, 8, 13)
                  and Path(exc.filename or "") == _EXECUTABLE)
            if not solver_missing and not backend_missing:
                raise
            report["failures"].append(dict(node=int(node.index), reason=str(exc)))
            report["skipped"] += sum(len(group[1]) for group in anchors[position:])
            report["status"] = "solver_unavailable" if solver_missing else "backend_unavailable"
            break
        report["solves"] += 1
        cut = build_routeopt_eta_cut(pd, node, result, anchor_flags=flags)
        _, _, upper = _certify_node(pd, tree, node.index, policy[1][0], policy)
        if result.get("lb_certified") is True and result.get("lb") is not None:
            if Fraction(result["lb"]) > Fraction(upper):
                raise RuntimeError("partial physical LB exceeds its certified feasible policy")
        added = 0
        if cut is not None:
            value = Fraction(cut[1]) + sum((Fraction(cut[0].get(
                f"z[{v},{node.info[1]}]", 0.)) * flag for v, flag in flags.items()), Fraction())
            if value > Fraction(upper):
                raise RuntimeError("partial eta cut exceeds its certified feasible policy")
            # Exact physical copies may use different investment decisions.
            # The affine row is globally valid; the primal routes are not
            # copied beyond the identical purchased-fleet targets below.
            for member in members:
                copied = build_routeopt_eta_cut(pd, member, result, anchor_flags=flags)
                if copied is None:
                    raise ValueError("partial route certificate changed across physical copies")
                added += int(add_unique_cut(cut_lag.setdefault(2, {}).setdefault(member.index, []), *copied))
            report["copied"] += len(members) - 1
            report["route_priors"].extend(build_route_priors(
                pd, tree, members, result, anchor_flags=flags))
        owner.improve_group(targets, min(deadline, group_deadline))
        report["added"] += added
        report["bounds"].append(dict(
            node=int(node.index), members=[int(n.index) for n in members],
            policy_members=[int(n.index) for n in targets], anchor_flags=flags,
            lb=result.get("lb"), ub=None, status=result.get("status"),
            accepted=cut is not None, added=added, requested_seconds=requested,
            seconds=time.monotonic() - call_started,
            certificate_source=result.get("certificate_source"),
            completed_pricing_calls=result.get("completed_pricing_calls", 0),
            pricing_calls=result.get("pricing_calls", 0), rmp_obj=result.get("rmp_obj"),
        ))
    report["policy_candidate"] = owner.report_candidate()
    report["policy_improvement"] = dict(owner.stats)
    report["seconds"] = time.monotonic() - started
    report["deadline_exhausted"] = time.monotonic() >= deadline
    if report["deadline_exhausted"] and report["status"] not in ("backend_unavailable", "solver_unavailable"):
        report["status"] = "time_limit"
    return report


def seed_routeopt_bounds(pd, tree, cut_lag, *, time_limit_s, per_node_limit_s=5.,
                        initial_policy=None, partial_node_limit_s=8.):
    """Preserve the full NG8 pass; use remaining time for partial NG4 roots.

    Only globally valid physical-Q rows modify eta. Returned route priors
    wait for the ordinary backward consumer. Restricted-route MIPs supply
    complete feasible policies, never lower bounds. No second Stage-1 solve
    or time reserve is introduced here.
    """
    started = time.monotonic()
    budget = _finite(time_limit_s, "time_limit_s")
    partial_limit = _finite(partial_node_limit_s, "partial_node_limit_s")
    if partial_limit <= 0:
        raise ValueError("partial_node_limit_s must be positive")
    deadline = started + budget
    report = _seed_full_fleet_bounds(pd, tree, cut_lag, time_limit_s=budget,
                                   per_node_limit_s=per_node_limit_s,
                                   initial_policy=initial_policy)
    report["full_pass_seconds"] = report["seconds"]
    report["full_pass_solves"] = report["solves"]
    report["full_pass_ng_size"] = 8
    if (initial_policy is None or not report["bounds"] or time.monotonic() >= deadline
            or report["status"] in ("backend_unavailable", "solver_unavailable")):
        return report
    from .restricted_master import certify_complete_policy

    normalized, initial_upper = certify_complete_policy(pd, tree, initial_policy)
    best = report.get("policy_candidate")
    upper = initial_upper
    if best is not None:
        if best.get("certified") is not True:
            raise ValueError("uncertified full-pass policy")
        current, upper = certify_complete_policy(pd, tree, best["policy"])
        if (current[1] != normalized[1] or upper != best["ub"] or upper > initial_upper):
            raise ValueError("full-pass policy changed purchases or reported cost")
    else:
        current = normalized
    partial_options = {"partial_node_limit_s": partial_limit} if partial_limit != 8. else {}
    partial = _seed_partial_fleet_bounds(pd, tree, cut_lag, current, deadline=deadline,
                                       eligible_nodes={row["node"] for row in report["bounds"]},
                                       **partial_options)
    candidate = partial.pop("policy_candidate", None)
    if candidate is not None:
        if candidate.get("certified") is not True:
            raise ValueError("uncertified partial-pass policy")
        checked, candidate_upper = certify_complete_policy(pd, tree, candidate["policy"])
        if (checked[1] != normalized[1] or candidate_upper != candidate["ub"] or candidate_upper > upper):
            raise ValueError("partial pass changed purchases, misreported cost, or worsened UB")
        report["policy_candidate"] = dict(policy=checked, ub=candidate_upper, certified=True)
    for field in ("attempts", "solves", "copied", "added", "skipped", "groups"):
        report[field] += partial[field]
    report["attempted"] |= bool(partial["attempts"])
    report["route_priors"].extend(partial.pop("route_priors"))
    report["failures"].extend(partial["failures"])
    report["partial_pass"] = partial
    report["partial_pass_ng_size"] = partial["ng_size"]
    if partial["status"] in ("backend_unavailable", "solver_unavailable"):
        report["status"] = partial["status"]
    report["seconds"] = time.monotonic() - started
    report["deadline_exhausted"] = time.monotonic() >= deadline
    if report["deadline_exhausted"] and report["status"] not in ("backend_unavailable", "solver_unavailable"):
        report["status"] = "time_limit"
    return report


__all__ = ["build_routeopt_eta_cut", "seed_routeopt_bounds"]
