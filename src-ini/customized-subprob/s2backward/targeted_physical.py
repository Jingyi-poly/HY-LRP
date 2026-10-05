"""Targeted physical-Q cuts and feasible policies; never surrogate-C bounds."""
from fractions import Fraction
import math
from numbers import Integral
import os
from pathlib import Path
import time


class PhysicalTargetsUnavailable(ValueError):
    """No trusted physical domain; preserve the ordinary Phase-2 path."""
    def __init__(self, failures):
        self.failures = failures
        super().__init__(str(failures))


def _checked_target_key(value):
    """Canonical scheduling identity, not a certificate or an oracle cache key."""
    try:
        members, flags = value
        members = tuple(members)
        flags = tuple(tuple(pair) for pair in flags)
        if (not members or any(isinstance(index, bool) or not isinstance(index, Integral)
                               or index < 0 for index in members)
                or len(set(members)) != len(members)):
            raise ValueError
        if any(len(pair) != 2 for pair in flags):
            raise ValueError
        if any(isinstance(vehicle, bool) or not isinstance(vehicle, Integral) or vehicle < 0
               or isinstance(bit, bool) or bit not in (0., 1.)
               for vehicle, bit in flags):
            raise ValueError
        if not flags or len({vehicle for vehicle, _ in flags}) != len(flags):
            raise ValueError
        return (tuple(sorted(int(index) for index in members)),
                tuple(sorted((int(vehicle), float(bit)) for vehicle, bit in flags)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("target key must contain unique node IDs and binary vehicle flags") from exc


def physical_target_key(target):
    """Identify one physical domain and purchased fleet in a target tuple."""
    members, _, flags, _ = target
    return _checked_target_key((tuple(node.index for node in members), flags.items()))


def physical_target_key_from_row(row):
    """Recover a scheduling key from a live or JSON-loaded target report."""
    try:
        flags = row["flags"]
        # JSON object keys are strings; only exact integer spellings are accepted.
        if isinstance(flags, dict):
            flags = [(int(vehicle) if isinstance(vehicle, str) and str(int(vehicle)) == vehicle
                      else vehicle, bit) for vehicle, bit in flags.items()]
        return _checked_target_key((row["members"], flags))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid physical target report identity") from exc


def _pricing_metadata(result):
    """Search completion is separate from validity of the returned physical LB."""
    residual = result.get("residual_outsourcing")
    underlying = residual.get("shadow_result") if isinstance(residual, dict) else result
    underlying = underlying if isinstance(underlying, dict) else {}
    status = underlying.get("status")
    return dict(pricing_status=status,
                pricing_search_complete=status in ("priced_stationary", "trivial"),
                pricing_calls=underlying.get("pricing_calls"),
                completed_pricing_calls=underlying.get("completed_pricing_calls"),
                pricing_iterations=underlying.get("iterations"),
                pricing_seconds=underlying.get("pricing_seconds"))


def eta_value(node, fleet, archive):
    return max([Fraction()] + [Fraction(intercept) + sum(
        (Fraction(value) * Fraction(fleet[name]) for name, value in slope.items()), Fraction())
        for slope, intercept in archive.get(2, {}).get(node.index, [])])


def current_fleet_targets(pd, tree, policy, archive):
    from s2backward.physical_route_seed import _physical_groups
    from s2backward.routeopt.restricted_master import _certify_node

    metadata = dict(failures=[], skipped=0)
    physical, _ = _physical_groups(pd, tree, metadata)
    if metadata["failures"]:
        raise PhysicalTargetsUnavailable(metadata["failures"])
    fleet, targets = policy[1][0], []
    for members in physical:
        by_flags = {}
        for node in members:
            bits = tuple(fleet[f"z[{v},{node.info[1]}]"] for v in pd.V)
            if any(bit not in (0., 1.) for bit in bits):
                raise ValueError("target fleet must be exactly binary")
            by_flags.setdefault(bits, []).append(node)
        for bits, exact_targets in by_flags.items():
            priority = Fraction()
            for node in exact_targets:
                _, _, upper = _certify_node(pd, tree, node.index, fleet, policy)
                priority += Fraction(node.multi_coeff) * max(
                    Fraction(), Fraction(upper) - eta_value(node, fleet, archive))
            targets.append((tuple(members), tuple(exact_targets), dict(zip(pd.V, bits)), priority))
    targets.sort(key=lambda item: (-item[3], item[1][0].index))
    return targets


def policy_deadline(seed_deadline, group_deadline, now, grace):
    """Optionally borrow a short policy slice, never from the master reserve."""
    if not math.isfinite(grace) or grace < 0:
        raise ValueError("policy grace must be finite and nonnegative")
    if grace == 0:
        return min(seed_deadline, group_deadline)
    return min(seed_deadline, max(group_deadline, now + grace))


def run_targeted(pd, tree, archive, policy, *, budget, reserve, progress,
                 max_targets=0, per_root_limit=None, policy_grace_s=0.,
                 portfolio=False, target_visits=None, pricing_method=None,
                 target_keys=None):
    from cuts.benders_cuts import add_unique_cut
    from s2backward.phase15 import refresh_stage1_bound
    from s2backward.routeopt.root_lp import solve_routeopt_root_lp
    from s2backward.routeopt.seed import build_routeopt_eta_cut, _EXECUTABLE, _UNSUPPORTED
    from gurobipy import GRB, GurobiError
    from s2backward.routeopt.route_priors import build_route_priors
    from s2backward.routeopt.restricted_master import RoutePolicyImprover, _certify_node

    started = time.monotonic()
    if not math.isfinite(budget) or budget <= 0 or not 0 <= reserve <= budget:
        raise ValueError("invalid targeted physical budget or reserve")
    deadline = started + budget
    seed_deadline = deadline - min(reserve, budget)
    policy_deadline(seed_deadline, seed_deadline, started, policy_grace_s)
    report = dict(backend="routeopt_current_fleet", requested_seconds=budget,
                  master_reserve_seconds=reserve, solves=0, added=0, bounds=[],
                  route_priors=[], master=None, status="complete",
                  policy_grace_s=policy_grace_s, failures=[])
    if isinstance(max_targets, bool) or not isinstance(max_targets, int) or max_targets < 0:
        raise ValueError("max_targets must be a nonnegative integer")
    if per_root_limit is not None and (not math.isfinite(per_root_limit) or per_root_limit <= 0):
        raise ValueError("per_root_limit must be finite and positive")
    if pricing_method is not None and pricing_method not in ("current", "residual"):
        raise ValueError("pricing_method must be None, 'current', or 'residual'")
    if target_keys is not None:
        try:
            target_keys = {_checked_target_key(value) for value in target_keys}
        except TypeError as exc:
            raise ValueError("target_keys must be an iterable of target keys") from exc
    if not _EXECUTABLE.is_file() or not os.access(_EXECUTABLE, os.X_OK):
        report.update(status='backend_unavailable', seconds=time.monotonic()-started)
        report['failures'].append(dict(node=None, reason='RouteOpt pricing executable unavailable'))
        return report
    try:
        targets = current_fleet_targets(pd, tree, policy, archive)
    except PhysicalTargetsUnavailable as exc:
        report.update(status='unsupported_physical_domain', failures=exc.failures,
                      seconds=time.monotonic()-started)
        return report
    # Visits belong to this solve only. Domain and fleet flags, not node rank
    # alone, identify a target; priority never supplies a lower bound.
    report["available_target_count"] = len(targets)
    if target_keys is not None:
        targets = [target for target in targets if physical_target_key(target) in target_keys]
    report["matched_target_count"] = len(targets)
    if target_visits is not None:
        targets.sort(key=lambda item: (target_visits.get(physical_target_key(item), 0),
                                       -item[3], item[1][0].index))
    if max_targets:
        targets = targets[:max_targets]
    report["target_count"] = len(targets)
    report["max_targets"] = max_targets
    report["per_root_limit"] = per_root_limit
    entries = []
    for target in targets:
        visits = (target_visits or {}).get(physical_target_key(target), 0)
        method = pricing_method or ("residual" if portfolio and visits % 2
                                    and not all(target[2].values()) else "current")
        entries.append((target, method))
    report["portfolio"] = portfolio
    report["pricing_method"] = pricing_method
    report["target_nodes"] = [int(target[1][0].index) for target in targets]
    if not entries:
        report.update(status="no_targets", seed_seconds=time.monotonic() - started,
                      seconds=time.monotonic() - started, deadline_exhausted=False,
                      policy_candidate=None, policy_improvement={})
        return report
    visited = set()
    owner = RoutePolicyImprover(pd, tree, policy)
    for position, (target, method) in enumerate(entries):
        members, exact_targets, flags, priority = target
        target_key = physical_target_key(target)
        remaining = seed_deadline - time.monotonic()
        if remaining <= 0:
            report["status"] = "time_limit"
            break
        node = exact_targets[0]
        group_deadline = min(seed_deadline, time.monotonic() + remaining / (len(entries) - position))
        consumer = owner.start_group(node)
        remaining = group_deadline - time.monotonic()
        primal_reserve = min(1., max(0., remaining / 3.))
        cap = per_root_limit if per_root_limit is not None else (5. if all(flags.values()) else 8.)
        requested = min(cap, remaining - primal_reserve)
        if requested <= 0:
            report["status"] = "time_limit"
            break
        called = time.monotonic()
        full = all(flags.values())
        solve = solve_routeopt_root_lp
        if method == "residual":
            from .routeopt.residual_outsourcing_dual import make_residual_outsourcing_solver
            solve = make_residual_outsourcing_solver(solve)
        try:
            result = solve(
                pd, node, flags, time_limit_s=requested, threads=1,
                ng_size=8 if full else 4, relaxed_columns=True,
                package_cover=not full, route_consumer=consumer)
        except (ValueError, RuntimeError, GurobiError, OSError) as exc:
            unsupported = isinstance(exc, ValueError) and str(exc) in _UNSUPPORTED
            solver_missing = isinstance(exc, GurobiError) and exc.errno in (
                GRB.Error.NO_LICENSE, GRB.Error.SIZE_LIMIT_EXCEEDED)
            backend_missing = (
                isinstance(exc, RuntimeError) and str(exc) == 'Build the adapter with routeopt/build.py first'
            ) or (isinstance(exc, OSError) and exc.errno in (2, 8, 13)
                  and Path(exc.filename or '') == _EXECUTABLE)
            if not (unsupported or solver_missing or backend_missing):
                raise
            report['failures'].append(dict(node=int(node.index), method=method, reason=str(exc)))
            if target_visits is not None:
                target_visits[target_key] = target_visits.get(target_key, 0) + 1
            if unsupported:
                continue
            report['status'] = 'solver_unavailable' if solver_missing else 'backend_unavailable'
            break
        report["solves"] += 1
        if target_visits is not None and target_key not in visited:
            target_visits[target_key] = target_visits.get(target_key, 0) + 1
            visited.add(target_key)
        _, _, certified_upper = _certify_node(pd, tree, node.index, policy[1][0], policy)
        if (result.get("lb_certified") is True and result.get("lb") is not None
                and Fraction(result["lb"]) > Fraction(certified_upper)):
            raise RuntimeError("physical anchor LB exceeds independently certified policy")
        cut = build_routeopt_eta_cut(pd, node, result, anchor_flags=flags)
        added = 0
        if cut is not None:
            anchor = Fraction(cut[1]) + sum((Fraction(cut[0].get(
                f"z[{v},{node.info[1]}]", 0.)) * Fraction(flag)
                for v, flag in flags.items()), Fraction())
            if anchor > Fraction(certified_upper):
                raise RuntimeError("physical eta cut exceeds independently certified policy")
            for member in members:
                copied = build_routeopt_eta_cut(pd, member, result, anchor_flags=flags)
                if copied is None:
                    raise RuntimeError("physical certificate failed for exact domain copy")
                added += int(add_unique_cut(archive.setdefault(2, {}).setdefault(member.index, []), *copied))
            report["route_priors"].extend(build_route_priors(
                pd, tree, members, result, anchor_flags=flags))
        policy_started = time.monotonic()
        policy_until = policy_deadline(seed_deadline, group_deadline,
                                       policy_started, policy_grace_s)
        owner.improve_group(exact_targets, policy_until)
        report["added"] += added
        row = dict(node=int(node.index), members=[int(n.index) for n in members],
                   method=method,
                   policy_members=[int(n.index) for n in exact_targets], flags=flags,
                   priority=float(priority), full_fleet=full, ng_size=8 if full else 4,
                   package_cover=not full, requested_seconds=requested,
                   seconds=time.monotonic() - called, accepted=cut is not None, added=added,
                   lb=result.get("lb"), certificate_source=result.get("certificate_source"),
                   lb_certified=result.get("lb_certified") is True,
                   physical_optimality_proven=result.get("physical_optimality_proven") is True,
                   status=result.get("status"), **_pricing_metadata(result),
                   policy_budget_s=max(0., policy_until - policy_started),
                   policy_borrowed_s=max(0., policy_until - group_deadline))
        report["bounds"].append(row)
        progress(dict(status="running", completed=position + 1, targets=len(entries),
                      latest=row, elapsed=time.monotonic() - started,
                      policy_improvement=owner.stats))
    report["seed_seconds"] = time.monotonic() - started
    report["policy_candidate"] = owner.report_candidate()
    report["policy_improvement"] = dict(owner.stats)
    remaining = deadline - time.monotonic()
    if report["added"] and remaining > 0:
        report["master"] = refresh_stage1_bound(pd, tree, archive, time_limit_s=remaining)
    report["seconds"] = time.monotonic() - started
    report["deadline_exhausted"] = report["seconds"] >= budget
    return report
