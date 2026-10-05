"""Use a physical route-pricing certificate in each private Stage-3 pool.

If c_v(R) - sum_j u_j a_j(R) >= beta_v for every feasible route and
beta_v <= 0, then theta_v >= sum_j u_j alpha[j,v] + beta_v y[v].
The row has zero intercept, including for an empty or inactive vehicle.
"""
from __future__ import annotations

from numbers import Integral

from cuts.benders_cuts import add_unique_cut
from .seed import _binary64, build_routeopt_eta_cut
from .physical_identity import physical_fingerprint


def _index(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("route-prior node/vehicle IDs must be integers")
    return int(value)


def _node(tree, stage, index):
    try:
        node = tree[stage][index]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("missing route-prior scenario node") from exc
    if _index(node.index) != index:
        raise ValueError("route-prior node index disagrees with tree placement")
    return node


def build_route_priors(pd, tree, members, result, *, anchor_flags=None):
    """Return immutable ``(target_ids, slope_items)`` rows from one certificate.

    ``members`` must be one exact physical period-dedup group in this solve.
    No pricing is repeated and no archive is changed. The caller keeps this
    small pending tuple locally until the next backward pass; it is not a
    cross-instance cache. Invalid certificates return (), while inconsistent
    scenario mappings raise ValueError before any row can be applied.
    """
    members = tuple(members)
    if not members:
        return ()
    source = members[0]
    options = {} if anchor_flags is None else {"anchor_flags": anchor_flags}
    if build_routeopt_eta_cut(pd, source, result, **options) is None:
        return ()
    vehicles = tuple(_index(v) for v in pd.V)
    member_ids = tuple(_index(node.index) for node in members)
    if len(set(member_ids)) != len(member_ids):
        raise ValueError("duplicate route-prior source node")
    source_data = tuple((source.active[j], _binary64(source.volume[j]),
                         _binary64(source.c_out[j])) for j in pd.J)
    ordinary_identity = result.get("physical_fingerprint")
    targets = {v: [] for v in vehicles}
    seen = set()
    for member, member_id in zip(members, member_ids):
        if _node(tree, 2, member_id) is not member:
            raise ValueError("route-prior source is not the current tree node")
        data = tuple((member.active[j], _binary64(member.volume[j]),
                      _binary64(member.c_out[j])) for j in pd.J)
        if data != source_data:
            raise ValueError("route-prior physical period-copy data differ")
        if ordinary_identity is not None and physical_fingerprint(pd, member) != ordinary_identity:
            raise ValueError("route-prior physical certificate domain differs")
        mapped = set()
        for third_id in member.successor:
            third_id = _index(third_id)
            child = _node(tree, 3, third_id)
            vehicle = _index(child.info)
            if vehicle not in targets or vehicle in mapped or third_id in seen:
                raise ValueError("route-prior vehicle mapping is not a bijection")
            if _index(child.predecessor) != member_id:
                raise ValueError("route-prior predecessor mismatch")
            if any(child.active[j] != source.active[j] or
                   _binary64(child.volume[j]) != _binary64(source.volume[j])
                   for j in pd.J):
                raise ValueError("Stage-3 domain differs from physical pricing")
            mapped.add(vehicle)
            seen.add(third_id)
            targets[vehicle].append(third_id)
        if mapped != set(vehicles):
            raise ValueError("missing route-prior vehicle successor")
    jobs = tuple(j for j in pd.J if source.active[j] == 1)
    prices = tuple((j, _binary64(u)) for j, u in zip(jobs, result["dual_u"]))
    bounds = {v: _binary64(beta)
              for group, beta in zip(result["groups"], result["dual_beta"])
              for v in group["vehicles"]}
    rows = []
    for vehicle in vehicles:
        slope = tuple((f"alpha[{j},{vehicle}]", u) for j, u in prices if u != 0.)
        if bounds[vehicle] != 0.:
            slope += ((f"y[{vehicle}]", bounds[vehicle]),)
        rows.append((tuple(targets[vehicle]), slope))
    return tuple(rows)


def apply_route_priors(cut_lag, priors):
    """Insert same-solve exported rows without cleaning their coefficients.

    Only private Stage-3 archives are changed. Clearing the local pending rows
    remains the caller's responsibility; repeated application is idempotent.
    """
    return sum(add_unique_cut(cut_lag.setdefault(3, {}).setdefault(third, []),
                              dict(slope), 0.)
               for targets, slope in priors for third in targets)


__all__ = ["build_route_priors", "apply_route_priors"]
