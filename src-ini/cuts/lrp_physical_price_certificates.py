"""Convert complete facility-pricing evidence into eta/theta archive rows."""
from __future__ import annotations

from fractions import Fraction as F
import math

from cuts.benders_cuts import add_unique_cut
from cuts.lrp_physical_bridge import (_exact_value, _paid_cut, _ratio, _state,
                                      physical_archive_value)
from models.stage_builder import _as_cut, _instance, _node_context, _route_pools, _state_keys
from solvers.lrp_facility_pricing_adapter import price_hash
from solvers.lrp_physical_cg_types import PhysicalPriceCertificate, PricingResult
from solvers.lrp_physical_types import node_signature


PRICE_SCHEMA = "lrp_physical_route_cg_v2"


def build_price_certificate(prob_data, node, A_mask, lambda_vector, pricing_results,
                            *, envelope_version):
    data = _instance(prob_data); ctx = _node_context(data, node, stage=2)
    lam = tuple(float(v) if ctx.active[j] else 0. for j, v in enumerate(lambda_vector))
    if len(lam) != ctx.n or any(not math.isfinite(v) for v in lam):
        raise ValueError("invalid complete customer price vector")
    for j in range(ctx.n):
        if lam[j] > float(ctx.outsourcing[j]) + 1e-9:
            raise ValueError("customer price exceeds outsourcing dual cap")
    by_i = {r.facility_id: r for r in pricing_results}
    if set(by_i) != set(range(ctx.m)):
        raise ValueError("one compatible pricing result is required for every facility")
    h = price_hash(lam); coeffs=[]; flags=[]; evidence=[]; theta=[]
    for i in range(ctx.m):
        result = by_i[i]
        if result.domain_signature != node_signature(data, node) or result.price_vector_hash != h:
            raise ValueError("pricing domains/prices cannot be mixed in one certificate")
        if result.safe_lower is None:
            raise ValueError("missing facility coefficient cannot default to zero")
        lower = F(float(result.safe_lower))
        b = min(F(), lower)
        coeffs.append(b); flags.append(result.certificate_source in
                                      {"nonnegative_cost_fallback","capacity_incoming_analytic"}
                                      or result.certificate_source.startswith("eq4_transported_"))
        evidence.append(f"{i}:{result.certificate_source}:{result.status}")
        # WITH_IDLE only proves the weaker equation (11); NONEMPTY preserves
        # a positive intercept when that stronger contract is available.
        route_intercept = lower if result.objective_kind == "NONEMPTY" else b
        meta = dict(schema=PRICE_SCHEMA, source="physical_price_theta",
            kind="THETA_NONEMPTY" if result.objective_kind == "NONEMPTY" else "THETA_IDLE",
            node_signature=node_signature(data,node), node_index=int(node.index), facility=i,
            envelope_version=int(envelope_version), lambda_hash=h,
            lambda_vector=lam, pricing_evidence_ids=(evidence[-1],),
            fallback_or_transport_flags=(flags[-1],), original_units=True,
            numerical_guard="binary64_fraction_fullbox_down", validity_scope="complete_active_route_domain")
        theta.append(_paid_cut("route", ctx.route_key(i),
            tuple(F(v) for v in lam)+(route_intercept,), F(), meta))
    intercept = sum((F(lam[j]) for j in range(ctx.n) if ctx.active[j]), F())
    eta_meta = dict(schema=PRICE_SCHEMA, source="physical_price_eta", kind="ETA_PHYSICAL",
        node_signature=node_signature(data,node), node_index=int(node.index),
        envelope_version=int(envelope_version), lambda_hash=h, lambda_vector=lam,
        all_facility_coeffs=tuple(float(v) for v in coeffs),
        pricing_evidence_ids=tuple(evidence), fallback_or_transport_flags=tuple(flags),
        original_units=True, numerical_guard="binary64_fraction_fullbox_down",
        generation_anchor=tuple(A_mask), validity_scope="availability_box")
    eta = _paid_cut("node", ctx.key, tuple(coeffs), intercept, eta_meta)
    value = _exact_value(eta, tuple(A_mask))
    return PhysicalPriceCertificate((ctx.period,ctx.scenario),node_signature(data,node),lam,h,
        tuple(float(v) for v in coeffs),tuple(evidence),tuple(flags),eta,tuple(theta),
        tuple(int(v) for v in A_mask),float(value),int(envelope_version))


def install_price_cut(prob_data, node, archive, cut, *, envelope_version, expected_version,
                      candidate_states=()):
    """Install a V2 price row after scope validation; do not discard off-anchor rows."""
    ctx = _node_context(_instance(prob_data),node,stage=2)
    if cut.certificate.get("schema") != PRICE_SCHEMA:
        raise ValueError("not a physical price certificate")
    meta=cut.certificate
    if (meta.get("node_signature") != node_signature(prob_data,node)
            or meta.get("node_index") != int(node.index)
            or meta.get("envelope_version") != int(expected_version)
            or int(envelope_version) < int(expected_version)):
        raise ValueError("price cut scope/version mismatch")
    if cut.level == "node":
        if meta.get("kind") != "ETA_PHYSICAL" or len(meta.get("all_facility_coeffs",())) != ctx.m:
            raise ValueError("eta cut lacks all facility coefficients")
        i=None; stage,target=2,node.index
    elif cut.level == "route":
        i=int(meta.get("facility")); _route_pools(ctx,node,archive)
        if meta.get("kind") not in {"THETA_NONEMPTY","THETA_IDLE"}:
            raise ValueError("invalid theta price certificate")
        _,mapping=_route_pools(ctx,node,archive); stage,target=3,mapping[i]
    else:
        raise ValueError("invalid physical price cut level")
    _as_cut(cut,ctx,i)
    gains=[]
    for state in candidate_states:
        values=_state(ctx,i,state)
        gains.append(float(_exact_value(cut,values)-physical_archive_value(
            prob_data,node,archive,values,facility_id=i)))
    pi={key:a for key,a in zip(_state_keys(ctx,i),cut.coefficients) if a}
    installed=add_unique_cut(archive.setdefault(stage,{}).setdefault(target,[]),pi,cut.intercept)
    return dict(installed=bool(installed),stage=stage,node=target,parent_node=node.index,facility=i,
        version_before=int(envelope_version),version_after=int(envelope_version)+int(bool(installed)),
        affected_s2_nodes=[node.index] if installed and i is not None else [],
        requires_same_A_reprice=bool(installed and i is not None), source=meta["source"],
        reason="INSTALLED" if installed else "DUPLICATE",
        maximum_candidate_gain=max(gains) if gains else None)
