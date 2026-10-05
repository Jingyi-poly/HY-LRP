"""Bounded pure-data persistence of complete, original physical price evidence.

No optimizer is called.  Native lower bounds retain the existing adapter's
complete-domain contract; this is not an independent proof of the native solver.
Equation (3) fallbacks and every optional primal route are independently checked.
Transported results are deliberately excluded: no missing proof chain can be
laundered into a new native certificate by checkpointing it.

One best price point is retained for one full instance/node/availability mask.
Physical evidence survives S3 archive changes; certificates are rebuilt with
the receiving archive version.  The caller owns all deadline checks, including
checks before and after these validation/audit operations.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
import math
import time

from cuts.lrp_physical_bridge import _exact_value
from cuts.lrp_physical_price_certificates import build_price_certificate
from models.stage_builder import _instance, _node_context
from solvers.lrp_facility_pricing_adapter import price_hash, trivial_idle_lower
from solvers.lrp_physical_cg_types import PricingResult
from solvers.lrp_physical_policy_pool import audit_route
from solvers.lrp_physical_types import node_signature


PRICE_STATE_SCHEMA = "lrp_cg_original_price_state_v1"
_NATIVE_SOURCE = "lrp_native_complete_idle_pricing"
_FALLBACK_SOURCE = "nonnegative_cost_fallback"


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative Python integer")
    return value


def _number(value, label, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} must be a finite real number")
    try:
        answer = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(answer):
        raise ValueError(f"{label} must be a finite real number")
    return answer


def _mask(values, count):
    values = tuple(values)
    if len(values) != count or any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError("availability mask must contain one exact Python binary integer per facility")
    return values


def _rebuild(data, node, mask, prices, results, envelope_version):
    """Audit an entire same-price batch before producing any persistent row."""
    _integer(envelope_version, "envelope_version")
    ctx = _node_context(data, node, stage=2)
    lam = tuple(_number(v, "customer price") for v in prices)
    if len(lam) != ctx.n:
        raise ValueError("price vector has the wrong customer dimension")
    if any((not ctx.active[j] and v != 0.) or v > float(ctx.outsourcing[j])
           for j, v in enumerate(lam)):
        raise ValueError("prices must have inactive zeroes and satisfy exact outsourcing caps")
    # Copy nested diagnostic dictionaries as well as immutable dataclass shells.
    # Callers cannot mutate the cache's evidence through their original result.
    results = deepcopy(tuple(results))
    if len(results) != ctx.m or any(not isinstance(r, PricingResult) for r in results):
        raise ValueError("one original PricingResult per physical facility is required")
    ids = [r.facility_id for r in results]
    if any(type(i) is not int for i in ids) or set(ids) != set(range(ctx.m)):
        raise ValueError("facility IDs must be unique exact integers covering every facility")
    signature, price_signature = node_signature(data, node), price_hash(lam)
    for result in results:
        # Dataclass validation is not automatically rerun after unpickling.
        result.__post_init__()
        if (result.domain_signature != signature or result.price_vector_hash != price_signature
                or result.objective_kind != "WITH_IDLE"
                or result.includes_idle_candidate is not True
                or result.certificate_domain != "EXACT_PHYSICAL"):
            raise ValueError("incompatible original full-domain idle-inclusive pricing evidence")
        lower = _number(result.safe_lower, "pricing safe lower")
        raw = _number(result.raw_lower, "pricing raw lower", optional=True)
        incumbent = _number(result.incumbent_price_value, "pricing incumbent", optional=True)
        original_cost = _number(result.incumbent_original_cost, "original route cost", optional=True)
        if lower > 0.:
            raise ValueError("idle-inclusive lower bound must not exceed zero")
        for field in ("elapsed_build", "elapsed_solve", "elapsed_audit"):
            if _number(getattr(result, field), field) < 0.:
                raise ValueError("pricing elapsed times must be nonnegative")
        if not isinstance(result.error_accounting, dict):
            raise ValueError("pricing error accounting must be a dictionary")
        meta = result.error_accounting
        if meta.get("transported") or any(str(k).startswith("transport") for k in meta):
            raise ValueError("transported evidence cannot enter the original-price cache")
        if result.certificate_source == _FALLBACK_SOURCE:
            if meta.get("formula") != "eq3" or F(lower) > trivial_idle_lower(ctx.active, lam):
                raise ValueError("equation (3) fallback is not downward safe")
        elif result.certificate_source == 'capacity_incoming_analytic':
            from solvers.lrp_capacity_price_bound import capacity_price_lower_bound, FORMULA
            analytic = capacity_price_lower_bound(ctx,result.facility_id,lam)
            if meta.get('formula') != FORMULA or F(lower) > F(*analytic['lower_exact']):
                raise ValueError('capacity/incoming analytic pricing bound is not downward safe')
        elif result.certificate_source == _NATIVE_SOURCE:
            analytic_idle = (meta.get("native_executed") is False
                and all(v <= 0. for v in lam) and result.status == "OPTIMAL"
                and meta.get("exact") is True and raw == 0.
                and F(lower) <= trivial_idle_lower(ctx.active, lam))
            if ((meta.get("native_executed") is not True and not analytic_idle)
                    or meta.get("fallback") or meta.get("formula") != "native_bound"
                    or result.status not in {"OPTIMAL", "LIMIT"}
                    or meta.get("native_status") != result.status
                    or raw is None or F(lower) > F(raw)):
                raise ValueError("native evidence does not match the original adapter contract")
        else:
            raise ValueError("only original native or independently checked analytic evidence may be persisted")
        if result.incumbent_route is not None:
            route = result.incumbent_route
            fresh = audit_route(data, node, result.facility_id, route.customers_in_order,
                                source=route.source, generation_id=route.generation_id)
            if fresh != route or original_cost != fresh.cost:
                raise ValueError("pricing route fails original-array audit")
            exact = F(*fresh.cost_exact) - sum((F(lam[j]) for j in fresh.customers_in_order), F())
            if F(lower) > exact:
                raise ValueError("pricing lower exceeds exact audited route objective")
            if incumbent is None or abs(float(exact)-incumbent) > 2e-6 + 1e-10*max(1., abs(float(exact))):
                raise ValueError("pricing incumbent does not match the audited route and prices")
        elif original_cost is not None or (incumbent is not None and incumbent != 0.):
            raise ValueError("nonidle pricing incumbent requires its original audited route")
    # Additional primal columns have a separate audited route-pool owner.
    # They are not part of the scalar pricing proof and must not be replayed
    # as implicitly trusted witnesses from a persisted evidence record.
    results = tuple(replace(r, additional_routes=())
                    for r in sorted(results, key=lambda r: r.facility_id))
    certificate = build_price_certificate(data, node, mask, lam, results,
                                          envelope_version=envelope_version)
    # Preserve the existing certificate byte-for-byte. Its display value uses
    # nearest rounding; consumers must floor _exact_value(eta_cut, A) for an LB.
    # Selection below uses exact rational comparisons, never that scalar.
    return certificate, results


class PhysicalCGPriceState:
    """One validated best complete price point, plus completed-round count.

    ``consider`` increments ``rounds`` once after each fully validated batch;
    the caller must not increment it again.  Invalid batches change nothing.
    A lower-scoring point is returned for optional cut installation, but does
    not replace the persistent center.  Comparison uses exact row evaluation.
    """

    def __init__(self, data, node, A):
        data = _instance(data)
        ctx = _node_context(data, node, stage=2)
        self.schema = PRICE_STATE_SCHEMA
        self.instance_sha256 = data.logical_hash()
        self.domain_signature = node_signature(data, node)
        self.node_index = _integer(node.index, "S2 node index")
        self.node_id = (ctx.period, ctx.scenario)
        self.anchor_mask = _mask(A, ctx.m)
        self.rounds = 0
        self._best = None
        self._validated = True

    def _scope(self, data, node, A):
        data = _instance(data)
        ctx = _node_context(data, node, stage=2)
        mask = _mask(A, ctx.m)
        if (self.schema != PRICE_STATE_SCHEMA or self.instance_sha256 != data.logical_hash()
                or self.domain_signature != node_signature(data, node)
                or type(self.node_index) is not int
                or self.node_index != _integer(node.index, "S2 node index")
                or not isinstance(self.node_id, tuple)
                or any(type(v) is not int for v in self.node_id)
                or self.node_id != (ctx.period, ctx.scenario)
                or _mask(self.anchor_mask, ctx.m) != mask):
            raise ValueError("persistent price state full-data/node/availability scope mismatch")
        _integer(self.rounds, "rounds")
        return data, mask

    def _read_best(self, data, node, mask, envelope_version):
        if self._best is None:
            return None
        if not isinstance(self._best, dict) or set(self._best) != {"lambda_vector", "raw_results", "generation_version"}:
            raise ValueError("invalid original price evidence payload")
        _integer(self._best["generation_version"], "original generation version")
        return _rebuild(data, node, mask, self._best["lambda_vector"],
                        self._best["raw_results"], envelope_version)

    def validate(self, data, node, A):
        """Revalidate scope and original proof; safe after plain pickle load."""
        data, mask = self._scope(data, node, A)
        self._read_best(data, node, mask, 0)
        self._validated = True
        return self

    @property
    def price_center(self):
        if not self._validated:
            raise ValueError("restored price state must be validated before reading its center")
        return None if self._best is None else tuple(self._best["lambda_vector"])

    def consider(self, data, node, A, lambda_vector, raw_results, envelope_version, *, deadline=None):
        """Return (fresh certificate, improved); commit only before the deadline.

        An already expired call returns ``(None, False)`` without any scope or
        proof audit. An audit that finishes late returns its candidate for
        diagnostics, but neither the best evidence nor ``rounds`` is updated.
        The caller must also reject that late candidate for cut installation.
        """
        if deadline is not None:
            deadline = _number(deadline, "deadline")
            if time.monotonic() >= deadline:
                return None, False
        data, mask = self._scope(data, node, A)
        candidate, results = _rebuild(data, node, mask, lambda_vector, raw_results, envelope_version)
        previous = self._read_best(data, node, mask, envelope_version)
        improved = previous is None or _exact_value(candidate.eta_cut, mask) > _exact_value(previous[0].eta_cut, mask)
        if previous is not None and not improved:
            old=previous[0]
            # Investment also retains equal-anchor slope improvements. A
            # closed facility's stronger coefficient matters when it opens.
            improved=(candidate.lambda_vector==old.lambda_vector
                and candidate.eta_cut.intercept==old.eta_cut.intercept
                and all(a>=b for a,b in zip(candidate.eta_cut.coefficients,old.eta_cut.coefficients))
                and any(a>b for a,b in zip(candidate.eta_cut.coefficients,old.eta_cut.coefficients)))
        if deadline is not None and time.monotonic() >= deadline:
            return candidate, False
        if improved:
            self._best = dict(lambda_vector=candidate.lambda_vector,
                              raw_results=results, generation_version=envelope_version)
        self.rounds += 1
        self._validated = True
        return candidate, improved

    def best(self, data, node, A, envelope_version):
        """Rebuild the physical proof; floor its exact row value when using an LB."""
        _integer(envelope_version, "envelope_version")
        data, mask = self._scope(data, node, A)
        result = self._read_best(data, node, mask, envelope_version)
        self._validated = True
        return result

    def export(self):
        """Portable data only: no model, cached scalar or trusted cut object."""
        return deepcopy(dict(schema=self.schema, instance_sha256=self.instance_sha256,
            domain_signature=self.domain_signature, node_index=self.node_index,
            node_id=self.node_id, anchor_mask=self.anchor_mask, rounds=self.rounds,
            best_original_evidence=self._best))

    @classmethod
    def from_export(cls, data, node, A, payload):
        result = cls.__new__(cls)
        result.__setstate__(payload)
        return result.validate(data, node, A)

    def __getstate__(self):
        return self.export()

    def __setstate__(self, payload):
        expected = {"schema", "instance_sha256", "domain_signature", "node_index", "node_id",
                    "anchor_mask", "rounds", "best_original_evidence"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("invalid persistent price state export schema")
        value = deepcopy(payload)
        for key in expected - {"best_original_evidence"}:
            setattr(self, key, value[key])
        self._best = value["best_original_evidence"]
        # No data is available during unpickle. Every consuming API rechecks
        # original scope/evidence, and center access is blocked until it does.
        self._validated = False
