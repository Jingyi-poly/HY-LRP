"""Typed contracts for facility pricing and joint physical route CG."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from models.stage_model_core import AffineCut
from solvers.lrp_physical_types import AuditedRoute


@dataclass(frozen=True)
class PricingResult:
    domain_signature: str
    facility_id: int
    price_vector_hash: str
    objective_kind: str
    certificate_domain: str
    status: str
    raw_lower: float | None
    safe_lower: float | None
    incumbent_route: AuditedRoute | None
    incumbent_original_cost: float | None
    incumbent_price_value: float | None
    includes_idle_candidate: bool
    certificate_source: str
    error_accounting: dict[str, Any] = field(default_factory=dict)
    elapsed_build: float = 0.
    elapsed_solve: float = 0.
    elapsed_audit: float = 0.
    additional_routes: tuple[AuditedRoute, ...] = ()

    def __post_init__(self):
        if self.objective_kind not in {"NONEMPTY", "WITH_IDLE"}:
            raise ValueError("invalid pricing objective kind")
        if self.certificate_domain not in {"EXACT_PHYSICAL", "PROVEN_SUPERSET", "HEURISTIC_ONLY"}:
            raise ValueError("invalid pricing certificate domain")
        if self.includes_idle_candidate != (self.objective_kind == "WITH_IDLE"):
            raise ValueError("idle flag and objective kind disagree")
        for value in (self.raw_lower, self.safe_lower, self.incumbent_original_cost,
                      self.incumbent_price_value):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("pricing values must be finite or None")
        if self.safe_lower is not None and self.certificate_domain == "HEURISTIC_ONLY":
            raise ValueError("heuristic pricing cannot supply a certified lower bound")
        if (self.safe_lower is not None and self.incumbent_price_value is not None
                and self.safe_lower > self.incumbent_price_value +
                2e-6 + 1e-10 * max(1., abs(self.safe_lower), abs(self.incumbent_price_value))):
            raise ValueError("pricing lower bound exceeds audited incumbent")


@dataclass(frozen=True)
class PhysicalPriceCertificate:
    node_id: tuple[int, int]
    domain_signature: str
    lambda_vector: tuple[float, ...]
    lambda_hash: str
    facility_coefficients: tuple[float, ...]
    evidence_ids: tuple[str, ...]
    fallback_flags: tuple[bool, ...]
    eta_cut: AffineCut
    theta_cuts: tuple[AffineCut, ...]
    anchor_mask: tuple[int, ...]
    value_at_anchor: float
    envelope_version: int

    def __post_init__(self):
        m = len(self.anchor_mask)
        if len(self.facility_coefficients) != m or len(self.fallback_flags) != m:
            raise ValueError("eta certificate must contain every facility coefficient")
        if any(v not in (0, 1) for v in self.anchor_mask):
            raise ValueError("invalid eta anchor mask")
        if any(v > 1e-12 for v in self.facility_coefficients):
            raise ValueError("eta availability coefficients must be nonpositive")


@dataclass(frozen=True)
class JointResult:
    node_id: tuple[int, int]
    facility_mask: tuple[int, ...]
    domain_signature: str
    status: str
    route_lp_lower: float | None
    rmp_lp_upper: float | None
    lp_certificate_gap: float | None
    combined_node_lower: float | None
    audited_node_upper: float | None
    audited_node_policy: object | None
    eta_certificates: tuple[PhysicalPriceCertificate, ...]
    theta_certificates: tuple[AffineCut, ...]
    added_route_ids: tuple[str, ...]
    pricing_status_counts: dict[str, int]
    fallback_facilities: tuple[int, ...]
    timings: dict[str, float]
    stop_reason: str
    round_trace: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        for value in (self.route_lp_lower, self.rmp_lp_upper,
                      self.lp_certificate_gap, self.combined_node_lower,
                      self.audited_node_upper):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("joint bounds must be finite or None")
        if (self.combined_node_lower is not None and self.audited_node_upper is not None
                and self.combined_node_lower > self.audited_node_upper +
                2e-6 + 1e-10 * max(1., abs(self.combined_node_lower), abs(self.audited_node_upper))):
            raise ValueError("joint lower bound exceeds audited upper bound")
