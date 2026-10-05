"""Level-Set oracle session backed by the fleet piece table.

One ``FleetPieceOracle`` serves one Stage-2 Level Set call (one node, one
fixed Stage-3 cut archive).  ``evaluate(pi)`` returns what the Level Set needs
from an oracle at ``pi``:

* ``outer_lb``  -- certified lower bound on ``D(pi)``:
  ``min_n [ lb(n) - credit(n, pi) ]`` rounded down, where ``lb(n) <= C(n)``;
* ``inner_value`` / ``inner_xcp`` -- a certified feasible point and its
  Lagrangian objective (rounded up), taken as the best stored policy with its
  optimal ``z`` for this ``pi``; this supports the bundle;
* ``exact`` -- whether the minimising piece is known to optimality, so the
  two coincide up to the checkpoint tolerance.

Pieces are refined lazily: when the minimising piece is not exact and there
is budget, its fixed-fleet model is solved (``piece_solver``) and the minimum
re-taken.  Every returned quantity is certified regardless of how much
refinement happened, so stopping early only weakens the cut.
"""

from __future__ import annotations

import math
import time
from fractions import Fraction
from typing import Callable, Dict, Mapping, Optional

from gurobipy import GRB

from core.exact_solver_log import log as exact_log
from core.backend_telemetry import backend_scope, record_backend_event
from core.stage2_tolerance import effective_abs_gap
from cuts import exact_subroutines as exact_sub

from s2forward.fleet_pieces import (
    Counts,
    FleetLayout,
    float_down,
    float_up,
    fraction_from_float,
)
from .piece_solver import certify_policy_dict, solve_piece
from .piece_table import FleetPieceTable, PolicyRecord, should_mark_stalled
from .direct_dp import DirectDPOracle
from .binary64_piece_values import Binary64PieceValues
from s2forward.subset_dp import (
    DEFAULT_MAX_CUSTOMERS,
    default_max_customers,
    subset_dp_solver_or_none,
)

BACKEND = "fleet_enum"
# Piece solver selection.  ``auto``: nodes with at most ``dp_only_max_customers``
# active customers are solved by the exact subset DP alone; larger nodes up to
# ``dp_max_customers`` use the hybrid below; beyond that Gurobi alone.  Both
# limits default (``None``) to the DP ceiling (20 with the C++ kernel, 16 with
# numpy), i.e. no hybrid band.  ``gurobi`` / ``dp`` force one backend.
PIECE_SOLVERS = ("auto", "gurobi", "dp")
DEFAULT_DP_ONLY_MAX_CUSTOMERS = DEFAULT_MAX_CUSTOMERS
# The free-fleet pass is useful for a one-shot oracle (Phase 1), while a
# multi-query Level-Set session is faster when it lazily closes fixed-fleet
# pieces and reuses them.  Callers therefore opt in explicitly.
DEFAULT_DIRECT_DP_MAX_CUSTOMERS = 0
# Hybrid (n > dp_only_max_customers): Gurobi runs first with a time limit of
# this fraction of the DP's predicted duration (it wins outright on loose
# pieces such as the full fleet, and ``BestBdStop`` exclusions usually end at
# the root); if it neither proves optimality nor the exclusion, the DP solves
# the piece exactly.  Worst case is therefore ~(1 + fraction) x DP time.
_HYBRID_GUROBI_FRACTION = 0.3
_HYBRID_GUROBI_MIN_SECONDS = 1.0


def _consistent_piece_lower_bound(lower_bound, *policy_upper_bounds):
    """Reject contradictory certificates without promoting a policy UB to LB.

    Search tolerances only control when to stop refining a valid interval.
    They cannot justify an inverted interval, even inside the numerical zero
    band. Missing policies do not invalidate an otherwise certified bound.
    """
    if lower_bound is None:
        return None
    try:
        lower = float(lower_bound)
        uppers = [float(upper) for upper in policy_upper_bounds if upper is not None]
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(lower) or any(not math.isfinite(upper) for upper in uppers):
        return None
    return None if any(lower > upper for upper in uppers) else lower


class FleetPieceOracle:
    def __init__(
        self,
        prob_data,
        node,
        cut_lag,
        table: FleetPieceTable,
        *,
        stage_builder_options: Optional[Mapping] = None,
        mip_gap: float = 1e-7,
        mip_gap_abs: Optional[float] = None,
        mip_gap_abs_rel_cap: Optional[float] = None,
        sub_time_limit: Optional[float] = 60.0,
        min_solve_time: float = 5.0,
        remaining_budget: Optional[Callable[[], Optional[float]]] = None,
        max_refinements_per_eval: Optional[int] = None,
        exact_abs_tol: float = 1e-6,
        exact_rel_tol: float = 0.0,
        binary_tolerance: float = 1e-5,
        node_tag: str = "",
        verbose: bool = True,
        piece_solver: str = "auto",
        dp_max_customers: Optional[int] = None,
        dp_only_max_customers: Optional[int] = None,
        direct_dp_max_customers: int = DEFAULT_DIRECT_DP_MAX_CUSTOMERS,
        archive_fingerprint: Optional[str] = None,
    ):
        if piece_solver not in PIECE_SOLVERS:
            raise ValueError(f"piece_solver={piece_solver!r}; expected one of {PIECE_SOLVERS}")
        self.piece_solver = piece_solver
        self.dp_only_max_customers = (
            default_max_customers() if dp_only_max_customers is None
            else int(dp_only_max_customers)
        )
        self.prob_data = prob_data
        self.node = node
        self.cut_lag = cut_lag
        self.table = table
        self.layout: FleetLayout = table.layout
        self.period = int(node.info[1])
        self.stage_builder_options = dict(stage_builder_options or {})
        self.mip_gap = float(mip_gap)
        # Scheduled absolute stop for Gurobi piece MIPs (None = MIPGap only);
        # ``exact_abs_tol`` should be at least as wide so a stopped piece
        # counts as exact instead of being refined again.
        self.mip_gap_abs = None if mip_gap_abs is None else float(mip_gap_abs)
        self.mip_gap_abs_rel_cap = (
            None if mip_gap_abs_rel_cap is None else float(mip_gap_abs_rel_cap)
        )
        self.sub_time_limit = sub_time_limit
        self.min_solve_time = float(min_solve_time)
        self.remaining_budget = remaining_budget
        self.max_refinements_per_eval = (
            None if max_refinements_per_eval is None else int(max_refinements_per_eval)
        )
        self.exact_abs_tol = float(exact_abs_tol)
        self.exact_rel_tol = float(exact_rel_tol)
        self.binary_tolerance = float(binary_tolerance)
        self.node_tag = node_tag
        self.verbose = verbose
        self.cuts_payload = exact_sub.build_s2_bp_cuts(
            prob_data, node, cut_lag,
            {successor: position for position, successor in enumerate(node.successor)},
        )
        # Policy cost under *this* archive, keyed by policy signature.
        self._policy_cost: Dict[tuple, float] = {}
        # Exact operating/outsource/theta objective at pi=0.  It is immutable
        # inside this oracle session because the Stage-3 archive is fixed.
        # New Level-Set multipliers therefore require only the affine
        # ``-pi*z`` update, not another full physical/cut certification.
        self._policy_base_exact: Dict[tuple, Fraction] = {}
        # For every purchased fleet n, the cheapest stored policy whose
        # required leading run r satisfies r <= n.  This is a multidimensional
        # prefix minimum over the fleet-count grid.  Rebuilding costs
        # O(number_of_pieces * number_of_types + number_of_policies), instead
        # of scanning every policy at every piece query.
        self._best_policy_prefix = None
        # Pieces whose time-limited Gurobi refinement moved the bound by less
        # than the exactness tolerance.  Their compact MIP has stalled (the
        # bound sits at the root relaxation); another capped solve in this
        # session would only burn the node budget, so the query answers from
        # the table's certified interval instead.  The set is shared through
        # the piece table with the forward/refresh solves of the *same*
        # archive (``archive_fingerprint``); a new archive tries again.
        if archive_fingerprint is not None:
            self.table.archive_fingerprint = str(archive_fingerprint)
        self._stalled: set = set(
            self.table.stalled_pieces(self.table.archive_fingerprint)
        )
        self.stats = {
            "evals": 0,
            "solves": 0,
            "solve_time": 0.0,
            "refinements": 0,
            "stalled_pieces": 0,
            "stalled_inherited": len(self._stalled),
            "exact_evals": 0,
            "dp_solves": 0,
            "direct_dp_evals": 0,
            "dp_prefix_cache_hits": 0,
            "dp_prefix_cache_misses": 0,
            "dp_prefix_cache_stores": 0,
            "dp_prefix_cache_evictions": 0,
            "dp_prefix_cache_bytes": 0,
            "dp_prefix_cache_budget_bytes": 0,
            "inner_affine_evals": 0,
            "inner_full_certifications": 0,
        }
        self.dp_solver = None
        if piece_solver in ("auto", "dp"):
            self.dp_solver = subset_dp_solver_or_none(
                prob_data, node, self.cuts_payload, self.layout,
                max_customers=(dp_max_customers if piece_solver == "auto" else 10**9),
                log=self._log if verbose else None,
            )
            if self.dp_solver is None and piece_solver == "dp":
                raise ValueError("piece_solver='dp' requested but the subset DP is not applicable")
        self.direct_dp = None
        if (
            piece_solver == "auto"
            and self.dp_solver is not None
            and int(direct_dp_max_customers) > 0
            and self.dp_solver.n <= int(direct_dp_max_customers)
        ):
            self.direct_dp = DirectDPOracle(
                prob_data,
                node,
                cut_lag,
                exact_abs_tol=self.exact_abs_tol,
                exact_rel_tol=self.exact_rel_tol,
                piece_solver=self.dp_solver,
            )
        self._ensure_all_outsource_policy()

    @property
    def piece_mode(self) -> str:
        """``dp`` | ``hybrid`` | ``gurobi`` for this node."""
        if self.direct_dp is not None:
            return "direct_dp"
        if self.dp_solver is None:
            return "gurobi"
        if self.piece_solver == "dp" or self.dp_solver.n <= self.dp_only_max_customers:
            return "dp"
        return "hybrid"

    # ------------------------------------------------------------ policies
    def _ensure_all_outsource_policy(self):
        zero = tuple(0 for _ in self.layout.groups)
        if any(p.counts == zero and not p.alpha for p in self.table.policies):
            return
        self._add_policy(
            PolicyRecord(counts=zero, alpha={}, y={}, source="all_outsource")
        )

    def _add_policy(self, policy: PolicyRecord) -> bool:
        """Add a policy and invalidate the exact prefix-min lookup if new."""
        added = self.table.add_policy(policy)
        if added:
            self._best_policy_prefix = None
        return added

    def seed_policy(self, x_dict: Mapping[str, float], source: str = "seed") -> Optional[PolicyRecord]:
        """Certify an external ``alpha``/``y`` point (e.g. the refreshed trial) and store it."""
        try:
            record, cost, certificate = certify_policy_dict(
                self.prob_data, self.node, self.cuts_payload, x_dict, self.layout,
                binary_tolerance=self.binary_tolerance, source=source,
                return_certificate=True,
            )
        except exact_sub.InvalidS2LagrangianPolicy as exc:
            self._log(f"seed policy rejected ({exc})")
            return None
        self._add_policy(record)
        self._remember_policy_certificate(record, cost, certificate)
        return record

    def _remember_policy_certificate(
        self, policy: PolicyRecord, cost: float, certificate: Mapping
    ) -> None:
        """Retain the exact pi=0 base produced by the full certifier."""
        signature = policy.signature()
        base_exact = certificate.get("objective_exact")
        if not isinstance(base_exact, Fraction):
            raise RuntimeError("Stage-2 certificate omitted its exact objective")
        self.table.assert_policy_upper_bound(
            policy.counts, float(cost), source=f"oracle policy {policy.source}"
        )
        self._policy_cost[signature] = float(cost)
        self._policy_base_exact[signature] = base_exact

    def policy_cost(self, policy: PolicyRecord) -> float:
        sig = policy.signature()
        cost = self._policy_cost.get(sig)
        if cost is None:
            _record, cost, certificate = certify_policy_dict(
                self.prob_data, self.node, self.cuts_payload, policy.as_x_dict(),
                self.layout, binary_tolerance=0.0, source=policy.source,
                return_certificate=True,
            )
            self._remember_policy_certificate(policy, cost, certificate)
        return cost

    def best_policy_for(self, counts: Counts):
        """Cheapest stored policy feasible for fleet ``counts`` (or None)."""
        counts = self.layout.check_counts(counts)
        if self._best_policy_prefix is None:
            self._build_best_policy_prefix()
        best = self._best_policy_prefix[counts]
        return (None, None) if best is None else (best[2], best[0])

    def _build_best_policy_prefix(self):
        """Build exact minima for ``required_counts <= purchased_counts``.

        A pair ``(cost, insertion_rank)`` is propagated so equal-cost ties
        retain the same first-insertion policy as the former linear scan.
        """
        best = {counts: None for counts in self.table.records}
        for rank, policy in enumerate(self.table.policies):
            item = (self.policy_cost(policy), rank, policy)
            current = best[policy.counts]
            if current is None or item[:2] < current[:2]:
                best[policy.counts] = item

        all_counts = tuple(best)
        for dimension, group in enumerate(self.layout.groups):
            for level in range(1, len(group) + 1):
                for counts in all_counts:
                    if counts[dimension] != level:
                        continue
                    previous = list(counts)
                    previous[dimension] -= 1
                    inherited = best[tuple(previous)]
                    current = best[counts]
                    if inherited is not None and (
                        current is None or inherited[:2] < current[:2]
                    ):
                        best[counts] = inherited
        self._best_policy_prefix = best

    # -------------------------------------------------------------- pieces
    def _exact_tol(self, scale: float) -> float:
        return self.exact_abs_tol + self.exact_rel_tol * max(1.0, abs(scale))

    def piece_is_exact(self, counts: Counts, lb: Optional[float] = None) -> bool:
        if lb is None:
            lb = self.table.lower_bound(counts)
        _policy, ub = self.best_policy_for(counts)
        if ub is None or lb > ub:
            return False
        return ub - lb <= self._exact_tol(ub)

    def closed_lower_bounds(self) -> Dict[Counts, float]:
        # A restored or newly merged table can contain bounds that were never
        # checked against this session's complete-policy costs. Fail before
        # exporting them into another oracle cut; old eta rows have no proof
        # dependency graph that would make silent rollback safe.
        for policy in self.table.policies:
            self.table.assert_policy_upper_bound(
                policy.counts, self.policy_cost(policy),
                source=f"oracle lower envelope / {policy.source}",
            )
        return {counts: self.table.lower_bound(counts) for counts in self.table.records}

    def outer_argmin(
        self,
        lbs: Mapping[Counts, float],
        piece_values: Binary64PieceValues,
    ):
        """Minimising piece, its value, and whether the minimum is certified.

        Ties are resolved in favour of a piece whose *feasible upper value*
        closes the global lower bound.  Checking only that its lower value is
        near the minimum and that the piece is individually exact would add
        two tolerances and could falsely call a gap of almost ``2 * tol``
        exact.

        Otherwise the *largest* tied fleet is the one to refine: ``C`` is
        non-increasing in the fleet, so its value is the lowest among the
        tied pieces and, through the monotone closure, one exact solve lifts
        the bound of every sub-fleet to it (typically settling the tie in one
        solve instead of one solve per tied piece).
        """
        values = piece_values.values
        best_scaled = min(values.values())
        best_value = piece_values.as_fraction(best_scaled)
        tol = Fraction(self._exact_tol(float(best_value)))
        tol_scaled = piece_values.tolerance_units(tol)
        candidates = [
            counts
            for counts, value in values.items()
            if value - best_scaled <= tol_scaled
        ]
        nonexact_candidates = []
        for counts in candidates:
            if not self.piece_is_exact(counts, lbs[counts]):
                nonexact_candidates.append(counts)
                continue
            _policy, upper_bound = self.best_policy_for(counts)
            upper_value = (
                fraction_from_float(upper_bound)
                - piece_values.credit_fraction(counts)
            )
            total_gap = upper_value - best_value
            if Fraction(0) <= total_gap <= tol:
                return counts, best_value, True

        # Do not spend another solve on a near-minimum piece that is already
        # individually exact but cannot close the *global* gap.  The true
        # lower-bound minimiser is necessarily among the remaining nonexact
        # candidates, unless the table contains inconsistent certificates.
        refinable = nonexact_candidates or candidates
        largest = max(refinable, key=lambda n: (sum(n), n))
        return largest, best_value, False

    def best_exact_value(
        self,
        lbs: Mapping[Counts, float],
        piece_values: Binary64PieceValues,
    ):
        """Smallest piece value among pieces known to optimality (or None)."""
        best_scaled = None
        for counts, lb in lbs.items():
            if not self.piece_is_exact(counts, lb):
                continue
            value = piece_values.values[counts]
            if best_scaled is None or value < best_scaled:
                best_scaled = value
        return (
            None
            if best_scaled is None
            else piece_values.as_fraction(best_scaled)
        )

    def _time_limit_for_solve(self) -> Optional[float]:
        limit = self.sub_time_limit
        if self.remaining_budget is not None:
            left = self.remaining_budget()
            if left is not None:
                if left <= 0.0:
                    return 0.0
                per = max(self.min_solve_time, left)
                limit = per if limit is None else min(float(limit), per)
        return limit

    def refine(self, counts: Counts, *, time_limit: Optional[float] = None,
               bound_stop: Optional[float] = None) -> bool:
        """Solve piece ``counts``; True if its bound or policy improved.

        ``bound_stop``: stop as soon as ``C(counts) >= bound_stop`` is proven
        (the piece is then excluded at the current multiplier).
        """
        if time_limit is None:
            time_limit = self._time_limit_for_solve()
        if time_limit is not None and time_limit <= 0.0:
            record_backend_event("fleet_table", "skip", "node_deadline")
            return False
        mode = self.piece_mode
        if mode == "dp":
            return self._refine_with_dp(counts)
        if mode == "gurobi":
            return self._refine_with_gurobi(counts, time_limit, bound_stop)
        # Hybrid: a short Gurobi attempt, then the exact DP if unsettled.
        predicted = self.dp_solver.predicted_seconds(counts)
        # The numpy fallback has no calibrated runtime predictor.  It is still
        # an exact piece solver, so use it directly instead of multiplying
        # ``None`` or inventing an unsafe/unstable Gurobi time budget.
        if predicted is None:
            return self._refine_with_dp(counts)
        grb_limit = max(_HYBRID_GUROBI_MIN_SECONDS, _HYBRID_GUROBI_FRACTION * predicted)
        if time_limit is not None:
            grb_limit = min(grb_limit, float(time_limit))
        improved = self._refine_with_gurobi(counts, grb_limit, bound_stop)
        lb = self.table.lower_bound(counts)
        settled = self.piece_is_exact(counts, lb) or (
            bound_stop is not None and lb >= bound_stop
        )
        if settled:
            return improved
        record_backend_event("gurobi", "fallback", "gap_open", target="subset_dp")
        return self._refine_with_dp(counts) or improved

    def _refine_with_gurobi(self, counts: Counts, time_limit, bound_stop) -> bool:
        start_policy, start_cost = self.best_policy_for(counts)
        result = solve_piece(
            self.prob_data, self.node, self.cut_lag, self.layout, counts,
            cuts_payload=self.cuts_payload,
            stage_builder_options=self.stage_builder_options,
            mip_gap=self.mip_gap, time_limit=time_limit, mip_start=start_policy,
            binary_tolerance=self.binary_tolerance, bound_stop=bound_stop,
            abs_gap=effective_abs_gap(
                self.mip_gap_abs, start_cost, self.mip_gap_abs_rel_cap
            ),
        )
        self.stats["solves"] += 1
        self.stats["solve_time"] += result["seconds"]
        self.stats["refinements"] += 1
        lb_before = self.table.lower_bound(counts)
        piece_lb = _consistent_piece_lower_bound(
            result["lb"], result["policy_cost"], start_cost
        )
        improved = self.table.update_lb(
            counts, piece_lb, status=f"grb_{result['status']}",
            seconds=result["seconds"],
            optimal_at=(self.table.archive_fingerprint
                        if result["optimal"] and piece_lb is not None else None),
        )
        if result["policy"] is not None:
            if self._add_policy(result["policy"]):
                improved = True
            self._remember_policy_certificate(
                result["policy"],
                result["policy_cost"],
                result["policy_certificate"],
            )
        lb_now = self.table.lower_bound(counts)
        _best_policy, ub_now = self.best_policy_for(counts)
        if piece_lb is not None and should_mark_stalled(
            time_limited=int(result["status"]) == int(GRB.TIME_LIMIT),
            solve_seconds=result["seconds"],
            min_solve_seconds=self.min_solve_time,
            tolerance=self._exact_tol(ub_now if ub_now is not None else lb_now),
            lb_before=lb_before,
            lb_after=lb_now,
            ub_after=ub_now,
            fingerprint=self.table.archive_fingerprint,
        ):
            if counts not in self._stalled:
                record_backend_event("fleet_table", "state", "stalled_piece")
                self._stalled.add(counts)
                self.stats["stalled_pieces"] += 1
                self.table.mark_stalled(counts, self.table.archive_fingerprint)
                if self.verbose:
                    self._log(
                        f"piece n={counts} stalled: time limit after "
                        f"{float(result['seconds']):.1f}s with open interval "
                        f"[{lb_now:.6f}, {float(ub_now):.6f}]; answered from "
                        "that certified interval for the rest of this archive "
                        "(session, forward and refresh)"
                    )
        if self.verbose:
            lb_after = self.table.lower_bound(counts)
            self._log(
                f"piece n={counts} solved: lb {lb_before:.6f} -> {lb_after:.6f}"
                f" ub={result['policy_cost'] if result['policy_cost'] is not None else 'n/a'}"
                f" status={result['status']} exact={int(self.piece_is_exact(counts))}"
                f" {result['seconds']:.1f}s nodes={result['node_count']:.0f}"
                + (f" (incumbent uncertified: {result['certification_failure']})"
                   if result["certification_failure"] else "")
            )
        return improved

    def _refine_with_dp(self, counts: Counts) -> bool:
        """Exact piece solve by subset DP; the optimum is re-costed by the certifier."""
        with backend_scope(operation="fleet_piece_dp"):
            result = self.dp_solver.solve(counts)
        for name in (
            "hits", "misses", "stores", "evictions", "bytes", "budget_bytes"
        ):
            self.stats[f"dp_prefix_cache_{name}"] = int(
                self.dp_solver.stats[f"prefix_cache_{name}"]
            )
        self.stats["solves"] += 1
        self.stats["dp_solves"] += 1
        self.stats["solve_time"] += result["seconds"]
        self.stats["refinements"] += 1
        lb_before = self.table.lower_bound(counts)
        policy, policy_cost, certificate = certify_policy_dict(
            self.prob_data, self.node, self.cuts_payload, result["x_dict"], self.layout,
            binary_tolerance=0.0, source=f"dp{counts}",
            return_certificate=True,
        )
        if policy.counts != counts and not self.layout.dominates(counts, policy.counts):
            raise RuntimeError("subset DP policy uses vehicles outside its fleet")
        _prior_policy, prior_cost = self.best_policy_for(counts)
        lb = _consistent_piece_lower_bound(result["lb"], policy_cost, prior_cost)
        improved = self.table.update_lb(
            counts, lb, status=result["status"], seconds=result["seconds"],
            optimal_at=self.table.archive_fingerprint if lb is not None else None,
        )
        if self._add_policy(policy):
            improved = True
        self._remember_policy_certificate(policy, policy_cost, certificate)
        if self.verbose:
            self._log(
                f"piece n={counts} subset-DP: lb {lb_before:.6f} -> "
                f"{self.table.lower_bound(counts):.6f} ub={policy_cost:.6f} "
                f"exact={int(self.piece_is_exact(counts))} {result['seconds']:.2f}s"
            )
        return improved

    def prepare(self, trial_z: Optional[Mapping[str, float]] = None):
        """Cold-table warm-up: solve the full fleet and the trial fleet first.

        ``C(full)`` is the smallest fixed-fleet value, so by monotonicity it
        lifts the lower bound of *every* piece in one solve; the trial fleet
        is the piece the Level Set target ``L`` refers to.  Without this the
        first query sees all pieces tied at 0 and refines them one by one.
        """
        if self.direct_dp is not None:
            return
        if any(r.solves > 0 for r in self.table.records.values()):
            return
        targets = [self.layout.full]
        if trial_z is not None:
            trial = self.layout.counts_from_z(trial_z, self.period)
            if trial != self.layout.full:
                targets.append(trial)
        for counts in targets:
            self.refine(counts)

    # ------------------------------------------------------------ evaluate
    def evaluate(self, pi_value: Mapping[str, float], *, finalize: bool = False) -> dict:
        """Certified bounds on ``D(pi)``.

        ``max_refinements_per_eval=None`` or ``finalize=True`` means no
        per-query refinement cap: the query is refined until the minimising
        piece is exact or the wall-clock budget is spent (``0`` = answer from
        the table as is).
        """
        if self.direct_dp is not None:
            with backend_scope(operation="direct_dp"):
                result = self.direct_dp.evaluate(pi_value)
            self.stats["evals"] += 1
            self.stats["solves"] += 1
            self.stats["dp_solves"] += 1
            self.stats["direct_dp_evals"] += 1
            self.stats["solve_time"] += float(result["dp_seconds"])
            if result["exact"]:
                self.stats["exact_evals"] += 1
            z_flags = {
                vehicle: result["policy"]["z"][position]
                for position, vehicle in enumerate(self.layout.vehicles)
            }
            return {
                "inner_value": result["inner_value"],
                "inner_xcp": result["xcp"],
                "outer_lb": result["outer_lb"],
                "exact": result["exact"],
                "status": "exact" if result["exact"] else "lb_only",
                "source": BACKEND,
                "argmin_counts": self.layout.counts_from_vehicle_flags(z_flags),
                "refinements": 0,
                "seconds": result["t_solve"],
            }

        started = time.time()
        self.stats["evals"] += 1
        pi = self.layout.pi_by_vehicle(pi_value, self.period)
        prefix_credits = self.layout.prefix_credit_tables(pi)
        refined_without_progress = set()
        refinements = 0
        cap = None if finalize else self.max_refinements_per_eval
        lbs = self.closed_lower_bounds()
        while True:
            piece_values = Binary64PieceValues(self.layout, lbs, pi)
            n_out, out_value, exact = self.outer_argmin(lbs, piece_values)
            if exact:
                if refinements == 0:
                    record_backend_event("fleet_table", "cache", "closed_interval")
                break
            if (
                (cap is not None and refinements >= cap)
                or n_out in refined_without_progress
                or n_out in self._stalled
            ):
                record_backend_event(
                    "fleet_table", "skip",
                    "refinement_limit" if cap is not None and refinements >= cap else (
                        "stalled_piece" if n_out in self._stalled else "no_progress"
                    ),
                )
                break
            # An exact piece already certifies a value at this pi; the
            # candidate only has to be bounded above it to be excluded.
            bound_stop = None
            reference = self.best_exact_value(lbs, piece_values)
            if reference is not None:
                bound_stop = float_up(
                    reference + piece_values.credit_fraction(n_out)
                )
            if not self.refine(n_out, bound_stop=bound_stop):
                refined_without_progress.add(n_out)
            refinements += 1
            lbs = self.closed_lower_bounds()

        # Inner: best stored feasible policy with its optimal z at this pi.
        best_policy = None
        best_inner = None
        best_counts = None
        for policy in self.table.policies:
            cost = fraction_from_float(self.policy_cost(policy))
            chosen = self.layout.best_extension_counts(policy.counts, prefix_credits)
            value = cost - piece_values.credit_fraction(chosen)
            if best_inner is None or value < best_inner:
                best_policy, best_inner, best_counts = policy, value, chosen
        inner = self._certify_inner(best_policy, pi_value, best_counts)

        if exact:
            self.stats["exact_evals"] += 1
        status = "exact" if exact else (
            "budget" if (cap is not None and refinements >= cap) else "lb_only"
        )
        return {
            "inner_value": inner["V"],
            "inner_xcp": inner["xcp"],
            "outer_lb": float_down(out_value),
            "exact": exact,
            "status": status,
            "source": BACKEND,
            "argmin_counts": n_out,
            "refinements": refinements,
            "seconds": time.time() - started,
        }

    def _certify_inner(self, policy: PolicyRecord, pi_value, purchased_counts) -> dict:
        purchased_counts = self.layout.check_counts(purchased_counts)
        if not self.layout.dominates(purchased_counts, policy.counts):
            raise exact_sub.InvalidS2LagrangianPolicy(
                "policy_used_vehicles_not_purchased"
            )
        signature = policy.signature()
        base_exact = self._policy_base_exact.get(signature)
        if base_exact is None:
            # Restored tables contain policies but deliberately do not persist
            # archive-dependent scores.  Certify once against this session's
            # archive, then every later pi is an exact affine update.
            _record, cost, certificate = certify_policy_dict(
                self.prob_data,
                self.node,
                self.cuts_payload,
                policy.as_x_dict(),
                self.layout,
                binary_tolerance=0.0,
                source=policy.source,
                return_certificate=True,
            )
            self._remember_policy_certificate(policy, cost, certificate)
            base_exact = self._policy_base_exact[signature]
            self.stats["inner_full_certifications"] += 1

        z = self.layout.flags_for_counts(purchased_counts)
        credit = sum(
            (
                fraction_from_float(
                    pi_value.get(f"z[{vehicle},{self.period}]", 0.0)
                )
                for vehicle in self.layout.vehicles
                if z[vehicle]
            ),
            Fraction(0),
        )
        objective = base_exact - credit
        self.stats["inner_affine_evals"] += 1
        return {
            "V": float_up(objective),
            "xcp": {
                f"z[{vehicle},{self.period}]": float(z[vehicle])
                for vehicle in self.layout.vehicles
            },
        }

    # ----------------------------------------------------------------- misc
    def _log(self, message: str):
        exact_log("backward", 2, BACKEND, f"{self.node_tag} | {message}", indent=6)

    def format_stats(self) -> str:
        s = self.stats
        cache = ""
        if s["dp_prefix_cache_budget_bytes"]:
            cache = (
                f" cache={s['dp_prefix_cache_hits']}h/"
                f"{s['dp_prefix_cache_misses']}m/"
                f"{s['dp_prefix_cache_evictions']}e "
                f"{s['dp_prefix_cache_bytes'] / (1 << 20):.0f}/"
                f"{s['dp_prefix_cache_budget_bytes'] / (1 << 20):.0f}MiB"
            )
        return (
            f"fleet_enum[{self.piece_mode}] evals={s['evals']} exact={s['exact_evals']} "
            f"piece_solves={s['solves']}({s['solve_time']:.1f}s"
            f"{', dp=' + str(s['dp_solves']) if s['dp_solves'] else ''}"
            f"{', direct=' + str(s['direct_dp_evals']) if s['direct_dp_evals'] else ''})"
            f"{cache} "
            f"| {self.table.summary()}"
        )
