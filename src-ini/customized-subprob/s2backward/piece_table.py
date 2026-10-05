"""Per-node table of fixed-fleet values ``C(n)`` and feasible policies.

Validity across SDDP iterations: the Stage-3 cut archive of a Stage-2 node
only ever grows (``cuts.benders_cuts.add_unique_cut`` appends), so the
fixed-fleet optimum ``C(n)`` never decreases and a certified lower bound
stored in one iteration is still a lower bound in every later one.  Policies
are feasible assignments; their *cost* depends on the archive and is
therefore re-scored by the oracle session, never stored as a number here.

Monotonicity: a superset fleet can only do better, ``C(n) >= C(n')`` for
``n <= n'`` componentwise, so a lower bound proven for ``n'`` is inherited by
every ``n <= n'``.  ``lower_bound(n)`` applies that closure on the fly.

Stall memory: forward Stage 2, the backward refresh and the S2->S1 oracle
solve the same piece MIP under the same archive within one backward/forward
round.  When one of them hit its time limit and moved the certified bound by
less than the scheduled tolerance, the compact MIP has stalled (its bound
sits at the root relaxation) and another capped solve of the *same* model
would only burn wall clock.  ``mark_stalled`` records the archive
fingerprint of that attempt; ``is_stalled`` lets every later caller under the
same fingerprint answer from the certified interval ``[lb, ub]`` instead.  A
new archive (new fingerprint) always gets a fresh attempt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from s2forward.fleet_pieces import Counts, FleetLayout


POLICY_BOUND_CONFLICT_TOL = 1e-6


class PieceBoundConflictError(RuntimeError):
    """An archived lower bound contradicts a freshly certified feasible policy.

    Previously exported cuts may depend on that bound.  Continuing after
    clipping it to an incumbent or resetting the table would not repair those
    cuts, so this error must propagate out of the solve.  Plain context data
    and the ordinary RuntimeError constructor keep it multiprocessing-safe.
    """

    def __init__(self, message, *, context=None):
        super().__init__(message)
        self.context = {} if context is None else context


def should_mark_stalled(
    *,
    time_limited: bool,
    solve_seconds: float,
    min_solve_seconds: float,
    tolerance: Optional[float],
    lb_before: Optional[float],
    lb_after: Optional[float],
    ub_after: Optional[float],
    fingerprint: Optional[str],
) -> bool:
    """Shared, conservative stall predicate for every piece-table caller.

    A stall mark is an archive-local performance hint, not a certificate.
    Requiring a meaningful solve duration, a pre-existing nontrivial bound and
    an open certified interval prevents a short budget tail from suppressing a
    later forward/refresh refinement.
    """
    if not time_limited or fingerprint is None or tolerance is None:
        return False
    try:
        seconds = float(solve_seconds)
        floor = max(0.0, float(min_solve_seconds))
        eps = float(tolerance)
        before = float(lb_before)
        after = float(lb_after)
        upper = float(ub_after)
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(map(math.isfinite, (seconds, floor, eps, before, after, upper))):
        return False
    if eps <= 0.0 or seconds < floor or before <= 0.0:
        return False
    if upper - after <= eps:
        return False
    return after - before <= eps


@dataclass
class PieceRecord:
    """State of one fleet piece ``n``."""

    # Certified lower bound on C(n) proven directly for this piece (any
    # archive state; still valid now).  ``0.0`` is the trivial bound.
    lb: float = 0.0
    # Archive fingerprint at which ``lb`` was proven optimal (C(n) == lb).
    # Only meaningful for that exact archive; the oracle re-checks exactness
    # via the policy re-score anyway.
    optimal_at: Optional[str] = None
    solves: int = 0
    solve_seconds: float = 0.0
    last_status: Optional[str] = None
    # Archive fingerprint under which a time-limited solve left the bound
    # (nearly) where it was; see the module docstring.
    stalled_at: Optional[str] = None


@dataclass
class PolicyRecord:
    """A certified feasible Stage-2 assignment (``alpha``/``y`` 0/1 values)."""

    counts: Counts                      # leading run of its used vehicles
    alpha: Dict[str, int]               # {"alpha[j,v]": 0/1}, only nonzero keys
    y: Dict[str, int]                   # {"y[v]": 0/1}, only nonzero keys
    source: str = "solve"

    def signature(self) -> Tuple[Tuple[str, int], ...]:
        return tuple(sorted(self.alpha.items()))

    def as_x_dict(self) -> Dict[str, float]:
        out = {k: float(v) for k, v in self.alpha.items()}
        out.update({k: float(v) for k, v in self.y.items()})
        return out


class FleetPieceTable:
    """Table of pieces for one Stage-2 node (one scenario/period model)."""

    def __init__(self, layout: FleetLayout):
        self.layout = layout
        self.records: Dict[Counts, PieceRecord] = {
            counts: PieceRecord() for counts in layout.all_counts()
        }
        self._closed_lb_cache: Optional[Dict[Counts, float]] = None
        self.policies: List[PolicyRecord] = []
        self._policy_signatures = set()
        self.archive_fingerprint: Optional[str] = None

    # ------------------------------------------------------------- bounds
    def direct_lb(self, counts: Counts) -> float:
        return self.records[counts].lb

    def closed_lower_bounds(self) -> Dict[Counts, float]:
        """Return the componentwise monotone closure of all direct bounds.

        A fleet inherits every bound proved for a componentwise larger fleet.
        Computing that maximum separately for every piece is quadratic in the
        number of fleet prefixes.  The prefix domain is a rectangular grid, so
        one reverse sweep per vehicle type computes the identical closure in
        ``O(types * pieces)``.  The cache is invalidated only when a direct
        bound improves.
        """
        if self._closed_lb_cache is None:
            closed = {
                counts: float(record.lb)
                for counts, record in self.records.items()
            }
            dimensions = len(self.layout.groups)
            for axis in range(dimensions):
                upper = self.layout.full[axis]
                # ``FleetLayout.all_counts`` is lexicographic; reverse order
                # guarantees the +1 neighbour has already been propagated.
                for counts in reversed(self.records):
                    if counts[axis] >= upper:
                        continue
                    successor = list(counts)
                    successor[axis] += 1
                    inherited = closed[tuple(successor)]
                    if inherited > closed[counts]:
                        closed[counts] = inherited
            self._closed_lb_cache = closed
        return dict(self._closed_lb_cache)

    def lower_bound(self, counts: Counts) -> float:
        """Monotone closure: best bound proven for ``counts`` or any superset fleet."""
        counts = self.layout.check_counts(counts)
        if self._closed_lb_cache is None:
            self.closed_lower_bounds()
        return self._closed_lb_cache[counts]

    def assert_policy_upper_bound(self, counts: Counts, upper_bound: float, *,
                                  source: str) -> None:
        """Fail closed if a certified policy refutes an existing table bound.

        ``counts`` must be a fleet that admits the policy, and ``upper_bound``
        its complete current-archive, upward-rounded objective.  Passing the
        policy's used leading run also checks every larger fleet through the
        monotone closure.  Binary64/solver differences within the model's
        absolute 1e-6 numerical contract are not treated as contradictions.
        This method never changes the table or promotes an incumbent to LB.
        """
        counts = self.layout.check_counts(counts)
        upper = float(upper_bound)
        if not math.isfinite(upper):
            raise ValueError("piece policy upper bound must be finite")
        lower = float(self.lower_bound(counts))
        if (
            math.isfinite(lower)
            and lower <= upper + POLICY_BOUND_CONFLICT_TOL
        ):
            return
        contributors = [
            {"fleet": fleet, "direct_lb": record.lb,
             "last_status": record.last_status, "optimal_at": record.optimal_at}
            for fleet, record in self.records.items()
            if self.layout.dominates(fleet, counts)
            and (
                not math.isfinite(record.lb)
                or record.lb > upper + POLICY_BOUND_CONFLICT_TOL
            )
        ]
        context = {
            "fleet": counts, "direct_lb": self.direct_lb(counts),
            "closed_lb": lower, "policy_ub": upper, "source": str(source),
            "conflict_tolerance": POLICY_BOUND_CONFLICT_TOL,
            "archive_fingerprint": self.archive_fingerprint,
            "contributors": contributors,
        }
        raise PieceBoundConflictError(
            f"Stage-2 archived bound conflict: fleet={counts} "
            f"closed_lb={lower!r} policy_ub={upper!r} "
            f"source={source!r} contributors={contributors!r}; "
            "historical cut dependencies are unavailable; stopping without "
            "clipping the bound or changing the archive",
            context=context,
        )

    def update_lb(self, counts: Counts, lb: float, *, status: Optional[str] = None,
                  seconds: float = 0.0, optimal_at: Optional[str] = None) -> bool:
        """Record a certified lower bound; returns True if the bound improved."""
        counts = self.layout.check_counts(counts)
        record = self.records[counts]
        record.solves += 1
        record.solve_seconds += float(seconds)
        if status is not None:
            record.last_status = status
        improved = False
        if lb is not None and float(lb) > record.lb:
            record.lb = float(lb)
            self._closed_lb_cache = None
            improved = True
        if optimal_at is not None:
            record.optimal_at = optimal_at
        return improved

    # -------------------------------------------------------- stall memory
    def mark_stalled(self, counts: Counts, fingerprint: Optional[str]) -> bool:
        """Remember that piece ``counts`` stalled under archive ``fingerprint``.

        Returns True when the mark is new.  Without a fingerprint nothing is
        recorded: the mark must never outlive the archive it was proven on.
        """
        if fingerprint is None:
            return False
        counts = self.layout.check_counts(counts)
        record = self.records[counts]
        if record.stalled_at == str(fingerprint):
            return False
        record.stalled_at = str(fingerprint)
        return True

    def is_stalled(self, counts: Counts, fingerprint: Optional[str]) -> bool:
        if fingerprint is None:
            return False
        counts = self.layout.check_counts(counts)
        return self.records[counts].stalled_at == str(fingerprint)

    def stalled_pieces(self, fingerprint: Optional[str]) -> List[Counts]:
        if fingerprint is None:
            return []
        return [
            counts for counts, record in self.records.items()
            if record.stalled_at == str(fingerprint)
        ]

    # ----------------------------------------------------------- policies
    def add_policy(self, policy: PolicyRecord) -> bool:
        sig = policy.signature()
        if sig in self._policy_signatures:
            return False
        self._policy_signatures.add(sig)
        self.policies.append(policy)
        return True

    def policies_for(self, counts: Counts) -> Iterable[PolicyRecord]:
        """Policies feasible for fleet ``counts`` (their used run is inside it)."""
        for policy in self.policies:
            if FleetLayout.dominates(counts, policy.counts):
                yield policy

    # ---------------------------------------------------------- transport
    def to_state(self) -> dict:
        return {
            "records": {
                counts: {
                    "lb": r.lb, "optimal_at": r.optimal_at, "solves": r.solves,
                    "solve_seconds": r.solve_seconds, "last_status": r.last_status,
                    "stalled_at": r.stalled_at,
                }
                for counts, r in self.records.items()
            },
            "policies": [
                {"counts": p.counts, "alpha": dict(p.alpha), "y": dict(p.y), "source": p.source}
                for p in self.policies
            ],
            "archive_fingerprint": self.archive_fingerprint,
        }

    @classmethod
    def from_state(cls, layout: FleetLayout, state: Optional[Mapping]) -> "FleetPieceTable":
        table = cls(layout)
        if not state:
            return table
        for counts, raw in state.get("records", {}).items():
            counts = layout.check_counts(counts)
            record = table.records[counts]
            record.lb = float(raw.get("lb", 0.0))
            record.optimal_at = raw.get("optimal_at")
            record.solves = int(raw.get("solves", 0))
            record.solve_seconds = float(raw.get("solve_seconds", 0.0))
            record.last_status = raw.get("last_status")
            record.stalled_at = raw.get("stalled_at")
        for raw in state.get("policies", []):
            table.add_policy(PolicyRecord(
                counts=layout.check_counts(raw["counts"]),
                alpha={str(k): int(v) for k, v in raw["alpha"].items()},
                y={str(k): int(v) for k, v in raw["y"].items()},
                source=str(raw.get("source", "state")),
            ))
        table.archive_fingerprint = state.get("archive_fingerprint")
        return table

    def summary(self) -> str:
        solved = sum(1 for r in self.records.values() if r.solves > 0)
        return (
            f"pieces={len(self.records)} solved={solved} policies={len(self.policies)} "
            f"solve_time={sum(r.solve_seconds for r in self.records.values()):.1f}s"
        )
