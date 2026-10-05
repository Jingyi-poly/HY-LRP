"""Wall-clock allocation for S3 cuts and fixed-fleet Stage-2 refinement."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class RefinementBudget:
    deadline: float
    seconds: float

    @classmethod
    def start(cls, seconds):
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("refinement seconds must be finite and nonnegative")
        return cls(time.monotonic() + seconds, seconds)

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    def batch(self, jobs, workers, *, reserve=0.0):
        """Return a batch deadline and a fair per-job allowance, queue included."""
        if not 0.0 <= reserve < 1.0:
            raise ValueError("reserve must be in [0, 1)")
        now = time.monotonic()
        available = max(0.0, self.deadline - now) * (1.0 - reserve)
        waves = max(1, math.ceil(max(0, jobs) / max(1, workers)))
        return now + available, available / waves


class RefreshSchedule:
    """Reuse trial assignments between expensive searches, without starving them."""

    def __init__(self):
        self._visits = {}
        self._slow = set()
        self.key = None
        self.rescore_only = False
        self.force_next = False
        self.budget_multiplier = 1

    def begin(self, fleet, *, force=False):
        self.key = tuple(sorted((key, float(value).hex()) for key, value in fleet.items()))
        visits = self._visits.get(self.key, 0) + 1
        self._visits[self.key] = visits
        if self.key in self._slow:
            # Repeated fleets eventually receive larger searches even if tiny
            # new cuts keep resetting the outer no-new-cut counter.
            self.budget_multiplier = max(self.budget_multiplier, visits // 3 + 1)
        force = bool(force or self.force_next)
        self.force_next = False
        self.rescore_only = (
            self.key in self._slow and visits % 3 != 1 and not force
        )

    def observed_refresh(self, seconds, budget_seconds):
        if seconds >= max(1.0, 0.1 * budget_seconds):
            self._slow.add(self.key)

    def request_refinement(self):
        self.force_next = True
        self.budget_multiplier *= 2
