"""Legacy per-node S2 Level Set allowance with a separate hard outer limit."""
from dataclasses import dataclass
import math
import os
import time


def _setting(suffix, default):
    value = float(os.environ.get('LRP_PHASE2_S2_LEVELSET_' + suffix,
                                 os.environ.get('VRP_PHASE2_S2_LEVELSET_' + suffix, default)))
    if not math.isfinite(value):
        raise ValueError('S2 Level Set time settings must be finite')
    return value


@dataclass(frozen=True)
class S2LevelSetBudget:
    soft_deadline: float | None
    hard_deadline: float | None
    min_solve_time: float = 30.

    @classmethod
    def from_environment(cls, hard_deadline=None):
        budget = _setting('TIME_BUDGET', 900.)
        minimum = _setting('MIN_SOLVE_TIME', 30.)
        if minimum < 0:
            raise ValueError('S2 Level Set MIN_SOLVE_TIME must be nonnegative')
        if hard_deadline is not None and not math.isfinite(float(hard_deadline)):
            raise ValueError('hard deadline must be finite or None')
        return cls(time.monotonic() + budget if budget > 0 else None,
                   hard_deadline, minimum)

    @property
    def search_deadline(self):
        values = [d for d in (self.soft_deadline, self.hard_deadline) if d is not None]
        return min(values) if values else None

    def solve_allowance(self, sub_time_limit):
        now = time.monotonic()
        soft_left = math.inf if self.soft_deadline is None else self.soft_deadline - now
        hard_left = math.inf if self.hard_deadline is None else self.hard_deadline - now
        if min(soft_left, hard_left) <= 0:
            return 0.
        # Original behavior: max(minimum, remaining node budget), capped by
        # subproblem time. A hard outer/diagnostic deadline always wins.
        node_limit = max(self.min_solve_time, soft_left)
        return max(0., min(float(sub_time_limit), node_limit, hard_left))
