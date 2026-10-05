"""Per-solve, LB-only scheduling for bounded physical-recourse refinement."""
from collections import deque
import math


class LatePhysicalSchedule:
    def __init__(self, *, total_seconds=0., per_call_seconds=180., window=3,
                 relative_gain=1e-4):
        for name, value in (("total_seconds", total_seconds),
                            ("per_call_seconds", per_call_seconds),
                            ("relative_gain", relative_gain)):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid late physical {name}")
        if per_call_seconds <= 0:
            raise ValueError("late physical per_call_seconds must be positive")
        if (isinstance(window, bool) or not math.isfinite(window)
                or int(window) != window or window < 1):
            raise ValueError("late physical window must be a positive integer")
        self.total_seconds = float(total_seconds)
        self.per_call_seconds = float(per_call_seconds)
        self.relative_gain = float(relative_gain)
        self.window = int(window)
        self.samples = deque(maxlen=self.window + 1)
        self.spent = 0.
        self.calls = 0
        self.last_iteration = None

    @property
    def remaining(self):
        return max(0., self.total_seconds - self.spent)

    def observe(self, iteration, lower):
        if self.last_iteration == iteration:
            return
        self.last_iteration = iteration
        if lower is None or not math.isfinite(lower):
            self.samples.clear()
            return
        self.samples.append(float(lower))

    def request(self, outer_remaining, *, continuation=False):
        if not math.isfinite(outer_remaining) or outer_remaining <= 0:
            return 0.
        if not continuation and len(self.samples) < self.window + 1:
            return 0.
        if not continuation:
            threshold = max(1e-6, self.relative_gain * max(1., abs(self.samples[-1])))
            if self.samples[-1] - self.samples[0] > threshold:
                return 0.
        # Do not launch physical pricing when no useful Stage-1 reserve fits.
        allowance = min(self.remaining, self.per_call_seconds, outer_remaining)
        return allowance if allowance >= 30. else 0.

    def charge(self, seconds, *, calls=1):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid physical refinement elapsed time")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise ValueError("invalid physical refinement batch count")
        self.spent += seconds
        self.calls += calls
        latest = self.samples[-1] if self.samples else None
        self.samples.clear()
        if latest is not None:
            self.samples.append(latest)

    def report(self):
        return dict(budget_seconds=self.total_seconds, spent_seconds=self.spent,
                    remaining_seconds=self.remaining, calls=self.calls,
                    window=self.window, relative_gain=self.relative_gain)
