"""Monotonic solve budgets shared by iterative subproblem solvers."""

from __future__ import annotations

import math
import time


def remaining_seconds(deadline):
    """Remaining wall time; ``None`` preserves an unbudgeted call."""
    if deadline is None:
        return math.inf
    value = float(deadline)
    if not math.isfinite(value):
        raise ValueError("solve deadline must be finite or None")
    return max(0.0, value - time.monotonic())


def clipped_time_limit(time_limit, deadline):
    """Never grant a new solve more than the node's remaining budget."""
    if deadline is None:
        return time_limit
    remaining = remaining_seconds(deadline)
    if time_limit is None:
        return remaining
    limit = float(time_limit)
    if math.isnan(limit) or limit < 0.0:
        raise ValueError("solve time limit must be nonnegative")
    return min(limit, remaining)


def prepare_model_solve(model, deadline, *, time_limit=None):
    """Clip a model's next optimize call, or return False without solving.

    With no deadline this does not modify model parameters. Gurobi stops
    cooperatively: checking the deadline does not kill a solver or a worker.
    """
    if deadline is None:
        return True
    if time_limit is None:
        try:
            time_limit = float(model.Params.TimeLimit)
        except (AttributeError, TypeError, ValueError):
            time_limit = math.inf
    limit = clipped_time_limit(time_limit, deadline)
    if limit <= 0.0:
        return False
    model.setParam("TimeLimit", limit)
    return True


def forward_retry_deadline(model, deadline=None):
    """Keep the first forward solve's allowance across certification retries.

    A caller's absolute deadline includes queue/build time. Without one, the
    first model's Runtime is deducted once from its original TimeLimit; later
    reset/optimize calls cannot replenish that allowance.
    """
    candidates = []
    for value in (deadline, getattr(model, "_forward_node_deadline", None)):
        if value is not None:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("forward deadline must be finite or None")
            candidates.append(value)
    try:
        limit = float(model.Params.TimeLimit)
        runtime = float(model.Runtime)
    except (AttributeError, TypeError, ValueError):
        limit, runtime = math.inf, 0.0
    if math.isfinite(limit) and 0.0 <= limit < 1e100:
        if not math.isfinite(runtime) or runtime < 0.0:
            runtime = 0.0
        candidates.append(time.monotonic() + max(0.0, limit - runtime))
    return min(candidates) if candidates else None
