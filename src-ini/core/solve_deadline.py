"""Optional solve deadlines; expiry is not a solver certificate."""
import math
import time

class SolveDeadlineReached(RuntimeError):
    pass

def bounded_solve_time(limit, deadline):
    limit = float(limit)
    if deadline is None:
        return limit
    deadline = float(deadline)
    if math.isnan(deadline):
        raise ValueError('deadline must not be NaN')
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SolveDeadlineReached('global solve deadline reached')
    return min(limit, remaining)
