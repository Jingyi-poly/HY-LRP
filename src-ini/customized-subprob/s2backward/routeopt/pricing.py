"""Conservative integer-grid pricing via a bounded RouteOpt subprocess.

The returned min_rc is min(0, min_nonempty_route(cost - sum prices - fleet_dual)).
The default domain is elementary. Positive ng_size enables the original
NG-route relaxation; repeated visits count repeatedly in both cost and demand.
It is a pricing lower bound, not the restricted master's objective. Original
binary64 costs and demands must be verified separately before adding columns.
"""
import json
import math
import numbers
from pathlib import Path
import subprocess
import time
from core.backend_telemetry import backend_call, record_backend_event


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("pricing inputs must be exact integers")
    return int(value)


def price_routes(scaled_costs, scaled_prices, integer_demands, integer_capacity,
                 *, fleet_dual=0, time_limit_s=1.0, max_routes=100, ng_size=0):
    started = time.perf_counter()
    costs = [[_integer(value) for value in row] for row in scaled_costs]
    prices = [_integer(value) for value in scaled_prices]
    demands = [_integer(value) for value in integer_demands]
    capacity, beta, count, ng_size = map(_integer, (integer_capacity, fleet_dual, max_routes, ng_size))
    n = len(prices)
    if not 1 <= n < 1002 or len(demands) != n or len(costs) != n + 1:
        raise ValueError("expected n prices/demands and a depot-inclusive (n+1) matrix")
    if any(len(row) != n + 1 for row in costs):
        raise ValueError("cost matrix must be square")
    if any(costs[i][j] != costs[j][i] for i in range(n + 1) for j in range(n + 1)):
        raise ValueError("RouteOpt CVRP pricing requires symmetric costs")
    if any(costs[i][i] for i in range(n + 1)):
        raise ValueError("cost diagonal must be zero")
    if any(d <= 0 for d in demands) or capacity <= 0:
        raise ValueError("pricing requires strictly positive integer resources")
    if not 0 <= ng_size <= 1001:
        raise ValueError("ng_size must be 0 (elementary) or a positive memory size")
    elementary = ng_size == 0 or ng_size >= n
    if max([capacity, *demands]) * 200 >= 2**31:
        raise ValueError("RouteOpt internal integer resources could overflow")
    # All arc RCs are exact half-integers. Bound even unsuccessful extensions
    # and concatenations, not only returned routes. NG routes can revisit nodes,
    # but every visit consumes positive demand, hence at most floor(Q/min d).
    visit_limit = capacity // min(demands)
    if elementary:
        visit_limit = min(n, visit_limit)
    vertex_prices = [beta, *prices]
    if max(abs(v) for row in costs for v in row) >= 2**51:
        raise ValueError("integer costs exceed exact arithmetic range")
    if max(map(abs, vertex_prices)) >= 2**51:
        raise ValueError("integer prices exceed exact arithmetic range")
    doubled_rc = max(abs(2 * costs[i][j] - vertex_prices[i] - vertex_prices[j])
                     for i in range(n + 1) for j in range(i + 1, n + 1))
    if (2 * visit_limit + 2) * doubled_rc >= 2**52:
        raise ValueError("half-integer label arithmetic could round")
    if (2 * visit_limit + 2) * max(abs(v) for row in costs for v in row) >= 2**52:
        raise ValueError("route cost arithmetic could round")
    if not math.isfinite(time_limit_s) or time_limit_s <= 0 or not 1 <= count <= 10000:
        raise ValueError("positive finite time limit and 1..10000 max_routes required")
    executable = Path(__file__).resolve().parent / "build/routeopt_pricing"
    if not executable.is_file():
        raise RuntimeError("Build the adapter with routeopt/build.py first")
    payload = " ".join(map(str, ["ROUTEOPT_GRID_PRICING_2", n, capacity, beta,
                                  time_limit_s, count, ng_size, *demands, *prices,
                                  *(v for row in costs for v in row)]))
    result = {"routes": [], "min_rc": None, "pricing_complete": False,
              "lb_certified": False, "status": "deadline", "ng_size": ng_size,
              "domain": "elementary_integer_grid" if elementary else "ng_relaxation_integer_grid"}
    remaining = time_limit_s - (time.perf_counter() - started)
    if remaining <= 0:
        record_backend_event("routeopt", "skip", "pricing_deadline")
        result["seconds"] = time.perf_counter() - started
        return result
    try:
        with backend_call("routeopt", "pricing_subprocess") as outcome:
            process = subprocess.run([str(executable)], input=payload, text=True,
                                     capture_output=True, timeout=remaining, check=False)
            outcome["status"] = process.returncode
    except subprocess.TimeoutExpired:
        result["seconds"] = time.perf_counter() - started
        return result
    try:
        native = json.loads(process.stdout)
    except (ValueError, TypeError):
        native = {}
    elapsed = time.perf_counter() - started
    complete = (process.returncode == 0 and native.get("pricing_complete") is True
                and elapsed <= time_limit_s)
    routes = native.get("routes", [])
    verified = []
    route_rcs = []
    for route in routes:
        if (not route or any(type(j) is not int or not 1 <= j <= n for j in route)
                or (elementary and len(set(route)) != len(route))
                or sum(demands[j - 1] for j in route) > capacity):
            complete = False
            continue
        sequence = [0, *route, 0]
        rc = sum(costs[i][j] for i, j in zip(sequence, sequence[1:]))
        rc -= sum(prices[j - 1] for j in route) + beta
        verified.append(route)
        route_rcs.append(rc)
    minimum = native.get("min_rc")
    if complete and (not isinstance(minimum, (int, float)) or not math.isfinite(minimum)
                     or int(minimum) != minimum or minimum != min([0, *route_rcs])):
        complete = False
    elapsed = time.perf_counter() - started
    complete = complete and elapsed <= time_limit_s
    result.update(routes=verified, pricing_complete=complete, lb_certified=complete,
                  min_rc=int(minimum) if complete else None, seconds=elapsed,
                  kernel_seconds=native.get("kernel_seconds"),
                  status="complete" if complete else "incomplete", stderr=process.stderr[-2000:])
    return result
