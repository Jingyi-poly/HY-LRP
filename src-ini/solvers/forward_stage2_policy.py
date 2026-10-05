"""Exact validation and upper-bound scoring for a Stage-2 forward policy.

The forward solvers may obtain a binary assignment from subset DP or Gurobi.
This module independently validates that assignment and evaluates its complete
learned-cut archive.  Every affine expression is accumulated as the exact
rational represented by its binary64 inputs, then rounded upward once.  The
result is therefore a certified feasible upper bound; it is never used as a
lower-bound certificate.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict

from cuts import exact_subroutines as exact_sub

__all__ = [
    "all_outsource_forward_solution",
    "score_forward_stage2_policy",
    "score_forward_stage2_policy_from_archive",
]


def _finite_binary64_fraction(raw_value, *, label: str) -> Fraction:
    """Return the exact rational represented by one finite binary64 input."""
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not a finite binary64 value") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite: {raw_value!r}")
    return Fraction.from_float(value)


def _fraction_to_finite_float_up(value: Fraction, *, label: str) -> float:
    """Return a finite binary64 upper endpoint for an exact rational score."""
    try:
        rounded = float(value)
    except OverflowError as exc:
        if value < 0:
            return -float.fromhex("0x1.fffffffffffffp+1023")
        raise ValueError(f"{label} exceeds the finite binary64 range") from exc
    if not math.isfinite(rounded):
        raise ValueError(f"{label} is non-finite after directed rounding")
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    if not math.isfinite(rounded):
        raise ValueError(f"{label} exceeds the finite binary64 range")
    return rounded


def _score_forward_s2_policy_exact(
    *, c_out, outsourced, successor, theta_lower_bound, cuts_payload, y, alpha
):
    """Score a fixed feasible policy and return upward-rounded endpoints."""
    successor = list(successor)
    theta_floor = max(
        Fraction(0),
        _finite_binary64_fraction(
            theta_lower_bound, label="Stage-2 theta lower bound"
        ),
    )
    y_bits = [int(value) for value in y]
    alpha_bits = [[int(value) for value in row] for row in alpha]
    m = len(y_bits)
    if len(alpha_bits) != m:
        raise ValueError("Stage-2 score has inconsistent vehicle shape")
    n = len(alpha_bits[0]) if alpha_bits else len(c_out)
    if any(len(row) != n for row in alpha_bits):
        raise ValueError("Stage-2 score has inconsistent customer shape")

    compiled = exact_sub.compiled_cut_payload(
        cuts_payload, m, n, error=ValueError
    )
    theta_list = compiled.exact_theta(
        y_bits, alpha_bits, theta_floor, len(successor)
    )
    theta_exact = {succ: theta_list[h] for h, succ in enumerate(successor)}

    if len(c_out) != len(outsourced):
        raise ValueError("Stage-2 outsourcing score has inconsistent shape")
    stage_cost_exact = sum(
        (
            _finite_binary64_fraction(cost, label=f"cOut[{j}]")
            * int(outsourced[j])
            for j, cost in enumerate(c_out)
        ),
        Fraction(0),
    )
    objective_exact = stage_cost_exact + sum(
        theta_exact.values(), Fraction(0)
    )
    return {
        "theta_by_succ": {
            succ: _fraction_to_finite_float_up(
                value, label=f"theta[{succ}] feasible value"
            )
            for succ, value in theta_exact.items()
        },
        "stage_cost": _fraction_to_finite_float_up(
            stage_cost_exact, label="Stage-2 feasible stage cost"
        ),
        "objective": _fraction_to_finite_float_up(
            objective_exact, label="Stage-2 feasible objective"
        ),
    }


def _as_binary(raw_value, *, label: str, tolerance: float) -> int:
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"non_numeric_{label}") from exc
    bit = int(round(value))
    if bit not in (0, 1) or abs(value - bit) > tolerance:
        raise ValueError(f"non_binary_{label}")
    return bit


def score_forward_stage2_policy(
    prob_data,
    node,
    x_star_dict,
    cuts_payload,
    *,
    vehicles=None,
    binary_tolerance: float = 1e-6,
):
    """Validate and upward-score a full-name Stage-2 feasible policy.

    ``cuts_payload`` must use the same explicit vehicle order supplied through
    ``vehicles``.  With ``vehicles=None`` it uses the complete fleet order.
    """
    customers = list(prob_data.J)
    all_vehicles = list(prob_data.V)
    used_vehicles = all_vehicles if vehicles is None else list(vehicles)
    if len(set(used_vehicles)) != len(used_vehicles):
        raise ValueError("Stage-2 score vehicle order contains duplicates")
    all_vehicle_set = set(all_vehicles)
    unknown = [vehicle for vehicle in used_vehicles if vehicle not in all_vehicle_set]
    if unknown:
        raise ValueError(f"Stage-2 score has unknown vehicles: {unknown!r}")

    try:
        y = [
            _as_binary(
                x_star_dict[f"y[{vehicle}]"],
                label=f"y[{vehicle}]",
                tolerance=binary_tolerance,
            )
            for vehicle in used_vehicles
        ]
        alpha = [
            [
                _as_binary(
                    x_star_dict[f"alpha[{customer},{vehicle}]"],
                    label=f"alpha[{customer},{vehicle}]",
                    tolerance=binary_tolerance,
                )
                for customer in customers
            ]
            for vehicle in used_vehicles
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Stage-2 score requires every y/alpha decision in the selected "
            "vehicle order"
        ) from exc

    outsourced = [0] * len(customers)
    for customer_pos, customer in enumerate(customers):
        assigned = sum(row[customer_pos] for row in alpha)
        if int(node.active[customer]) == 1:
            if assigned not in (0, 1):
                raise ValueError(
                    f"invalid Stage-2 feasible policy score: customer_"
                    f"{customer}_assigned_{assigned}_times"
                )
            outsourced[customer_pos] = 1 - assigned
        else:
            if assigned != 0:
                raise ValueError(
                    f"invalid Stage-2 feasible policy score: inactive_customer_"
                    f"{customer}_assigned"
                )
            # The Stage-2 formulation has fulfillment for every customer.
            outsourced[customer_pos] = 1

    for vehicle_pos, vehicle in enumerate(used_vehicles):
        expected_y = int(any(alpha[vehicle_pos]))
        if y[vehicle_pos] != expected_y:
            raise ValueError(
                f"invalid Stage-2 feasible policy score: vehicle_{vehicle}_"
                "activation_mismatch"
            )
        load = sum(
            (
                Fraction.from_float(float(node.volume[customer]))
                for customer_pos, customer in enumerate(customers)
                if alpha[vehicle_pos][customer_pos]
            ),
            Fraction(0),
        )
        capacity = Fraction.from_float(float(prob_data.Qv[vehicle]))
        if load > capacity:
            raise ValueError(
                f"invalid Stage-2 feasible policy score: vehicle_{vehicle}_"
                "capacity_violation"
            )

    score = _score_forward_s2_policy_exact(
        c_out=[node.c_out[customer] for customer in customers],
        outsourced=outsourced,
        successor=node.successor,
        theta_lower_bound=exact_sub._S2_BPC_THETA_LOWER_BOUND,
        cuts_payload=(
            cuts_payload if isinstance(cuts_payload, list) else list(cuts_payload)
        ),
        y=y,
        alpha=alpha,
    )
    return {
        "theta_by_succ": dict(score["theta_by_succ"]),
        "stage_cost_value": float(score["stage_cost"]),
        "cost_star_value": float(score["objective"]),
    }


def score_forward_stage2_policy_from_archive(
    prob_data,
    node,
    x_star_dict,
    cut_lag,
):
    """Score a policy against learned cuts and always-explicit RouteCuts."""
    successor_to_position = {
        successor: position
        for position, successor in enumerate(node.successor)
    }
    cuts_payload = exact_sub.build_s2_bp_cuts(
        prob_data,
        node,
        cut_lag,
        successor_to_position,
    )
    return score_forward_stage2_policy(
        prob_data,
        node,
        x_star_dict,
        cuts_payload,
    )


def all_outsource_forward_solution(prob_data, node, cut_lag) -> Dict[str, Any]:
    """Construct and score the always-feasible all-outsource policy."""
    x_star_dict: Dict[str, float] = {}
    for vehicle in prob_data.V:
        x_star_dict[f"y[{vehicle}]"] = 0.0
    for customer in prob_data.J:
        for vehicle in prob_data.V:
            x_star_dict[f"alpha[{customer},{vehicle}]"] = 0.0

    successor = list(node.successor)
    successor_to_position = {
        successor_node: position
        for position, successor_node in enumerate(successor)
    }
    # Slopes vanish at the all-zero assignment, so parsing the archive on an
    # empty projected fleet is both exact and avoids needless capacity data.
    learned_payload = exact_sub.build_s2_bp_learned_cuts(
        prob_data,
        cut_lag,
        successor_to_position,
        vehicles=[],
    )
    score = _score_forward_s2_policy_exact(
        c_out=[node.c_out[customer] for customer in prob_data.J],
        outsourced=[1 for _customer in prob_data.J],
        successor=successor,
        theta_lower_bound=exact_sub._S2_BPC_THETA_LOWER_BOUND,
        cuts_payload=learned_payload,
        y=[],
        alpha=[],
    )
    return {
        "ok": True,
        "V": score["objective"],
        "cost_star_value": score["objective"],
        "stage_cost_value": score["stage_cost"],
        "theta_by_succ": score["theta_by_succ"],
        "x_star_dict": x_star_dict,
        "timed_out": False,
        "exact": True,
        "optimality_certified": True,
        "ub_certified": True,
        "solve_status": "optimal_certified",
        "backend": "forward_s2_all_outsource",
        "t_solve": 0.0,
        "num_cuts": len(learned_payload),
        "num_cuts_raw": len(learned_payload),
    }
