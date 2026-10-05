"""Exact free-fleet Stage-2 Lagrangian oracle by subset dynamic programming."""

from __future__ import annotations

import math
import time
from typing import Mapping, Optional

from cuts import exact_subroutines as exact_sub
from s2forward.fleet_pieces import FleetLayout
from s2forward.subset_dp import SubsetDPPieceSolver, default_max_customers


def _dense_policy(prob_data, node, sparse):
    vehicles = list(prob_data.V)
    customers = list(prob_data.J)
    period = int(node.info[1])
    return {
        "z": [int(sparse.get(f"z[{vehicle},{period}]", 0)) for vehicle in vehicles],
        "y": [int(sparse.get(f"y[{vehicle}]", 0)) for vehicle in vehicles],
        "alpha": [
            [int(sparse.get(f"alpha[{customer},{vehicle}]", 0)) for customer in customers]
            for vehicle in vehicles
        ],
    }


class DirectDPOracle:
    """One fixed-node/archive session with reusable subset and theta tables."""

    def __init__(
        self,
        prob_data,
        node,
        cut_lag,
        *,
        max_customers: Optional[int] = None,
        exact_abs_tol: float = 1e-6,
        exact_rel_tol: float = 0.0,
        piece_solver: Optional[SubsetDPPieceSolver] = None,
    ):
        started = time.perf_counter()
        self.prob_data = prob_data
        self.node = node
        self.cut_lag = cut_lag
        self.vehicles = list(prob_data.V)
        successors = list(node.successor)
        self.cuts_payload = exact_sub.build_s2_bp_cuts(
            prob_data,
            node,
            cut_lag,
            {successor: position for position, successor in enumerate(successors)},
        )
        self.layout = FleetLayout(prob_data)
        self.exact_abs_tol = float(exact_abs_tol)
        self.exact_rel_tol = float(exact_rel_tol)
        if piece_solver is None:
            self.__solver = SubsetDPPieceSolver(
                prob_data,
                node,
                self.cuts_payload,
                self.layout,
                max_customers=(
                    default_max_customers()
                    if max_customers is None
                    else int(max_customers)
                ),
            )
        else:
            if (
                piece_solver.prob_data is not prob_data
                or piece_solver.node is not node
                or piece_solver.cuts_payload != self.cuts_payload
                or piece_solver.layout.groups != self.layout.groups
            ):
                raise ValueError("piece_solver does not match the node/archive")
            self.__solver = piece_solver
        self.initialization_seconds = time.perf_counter() - started
        self.evaluations = 0

    def evaluate(self, pi_value: Mapping[str, float]) -> dict:
        started = time.perf_counter()
        raw = self.__solver.solve_lagrangian(pi_value)
        dense = _dense_policy(self.prob_data, self.node, raw["x_dict"])
        certified = exact_sub.certify_s2_lagrangian_policy(
            self.prob_data,
            self.node,
            self.cut_lag,
            pi_value,
            dense,
            binary_tolerance=0.0,
            cuts_payload=self.cuts_payload,
            require_assignment_order=True,
            require_purchase_order=True,
        )
        position = {vehicle: index for index, vehicle in enumerate(self.vehicles)}
        for group in self.layout.groups:
            flags = [dense["y"][position[vehicle]] for vehicle in group]
            if any(left < right for left, right in zip(flags, flags[1:])):
                raise RuntimeError("subset DP policy violates activation order")

        incumbent = float(certified["V"])
        lower_bound = min(float(raw["lb"]), incumbent)
        if not math.isfinite(lower_bound):
            raise RuntimeError("subset DP returned a non-finite lower bound")
        exact_tolerance = self.exact_abs_tol + self.exact_rel_tol * max(
            1.0, abs(incumbent)
        )
        exact = incumbent - lower_bound <= exact_tolerance
        self.evaluations += 1
        elapsed = time.perf_counter() - started
        return {
            "ok": True,
            "V": incumbent,
            "lb": lower_bound,
            "inner_value": incumbent,
            "outer_lb": lower_bound,
            "lb_certified": True,
            "ub_certified": True,
            "optimality_certified": exact,
            "exact": exact,
            "xcp": dict(certified["xcp"]),
            "policy": dense,
            "theta": list(certified["theta"]),
            "status": raw["status"],
            "t_solve": elapsed,
            "dp_seconds": float(raw["seconds"]),
            "raw_value": float(raw["value"]),
            "initialization_seconds": self.initialization_seconds,
            "evaluation": self.evaluations,
        }


def solve_lagrangian_by_dp(
    prob_data,
    node,
    cut_lag,
    pi_value: Mapping[str, float],
    *,
    max_customers: Optional[int] = None,
) -> dict:
    """One-shot convenience wrapper around :class:`DirectDPOracle`."""
    started = time.perf_counter()
    oracle = DirectDPOracle(
        prob_data, node, cut_lag, max_customers=max_customers
    )
    result = oracle.evaluate(pi_value)
    result["t_solve"] = time.perf_counter() - started
    return result


__all__ = ["DirectDPOracle", "solve_lagrangian_by_dp"]
