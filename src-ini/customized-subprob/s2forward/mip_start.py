"""Exact-safe reuse of a Stage-2 forward policy after cut-pool growth.

If the current archive is an exact append-only extension of the previous one,
its represented objective ``F_new`` satisfies ``F_new(x) >= F_old(x)`` for
every policy.  Hence a previously certified policy can be returned without a
new solve when its exact binary64 score did not move: its old certified lower
bound and unchanged upper bound still bracket the new optimum.  The same
predicate also gates a cheaper Gurobi MIP start when no reusable solve
certificate is available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

from cuts import exact_subroutines as exact_sub
from solvers.forward_policy_certification import certify_stage2_forward_policy


@dataclass(frozen=True)
class ForwardStartDecision:
    installed: int
    reason: str
    old_score: float | None = None
    new_score: float | None = None
    uplift: float | None = None
    previous_optimality_certified: bool = False
    still_optimal: bool = False


@dataclass(frozen=True)
class ForwardReuseDecision:
    reused: bool
    reason: str
    policy: Mapping[str, float] | None = None
    old_score: float | None = None
    new_score: float | None = None
    uplift: float | None = None
    preserved_lower_bound: float | None = None
    preserved_upper_bound: float | None = None
    preserved_relative_gap: float | None = None
    still_exact_optimal: bool = False


@dataclass(frozen=True)
class _PolicyTransition:
    reason: str
    policy: Mapping[str, float] | None = None
    old_exact: Fraction | None = None
    new_exact: Fraction | None = None


def _finite_fraction(value, *, label: str) -> Fraction:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return Fraction.from_float(number)


def _cut_signature(cut):
    slope, intercept = cut
    return (
        tuple(sorted(
            (str(name), _finite_fraction(value, label=f"cut coefficient {name}"))
            for name, value in slope.items()
        )),
        _finite_fraction(intercept, label="cut intercept"),
    )


def archive_is_prefix(previous_cut_lag, current_cut_lag, successors) -> bool:
    """Whether every relevant previous cut list is an exact current prefix."""
    previous = previous_cut_lag.get(3, {}) or {}
    current = current_cut_lag.get(3, {}) or {}
    for successor in successors:
        old_pool = list(previous.get(successor, ()))
        new_pool = list(current.get(successor, ()))
        if len(new_pool) < len(old_pool):
            return False
        if any(
            _cut_signature(old) != _cut_signature(new)
            for old, new in zip(old_pool, new_pool)
        ):
            return False
    return True


def _same_fixed_fleet(prob_data, node, previous_x_prev, current_x_prev) -> bool:
    period = int(node.info[1])
    for vehicle in prob_data.V:
        name = f"z[{vehicle},{period}]"
        old = float(previous_x_prev.get(name, 0.0)) > 0.5
        new = float(current_x_prev.get(name, 0.0)) > 0.5
        if old != new:
            return False
    return True


def exact_forward_policy_score(prob_data, node, cut_lag, policy) -> Fraction:
    """Exact rational score over the represented binary64 cut coefficients."""
    customers = list(prob_data.J)
    vehicles = list(prob_data.V)
    successor_to_position = {
        successor: position for position, successor in enumerate(node.successor)
    }
    payload = exact_sub.build_s2_bp_cuts(
        prob_data, node, cut_lag, successor_to_position
    )
    y = [int(float(policy[f"y[{vehicle}]"]) > 0.5) for vehicle in vehicles]
    alpha = [
        [
            int(float(policy[f"alpha[{customer},{vehicle}]"]) > 0.5)
            for customer in customers
        ]
        for vehicle in vehicles
    ]
    compiled = exact_sub.compiled_cut_payload(
        payload, len(vehicles), len(customers), error=ValueError
    )
    theta = compiled.exact_theta(
        y, alpha, Fraction(0), len(node.successor)
    )
    stage_cost = sum(
        (
            _finite_fraction(node.c_out[customer], label=f"c_out[{customer}]")
            for customer_position, customer in enumerate(customers)
            if sum(alpha[vehicle_position][customer_position]
                   for vehicle_position in range(len(vehicles))) == 0
        ),
        Fraction(0),
    )
    return stage_cost + sum(theta, Fraction(0))


def _analyze_policy_transition(
    prob_data,
    node,
    *,
    policy,
    previous_x_prev,
    current_x_prev,
    previous_cut_lag,
    current_cut_lag,
) -> _PolicyTransition:
    if not _same_fixed_fleet(
        prob_data, node, previous_x_prev, current_x_prev
    ):
        return _PolicyTransition("fleet_changed")
    if not archive_is_prefix(
        previous_cut_lag, current_cut_lag, node.successor
    ):
        return _PolicyTransition("archive_not_prefix")

    certified, _ = certify_stage2_forward_policy(
        prob_data, node, current_x_prev, policy
    )
    old_exact = exact_forward_policy_score(
        prob_data, node, previous_cut_lag, certified
    )
    new_exact = exact_forward_policy_score(
        prob_data, node, current_cut_lag, certified
    )
    return _PolicyTransition(
        "unchanged_policy_score" if new_exact == old_exact else "archive_uplift",
        dict(certified),
        old_exact,
        new_exact,
    )


def certify_unchanged_policy_reuse(
    prob_data,
    node,
    *,
    policy: Mapping[str, object],
    previous_x_prev,
    current_x_prev,
    previous_cut_lag,
    current_cut_lag,
    previous_exact_optimal: bool = False,
    previous_status_optimal: bool = False,
    previous_objective_lower_bound: float | None = None,
) -> ForwardReuseDecision:
    """Certify that ``policy`` may replace a new Stage-2 solve.

    ``previous_exact_optimal`` is for a proof-producing exact solver such as
    subset DP.  A Gurobi ``OPTIMAL`` result with a nonzero requested MIP gap is
    also accepted, but only together with its *certified* directed lower
    bound.  In that case the returned policy preserves the old certified gap;
    it is deliberately not labelled exact-optimal.
    """
    transition = _analyze_policy_transition(
        prob_data,
        node,
        policy=policy,
        previous_x_prev=previous_x_prev,
        current_x_prev=current_x_prev,
        previous_cut_lag=previous_cut_lag,
        current_cut_lag=current_cut_lag,
    )
    if transition.policy is None:
        return ForwardReuseDecision(False, transition.reason)

    old_exact = transition.old_exact
    new_exact = transition.new_exact
    assert old_exact is not None and new_exact is not None
    old_score = float(old_exact)
    new_score = float(new_exact)
    uplift = float(new_exact - old_exact)
    if transition.reason != "unchanged_policy_score":
        return ForwardReuseDecision(
            False,
            transition.reason,
            old_score=old_score,
            new_score=new_score,
            uplift=uplift,
        )

    if previous_exact_optimal:
        lower_bound = new_score
        still_exact = True
    elif previous_status_optimal:
        if previous_objective_lower_bound is None:
            return ForwardReuseDecision(
                False,
                "missing_previous_lower_bound",
                old_score=old_score,
                new_score=new_score,
                uplift=uplift,
            )
        try:
            lower_bound = float(previous_objective_lower_bound)
        except (TypeError, ValueError, OverflowError):
            lower_bound = float("nan")
        if (
            not math.isfinite(lower_bound)
            or Fraction.from_float(lower_bound) > old_exact
        ):
            return ForwardReuseDecision(
                False,
                "invalid_previous_lower_bound",
                old_score=old_score,
                new_score=new_score,
                uplift=uplift,
            )
        still_exact = Fraction.from_float(lower_bound) == new_exact
    else:
        return ForwardReuseDecision(
            False,
            "previous_solve_not_certified",
            old_score=old_score,
            new_score=new_score,
            uplift=uplift,
        )

    denominator = max(abs(new_score), 1.0)
    relative_gap = max(0.0, new_score - lower_bound) / denominator
    return ForwardReuseDecision(
        True,
        "unchanged_certified_policy",
        dict(transition.policy),
        old_score,
        new_score,
        uplift,
        lower_bound,
        new_score,
        relative_gap,
        still_exact,
    )


def apply_forward_binary_start(model, prob_data, node, x_prev, policy) -> int:
    """Physically certify and install alpha/y/s starts present in ``model``."""
    certified, _ = certify_stage2_forward_policy(
        prob_data, node, x_prev, policy
    )
    model.update()
    installed = 0
    for name, value in certified.items():
        variable = model.getVarByName(name)
        selected = float(value) > 0.5
        if variable is None:
            if selected:
                raise ValueError(f"model omitted selected start variable {name}")
            continue
        variable.Start = float(selected)
        installed += 1
    for customer in prob_data.J:
        variable = model.getVarByName(f"s[{customer}]")
        if variable is None:
            continue
        assigned = sum(
            int(certified[f"alpha[{customer},{vehicle}]"] > 0.5)
            for vehicle in prob_data.V
        )
        variable.Start = float(1 - assigned)
        installed += 1
    # Gurobi batches Start-attribute updates; make them observable before the
    # caller optimizes or inspects the model.
    model.update()
    return installed


def apply_unchanged_exact_policy_start(
    model,
    prob_data,
    node,
    *,
    policy: Mapping[str, object],
    previous_x_prev,
    current_x_prev,
    previous_cut_lag,
    current_cut_lag,
    previous_optimality_certified: bool = False,
) -> ForwardStartDecision:
    """Install when an append-only archive leaves a certified policy unchanged.

    For an archive extension every policy objective can only increase.  If a
    previously exact-optimal policy has exactly the same rational score under
    the extended archive, it is still globally optimal.  A merely feasible
    prior incumbent is still a safe MIP start, but is not labelled
    optimal.  Comparing Fractions eliminates scale-dependent roundoff.
    """
    transition = _analyze_policy_transition(
        prob_data,
        node,
        policy=policy,
        previous_x_prev=previous_x_prev,
        current_x_prev=current_x_prev,
        previous_cut_lag=previous_cut_lag,
        current_cut_lag=current_cut_lag,
    )
    if transition.policy is None:
        return ForwardStartDecision(
            0,
            transition.reason,
            previous_optimality_certified=previous_optimality_certified,
        )
    old_exact = transition.old_exact
    new_exact = transition.new_exact
    assert old_exact is not None and new_exact is not None
    old_score = float(old_exact)
    new_score = float(new_exact)
    uplift = float(new_exact - old_exact)
    if transition.reason != "unchanged_policy_score":
        return ForwardStartDecision(
            0,
            "archive_uplift",
            old_score,
            new_score,
            uplift,
            previous_optimality_certified,
            False,
        )
    installed = apply_forward_binary_start(
        model, prob_data, node, current_x_prev, transition.policy
    )
    return ForwardStartDecision(
        installed,
        "unchanged_policy_score",
        old_score,
        new_score,
        uplift,
        previous_optimality_certified,
        bool(previous_optimality_certified),
    )
