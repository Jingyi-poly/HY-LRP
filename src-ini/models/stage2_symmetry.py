"""Stage-2 canonical ordering for same-type vehicles.

Vehicles inside one ``prob_data.V_k[k]`` group are physically identical, so any
permutation of their assignments is another solution of the same cost.  These
rows keep only the representative whose assignment scores are non-increasing
along the group, in ``prob_data.V_k[k]`` order -- the canonical order that
``customized-subprob/s2forward`` builds its leading-run fleet pieces on.

The same canonical domain is used in forward and backward models.  Private
Stage-3 cut pools can make the temporary surrogate asymmetric, but each pool
is still a pointwise lower bound on the true routing value.  Therefore the
minimum over canonical assignments remains below true physical recourse; the
surrogate itself does not need to be permutation invariant.

Backward oracles additionally keep their free copy of the Stage-1 purchase
state in this rank order, so each same-type ``z`` vector is a leading prefix.
"""
from fractions import Fraction

import gurobipy as gp
import numpy as np


def assignment_score_weight(customer):
    """Lexicographic weight of ``customer``; ``log(j+2)`` keeps ``j=0`` nonzero."""
    return np.round(np.log(customer + 2), 4)


def exact_assignment_score(customers, assignment_row):
    """Exact score of one binary row over the represented float weights."""
    if len(customers) != len(assignment_row):
        raise ValueError("assignment row/customer shape mismatch")
    return sum(
        (
            Fraction.from_float(float(assignment_score_weight(customer)))
            for customer, assigned in zip(customers, assignment_row)
            if int(assigned) == 1
        ),
        Fraction(0),
    )


def canonicalize_assignment_rows(prob_data, vehicles, alpha_rows, y_values):
    """Sort same-type assignment rows into the model's canonical rank order.

    ``vehicles`` is the explicit row order (for forward BPC it contains only
    purchased vehicles). Rows and activation bits move together. The sort is
    stable on equal scores because the model imposes only a non-strict order.
    """
    vehicles = list(vehicles)
    customers = list(prob_data.J)
    rows = [list(row) for row in alpha_rows]
    activations = list(y_values)
    if len(rows) != len(vehicles) or len(activations) != len(vehicles):
        raise ValueError("vehicle/assignment shape mismatch")
    if any(len(row) != len(customers) for row in rows):
        raise ValueError("customer/assignment shape mismatch")

    try:
        groups = [list(prob_data.V_k[k]) for k in prob_data.K]
    except (AttributeError, KeyError, TypeError):
        # Tiny external fixtures without type metadata declare no symmetry.
        return rows, activations, False

    position = {vehicle: pos for pos, vehicle in enumerate(vehicles)}
    if len(position) != len(vehicles):
        raise ValueError("duplicate vehicle identifier")
    changed = False
    for group in groups:
        group_positions = [position[v] for v in group if v in position]
        if len(group_positions) < 2:
            continue
        ranked = [
            (
                exact_assignment_score(customers, rows[pos]),
                original_order,
                list(rows[pos]),
                activations[pos],
            )
            for original_order, pos in enumerate(group_positions)
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)
        for target_pos, (_score, _order, row, activation) in zip(
            group_positions, ranked
        ):
            if rows[target_pos] != row or activations[target_pos] != activation:
                changed = True
            rows[target_pos] = row
            activations[target_pos] = activation
    return rows, activations, changed


def assignment_order_holds(prob_data, vehicles, alpha_rows):
    """Whether binary rows satisfy every applicable same-type score order."""
    vehicles = list(vehicles)
    customers = list(prob_data.J)
    rows = [list(row) for row in alpha_rows]
    try:
        groups = [list(prob_data.V_k[k]) for k in prob_data.K]
    except (AttributeError, KeyError, TypeError):
        return True
    position = {vehicle: pos for pos, vehicle in enumerate(vehicles)}
    for group in groups:
        group_positions = [position[v] for v in group if v in position]
        scores = [
            exact_assignment_score(customers, rows[pos])
            for pos in group_positions
        ]
        if any(left < right for left, right in zip(scores, scores[1:])):
            return False
    return True


def add_stage2_symmetry_rows(model, prob_data, alpha, y=None):
    """Add ``assignment_order`` (and ``activation_order`` when ``y`` is given).

    ``alpha`` is indexed ``[customer, vehicle]`` and ``y`` by vehicle.
    """
    model.addConstrs(
        (gp.quicksum(alpha[j, prob_data.V_k[k][v_ind]] * assignment_score_weight(j)
                     for j in prob_data.J) >=
         gp.quicksum(alpha[j, prob_data.V_k[k][v_ind + 1]] * assignment_score_weight(j)
                     for j in prob_data.J)
         for k in prob_data.K
         for v_ind in range(len(prob_data.V_k[k]) - 1)),
        name="assignment_order"
    )
    if y is None:
        return
    model.addConstrs(
        (y[prob_data.V_k[k][v_ind]] >= y[prob_data.V_k[k][v_ind + 1]]
         for k in prob_data.K
         for v_ind in range(len(prob_data.V_k[k]) - 1)),
        name="activation_order"
    )


def add_stage2_purchase_order_rows(model, prob_data, z):
    """Restrict each same-type purchase vector to its canonical prefix.

    Stage 1 purchases interchangeable vehicles in ``V_k[k]`` order.  A
    backward Stage-2 oracle makes its copied purchase variables free, so it
    must impose that domain explicitly: ``z[r] >= z[r+1]``.
    """
    model.addConstrs(
        (z[prob_data.V_k[k][v_ind]] >= z[prob_data.V_k[k][v_ind + 1]]
         for k in prob_data.K
         for v_ind in range(len(prob_data.V_k[k]) - 1)),
        name="purchase_order"
    )
