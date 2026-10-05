"""Cheap Stage-1 separation from already certified fixed-fleet bounds.

An improving hard-table cut or a proof of nonseparation can bypass Level
Set. Otherwise the ordinary oracle still refines its bounds. Neither shortcut
changes the physical piece domain or the Level-Set bundle.
"""

from __future__ import annotations

import math
import os
from fractions import Fraction

from .budget_fleet_domain import build_budget_fleet_domain
from .piece_table import FleetPieceTable
from s2forward.fleet_pieces import FleetLayout, float_down, float_up


def table_support_enabled() -> bool:
    return os.environ.get("VRP_PHASE2_S2_TABLE_SUPPORT", "1").lower() not in (
        "0", "false", "off", "",
    )


def budget_domain_fingerprint(prob_data, period):
    layout = FleetLayout(prob_data)
    return build_budget_fleet_domain(prob_data, period, layout).fingerprint


def stage2_cut_envelope(cuts, trial_z, *, lower=False):
    """Upward endpoint for separation; downward endpoint for nonseparation."""
    try:
        value = max((
            Fraction.from_float(float(intercept)) + sum(
                (Fraction.from_float(float(coefficient))
                 * Fraction.from_float(float(trial_z[key]))
                 for key, coefficient in slope.items()), Fraction(0),
            ) for slope, intercept in cuts
        ), default=Fraction(0))
        return (float_down if lower else float_up)(max(Fraction(0), value))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _validate_table_policy_bounds(prob_data, node, cut_lag, table, seed_policies):
    """Check every known policy before using any fleet's bound in a support.

    Stored policies must still be feasible under the same physical model.
    Invalid optional seeds are ignored, as by the regular fleet oracle.  A
    bound conflict is a RuntimeError, not an invalid-policy fallback, and
    propagates without altering the table or the existing cut archive.
    """
    from cuts import exact_subroutines as exact_sub
    from .piece_solver import certify_policy_dict

    witnesses = [
        (policy.as_x_dict(), f"table support stored policy ({policy.source})", True)
        for policy in table.policies
    ]
    witnesses.extend((seed, f"table support seed {index}", False)
                     for index, seed in enumerate(seed_policies or ()))
    if not witnesses:
        return
    payload = exact_sub.build_s2_bp_cuts(
        prob_data, node, cut_lag,
        {successor: pos for pos, successor in enumerate(node.successor)},
    )
    for decisions, source, required in witnesses:
        try:
            policy, upper = certify_policy_dict(
                prob_data, node, payload, decisions, table.layout,
                binary_tolerance=0.0, source=source,
            )
        except exact_sub.InvalidS2LagrangianPolicy:
            if required:
                raise
            continue
        table.assert_policy_upper_bound(policy.counts, upper, source=source)


def improving_table_cut(prob_data, node, cut_lag, state, trial_z):
    """Return a certified cut strictly above the current eta, or ``None``.

    The caller must copy this Stage-1 cut only to nodes with the same budget
    domain. A restricted-domain intercept must not enter an unrestricted
    Lagrangian oracle or its bundle as a value of D(pi).
    """
    if not table_support_enabled() or not state or not state.get("table_state"):
        return None
    try:
        eta = float(state["trial_eta"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(eta):
        return None
    from .fleet_support_cut import build_fleet_support_cut

    layout = FleetLayout(prob_data)
    table = FleetPieceTable.from_state(layout, state["table_state"])
    _validate_table_policy_bounds(
        prob_data, node, cut_lag, table, state.get("seed_policies"),
    )
    # On easy exact pieces the Level-Set slope produced fewer outer rounds
    # than the minimum-L1 table slope. Use the latter only after observed
    # time-limited refinement, not merely because the table exists.
    if not any(record.last_status == "grb_9" or record.stalled_at is not None
               for record in table.records.values()):
        return None
    try:
        flags = {
            vehicle: float(trial_z[f"z[{vehicle},{int(node.info[1])}]"])
            for vehicle in layout.vehicles
        }
        if any(value not in (0.0, 1.0) for value in flags.values()):
            return None
        trial = layout.counts_from_vehicle_flags(flags)
        if Fraction(float(table.lower_bound(trial))) <= Fraction(eta) + Fraction(1e-6):
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    domain = build_budget_fleet_domain(prob_data, int(node.info[1]), layout)
    candidate = build_fleet_support_cut(
        table, trial_z, int(node.info[1]), allowed_counts=domain.allowed_counts,
    )
    if candidate is None:
        return None
    pi, intercept = candidate["pi_value"], candidate["intercept"]
    score = Fraction.from_float(float(intercept)) + sum(
        (Fraction.from_float(float(value)) * Fraction.from_float(float(trial_z[key]))
         for key, value in pi.items()), Fraction(0),
    )
    if score <= Fraction.from_float(eta) + Fraction.from_float(1e-6):
        return None
    return dict(candidate, previous_eta=eta, budget_domain=domain.fingerprint)


def nonseparating_trial_policy(prob_data, node, cut_lag, state, trial_z):
    """Prove no current-archive Lagrangian cut can separate this master point.

    A freshly certified policy gives C(trial) <= U <= eta. Weak duality then
    gives D(pi) + pi*trial <= eta for every pi. This says nothing about true
    downstream routing costs or the next, strengthened Stage-3 archive.
    """
    if not table_support_enabled() or not state:
        return None
    try:
        eta = float(state.get("trial_eta_lower", state["trial_eta"]))
        if not math.isfinite(eta):
            return None
        layout = FleetLayout(prob_data)
        flags = {v: float(trial_z[f"z[{v},{int(node.info[1])}]"])
                 for v in layout.vehicles}
        if any(value not in (0.0, 1.0) for value in flags.values()):
            return None
        if not layout.is_leading_run(flags):
            return None
        trial = layout.counts_from_vehicle_flags(flags)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    from cuts import exact_subroutines as exact_sub
    from .piece_solver import certify_policy_dict

    payload = None
    for seed in state.get("seed_policies") or ():
        if payload is None:
            payload = exact_sub.build_s2_bp_cuts(
                prob_data, node, cut_lag,
                {successor: pos for pos, successor in enumerate(node.successor)},
            )
        try:
            policy, upper = certify_policy_dict(
                prob_data, node, payload, seed, layout,
                binary_tolerance=0.0, source="separation_check",
            )
        except exact_sub.InvalidS2LagrangianPolicy:
            continue
        if layout.dominates(trial, policy.counts) and upper <= eta:
            return {"policy_upper": upper, "trial_eta": eta}
    return None
