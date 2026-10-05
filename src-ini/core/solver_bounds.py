"""Helpers for keeping solver incumbents separate from certified bounds."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from fractions import Fraction
from typing import Any


# Gurobi statuses where ObjBound is not a usable certificate (no gurobipy import).
_GUROBI_INVALID_BOUND_STATUSES = frozenset({
    1,   # LOADED
    3,   # INFEASIBLE
    4,   # INF_OR_UNBD
    5,   # UNBOUNDED
    12,  # NUMERIC
})
_GUROBI_INFINITY = 1e100
_GUROBI_OPTIMAL = 2
_DUAL_RESIDUAL_REPAIR_TOL = 1e-6


def _finite_box_endpoint(value: float) -> bool:
    return math.isfinite(value) and abs(value) < 0.5 * _GUROBI_INFINITY


def _fraction_to_finite_float_down(value: Fraction, *, label: str) -> float:
    """Return a finite binary64 value no greater than an exact rational."""
    try:
        rounded = float(value)
    except OverflowError as exc:
        if value > 0:
            return math.nextafter(math.inf, 0.0)
        raise ValueError(f"{label} is below the finite binary64 range") from exc
    if Fraction.from_float(rounded) > value:
        rounded = math.nextafter(rounded, -math.inf)
    if not math.isfinite(rounded):
        raise ValueError(f"{label} is non-finite after directed rounding")
    return rounded


def _repair_unbounded_dual_residual(
    index, reduced, bounds, candidates, slope, intercept, *, label,
):
    """Repair a tiny residual without changing any other unbounded column.

    Each candidate row touches just this one unbounded column. Directed
    rounding preserves its required residual sign; changes on finite boxes
    are charged to the intercept by the caller after all repairs finish.
    None means no small, sign-feasible repair exists.
    """
    residual = reduced[index]
    lower_finite, upper_finite = map(_finite_box_endpoint, bounds)
    if ((residual == 0) or (residual > 0 and lower_finite)
            or (residual < 0 and upper_finite)):
        return intercept
    tolerance = Fraction.from_float(_DUAL_RESIDUAL_REPAIR_TOL)
    if abs(residual) > tolerance:
        return None

    choices = []
    for terms, multiplier, sense, rhs, linking_variable in candidates:
        coefficient = terms[index]
        target = multiplier + residual / coefficient
        # Upper-unbounded requires r >= 0; lower-unbounded requires r <= 0.
        round_down = ((not upper_finite and coefficient > 0)
                      or (upper_finite and coefficient < 0))
        try:
            new_float = (_fraction_to_finite_float_down(target, label=label)
                         if round_down else
                         -_fraction_to_finite_float_down(-target, label=label))
        except (OverflowError, ValueError):
            continue
        new_multiplier = Fraction.from_float(new_float)
        if ((sense == "<" and new_multiplier > 0)
                or (sense == ">" and new_multiplier < 0)):
            continue
        change = new_multiplier - multiplier
        if abs(change) > tolerance:
            continue
        new_residual = residual - coefficient * change
        if ((not upper_finite and new_residual < 0)
                or (not lower_finite and new_residual > 0)):
            continue
        choices.append((linking_variable is None, abs(change), terms, change,
                        rhs, linking_variable, new_float))
    if not choices:
        return None
    _, _, terms, change, rhs, linking_variable, new_float = min(
        choices, key=lambda item: item[:2],
    )
    for column, coefficient in terms.items():
        reduced[column] -= coefficient * change
    if linking_variable is None:
        intercept += rhs * change
    else:
        slope[linking_variable] = new_float
    return intercept


def extract_optimal_fixed_rhs_dual_cut(
    model: Any,
    bindings: Iterable[tuple[str, str]],
    *,
    label: str,
) -> tuple[dict[str, float], float]:
    """Extract a Benders slope/intercept from an optimal fixed-state LP.

    Anchor at the linking equalities' actual RHS in ``model``.
    ``bindings``: ``(cut_variable_name, linking_constraint_name)`` pairs;
    duals and anchor come from the solved model only (no caller ``x_prev``).
    """
    status = int(getattr(model, "Status", -1))
    if status != _GUROBI_OPTIMAL:
        raise ValueError(
            f"{label} LP relaxation is not optimal; status={status}"
        )

    try:
        objective = float(model.ObjVal)
    except Exception as exc:
        raise ValueError(f"{label} LP has no readable objective") from exc
    if not math.isfinite(objective):
        raise ValueError(f"{label} LP has non-finite objective: {objective!r}")

    slope: dict[str, float] = {}
    linking_constraints: set[str] = set()
    anchor_exact = Fraction(0)
    for raw_variable_name, raw_constraint_name in bindings:
        variable_name = str(raw_variable_name)
        constraint_name = str(raw_constraint_name)
        if variable_name in slope:
            raise ValueError(
                f"{label} LP has duplicate cut variable {variable_name!r}"
            )
        if constraint_name in linking_constraints:
            raise ValueError(
                f"{label} LP has duplicate linking constraint "
                f"{constraint_name!r}"
            )
        linking_constraints.add(constraint_name)

        constraint = model.getConstrByName(constraint_name)
        if constraint is None:
            raise KeyError(
                f"{label} LP is missing linking constraint "
                f"{constraint_name!r}"
            )
        if str(constraint.Sense) != "=":
            raise ValueError(
                f"{label} linking constraint {constraint_name!r} "
                "is not an equality"
            )

        try:
            dual = float(constraint.Pi)
            rhs = float(constraint.RHS)
        except Exception as exc:
            raise ValueError(
                f"{label} linking constraint {constraint_name!r} "
                "has unreadable dual/RHS"
            ) from exc
        if not math.isfinite(dual) or not math.isfinite(rhs):
            raise ValueError(
                f"{label} linking constraint {constraint_name!r} has "
                f"non-finite dual/RHS: Pi={dual!r}, RHS={rhs!r}"
            )

        slope[variable_name] = dual
        anchor_exact += Fraction.from_float(dual) * Fraction.from_float(rhs)

    intercept_exact = Fraction.from_float(objective) - anchor_exact
    intercept = _fraction_to_finite_float_down(
        intercept_exact,
        label=f"{label} LP cut intercept",
    )
    return slope, intercept


def extract_verified_fixed_rhs_dual_cut(
    model: Any,
    bindings: Iterable[tuple[str, str]],
    *,
    label: str,
) -> tuple[dict[str, float], float]:
    """Return an exact-residual-valid parametric cut from a solved LP.

    Let the named linking equalities be ``E x = b``.  Their reported duals
    provide the cut slope. Every other row multiplier is projected onto its
    exact sign cone. Tiny (<=1e-6) invalid unbounded-direction residuals are
    repaired by local, directed-rounded multiplier adjustments; a repair
    touching a linking row also updates the returned slope. The complete
    residual is then minimized over each variable's represented box. The
    tolerance permits repair, never omission of a residual. Consequently the
    returned binary64 row

    ``value(b) >= intercept + sum(slope[name] * b[name])``

    is valid for every right-hand side represented by the linking variables,
    even when Gurobi's floating-point dual has small feasibility residuals.
    The final intercept is rounded once toward ``-inf``.
    """
    if int(getattr(model, "Status", -1)) != _GUROBI_OPTIMAL:
        raise ValueError(
            f"{label} LP relaxation is not optimal; "
            f"status={getattr(model, 'Status', -1)}"
        )
    try:
        if int(model.ModelSense) != 1 or int(model.IsMIP) != 0:
            raise ValueError(f"{label} must be a continuous minimization LP")
        unsupported_counts = (
            int(getattr(model, "NumQNZs", 0)),
            int(getattr(model, "NumQConstrs", 0)),
            int(getattr(model, "NumGenConstrs", 0)),
            int(getattr(model, "NumSOS", 0)),
            int(getattr(model, "NumPWLObjVars", 0)),
            max(0, int(getattr(model, "NumObj", 1)) - 1),
            int(getattr(model, "NumScenarios", 0)),
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{label} has unreadable model structure") from exc
    if any(unsupported_counts):
        raise ValueError(f"{label} contains unsupported non-linear structure")

    # Every coefficient below is an exact binary64 dyadic.  Routing LPs reuse
    # a small set of values (especially 0, +/-1 and one demand per incoming
    # arc), so converting the same float to Fraction thousands of times is a
    # substantial avoidable cost. This call-local cache preserves the exact
    # arithmetic while avoiding repeated conversions.
    dyadic_cache: dict[float, Fraction] = {}

    def as_dyadic(value: float) -> Fraction:
        binary64 = float(value)
        try:
            return dyadic_cache[binary64]
        except KeyError:
            exact = Fraction.from_float(binary64)
            dyadic_cache[binary64] = exact
            return exact

    def attribute_values(attribute: str, objects: list[Any]) -> list[Any]:
        """Bulk-read a Gurobi attribute, with a fake-model-safe fallback."""
        get_attribute = getattr(model, "getAttr", None)
        if callable(get_attribute):
            try:
                values = list(get_attribute(attribute, objects))
                if len(values) == len(objects):
                    return values
            except Exception:
                # Unit-test fakes and solver-compatible wrappers need not
                # implement Gurobi's list-valued getAttr overload.
                pass
        return [getattr(item, attribute) for item in objects]

    binding_items = tuple(bindings)
    slope: dict[str, float] = {}
    constraint_to_variable: dict[str, str] = {}
    for raw_variable_name, raw_constraint_name in binding_items:
        variable_name = str(raw_variable_name)
        constraint_name = str(raw_constraint_name)
        if variable_name in slope:
            raise ValueError(
                f"{label} LP has duplicate cut variable {variable_name!r}"
            )
        if constraint_name in constraint_to_variable:
            raise ValueError(
                f"{label} LP has duplicate linking constraint "
                f"{constraint_name!r}"
            )
        constraint = model.getConstrByName(constraint_name)
        if constraint is None:
            raise KeyError(
                f"{label} LP is missing linking constraint "
                f"{constraint_name!r}"
            )
        if str(constraint.Sense) != "=":
            raise ValueError(
                f"{label} linking constraint {constraint_name!r} "
                "is not an equality"
            )
        try:
            multiplier = float(constraint.Pi)
            rhs = float(constraint.RHS)
        except Exception as exc:
            raise ValueError(
                f"{label} linking constraint {constraint_name!r} "
                "has unreadable dual/RHS"
            ) from exc
        if not math.isfinite(multiplier) or not math.isfinite(rhs):
            raise ValueError(
                f"{label} linking constraint {constraint_name!r} has "
                f"non-finite dual/RHS: Pi={multiplier!r}, RHS={rhs!r}"
            )
        slope[variable_name] = multiplier
        constraint_to_variable[constraint_name] = variable_name

    try:
        variables = list(model.getVars())
        constraints = list(model.getConstrs())
        objective_constant = float(getattr(model, "ObjCon", 0.0))
    except Exception as exc:
        raise ValueError(f"{label} has unreadable LP matrix") from exc
    if not math.isfinite(objective_constant):
        raise ValueError(f"{label} has non-finite objective constant")

    try:
        variable_objectives = attribute_values("Obj", variables)
        variable_lowers = attribute_values("LB", variables)
        variable_uppers = attribute_values("UB", variables)
    except Exception as exc:
        raise ValueError(f"{label} has unreadable variable data") from exc

    reduced: dict[int, Fraction] = {}
    objective_by_index: dict[int, Fraction] = {}
    variable_data: dict[int, tuple[float, float]] = {}
    for variable, raw_objective, raw_lower, raw_upper in zip(
        variables,
        variable_objectives,
        variable_lowers,
        variable_uppers,
    ):
        try:
            index = int(variable.index)
            objective = float(raw_objective)
            lower = float(raw_lower)
            upper = float(raw_upper)
        except Exception as exc:
            raise ValueError(f"{label} has unreadable variable data") from exc
        if not math.isfinite(objective) or index in reduced:
            raise ValueError(f"{label} has invalid variable data")
        objective_exact = as_dyadic(objective)
        reduced[index] = objective_exact
        objective_by_index[index] = objective_exact
        variable_data[index] = (lower, upper)

    open_columns = {
        index for index, bounds in variable_data.items()
        if not all(_finite_box_endpoint(endpoint) for endpoint in bounds)
    }
    repair_rows: dict[int, list] = {index: [] for index in open_columns}

    def objective_box_fallback() -> tuple[dict[str, float], float]:
        """Return a rigorous RHS-independent zero-multiplier certificate."""
        fallback_exact = as_dyadic(objective_constant)
        for index, coefficient in objective_by_index.items():
            if coefficient == 0:
                continue
            lower, upper = variable_data[index]
            endpoint = lower if coefficient > 0 else upper
            if (
                not math.isfinite(endpoint)
                or endpoint <= -0.5 * _GUROBI_INFINITY
                or endpoint >= 0.5 * _GUROBI_INFINITY
            ):
                raise ValueError(
                    f"{label} zero-multiplier objective box is unbounded"
                )
            fallback_exact += coefficient * as_dyadic(endpoint)
        warnings.warn(
            f"{label} floating dual is unbounded in an exact variable-box "
            "direction; using the rigorous zero-multiplier objective-box cut",
            RuntimeWarning,
            stacklevel=2,
        )
        return (
            {variable_name: 0.0 for variable_name in slope},
            _fraction_to_finite_float_down(
                fallback_exact,
                label=f"{label} zero-multiplier cut intercept",
            ),
        )

    try:
        constraint_names = attribute_values("ConstrName", constraints)
        constraint_senses = attribute_values("Sense", constraints)
        constraint_rhss = attribute_values("RHS", constraints)
        constraint_multipliers = attribute_values("Pi", constraints)
    except Exception as exc:
        raise ValueError(f"{label} has unreadable row data") from exc

    intercept_exact = as_dyadic(objective_constant)
    seen_linking: set[str] = set()
    for constraint, raw_name, raw_sense, raw_rhs, raw_multiplier in zip(
        constraints,
        constraint_names,
        constraint_senses,
        constraint_rhss,
        constraint_multipliers,
    ):
        try:
            name = str(raw_name)
            sense = str(raw_sense)
            rhs = float(raw_rhs)
            multiplier = float(raw_multiplier)
            row = model.getRow(constraint)
        except Exception as exc:
            raise ValueError(f"{label} has unreadable row data") from exc
        if not math.isfinite(rhs) or not math.isfinite(multiplier):
            raise ValueError(f"{label} has non-finite row data")
        if name in constraint_to_variable:
            if sense != "=":
                raise ValueError(f"{label} linking row changed sense")
            multiplier = slope[constraint_to_variable[name]]
            seen_linking.add(name)
        elif sense == "<":
            multiplier = min(multiplier, 0.0)
            intercept_exact += (
                as_dyadic(rhs) * as_dyadic(multiplier)
            )
        elif sense == ">":
            multiplier = max(multiplier, 0.0)
            intercept_exact += (
                as_dyadic(rhs) * as_dyadic(multiplier)
            )
        elif sense == "=":
            intercept_exact += (
                as_dyadic(rhs) * as_dyadic(multiplier)
            )
        else:
            raise ValueError(f"{label} has unsupported row sense {sense!r}")

        multiplier_exact = as_dyadic(multiplier)
        row_entries = []
        try:
            for position in range(int(row.size())):
                variable = row.getVar(position)
                coefficient = float(row.getCoeff(position))
                index = int(variable.index)
                if index not in reduced or not math.isfinite(coefficient):
                    raise ValueError
                coefficient_exact = as_dyadic(coefficient)
                reduced[index] -= coefficient_exact * multiplier_exact
                if open_columns and coefficient_exact:
                    row_entries.append((index, coefficient_exact))
        except Exception as exc:
            raise ValueError(f"{label} has invalid sparse row data") from exc
        # Only local pivots are retained: no repair may disturb another
        # unbounded column. Aggregate duplicate entries before classifying.
        open_terms: dict[int, Fraction] = {}
        for index, coefficient in row_entries:
            if index in open_columns:
                open_terms[index] = open_terms.get(index, Fraction(0)) + coefficient
        open_terms = {index: value for index, value in open_terms.items() if value}
        if len(open_terms) == 1:
            terms: dict[int, Fraction] = {}
            for index, coefficient in row_entries:
                terms[index] = terms.get(index, Fraction(0)) + coefficient
            terms = {index: value for index, value in terms.items() if value}
            index = next(iter(open_terms))
            repair_rows[index].append((
                terms, multiplier_exact, sense, as_dyadic(rhs),
                constraint_to_variable.get(name),
            ))

    if seen_linking != set(constraint_to_variable):
        missing = sorted(set(constraint_to_variable) - seen_linking)
        raise KeyError(f"{label} LP is missing linking rows in matrix: {missing}")

    # Repair first; a pivot can change residuals on finite columns that will
    # subsequently contribute to the exact intercept compensation.
    for index in open_columns:
        repaired = _repair_unbounded_dual_residual(
            index, reduced, variable_data[index], repair_rows[index],
            slope, intercept_exact, label=f"{label} dual residual repair",
        )
        if repaired is None:
            return objective_box_fallback()
        intercept_exact = repaired

    for index, coefficient in reduced.items():
        if coefficient == 0:
            continue
        lower, upper = variable_data[index]
        endpoint = lower if coefficient > 0 else upper
        if (
            not math.isfinite(endpoint)
            or endpoint <= -0.5 * _GUROBI_INFINITY
            or endpoint >= 0.5 * _GUROBI_INFINITY
        ):
            return objective_box_fallback()
        intercept_exact += coefficient * as_dyadic(endpoint)

    return slope, _fraction_to_finite_float_down(
        intercept_exact,
        label=f"{label} verified LP cut intercept",
    )


def certified_gurobi_minimization_lower_bound(model: Any) -> float | None:
    """Finite, order-consistent Gurobi minimization dual bound from ObjBound.

    Never promotes ObjVal to an LB. Any ObjBound>ObjVal contradicts the
    reported incumbent and is rejected, including a one-ULP inversion on
    OPTIMAL. Without an incumbent, a finite usable ObjBound is retained.
    """
    status = getattr(model, "Status", None)
    if status in _GUROBI_INVALID_BOUND_STATUSES:
        return None

    try:
        bound = float(model.ObjBound)
    except Exception:
        return None
    if not math.isfinite(bound) or abs(bound) >= 0.5 * _GUROBI_INFINITY:
        return None

    try:
        has_incumbent = int(getattr(model, "SolCount", 0)) > 0
    except Exception:
        has_incumbent = False
    if not has_incumbent:
        return bound

    try:
        incumbent = float(model.ObjVal)
    except Exception:
        return bound
    if not math.isfinite(incumbent):
        return bound
    if bound <= incumbent:
        return bound
    return None


def certified_minimization_lower_bound(model: Any) -> float:
    """Solver-certified minimization LB (ObjBound; ObjVal is only an UB)."""
    certified = certified_gurobi_minimization_lower_bound(model)
    return float("-inf") if certified is None else certified


def minimization_bounds_inverted(
    lower_bound: float,
    upper_bound: float,
    *,
    absolute_tolerance: float = 0.0,
) -> bool:
    """True if minimization LB strictly exceeds feasible UB (tol must be 0)."""
    tolerance = float(absolute_tolerance)
    if not math.isfinite(tolerance) or tolerance != 0.0:
        raise ValueError("exact bound inversion checks require zero tolerance")
    if not (math.isfinite(lower_bound) and math.isfinite(upper_bound)):
        return False
    return lower_bound > upper_bound


def minimization_gap_percent(lower_bound: float, upper_bound: float) -> float:
    """One-sided gap; inf if inverted."""
    if (
        not math.isfinite(lower_bound)
        or not math.isfinite(upper_bound)
        or minimization_bounds_inverted(lower_bound, upper_bound)
    ):
        return float("inf")
    # LRP permits the all-closed, zero-demand policy with objective zero.
    # max(1, |LB|) also gives a stable absolute scale close to zero.
    return max(0.0, upper_bound - lower_bound) / max(1.0, abs(lower_bound)) * 100.0
